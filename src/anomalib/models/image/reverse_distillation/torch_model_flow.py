# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reverse Distillation with a normalizing flow on the teacher-student residual (staged).

Training is staged, PatchCore-style:

    Phase 1 -- Reverse Distillation (gradient, standard RD cosine loss):
        encoder (frozen) -> bottleneck -> decoder
        loss = ReverseDistillationLoss(encoder_features, decoder_features)

    Phase 2 -- Flow (gradient, PatchFlow-style NLL), run once after Phase 1
    completes, on the now-frozen residual features:
        residual = concat_l [encoder_feature_l - decoder_feature_l]
        residual -> 1x1 conv adaptor -> normalizing flow -> (z, log|det J|)
        loss = PatchflowLoss(z, log|det J|)

Inference:
    residual -> adaptor -> flow -> z -> PatchFlow's own NLL-based anomaly map
"""

from collections.abc import Callable, Sequence

import torch
from FrEIA.framework import SequenceINN
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.data import InferenceBatch
from anomalib.models.components.flow import AllInOneBlock
from anomalib.models.image.patchflow.anomaly_map import AnomalyMapGenerator as PatchflowAnomalyMapGenerator
from anomalib.models.image.patchflow.loss import PatchflowLoss

from .anomaly_map import AnomalyMapGenerationMode
from .torch_model import ReverseDistillationModel


def _build_subnet_constructor(hidden_dim: int) -> Callable:
    """Build a subnet constructor for the normalizing flow coupling blocks."""

    def subnet_conv(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    return subnet_conv


class ReverseDistillationFlowModel(ReverseDistillationModel):
    """Reverse Distillation model with a PatchFlow-style flow on the residual.

    Args:
        backbone: Timm backbone name for the (frozen) teacher encoder.
        input_size: Model input size ``(H, W)``.
        layers: Encoder layers. The bottleneck requires exactly 3.
            Defaults to ``("layer1", "layer2", "layer3")``.
        pre_trained: Use pretrained encoder weights. Defaults to ``True``.
        flow_steps: Number of coupling blocks in the flow. Defaults to ``1``.
        flow_feature_dim: Channel dimension after the 1x1 adaptor. Defaults to ``128``.
        flow_hidden_dim: Hidden channels in the flow subnet. Defaults to ``128``.
        flow_epochs: Epochs to train the flow in Phase 2. Defaults to ``50``.
        flow_lr: Adam learning rate for adaptor + flow. Defaults to ``1e-3``.
        flow_batch_size: Mini-batch size used during flow training. Defaults to ``64``.
    """

    def __init__(
        self,
        backbone: str,
        input_size: tuple[int, int],
        layers: Sequence[str] = ("layer1", "layer2", "layer3"),
        pre_trained: bool = True,
        flow_steps: int = 1,
        flow_feature_dim: int = 128,
        flow_hidden_dim: int = 128,
        flow_epochs: int = 50,
        flow_lr: float = 1e-3,
        flow_batch_size: int = 64,
    ) -> None:
        super().__init__(
            backbone=backbone,
            input_size=input_size,
            layers=layers,
            anomaly_map_mode=AnomalyMapGenerationMode.ADD,
            pre_trained=pre_trained,
        )
        self.flow_epochs = flow_epochs
        self.flow_lr = flow_lr
        self.flow_batch_size = flow_batch_size

        total_channels = sum(self.encoder.out_dims)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *input_size)
            residual_size = tuple(next(iter(self.encoder(dummy).values())).shape[-2:])

        self.feature_adaptor = nn.Conv2d(total_channels, flow_feature_dim, kernel_size=1)
        self.flow = SequenceINN(flow_feature_dim, *residual_size)
        for _ in range(flow_steps):
            self.flow.append(AllInOneBlock, subnet_constructor=_build_subnet_constructor(flow_hidden_dim))
        self.anomaly_map_generator = PatchflowAnomalyMapGenerator(input_size=input_size)

        # Populated once Phase 2 (train_flow) has run; used by visualize_nll_space.py.
        self.viz_pre_flow: torch.Tensor | None = None
        self.viz_z_all: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def encode_decode(self, images: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run the frozen encoder, then the bottleneck/decoder, handling tiling."""
        self.encoder.eval()
        if self.tiler:
            images = self.tiler.tile(images)
        encoder_features = list(self.encoder(images).values())
        decoder_features = self.decoder(self.bottleneck(encoder_features))
        if self.tiler:
            encoder_features = [self.tiler.untile(feature) for feature in encoder_features]
            decoder_features = [self.tiler.untile(feature) for feature in decoder_features]
        return encoder_features, decoder_features

    @staticmethod
    def generate_residual_embedding(
        encoder_features: list[torch.Tensor],
        decoder_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Concatenate multi-scale teacher-student residuals into one embedding."""
        embeddings = encoder_features[0] - decoder_features[0]
        for enc_feature, dec_feature in zip(encoder_features[1:], decoder_features[1:], strict=True):
            residual = enc_feature - dec_feature
            residual = F.interpolate(residual, size=embeddings.shape[-2:], mode="bilinear")
            embeddings = torch.cat((embeddings, residual), 1)
        return embeddings

    @staticmethod
    def _flatten(embedding: torch.Tensor) -> torch.Tensor:
        """Flatten a ``(B, C, H, W)`` tensor into ``(B*H*W, C)`` patch vectors (for viz only)."""
        embedding_size = embedding.size(1)
        return embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)

    # ------------------------------------------------------------------
    # Phase 2: flow training (called once, after Phase 1 RD training)
    # ------------------------------------------------------------------

    def train_flow(self, residuals: torch.Tensor) -> None:
        """Train the 1x1 adaptor + flow via PatchFlow's NLL loss.

        Args:
            residuals: All training-set residual maps, shape (N_images, C, H, W).
        """
        device = residuals.device
        self.feature_adaptor = self.feature_adaptor.to(device)
        self.flow = self.flow.to(device)
        self.feature_adaptor.train()
        self.flow.train()

        loss_fn = PatchflowLoss()
        optimizer = torch.optim.Adam(
            list(self.feature_adaptor.parameters()) + list(self.flow.parameters()),
            lr=self.flow_lr,
        )
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(residuals),
            batch_size=self.flow_batch_size,
            shuffle=True,
            drop_last=False,
        )
        with torch.enable_grad():
            for epoch in range(self.flow_epochs):
                epoch_loss = 0.0
                for (batch,) in loader:
                    optimizer.zero_grad()
                    adapted = self.feature_adaptor(batch)
                    hidden_variables, log_jacobians = self.flow(adapted)
                    loss = loss_fn(hidden_variables, log_jacobians)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.feature_adaptor.parameters()) + list(self.flow.parameters()), 1.0
                    )
                    optimizer.step()
                    epoch_loss += loss.item()
                if (epoch + 1) % 10 == 0:
                    n_batches = max(len(loader), 1)
                    print(
                        f"[ReverseDistillationFlow] epoch {epoch + 1}/{self.flow_epochs}"
                        f"  loss={epoch_loss / n_batches:.4f}"
                    )

        self.feature_adaptor.eval()
        self.flow.eval()
        with torch.no_grad():
            adapted = self.feature_adaptor(residuals)
            z_all, _ = self.flow(adapted)
        self.viz_pre_flow = self._flatten(adapted).cpu()
        self.viz_z_all = self._flatten(z_all).cpu()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]] | InferenceBatch:
        """Forward pass.

        Training (Phase 1): returns ``(encoder_features, decoder_features)``,
            identical to the plain :class:`ReverseDistillationModel`.
        Inference: returns an :class:`InferenceBatch` scored by PatchFlow's NLL map.
        """
        encoder_features, decoder_features = self.encode_decode(images)

        if self.training:
            return encoder_features, decoder_features

        residual = self.generate_residual_embedding(encoder_features, decoder_features)
        self.feature_adaptor.eval()
        self.flow.eval()
        with torch.no_grad():
            adapted = self.feature_adaptor(residual)
            hidden_variables, _ = self.flow(adapted)

        anomaly_map = self.anomaly_map_generator(hidden_variables)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
