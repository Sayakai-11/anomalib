# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reverse Distillation with a VAE-then-flow head on the residual (staged).

Training is staged, PatchCore-style (mirrors
:class:`anomalib.models.image.patchcore.torch_model_vae_flow_nll.PatchcoreVAEFlowNLLModel`):

    Phase 1 -- Reverse Distillation (gradient, standard RD cosine loss):
        encoder (frozen) -> bottleneck -> decoder
        loss = ReverseDistillationLoss(encoder_features, decoder_features)

    Phase 2 -- VAE (gradient, ELBO), run once after Phase 1 completes, on the
    now-frozen residual features:
        residual = concat_l [encoder_feature_l - decoder_feature_l]
        residual -> VAE -> (reconstruction, mu, logvar)
        loss = MSE(reconstruction, residual) + KL(mu, logvar)

    Phase 3 -- Flow (gradient, PatchFlow-style NLL), run once after Phase 2
    completes, on the now-frozen VAE posterior mean:
        mu (reshaped back to a spatial map) -> 1x1 conv adaptor
            -> normalizing flow -> (z, log|det J|)
        loss = PatchflowLoss(z, log|det J|)

Inference:
    residual -> VAE encoder -> mu -> adaptor -> flow -> z
             -> PatchFlow's own NLL-based anomaly map
"""

from collections.abc import Sequence

import torch
from FrEIA.framework import SequenceINN
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.data import InferenceBatch
from anomalib.models.components.flow import AllInOneBlock
from anomalib.models.image.patchcore.torch_model_vae import PatchcoreVAE
from anomalib.models.image.patchflow.anomaly_map import AnomalyMapGenerator as PatchflowAnomalyMapGenerator
from anomalib.models.image.patchflow.loss import PatchflowLoss

from .anomaly_map import AnomalyMapGenerationMode
from .torch_model import ReverseDistillationModel
from .torch_model_flow import _build_subnet_constructor


class ReverseDistillationVAEFlowModel(ReverseDistillationModel):
    """Reverse Distillation model with a VAE-then-flow head on the residual.

    Args:
        backbone: Timm backbone name for the (frozen) teacher encoder.
        input_size: Model input size ``(H, W)``.
        layers: Encoder layers. The bottleneck requires exactly 3.
            Defaults to ``("layer1", "layer2", "layer3")``.
        pre_trained: Use pretrained encoder weights. Defaults to ``True``.
        latent_dim: VAE latent dimensionality. Defaults to ``512``.
        vae_epochs: Epochs to train the VAE in Phase 2. Defaults to ``500``.
        vae_lr: Adam learning rate for VAE training. Defaults to ``1e-3``.
        vae_batch_size: Mini-batch size used during VAE training. Defaults to ``512``.
        flow_steps: Number of coupling blocks in the flow. Defaults to ``1``.
        flow_feature_dim: Channel dimension after the 1x1 adaptor. Defaults to ``128``.
        flow_hidden_dim: Hidden channels in the flow subnet. Defaults to ``128``.
        flow_epochs: Epochs to train the flow in Phase 3. Defaults to ``50``.
        flow_lr: Adam learning rate for adaptor + flow. Defaults to ``1e-3``.
        flow_batch_size: Mini-batch size used during flow training. Defaults to ``64``.
    """

    def __init__(
        self,
        backbone: str,
        input_size: tuple[int, int],
        layers: Sequence[str] = ("layer1", "layer2", "layer3"),
        pre_trained: bool = True,
        latent_dim: int = 512,
        vae_epochs: int = 500,
        vae_lr: float = 1e-3,
        vae_batch_size: int = 512,
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
        self.vae_epochs = vae_epochs
        self.vae_lr = vae_lr
        self.vae_batch_size = vae_batch_size
        self.flow_epochs = flow_epochs
        self.flow_lr = flow_lr
        self.flow_batch_size = flow_batch_size

        total_channels = sum(self.encoder.out_dims)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *input_size)
            self.residual_hw = tuple(next(iter(self.encoder(dummy).values())).shape[-2:])

        self.vae = PatchcoreVAE(input_dim=total_channels, latent_dim=latent_dim)
        self.feature_adaptor = nn.Conv2d(latent_dim, flow_feature_dim, kernel_size=1)
        self.flow = SequenceINN(flow_feature_dim, *self.residual_hw)
        for _ in range(flow_steps):
            self.flow.append(AllInOneBlock, subnet_constructor=_build_subnet_constructor(flow_hidden_dim))
        self.anomaly_map_generator = PatchflowAnomalyMapGenerator(input_size=input_size)

        # Populated once Phase 2/3 (train_vae/train_flow) have run; used by visualize_nll_space.py.
        self.viz_mu_all: torch.Tensor | None = None
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
    def reshape_embedding(embedding: torch.Tensor) -> torch.Tensor:
        """Flatten a ``(B, C, H, W)`` embedding into ``(B*H*W, C)`` patch vectors."""
        embedding_size = embedding.size(1)
        return embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)

    # ------------------------------------------------------------------
    # Phase 2: VAE training (called once, after Phase 1 RD training)
    # ------------------------------------------------------------------

    def train_vae(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Train the VAE via ELBO on collected residual embeddings.

        Args:
            embeddings: All training-set residual patch vectors, shape (N, D).

        Returns:
            torch.Tensor: The VAE posterior mean ``mu`` for every input row,
                shape (N, latent_dim), for chaining into Phase 3 (flow training).
        """
        self.vae = self.vae.to(embeddings.device)
        self.vae.train()
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.vae_lr)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(embeddings),
            batch_size=self.vae_batch_size,
            shuffle=True,
            drop_last=False,
        )
        with torch.enable_grad():
            for epoch in range(self.vae_epochs):
                epoch_loss = 0.0
                for (batch,) in loader:
                    optimizer.zero_grad()
                    recon, mu, logvar = self.vae(batch)
                    recon_loss = F.mse_loss(recon, batch)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + kl_loss
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                if (epoch + 1) % 20 == 0:
                    n_batches = max(len(loader), 1)
                    print(
                        f"[ReverseDistillationVAEFlow] VAE epoch {epoch + 1}/{self.vae_epochs}"
                        f"  loss={epoch_loss / n_batches:.4f}"
                    )

        self.vae.eval()
        with torch.no_grad():
            mu, _ = self.vae.encode(embeddings.float())
        self.viz_mu_all = mu.cpu()
        return mu

    # ------------------------------------------------------------------
    # Phase 3: flow training (called once, after Phase 2 VAE training)
    # ------------------------------------------------------------------

    def train_flow(self, mu_all: torch.Tensor) -> None:
        """Train the 1x1 adaptor + flow via PatchFlow's NLL loss on VAE's mu.

        Args:
            mu_all: All training-set VAE posterior means, shape (N_patches, latent_dim),
                where N_patches = n_images * H * W for the fixed residual grid (H, W).
        """
        height, width = self.residual_hw
        n_images = mu_all.shape[0] // (height * width)
        mu_spatial = mu_all[: n_images * height * width].reshape(n_images, height, width, -1)
        mu_spatial = mu_spatial.permute(0, 3, 1, 2).contiguous()

        device = mu_spatial.device
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
            torch.utils.data.TensorDataset(mu_spatial),
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
                        f"[ReverseDistillationVAEFlow] Flow epoch {epoch + 1}/{self.flow_epochs}"
                        f"  loss={epoch_loss / n_batches:.4f}"
                    )

        self.feature_adaptor.eval()
        self.flow.eval()
        with torch.no_grad():
            adapted = self.feature_adaptor(mu_spatial)
            z_all, _ = self.flow(adapted)
        self.viz_z_all = self.reshape_embedding(z_all).cpu()

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
        batch_size, _, height, width = residual.shape
        flat_residual = self.reshape_embedding(residual)

        self.vae.eval()
        with torch.no_grad():
            mu, _ = self.vae.encode(flat_residual.float())
        mu_spatial = mu.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()

        self.feature_adaptor.eval()
        self.flow.eval()
        with torch.no_grad():
            adapted = self.feature_adaptor(mu_spatial)
            hidden_variables, _ = self.flow(adapted)

        anomaly_map = self.anomaly_map_generator(hidden_variables)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
