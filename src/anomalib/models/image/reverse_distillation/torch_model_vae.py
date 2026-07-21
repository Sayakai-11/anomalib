# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reverse Distillation with a VAE trained on the teacher-student residual (staged).

Training is staged, PatchCore-style:

    Phase 1 -- Reverse Distillation (gradient, standard RD cosine loss):
        encoder (frozen) -> bottleneck -> decoder
        loss = ReverseDistillationLoss(encoder_features, decoder_features)

    Phase 2 -- VAE (gradient, ELBO), run once after Phase 1 completes, on the
    now-frozen residual features:
        residual = concat_l [encoder_feature_l - decoder_feature_l]   (per patch)
        residual -> VAE -> (reconstruction, mu, logvar)
        loss = MSE(reconstruction, residual) + KL(mu, logvar)

Inference:
    residual -> VAE encoder -> mu -> score = 1/2 * ||mu||^2 per patch
    (NLL of mu under the standard normal prior, no k-NN / memory bank)
"""

from collections.abc import Sequence

import torch
from torch.nn import functional as F  # noqa: N812

from anomalib.data import InferenceBatch
from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator as PatchcoreAnomalyMapGenerator
from anomalib.models.image.patchcore.torch_model_vae import PatchcoreVAE

from .anomaly_map import AnomalyMapGenerationMode
from .torch_model import ReverseDistillationModel


class ReverseDistillationVAEModel(ReverseDistillationModel):
    """Reverse Distillation model with a VAE head on the teacher-student residual.

    Args:
        backbone: Timm backbone name for the (frozen) teacher encoder.
        input_size: Model input size ``(H, W)``.
        layers: Encoder layers to extract features from. The bottleneck
            architecture requires exactly 3 layers. Defaults to
            ``("layer1", "layer2", "layer3")``.
        pre_trained: Use pretrained encoder weights. Defaults to ``True``.
        latent_dim: VAE latent dimensionality. Defaults to ``512``.
        vae_epochs: Epochs to train the VAE in Phase 2. Defaults to ``500``.
        vae_lr: Adam learning rate for VAE training. Defaults to ``1e-3``.
        vae_batch_size: Mini-batch size used during VAE training. Defaults to ``512``.
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

        total_channels = sum(self.encoder.out_dims)
        self.vae = PatchcoreVAE(input_dim=total_channels, latent_dim=latent_dim)
        self.anomaly_map_generator = PatchcoreAnomalyMapGenerator()

        # Populated once Phase 2 (train_vae) has run; used by visualize_nll_space.py.
        self.viz_raw_embeddings: torch.Tensor | None = None
        self.viz_mu_all: torch.Tensor | None = None

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

    def train_vae(self, embeddings: torch.Tensor) -> None:
        """Train the VAE via ELBO on collected residual embeddings.

        Args:
            embeddings: All training-set residual patch vectors, shape (N, D).
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
                        f"[ReverseDistillationVAE] epoch {epoch + 1}/{self.vae_epochs}"
                        f"  loss={epoch_loss / n_batches:.4f}"
                    )

        self.vae.eval()
        with torch.no_grad():
            mu, _ = self.vae.encode(embeddings.float())
        self.viz_raw_embeddings = embeddings.cpu()
        self.viz_mu_all = mu.cpu()

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
        Inference: returns an :class:`InferenceBatch` scored by ``1/2 ||mu||^2``.
        """
        output_size = images.shape[-2:]
        encoder_features, decoder_features = self.encode_decode(images)

        if self.training:
            return encoder_features, decoder_features

        residual = self.generate_residual_embedding(encoder_features, decoder_features)
        batch_size, _, width, height = residual.shape
        flat_residual = self.reshape_embedding(residual)

        self.vae.eval()
        with torch.no_grad():
            mu, _ = self.vae.encode(flat_residual.float())

        # NLL of mu under N(0, I): 1/2 ||mu||^2 per patch.
        nll_per_patch = 0.5 * mu.pow(2).sum(dim=1)
        patch_scores = nll_per_patch.reshape(batch_size, -1)
        pred_score = patch_scores.amax(1)
        patch_scores = patch_scores.reshape(batch_size, 1, width, height)
        anomaly_map = self.anomaly_map_generator(patch_scores, output_size)

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
