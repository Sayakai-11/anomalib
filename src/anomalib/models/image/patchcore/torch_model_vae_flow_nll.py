# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PatchCore + VAE + RealNVP with ½‖z‖² anomaly scoring.

Training pipeline:
    Phase 1 — VAE (ELBO):
        backbone features (N, D)
        → VAE encoder → (μ, logvar)
        → ELBO = recon_loss + KL

    Phase 2 — RealNVP on μ (NLL, Jacobian included in loss):
        μ (N, latent_dim)
        → RealNVP: maximize log p(μ) = log p_z(z) + log|det J|

Inference scoring (Jacobian excluded, same convention as PatchFlow):
    backbone → VAE encoder → μ → RealNVP → z
    score = ½‖z‖²  per patch
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models.components import KCenterGreedy

from .torch_model_flow import RealNVP
from .torch_model_vae import PatchcoreVAEModel


class PatchcoreVAEFlowNLLModel(PatchcoreVAEModel):
    """PatchCore + VAE + RealNVP with ½‖z‖² anomaly score.

    Extends PatchcoreVAEModel by training a RealNVP normalizing flow on
    the VAE posterior means after ELBO training is complete.

    Args:
        layers: Backbone layer names.
        backbone: Timm backbone name. Defaults to ``"wide_resnet50_2"``.
        pre_trained: Use pretrained backbone weights. Defaults to ``True``.
        num_neighbors: Kept for API compatibility; not used in scoring.
        latent_dim: VAE latent dimensionality (= RealNVP input dim).
            Defaults to ``512``.
        vae_epochs: VAE training epochs. Defaults to ``500``.
        vae_lr: VAE Adam learning rate. Defaults to ``1e-3``.
        vae_batch_size: VAE mini-batch size. Defaults to ``512``.
        n_flow_layers: Number of RealNVP coupling layers. Defaults to ``8``.
        flow_hidden_dim: Hidden units in coupling MLPs. Defaults to ``512``.
        flow_epochs: RealNVP training epochs. Defaults to ``200``.
        flow_lr: RealNVP Adam learning rate. Defaults to ``1e-4``.
        flow_batch_size: RealNVP mini-batch size. Defaults to ``512``.
    """

    def __init__(
        self,
        layers: Sequence[str],
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        num_neighbors: int = 9,
        latent_dim: int = 512,
        vae_epochs: int = 500,
        vae_lr: float = 1e-3,
        vae_batch_size: int = 512,
        n_flow_layers: int = 8,
        flow_hidden_dim: int = 512,
        flow_epochs: int = 200,
        flow_lr: float = 1e-4,
        flow_batch_size: int = 512,
    ) -> None:
        super().__init__(
            layers=layers,
            backbone=backbone,
            pre_trained=pre_trained,
            num_neighbors=num_neighbors,
            latent_dim=latent_dim,
            vae_epochs=vae_epochs,
            vae_lr=vae_lr,
            vae_batch_size=vae_batch_size,
        )
        self.n_flow_layers = n_flow_layers
        self.flow_hidden_dim = flow_hidden_dim
        self.flow_epochs = flow_epochs
        self.flow_lr = flow_lr
        self.flow_batch_size = flow_batch_size
        self.flow: RealNVP | None = None

    # ------------------------------------------------------------------
    # RealNVP training helper
    # ------------------------------------------------------------------

    def _train_flow(self, mu_embeddings: torch.Tensor) -> None:
        """Train RealNVP on VAE posterior means.

        Loss = −E[log p(μ)] = −E[log p_z(z) + log|det J|]
             = mean(½‖z‖² − log|det J|)

        Jacobian is included in training to prevent the trivial collapse
        of mapping everything to z=0.

        Args:
            mu_embeddings: All training μ values, shape (N, latent_dim).
        """
        dim = mu_embeddings.shape[1]
        self.flow = RealNVP(dim=dim, hidden_dim=self.flow_hidden_dim, n_layers=self.n_flow_layers)
        self.flow = self.flow.to(mu_embeddings.device)
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.flow_lr)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(mu_embeddings),
            batch_size=self.flow_batch_size,
            shuffle=True,
            drop_last=False,
        )
        with torch.enable_grad():
            for epoch in range(self.flow_epochs):
                epoch_loss = 0.0
                for (batch,) in loader:
                    optimizer.zero_grad()
                    _z, log_px = self.flow(batch)
                    loss = -log_px.mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.flow.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                if (epoch + 1) % 20 == 0:
                    n = max(len(loader), 1)
                    print(
                        f"[VAEFlow] epoch {epoch + 1}/{self.flow_epochs}"
                        f"  nll={epoch_loss / n:.4f}"
                    )

    # ------------------------------------------------------------------
    # Memory-bank construction
    # ------------------------------------------------------------------

    def subsample_embedding(self, sampling_ratio: float, embeddings: torch.Tensor = None) -> None:
        """Two-phase training: VAE then RealNVP on μ.

        Phase 1: delegates to parent (train VAE, encode to μ, coreset).
        Phase 2: trains RealNVP on all μ (viz_mu_all set by parent).

        Args:
            sampling_ratio: Coreset subsampling ratio (used in Phase 1).
            embeddings: Unused; kept for API compatibility.
        """
        # Phase 1: VAE training + μ-bank via parent
        super().subsample_embedding(sampling_ratio, embeddings)

        # Phase 2: train RealNVP on all μ values
        mu_all = self.viz_mu_all.to(self.memory_bank.device)
        print(
            f"[VAEFlow] Training RealNVP on μ  dim={mu_all.shape[1]}"
            f"  n_layers={self.n_flow_layers}  epochs={self.flow_epochs}"
        )
        self._train_flow(mu_all)

        # Visualisation: z after flow
        self.flow.eval()
        with torch.no_grad():
            z_all, _ = self.flow(mu_all)
        self.viz_z_all: torch.Tensor = z_all.cpu()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(self, input_tensor: torch.Tensor):
        """backbone → VAE → μ → RealNVP → ½‖z‖² anomaly score.

        Jacobian is excluded from scoring (same convention as PatchFlow).
        """
        input_tensor = input_tensor.type(self.memory_bank.dtype)
        output_size = input_tensor.shape[-2:]
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = {layer: self.feature_pooler(f) for layer, f in features.items()}
        embedding = self.generate_embedding(features)

        if self.tiler:
            embedding = self.tiler.untile(embedding)

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        if self.training:
            self.embedding_store.append(embedding)
            return embedding

        if self.vae is None or self.flow is None:
            msg = "VAE and Flow must be trained before inference."
            raise RuntimeError(msg)

        # VAE → μ
        self.vae.eval()
        with torch.no_grad():
            mu, _ = self.vae.encode(embedding.float())

        # RealNVP → z  (score = ½‖z‖², Jacobian excluded)
        self.flow.eval()
        with torch.no_grad():
            z, _ = self.flow(mu.to(self.memory_bank.device))

        nll_per_patch = 0.5 * z.pow(2).sum(dim=1)  # (N_patches,)

        patch_scores = nll_per_patch.reshape(batch_size, -1)
        pred_score = patch_scores.amax(1)
        patch_scores = patch_scores.reshape(batch_size, 1, width, height)
        anomaly_map = self.anomaly_map_generator(patch_scores, output_size)

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
