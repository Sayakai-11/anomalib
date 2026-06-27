# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PatchCore + VAE with prior likelihood anomaly scoring.

Extends PatchcoreVAEModel by replacing kNN scoring with prior likelihood.

Anomaly score per patch:
    score = 0.5 * sum_d(μ_d² + σ_d² - log σ_d² - 1)
          = KL( N(μ, σ²) || N(0, I) )

When σ → 0 (point estimate), this reduces to ||μ||² / 2,
which is -log N(μ; 0, I) up to a constant.

No memory bank is needed at inference time.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from anomalib.data import InferenceBatch

from .torch_model_vae import PatchcoreVAEModel


class PatchcoreVAEPriorModel(PatchcoreVAEModel):
    """PatchCore + VAE using KL divergence from prior as anomaly score.

    Inherits VAE training from PatchcoreVAEModel.
    Only forward() at inference is overridden to use:

        score_p = KL( q(z|x_p) || N(0,I) )
                = 0.5 * Σ_d ( μ_d² + σ_d² - log σ_d² - 1 )

    Args:
        layers: Backbone layer names.
        backbone: Timm backbone name. Defaults to ``"wide_resnet50_2"``.
        pre_trained: Use pretrained backbone weights. Defaults to ``True``.
        num_neighbors: Kept for API compatibility; not used in scoring.
        latent_dim: VAE latent dimensionality.
        vae_epochs: VAE training epochs.
        vae_lr: VAE Adam learning rate.
        vae_batch_size: VAE mini-batch size.
    """

    def __init__(
        self,
        layers: Sequence[str],
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        num_neighbors: int = 9,
        latent_dim: int = 256,
        vae_epochs: int = 500,
        vae_lr: float = 1e-3,
        vae_batch_size: int = 512,
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

    def forward(self, input_tensor: torch.Tensor):
        """Backbone → VAE encoder → KL score → anomaly map.

        Training: stores raw embeddings (same as parent).
        Inference: KL( q(z|x) || N(0,I) ) per patch, no kNN.
        """
        input_tensor = input_tensor.type(self.memory_bank.dtype)
        output_size = input_tensor.shape[-2:]
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)

        if self.tiler:
            embedding = self.tiler.untile(embedding)

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        if self.training:
            self.embedding_store.append(embedding)
            return embedding

        # KL divergence from prior as patch-level anomaly score
        self.vae.eval()
        with torch.no_grad():
            mu, logvar = self.vae.encode(embedding.float())

        # KL = 0.5 * sum_d( μ² + σ² - log σ² - 1 )  shape: (N_patches,)
        kl_per_patch = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        # also store for visualization (only first call in a session)
        if not hasattr(self, "viz_kl_train") and self.training is False:
            pass  # populated separately via visualize_prior_space.py

        patch_scores = kl_per_patch.reshape(batch_size, -1)          # (B, H*W)
        pred_score   = patch_scores.amax(1)                           # (B,)
        patch_scores = patch_scores.reshape(batch_size, 1, width, height)
        anomaly_map  = self.anomaly_map_generator(patch_scores, output_size)

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
