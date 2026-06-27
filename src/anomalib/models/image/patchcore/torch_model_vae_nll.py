# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PatchCore + VAE with NLL(μ) anomaly scoring.

Identical to PatchcoreVAEModel in training (ELBO with KL term).
Inference replaces kNN with the closed-form NLL of μ under N(0, I):

    score = ½ ‖μ‖²   ( = −log N(μ; 0, I) up to a constant )

Unlike PatchcoreVAEPriorModel (KL scoring), σ is not used in the score.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from anomalib.data import InferenceBatch

from .torch_model_vae import PatchcoreVAEModel


class PatchcoreVAENLLModel(PatchcoreVAEModel):
    """PatchCore + VAE with ½‖μ‖² anomaly score.

    Training is identical to :class:`PatchcoreVAEModel` (ELBO).
    Inference uses the NLL of the posterior mean μ under the standard
    normal prior, with no kNN lookup.

    Args:
        layers: Backbone layer names.
        backbone: Timm backbone name. Defaults to ``"wide_resnet50_2"``.
        pre_trained: Use pretrained backbone weights. Defaults to ``True``.
        num_neighbors: Kept for API compatibility; not used in scoring.
        latent_dim: VAE latent dimensionality. Defaults to ``512``.
        vae_epochs: VAE training epochs. Defaults to ``500``.
        vae_lr: VAE Adam learning rate. Defaults to ``1e-3``.
        vae_batch_size: VAE mini-batch size. Defaults to ``512``.
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
        """Backbone → VAE encoder → ½‖μ‖² → anomaly map.

        Training: stores raw embeddings (same as parent).
        Inference: score = ½‖μ‖² per patch, no kNN.
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

        if self.vae is None:
            msg = "VAE has not been trained yet."
            raise RuntimeError(msg)

        self.vae.eval()
        with torch.no_grad():
            mu, _ = self.vae.encode(embedding.float())

        # NLL of μ under N(0, I): ½‖μ‖² per patch
        nll_per_patch = 0.5 * mu.pow(2).sum(dim=1)  # (N_patches,)

        patch_scores = nll_per_patch.reshape(batch_size, -1)
        pred_score = patch_scores.amax(1)
        patch_scores = patch_scores.reshape(batch_size, 1, width, height)
        anomaly_map = self.anomaly_map_generator(patch_scores, output_size)

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
