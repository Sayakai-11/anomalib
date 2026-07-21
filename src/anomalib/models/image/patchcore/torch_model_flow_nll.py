# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PatchCore + RealNVP (no VAE) with ½‖z‖² anomaly scoring.

Identical to PatchcoreFlowModel in training (PCA → standardize → RealNVP via
NLL). Inference replaces kNN with the closed-form NLL of z under N(0, I):

    score = ½ ‖z‖²   ( = −log N(z; 0, I) up to a constant )

Unlike PatchcoreFlowModel (kNN-in-flow-space scoring), the coreset memory
bank built by the inherited ``subsample_embedding`` is not used for scoring
here; it is kept only so ``self.memory_bank`` still provides a valid
dtype/device reference (same convention as PatchcoreVAENLLModel).
"""

from __future__ import annotations

import torch

from anomalib.data import InferenceBatch

from .torch_model_flow import PatchcoreFlowModel


class PatchcoreFlowNLLModel(PatchcoreFlowModel):
    """PatchCore + RealNVP with ½‖z‖² anomaly score (no VAE, no kNN).

    Training is identical to :class:`PatchcoreFlowModel` (PCA + RealNVP via
    NLL). Inference uses the NLL of z under the standard normal prior, with
    no kNN lookup.

    Args:
        layers: Backbone layer names.
        backbone: Timm backbone name. Defaults to ``"wide_resnet50_2"``.
        pre_trained: Use pretrained backbone weights. Defaults to ``True``.
        num_neighbors: Kept for API compatibility; not used in scoring.
        latent_dim: PCA output / flow input dimension. Defaults to ``256``.
        n_flow_layers: Number of RealNVP coupling layers. Defaults to ``8``.
        flow_hidden_dim: Hidden units in coupling MLPs. Defaults to ``256``.
        flow_epochs: Flow training epochs. Defaults to ``200``.
        flow_lr: Adam learning rate. Defaults to ``1e-4``.
        flow_batch_size: Mini-batch size for flow training. Defaults to ``512``.
    """

    def forward(self, input_tensor: torch.Tensor):
        """Backbone → PCA → RealNVP → ½‖z‖² → anomaly map.

        Training: stores raw embeddings (same as parent).
        Inference: score = ½‖z‖² per patch, no kNN.
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

        if self.flow is None or self.pca is None:
            msg = "Flow has not been trained yet."
            raise RuntimeError(msg)

        z = self._to_flow_space(embedding).to(self.pca_mean.device)

        # NLL of z under N(0, I): ½‖z‖² per patch
        nll_per_patch = 0.5 * z.pow(2).sum(dim=1)  # (N_patches,)

        patch_scores = nll_per_patch.reshape(batch_size, -1)
        pred_score = patch_scores.amax(1)
        patch_scores = patch_scores.reshape(batch_size, 1, width, height)
        anomaly_map = self.anomaly_map_generator(patch_scores, output_size)

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
