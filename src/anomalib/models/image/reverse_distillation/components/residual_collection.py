# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared helper for staged Reverse-Distillation training.

The RD + VAE / Flow variants train in stages, PatchCore-style: first the
teacher-student network (Reverse Distillation's own cosine loss), then a
VAE and/or normalizing flow on the *frozen* teacher-student residual. This
mixin provides the "collect residual embeddings over the whole training set"
step shared by all of those Lightning modules.
"""

import torch


class ResidualCollectionMixin:
    """Collects teacher-student residual embeddings for a post-hoc training stage.

    Expects the host Lightning module to provide ``self.model`` with
    ``encode_decode``, ``generate_residual_embedding`` and ``reshape_embedding``
    (all implemented identically across the RD + VAE/Flow torch models), plus
    the usual ``self.trainer``, ``self.device`` and ``self.pre_processor``
    attributes of an :class:`~anomalib.models.components.AnomalibModule`.
    """

    def _collect_residuals(self, *, flatten: bool) -> torch.Tensor:
        """Run the frozen encoder-bottleneck-decoder over the training set.

        Args:
            flatten: If ``True``, flatten each ``(C, H, W)`` residual map into
                per-patch ``(H*W, C)`` rows (for VAE training). If ``False``,
                keep the per-image ``(C, H, W)`` spatial map (for flow training).

        Returns:
            torch.Tensor: Concatenated residuals over the whole training set.
        """
        self.model.eval()
        transform = self.pre_processor.transform if self.pre_processor else None
        loader = self.trainer.datamodule.train_dataloader()

        chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                images = batch.image.to(self.device)
                if transform is not None:
                    images = transform(images)
                encoder_features, decoder_features = self.model.encode_decode(images)
                residual = self.model.generate_residual_embedding(encoder_features, decoder_features)
                if flatten:
                    residual = self.model.reshape_embedding(residual)
                chunks.append(residual.cpu())

        return torch.cat(chunks, dim=0).to(self.device)
