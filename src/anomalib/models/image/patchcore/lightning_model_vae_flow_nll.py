# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning module for PatchCore + VAE + RealNVP with ½‖z‖² scoring."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from anomalib import LearningType, PrecisionType
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from anomalib.post_processing import PostProcessor
from anomalib.visualization import Visualizer

from .lightning_model import Patchcore
from .torch_model_vae_flow_nll import PatchcoreVAEFlowNLLModel

logger = logging.getLogger(__name__)


class PatchcoreVAEFlowNLLModule(MemoryBankMixin, AnomalibModule):
    """PatchCore Lightning module: VAE + RealNVP with ½‖z‖² anomaly scoring.

    Training is two-phase:
      1. ELBO on backbone features (VAE).
      2. NLL on VAE posterior means μ (RealNVP, Jacobian in loss).

    Inference scoring uses ½‖z‖² where z = flow(μ). Jacobian is excluded
    from the score, matching the PatchFlow convention.

    Args:
        backbone: Timm backbone name. Defaults to ``"wide_resnet50_2"``.
        layers: Backbone layers. Defaults to ``("layer2", "layer3")``.
        pre_trained: Use pretrained backbone weights. Defaults to ``True``.
        coreset_sampling_ratio: Kept for API compatibility. Defaults to ``0.1``.
        num_neighbors: Kept for API compatibility. Defaults to ``9``.
        latent_dim: VAE latent / RealNVP input dimensionality.
            Defaults to ``512``.
        vae_epochs: VAE training epochs. Defaults to ``500``.
        vae_lr: VAE Adam learning rate. Defaults to ``1e-3``.
        vae_batch_size: VAE mini-batch size. Defaults to ``512``.
        n_flow_layers: RealNVP coupling layers. Defaults to ``8``.
        flow_hidden_dim: Hidden units in coupling MLPs. Defaults to ``512``.
        flow_epochs: RealNVP training epochs. Defaults to ``200``.
        flow_lr: RealNVP Adam learning rate. Defaults to ``1e-4``.
        flow_batch_size: RealNVP mini-batch size. Defaults to ``512``.
        precision: Float precision for the backbone. Defaults to ``"float32"``.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3"),
        pre_trained: bool = True,
        coreset_sampling_ratio: float = 0.1,
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
        precision: str | PrecisionType = PrecisionType.FLOAT32,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.model = PatchcoreVAEFlowNLLModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            num_neighbors=num_neighbors,
            latent_dim=latent_dim,
            vae_epochs=vae_epochs,
            vae_lr=vae_lr,
            vae_batch_size=vae_batch_size,
            n_flow_layers=n_flow_layers,
            flow_hidden_dim=flow_hidden_dim,
            flow_epochs=flow_epochs,
            flow_lr=flow_lr,
            flow_batch_size=flow_batch_size,
        )
        self.coreset_sampling_ratio = coreset_sampling_ratio

        if isinstance(precision, str):
            precision = PrecisionType(precision.lower())
        if precision == PrecisionType.FLOAT16:
            self.model = self.model.half()
        elif precision == PrecisionType.FLOAT32:
            self.model = self.model.float()
        else:
            msg = f"Unsupported precision: {precision}"
            raise ValueError(msg)

    @classmethod
    def configure_pre_processor(cls, image_size=None, center_crop_size=None):
        return Patchcore.configure_pre_processor(
            image_size=image_size, center_crop_size=center_crop_size
        )

    @staticmethod
    def configure_optimizers() -> None:
        return

    def training_step(self, batch, *args, **kwargs):
        del args, kwargs
        _ = self.model(batch.image)
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        logger.info("Phase 1: training VAE. Phase 2: training RealNVP on μ.")
        self.model.subsample_embedding(self.coreset_sampling_ratio)

    def validation_step(self, batch, *args, **kwargs):
        del args, kwargs
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0, "devices": 1}

    @property
    def learning_type(self) -> LearningType:
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        return PostProcessor()
