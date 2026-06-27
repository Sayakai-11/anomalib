# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning module for PatchCore + Normalizing Flow (RealNVP)."""

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
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .lightning_model import Patchcore
from .torch_model_flow import PatchcoreFlowModel

logger = logging.getLogger(__name__)


class PatchcoreFlowModule(MemoryBankMixin, AnomalibModule):
    """PatchCore Lightning module with RealNVP feature normalization.

    Drop-in replacement for Patchcore / PatchcoreVAEModule.
    Uses PatchcoreFlowModel: backbone → PCA → RealNVP → kNN.

    Args:
        backbone: Timm backbone name. Defaults to ``"wide_resnet50_2"``.
        layers: Backbone layers. Defaults to ``("layer2", "layer3")``.
        pre_trained: Pretrained backbone weights. Defaults to ``True``.
        coreset_sampling_ratio: Coreset subsampling ratio. Defaults to ``0.1``.
        num_neighbors: kNN neighbours. Defaults to ``9``.
        latent_dim: PCA output / flow input dimension. Defaults to ``256``.
        n_flow_layers: Number of RealNVP coupling layers. Defaults to ``8``.
        flow_hidden_dim: Hidden units in coupling MLPs. Defaults to ``256``.
        flow_epochs: Flow training epochs. Defaults to ``200``.
        flow_lr: Adam learning rate. Defaults to ``1e-4``.
        flow_batch_size: Mini-batch size for flow training. Defaults to ``512``.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3"),
        pre_trained: bool = True,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        latent_dim: int = 256,
        n_flow_layers: int = 8,
        flow_hidden_dim: int = 256,
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

        self.model = PatchcoreFlowModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            num_neighbors=num_neighbors,
            latent_dim=latent_dim,
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
            raise ValueError(f"Unsupported precision: {precision}")

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
        logger.info("Training RealNVP flow and building z-space memory bank.")
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
