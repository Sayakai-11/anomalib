# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning module for PatchCore with softmin-NLL (fixed memory-bank density) scoring."""

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
from .torch_model_softmin_nll import PatchcoreSoftminNLLModel

logger = logging.getLogger(__name__)


class PatchcoreSoftminNLLModule(MemoryBankMixin, AnomalibModule):
    """PatchCore Lightning module with fixed-density softmin-NLL anomaly scoring.

    The coreset memory bank is built exactly as in vanilla PatchCore and
    never updated afterwards; it defines a fixed mixture-of-Gaussians density.
    A residual mapping ``g_theta`` (identity-initialized) is then trained so
    that normal features attain high likelihood under this fixed density.
    Inference scores patches with ``-log p_M(g_theta(z))`` instead of kNN.

    Args:
        backbone: Timm backbone name. Defaults to ``"wide_resnet50_2"``.
        layers: Backbone layers. Defaults to ``("layer2", "layer3")``.
        pre_trained: Use pretrained backbone weights. Defaults to ``True``.
        coreset_sampling_ratio: Coreset sampling ratio for the fixed memory
            bank. Defaults to ``0.1``.
        num_neighbors: Kept for API compatibility. Defaults to ``9``.
        sigma: Width of each Gaussian component in the memory-bank mixture.
            If ``None`` (default), auto-estimated from the memory bank as
            the median nearest-neighbor distance (see
            :meth:`PatchcoreSoftminNLLModel._estimate_sigma`).
        lambda_reg: Weight of the ``||g_theta(z) - z||^2`` regularizer.
            Defaults to ``1.0``.
        hidden_dim: Hidden units of the residual MLP. Defaults to ``512``.
        map_epochs: Training epochs for ``g_theta``. Defaults to ``100``.
        map_lr: Adam learning rate for ``g_theta``. Defaults to ``1e-3``.
        map_batch_size: Mini-batch size for ``g_theta`` training. Defaults to ``512``.
        precision: Float precision for the backbone. Defaults to ``"float32"``.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3"),
        pre_trained: bool = True,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        sigma: float | None = None,
        lambda_reg: float = 1.0,
        hidden_dim: int = 512,
        map_epochs: int = 100,
        map_lr: float = 1e-3,
        map_batch_size: int = 512,
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

        self.model = PatchcoreSoftminNLLModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            num_neighbors=num_neighbors,
            sigma=sigma,
            lambda_reg=lambda_reg,
            hidden_dim=hidden_dim,
            map_epochs=map_epochs,
            map_lr=map_lr,
            map_batch_size=map_batch_size,
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
        logger.info(
            "Building fixed memory-bank density and training residual map g_theta "
            "for softmin-NLL scoring."
        )
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
