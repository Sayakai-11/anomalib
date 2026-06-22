# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PatchCore Lightning module with VAE-based feature normalization.

Drop-in replacement for the standard Patchcore lightning module.  Wraps
PatchcoreVAEModel instead of PatchcoreModel so that the memory bank is built in
VAE latent (μ) space rather than raw feature space.

Usage::

    from anomalib.models.image.patchcore.lightning_model_vae import PatchcoreVAEModule
    from anomalib.engine import Engine
    from anomalib.data import MVTecAD

    datamodule = MVTecAD()
    model = PatchcoreVAEModule(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        latent_dim=256,
        vae_epochs=50,
    )
    engine = Engine()
    engine.fit(model=model, datamodule=datamodule)
"""

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
from .torch_model_vae import PatchcoreVAEModel

logger = logging.getLogger(__name__)


class PatchcoreVAEModule(MemoryBankMixin, AnomalibModule):
    """PatchCore Lightning module with VAE feature-space normalisation.

    Identical training/inference API to :class:`Patchcore`, but uses
    :class:`PatchcoreVAEModel` so the memory bank lives in μ-space and the
    kNN scoring is done there too.

    Args:
        backbone: Timm backbone name. Defaults to ``"wide_resnet50_2"``.
        layers: Backbone layers to extract features from.
            Defaults to ``("layer2", "layer3")``.
        pre_trained: Use pretrained backbone weights. Defaults to ``True``.
        coreset_sampling_ratio: Coreset subsampling ratio. Defaults to ``0.1``.
        num_neighbors: Number of kNN neighbours for scoring. Defaults to ``9``.
        latent_dim: VAE latent dimensionality. Defaults to ``256``.
        vae_epochs: Epochs to train the VAE after feature collection.
            Defaults to ``50``.
        vae_lr: Adam learning rate for VAE training. Defaults to ``1e-3``.
        vae_batch_size: Mini-batch size used during VAE training.
            Defaults to ``512``.
        precision: Float precision for the backbone. Defaults to ``"float32"``.
        pre_processor: Pre-processor or ``True`` to use default.
        post_processor: Post-processor or ``True`` to use default.
        evaluator: Evaluator or ``True`` to use default.
        visualizer: Visualizer or ``True`` to use default.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3"),
        pre_trained: bool = True,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        latent_dim: int = 256,
        vae_epochs: int = 50,
        vae_lr: float = 1e-3,
        vae_batch_size: int = 512,
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

        self.model = PatchcoreVAEModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            num_neighbors=num_neighbors,
            latent_dim=latent_dim,
            vae_epochs=vae_epochs,
            vae_lr=vae_lr,
            vae_batch_size=vae_batch_size,
        )
        self.coreset_sampling_ratio = coreset_sampling_ratio

        if isinstance(precision, str):
            precision = PrecisionType(precision.lower())

        if precision == PrecisionType.FLOAT16:
            self.model = self.model.half()
        elif precision == PrecisionType.FLOAT32:
            self.model = self.model.float()
        else:
            msg = (
                f"Unsupported precision type: {precision}. "
                f"Supported: {PrecisionType.FLOAT16}, {PrecisionType.FLOAT32}."
            )
            raise ValueError(msg)

    @classmethod
    def configure_pre_processor(cls, image_size=None, center_crop_size=None):
        return Patchcore.configure_pre_processor(image_size=image_size, center_crop_size=center_crop_size)

    @staticmethod
    def configure_optimizers() -> None:
        """No gradient-based optimisation of the backbone."""
        return

    def training_step(self, batch, *args, **kwargs):
        """Collect backbone embeddings for later VAE training."""
        del args, kwargs
        _ = self.model(batch.image)
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        """Train VAE, encode embeddings to μ, apply coreset subsampling."""
        logger.info("Training VAE on collected embeddings and building μ-bank.")
        self.model.subsample_embedding(self.coreset_sampling_ratio)

    def validation_step(self, batch, *args, **kwargs):
        """Run inference and attach predictions to the batch."""
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
