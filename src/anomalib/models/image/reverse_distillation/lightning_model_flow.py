# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reverse Distillation + PatchFlow-style flow Lightning module (staged, PatchCore-style).

Training is staged:
    Phase 1 -- the teacher-student network is trained for ``rd_epochs`` with
        the standard Reverse Distillation cosine loss (identical to
        :class:`~anomalib.models.image.reverse_distillation.lightning_model.
        ReverseDistillation`).
    Phase 2 -- once Phase 1 completes, the (now frozen) teacher-student
        residual is collected over the whole training set and a PatchFlow-style
        normalizing flow is trained via NLL on it.

There is no memory bank. Inference uses PatchFlow's own NLL-based anomaly map.

Usage::

    from anomalib.models.image.reverse_distillation.lightning_model_flow import ReverseDistillationFlow
    from anomalib.engine import Engine
    from anomalib.data import MVTecAD

    datamodule = MVTecAD()
    model = ReverseDistillationFlow(backbone="wide_resnet50_2", rd_epochs=200)
    engine = Engine()
    engine.fit(model=model, datamodule=datamodule)
"""

import logging
from collections.abc import Sequence
from typing import Any

import torch
from torch import optim

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.models.image.reverse_distillation.components import ResidualCollectionMixin
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .loss import ReverseDistillationLoss
from .torch_model_flow import ReverseDistillationFlowModel

logger = logging.getLogger(__name__)


class ReverseDistillationFlow(ResidualCollectionMixin, AnomalibModule):
    """Reverse Distillation with a flow trained after RD training completes.

    Args:
        backbone: Timm backbone name for the (frozen) teacher encoder.
            Defaults to ``"wide_resnet50_2"``.
        layers: Encoder layers. The bottleneck requires exactly 3.
            Defaults to ``("layer1", "layer2", "layer3")``.
        pre_trained: Use pretrained encoder weights. Defaults to ``True``.
        flow_steps: Number of coupling blocks in the flow. Defaults to ``1``.
        flow_feature_dim: Channel dimension after the 1x1 adaptor. Defaults to ``128``.
        flow_hidden_dim: Hidden channels in the flow subnet. Defaults to ``128``.
        rd_epochs: Number of Phase-1 (Reverse Distillation) training epochs.
            Defaults to ``200``.
        rd_lr: Adam learning rate for bottleneck/decoder. Defaults to ``0.005``.
        flow_epochs: Number of Phase-2 (flow) training epochs. Defaults to ``50``.
        flow_lr: Adam learning rate for adaptor + flow. Defaults to ``1e-3``.
        flow_batch_size: Mini-batch size used during flow training. Defaults to ``64``.
        pre_processor: Pre-processor or ``True`` to use default.
        post_processor: Post-processor or ``True`` to use default.
        evaluator: Evaluator or ``True`` to use default.
        visualizer: Visualizer or ``True`` to use default.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer1", "layer2", "layer3"),
        pre_trained: bool = True,
        flow_steps: int = 1,
        flow_feature_dim: int = 128,
        flow_hidden_dim: int = 128,
        rd_epochs: int = 200,
        rd_lr: float = 0.005,
        flow_epochs: int = 50,
        flow_lr: float = 1e-3,
        flow_batch_size: int = 64,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        if self.input_size is None:
            msg = "Input size is required for Reverse Distillation models."
            raise ValueError(msg)

        self.rd_epochs = rd_epochs
        self.rd_lr = rd_lr
        self.model = ReverseDistillationFlowModel(
            backbone=backbone,
            input_size=self.input_size,
            layers=layers,
            pre_trained=pre_trained,
            flow_steps=flow_steps,
            flow_feature_dim=flow_feature_dim,
            flow_hidden_dim=flow_hidden_dim,
            flow_epochs=flow_epochs,
            flow_lr=flow_lr,
            flow_batch_size=flow_batch_size,
        )
        self.rd_loss = ReverseDistillationLoss()

    def configure_optimizers(self) -> optim.Adam:
        """Phase 1 only: optimise bottleneck and decoder."""
        return optim.Adam(
            params=list(self.model.decoder.parameters()) + list(self.model.bottleneck.parameters()),
            lr=self.rd_lr,
            betas=(0.5, 0.99),
        )

    def training_step(self, batch: Batch, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Phase 1: standard Reverse Distillation cosine loss."""
        del args, kwargs
        encoder_features, decoder_features = self.model(batch.image)
        loss = self.rd_loss(encoder_features, decoder_features)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> dict:
        """Score the batch using the trained model."""
        del args, kwargs
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    def on_train_epoch_end(self) -> None:
        """After the last Phase-1 epoch, run Phase 2 (flow training) once."""
        if self.current_epoch != self.rd_epochs - 1:
            return
        logger.info("Reverse Distillation training complete -- collecting residuals for flow training.")
        residuals = self._collect_residuals(flatten=False)
        self.model.train_flow(residuals)

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        return {
            "gradient_clip_val": 0,
            "num_sanity_val_steps": 0,
            "max_epochs": self.rd_epochs,
            # Only validate once, after Phase 2 has trained the flow -- otherwise
            # Lightning's default per-epoch validation would run against an
            # untrained flow during Phase 1.
            "check_val_every_n_epoch": self.rd_epochs,
        }

    @property
    def learning_type(self) -> LearningType:
        return LearningType.ONE_CLASS
