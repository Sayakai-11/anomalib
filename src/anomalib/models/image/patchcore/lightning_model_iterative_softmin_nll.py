"""Lightning wrapper for iterative re-coreset softmin-NLL PatchCore."""

from __future__ import annotations

from collections.abc import Sequence

from anomalib import PrecisionType

from .lightning_model_softmin_nll import PatchcoreSoftminNLLModule
from .torch_model_iterative_softmin_nll import PatchcoreIterativeSoftminNLLModel


class PatchcoreIterativeSoftminNLLModule(PatchcoreSoftminNLLModule):
    """Drop-in Lightning module using the iterative Torch model."""

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3"),
        pre_trained: bool = True,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        sigma: float | None = None,
        lambda_reg: float = 0.1,
        hidden_dim: int = 512,
        map_epochs: int = 50,
        map_lr: float = 1e-3,
        map_batch_size: int = 512,
        num_recore_stages: int = 3,
        precision: str | PrecisionType = PrecisionType.FLOAT32,
        **kwargs,
    ) -> None:
        super().__init__(
            backbone=backbone,
            layers=layers,
            pre_trained=pre_trained,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
            sigma=sigma,
            lambda_reg=lambda_reg,
            hidden_dim=hidden_dim,
            map_epochs=map_epochs,
            map_lr=map_lr,
            map_batch_size=map_batch_size,
            precision=precision,
            **kwargs,
        )
        self.model = PatchcoreIterativeSoftminNLLModel(
            backbone=backbone,
            layers=layers,
            pre_trained=pre_trained,
            num_neighbors=num_neighbors,
            sigma=sigma,
            lambda_reg=lambda_reg,
            hidden_dim=hidden_dim,
            map_epochs=map_epochs,
            map_lr=map_lr,
            map_batch_size=map_batch_size,
            num_recore_stages=num_recore_stages,
        )
        if isinstance(precision, str):
            precision = PrecisionType(precision.lower())
        if precision == PrecisionType.FLOAT16:
            self.model = self.model.half()
        elif precision == PrecisionType.FLOAT32:
            self.model = self.model.float()
        else:
            raise ValueError(f"Unsupported precision: {precision}")

