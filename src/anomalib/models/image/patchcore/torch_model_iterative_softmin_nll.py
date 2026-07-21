"""Iterative re-coreset PatchCore with softmin-NLL scoring.

At stage k, Z(k-1) and its KCenterGreedy coreset R(k-1) are frozen.  A newly
created residual map g(k) alone is optimized.  The transformed normal features
Z(k) are then used to construct the coreset for the next stage.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from anomalib.models.components import KCenterGreedy

from .torch_model_softmin_nll import (
    PatchcoreSoftminNLLModel,
    ResidualMLP,
)


class ResidualMapChain(nn.Module):
    """Sequential composition g(K) o ... o g(1)."""

    def __init__(self) -> None:
        super().__init__()
        self.maps = nn.ModuleList()

    def append(self, residual_map: nn.Module) -> None:
        self.maps.append(residual_map)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for residual_map in self.maps:
            z = residual_map(z)
        return z


class PatchcoreIterativeSoftminNLLModel(PatchcoreSoftminNLLModel):
    """Softmin-NLL PatchCore that alternates map fitting and coreset selection."""

    def __init__(
        self,
        layers: Sequence[str],
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        num_neighbors: int = 9,
        sigma: float | None = None,
        lambda_reg: float = 0.1,
        hidden_dim: int = 512,
        map_epochs: int = 50,
        map_lr: float = 1e-3,
        map_batch_size: int = 512,
        num_recore_stages: int = 3,
    ) -> None:
        super().__init__(
            layers=layers,
            backbone=backbone,
            pre_trained=pre_trained,
            num_neighbors=num_neighbors,
            sigma=sigma,
            lambda_reg=lambda_reg,
            hidden_dim=hidden_dim,
            map_epochs=map_epochs,
            map_lr=map_lr,
            map_batch_size=map_batch_size,
        )
        if num_recore_stages < 1:
            raise ValueError("num_recore_stages must be at least 1")
        self.num_recore_stages = num_recore_stages
        self.configured_sigma = sigma
        self.residual_map = ResidualMapChain()
        self.coreset_indices_history: list[list[int]] = []
        self.sigma_history: list[float] = []

    @staticmethod
    def _make_coreset(
        embeddings: torch.Tensor,
        sampling_ratio: float,
    ) -> tuple[torch.Tensor, list[int]]:
        """Run the same KCenterGreedy selection used by vanilla PatchCore."""
        sampler = KCenterGreedy(embedding=embeddings, sampling_ratio=sampling_ratio)
        indices = sampler.select_coreset_idxs()
        return embeddings[indices].detach(), indices

    def _fit_one_map(
        self,
        stage_input: torch.Tensor,
        fixed_reference: torch.Tensor,
        stage_sigma: float,
        stage_number: int,
    ) -> ResidualMLP:
        """Fit only a fresh g(k); inputs, reference points and earlier maps stay frozen."""
        stage_input = stage_input.detach()
        fixed_reference = fixed_reference.detach()
        stage_input.requires_grad_(False)
        fixed_reference.requires_grad_(False)

        residual_map = ResidualMLP(
            stage_input.shape[1], hidden_dim=self.hidden_dim
        ).to(stage_input.device)
        optimizer = torch.optim.Adam(residual_map.parameters(), lr=self.map_lr)
        loader = DataLoader(
            TensorDataset(stage_input),
            batch_size=self.map_batch_size,
            shuffle=True,
        )
        two_sigma_sq = 2.0 * stage_sigma**2

        residual_map.train()
        with torch.enable_grad():
            for epoch in range(self.map_epochs):
                epoch_nll = 0.0
                epoch_reg = 0.0
                for (batch,) in loader:
                    optimizer.zero_grad()
                    transformed = residual_map(batch)
                    dist2 = self._squared_dist(transformed, fixed_reference)
                    nll = -torch.logsumexp(-dist2 / two_sigma_sq, dim=1).mean()
                    reg = (transformed - batch).pow(2).sum(dim=1).mean()
                    loss = nll + self.lambda_reg * reg
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"non-finite loss at stage={stage_number}, epoch={epoch + 1}"
                        )
                    loss.backward()
                    nn.utils.clip_grad_norm_(residual_map.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_nll += nll.item()
                    epoch_reg += reg.item()

                if (epoch + 1) % 20 == 0 or epoch + 1 == self.map_epochs:
                    count = max(len(loader), 1)
                    print(
                        f"[IterSoftminNLL] stage={stage_number}/{self.num_recore_stages} "
                        f"epoch={epoch + 1}/{self.map_epochs} "
                        f"nll={epoch_nll / count:.4f} reg={epoch_reg / count:.4f}"
                    )

        residual_map.eval()
        for parameter in residual_map.parameters():
            parameter.requires_grad_(False)
        return residual_map

    def _sigma_for_bank(self, bank: torch.Tensor) -> float:
        if self.configured_sigma is not None:
            return float(self.configured_sigma)
        self.memory_bank = bank
        return self._estimate_sigma()

    def subsample_embedding(
        self,
        sampling_ratio: float,
        embeddings: torch.Tensor | None = None,
    ) -> None:
        """Alternately select a fixed coreset and fit one new residual map."""
        if embeddings is not None:
            del embeddings
        if not self.embedding_store:
            raise ValueError("Embedding store is empty.")

        raw_embeddings = torch.vstack(self.embedding_store).float()
        self.embedding_store.clear()
        self._log_feature_scale_diagnostics(raw_embeddings)

        # self.feature_mean = raw_embeddings.mean(dim=0, keepdim=True)
        # z_current = (raw_embeddings - self.feature_mean).detach()
        z_current = raw_embeddings.detach()
        self.viz_raw_embeddings = z_current.cpu()

        self.residual_map = ResidualMapChain().to(z_current.device)
        self.coreset_indices_history.clear()
        self.sigma_history.clear()

        for stage_idx in range(self.num_recore_stages):
            fixed_bank, indices = self._make_coreset(z_current, sampling_ratio)
            stage_sigma = self._sigma_for_bank(fixed_bank)
            self.coreset_indices_history.append(indices)
            self.sigma_history.append(stage_sigma)
            print(
                f"[IterSoftminNLL] stage={stage_idx + 1} "
                f"input={z_current.shape[0]} bank={fixed_bank.shape[0]} "
                f"sigma={stage_sigma:.4f}"
            )

            new_map = self._fit_one_map(
                stage_input=z_current,
                fixed_reference=fixed_bank,
                stage_sigma=stage_sigma,
                stage_number=stage_idx + 1,
            )
            self.residual_map.append(new_map)
            with torch.no_grad():
                z_current = new_map(z_current).detach()

        # The last training bank lies in Z(K-1).  Inference queries lie in Z(K),
        # so construct one final coreset in exactly that final coordinate space.
        final_bank, final_indices = self._make_coreset(z_current, sampling_ratio)
        self.memory_bank = final_bank
        self.sigma = self._sigma_for_bank(final_bank)
        self.coreset_indices_history.append(final_indices)
        self.sigma_history.append(float(self.sigma))
        self.viz_z_tilde_all = z_current.cpu()
        print(
            f"[IterSoftminNLL] final bank={self.memory_bank.shape[0]} "
            f"sigma={self.sigma:.4f} maps={len(self.residual_map.maps)}"
        )
