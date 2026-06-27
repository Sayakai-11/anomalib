# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PatchCore with Normalizing Flow (RealNVP) feature normalization.

Pipeline:
    Train:
        backbone features (N, D)
        → PCA (D → latent_dim)
        → standardize (zero-mean, unit-std per dim)
        → RealNVP: maximize log p(z) s.t. z ~ N(0, I)
        → memory_bank = coreset( flow(pca(x)) )

    Inference:
        backbone → PCA → standardize → flow → z → kNN against memory_bank
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from anomalib.data import InferenceBatch
from anomalib.models.components import KCenterGreedy

from .torch_model import PatchcoreModel


# ---------------------------------------------------------------------------
# RealNVP
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CouplingLayer(nn.Module):
    """Affine coupling layer (RealNVP).

    reverse=False: first half unchanged, second half transformed.
    reverse=True : second half unchanged, first half transformed.
    """

    def __init__(self, dim: int, hidden_dim: int, reverse: bool) -> None:
        super().__init__()
        self.reverse = reverse
        half = dim // 2
        rest = dim - half
        in_dim  = half if not reverse else rest
        out_dim = rest if not reverse else half
        self.net_s = nn.Sequential(_MLP(in_dim, out_dim, hidden_dim), nn.Tanh())
        self.net_t = _MLP(in_dim, out_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        half = x.shape[1] // 2
        x_a, x_b = (x[:, :half], x[:, half:]) if not self.reverse else (x[:, half:], x[:, :half])
        s = self.net_s(x_a)
        t = self.net_t(x_a)
        y_b = x_b * s.exp() + t
        log_det = s.sum(dim=1)
        y = torch.cat([x_a, y_b], dim=1) if not self.reverse else torch.cat([y_b, x_a], dim=1)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        half = y.shape[1] // 2
        y_a, y_b = (y[:, :half], y[:, half:]) if not self.reverse else (y[:, half:], y[:, :half])
        s = self.net_s(y_a)
        t = self.net_t(y_a)
        x_b = (y_b - t) * (-s).exp()
        return torch.cat([y_a, x_b], dim=1) if not self.reverse else torch.cat([x_b, y_a], dim=1)


class RealNVP(nn.Module):
    """Stack of affine coupling layers.

    Args:
        dim: Input / output dimensionality.
        hidden_dim: Hidden units in scale/translate MLPs.
        n_layers: Number of coupling layers.
    """

    LOG2PI = math.log(2 * math.pi)

    def __init__(self, dim: int, hidden_dim: int = 256, n_layers: int = 8) -> None:
        super().__init__()
        self.flows = nn.ModuleList([
            _CouplingLayer(dim, hidden_dim, reverse=(i % 2 == 1))
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x → (z, log p(x)).  Includes log|det J| and log N(z; 0, I)."""
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        for flow in self.flows:
            z, ld = flow(z)
            log_det += ld
        log_pz = -0.5 * (z.pow(2).sum(dim=1) + x.shape[1] * self.LOG2PI)
        return z, log_pz + log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for flow in reversed(self.flows):
            x = flow.inverse(x)
        return x


# ---------------------------------------------------------------------------
# PatchcoreFlowModel
# ---------------------------------------------------------------------------

class PatchcoreFlowModel(PatchcoreModel):
    """PatchCore with RealNVP feature normalization.

    Inherits the full PatchcoreModel pipeline. Overrides only
    subsample_embedding() and forward() to use flow-normalized features.

    Args:
        layers: Backbone layer names.
        backbone: Timm backbone. Defaults to ``"wide_resnet50_2"``.
        pre_trained: Pretrained backbone weights. Defaults to ``True``.
        num_neighbors: kNN neighbours for scoring. Defaults to ``9``.
        latent_dim: PCA output dimension (flow input). Defaults to ``256``.
        n_flow_layers: Number of RealNVP coupling layers. Defaults to ``8``.
        flow_hidden_dim: Hidden units in coupling MLPs. Defaults to ``256``.
        flow_epochs: Flow training epochs. Defaults to ``200``.
        flow_lr: Adam learning rate. Defaults to ``1e-4``.
        flow_batch_size: Mini-batch size for flow training. Defaults to ``512``.
    """

    def __init__(
        self,
        layers: Sequence[str],
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        num_neighbors: int = 9,
        latent_dim: int = 256,
        n_flow_layers: int = 8,
        flow_hidden_dim: int = 256,
        flow_epochs: int = 200,
        flow_lr: float = 1e-4,
        flow_batch_size: int = 512,
    ) -> None:
        super().__init__(
            layers=layers,
            backbone=backbone,
            pre_trained=pre_trained,
            num_neighbors=num_neighbors,
        )
        self.latent_dim      = latent_dim
        self.n_flow_layers   = n_flow_layers
        self.flow_hidden_dim = flow_hidden_dim
        self.flow_epochs     = flow_epochs
        self.flow_lr         = flow_lr
        self.flow_batch_size = flow_batch_size

        self.flow: RealNVP | None = None
        self.pca:  PCA    | None = None
        # Normalization stats (registered as buffers → move with .to(device))
        self.register_buffer("pca_mean", torch.zeros(1))
        self.register_buffer("pca_std",  torch.ones(1))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_pca_normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Fit PCA, standardize, return tensor on same device."""
        n_comp = min(self.latent_dim, embeddings.shape[1], embeddings.shape[0])
        self.pca = PCA(n_components=n_comp)
        pca_np = self.pca.fit_transform(embeddings.cpu().numpy()).astype(np.float32)
        pca_t  = torch.from_numpy(pca_np).to(embeddings.device)
        self.pca_mean = pca_t.mean(dim=0)
        self.pca_std  = pca_t.std(dim=0).clamp(min=1e-8)
        return (pca_t - self.pca_mean) / self.pca_std

    def _train_flow(self, pca_embeddings: torch.Tensor) -> None:
        dim = pca_embeddings.shape[1]
        self.flow = RealNVP(dim=dim, hidden_dim=self.flow_hidden_dim, n_layers=self.n_flow_layers)
        self.flow = self.flow.to(pca_embeddings.device)
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.flow_lr)
        loader = DataLoader(
            TensorDataset(pca_embeddings),
            batch_size=self.flow_batch_size,
            shuffle=True,
        )
        with torch.enable_grad():
            for epoch in range(self.flow_epochs):
                epoch_loss = 0.0
                for (batch,) in loader:
                    optimizer.zero_grad()
                    _, log_px = self.flow(batch)
                    loss = -log_px.mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.flow.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                if (epoch + 1) % 20 == 0:
                    n = max(len(loader), 1)
                    print(f"[Flow] epoch {epoch+1}/{self.flow_epochs}  nll={epoch_loss/n:.4f}")

    def _to_flow_space(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Raw backbone embeddings → PCA → standardize → flow z."""
        assert self.pca is not None and self.flow is not None
        pca_np = self.pca.transform(embeddings.cpu().numpy()).astype(np.float32)
        pca_t  = torch.from_numpy(pca_np).to(self.pca_mean.device)
        pca_norm = (pca_t - self.pca_mean) / self.pca_std
        self.flow.eval()
        with torch.no_grad():
            z, _ = self.flow(pca_norm)
        return z

    # ------------------------------------------------------------------
    # Memory-bank construction
    # ------------------------------------------------------------------

    def subsample_embedding(self, sampling_ratio: float, embeddings: torch.Tensor = None) -> None:
        """Build flow-normalized memory bank.

        Steps:
            1. Stack raw backbone embeddings.
            2. PCA (D → latent_dim) + standardize.
            3. Train RealNVP on PCA output (maximize log-likelihood).
            4. Encode all embeddings to z = flow(pca(x)).
            5. Coreset subsample and store as memory_bank.
        """
        if embeddings is not None:
            del embeddings

        if not self.embedding_store:
            raise ValueError("Embedding store is empty.")

        all_embeddings = torch.vstack(self.embedding_store).float()
        self.embedding_store.clear()

        print(
            f"[Flow] Training  raw_dim={all_embeddings.shape[1]}"
            f"  latent_dim={self.latent_dim}  n_layers={self.n_flow_layers}"
            f"  epochs={self.flow_epochs}"
        )

        pca_norm = self._fit_pca_normalize(all_embeddings)
        self._train_flow(pca_norm)

        self.flow.eval()
        with torch.no_grad():
            z, _ = self.flow(pca_norm)

        # 可視化用
        self.viz_raw_embeddings: torch.Tensor = all_embeddings.cpu()
        self.viz_z_all: torch.Tensor          = z.cpu()

        self.memory_bank = z
        sampler = KCenterGreedy(embedding=self.memory_bank, sampling_ratio=sampling_ratio)
        self.memory_bank = sampler.sample_coreset()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(self, input_tensor: torch.Tensor):
        """backbone → PCA → flow → kNN in z-space → anomaly score."""
        input_tensor = input_tensor.type(self.memory_bank.dtype)
        output_size  = input_tensor.shape[-2:]
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features  = {layer: self.feature_pooler(f) for layer, f in features.items()}
        embedding = self.generate_embedding(features)

        if self.tiler:
            embedding = self.tiler.untile(embedding)

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        if self.training:
            self.embedding_store.append(embedding)
            return embedding

        if self.memory_bank.size(0) == 0:
            raise ValueError("Memory bank is empty.")

        z = self._to_flow_space(embedding).to(self.memory_bank.device)

        patch_scores, locations = self.nearest_neighbors(embedding=z, n_neighbors=1)
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations    = locations.reshape((batch_size, -1))
        pred_score   = self.compute_anomaly_score(patch_scores, locations, z)
        patch_scores = patch_scores.reshape((batch_size, 1, width, height))
        anomaly_map  = self.anomaly_map_generator(patch_scores, output_size)

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
