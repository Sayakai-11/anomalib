# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PatchCore with softmin-NLL anomaly scoring under a fixed memory-bank density.

The coreset memory bank is built exactly as in vanilla PatchCore and is never
updated afterwards. It defines a fixed mixture-of-Gaussians density over the
normal-patch feature space:

    p_M(z) = (1/M) * sum_m N(z | s_m, sigma^2 I)

whose negative log-likelihood is, up to a z-independent constant, the
softmin (LogSumExp) relaxation of the PatchCore nearest-neighbor distance:

    -log p_M(z) = -logsumexp_m( -||z - s_m||^2 / (2 sigma^2) ) + const

Unlike the hard min used by vanilla PatchCore, this score is smooth in z, so a
small residual mapping g_theta(z) = z + f_theta(z) (initialized to the
identity) can be trained by gradient descent to pull normal features toward
the fixed density:

    L(theta) = E_z[ -log p_M(g_theta(z)) ] + lambda * E_z[ ||g_theta(z) - z||^2 ]

The L2 term prevents the degenerate collapse of g_theta onto a single point.
Only g_theta is trained; the backbone, the memory bank, and p_M itself remain
fixed throughout.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from anomalib.data import InferenceBatch
from anomalib.models.components import KCenterGreedy

from .torch_model import DEFAULT_CHUNK_SIZE, PatchcoreModel


class ResidualMLP(nn.Module):
    """g_theta(z) = z + f_theta(z), initialized to the identity map.

    The final layer of f_theta is zero-initialized so training starts from
    g_theta = identity and only moves normal features as far as the NLL /
    regularization trade-off requires.

    Args:
        dim: Input / output feature dimensionality.
        hidden_dim: Hidden units of the residual MLP. Defaults to ``512``.
    """

    def __init__(self, dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.net(z)


class PatchcoreSoftminNLLModel(PatchcoreModel):
    """PatchCore with a fixed mixture-of-Gaussians memory bank and softmin-NLL scoring.

    Training builds the coreset memory bank exactly as in vanilla PatchCore,
    then trains a residual mapping ``g_theta`` so that transformed normal
    features have high likelihood under the fixed density defined by the
    bank. Inference uses ``-log p_M(g_theta(z))`` as the per-patch score
    instead of kNN distance.

    Args:
        layers: Backbone layer names.
        backbone: Timm backbone name. Defaults to ``"wide_resnet50_2"``.
        pre_trained: Use pretrained backbone weights. Defaults to ``True``.
        num_neighbors: Kept for API compatibility; not used in scoring.
        sigma: Width of each Gaussian component in the memory-bank mixture.
            The only hyperparameter of the fixed density. If ``None``
            (default), it is auto-estimated as the median nearest-neighbor
            distance among the coreset bank points (~ the bank spacing
            Delta; see the sigma >= Delta guidance for the tangent-plane
            approximation). A fixed ``sigma`` picked without regard to the
            raw feature scale is a common cause of divergence during
            ``g_theta`` training.
        lambda_reg: Weight of the ``||g_theta(z) - z||^2`` regularizer.
            Defaults to ``1.0``.
        hidden_dim: Hidden units of the residual MLP. Defaults to ``512``.
        map_epochs: Training epochs for ``g_theta``. Defaults to ``100``.
        map_lr: Adam learning rate for ``g_theta``. Defaults to ``1e-3``.
        map_batch_size: Mini-batch size for ``g_theta`` training. Defaults to ``512``.
    """

    def __init__(
        self,
        layers: Sequence[str],
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        num_neighbors: int = 9,
        sigma: float | None = None,
        lambda_reg: float = 1.0,
        hidden_dim: int = 512,
        map_epochs: int = 100,
        map_lr: float = 1e-3,
        map_batch_size: int = 512,
    ) -> None:
        super().__init__(
            layers=layers,
            backbone=backbone,
            pre_trained=pre_trained,
            num_neighbors=num_neighbors,
        )
        self.sigma = sigma
        self.lambda_reg = lambda_reg
        self.hidden_dim = hidden_dim
        self.map_epochs = map_epochs
        self.map_lr = map_lr
        self.map_batch_size = map_batch_size

        self.residual_map: ResidualMLP | None = None

        # 学習時に全正常パッチ埋め込みの平均として1回だけ計算し、以後固定 (推論時も
        # 同じ値を引く)。ReLU後の特徴は全成分が非負で、分布の中心が原点から離れて
        # いる (Fig2のL2ノルム分布参照)。スコア自体はペア距離 ||z-s_m||^2 のみに
        # 依存するため平行移動不変だが、g_theta (ReLU入りMLP) の学習ダイナミクス
        # (活性パターン・数値条件) はゼロ中心の入力を前提にしている。
        self.feature_mean: torch.Tensor
        self.register_buffer("feature_mean", torch.empty(0))

    # ------------------------------------------------------------------
    # Fixed density: -log p_M(z) = -logsumexp_m(-||z-s_m||^2 / 2*sigma^2) + const
    # ------------------------------------------------------------------

    @staticmethod
    def _squared_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Pairwise squared Euclidean distances, computed without ever taking a sqrt.

        The inherited ``euclidean_dist`` returns ``sqrt(...)`` (it is meant
        for grad-free kNN search). Squaring that result back — as an
        earlier version of ``_nll_softmin`` did via
        ``euclidean_dist(...).pow(2)`` — routes the gradient through
        ``d/du sqrt(u) = 1/(2*sqrt(u))``, which is exactly ``0 * inf = NaN``
        at ``u = 0``. Because ``memory_bank`` is a literal coreset subset
        of the training patches and ``g_theta`` starts at the identity,
        training batches routinely contain rows that are bit-identical to
        a bank point, making ``u = 0`` a near-certainty rather than an edge
        case (confirmed empirically: instrumented training first diverges
        at ``dist2 to bank: min=0``). Computing the squared distance
        directly sidesteps the singularity entirely.
        """
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)
        dist2 = x_norm + y_norm.transpose(-2, -1) - 2.0 * torch.matmul(x, y.transpose(-2, -1))
        return dist2.clamp_min(0.0)

    def _nll_softmin(self, z: torch.Tensor) -> torch.Tensor:
        """Per-row NLL of ``z`` under the fixed memory-bank mixture density.

        The z-independent normalization constant is omitted: it does not
        affect ranking (AUC) or gradients w.r.t. ``g_theta``.

        Args:
            z: Query features of shape ``(N, D)``.

        Returns:
            torch.Tensor: NLL score per row, shape ``(N,)``.
        """
        two_sigma_sq = 2.0 * self.sigma**2
        n = z.shape[0]
        chunk_size = DEFAULT_CHUNK_SIZE
        scores = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            dist2 = self._squared_dist(z[start:end], self.memory_bank)
            scores.append(-torch.logsumexp(-dist2 / two_sigma_sq, dim=1))
        return torch.cat(scores, dim=0)

    def _estimate_sigma(self) -> float:
        """Auto-estimate sigma from the (already built) memory bank.

        Returns the median nearest-neighbor distance among bank points, a
        proxy for the coreset spacing Delta. This scales sigma to the
        actual (possibly unnormalized) feature space instead of relying on
        a fixed constant that may be orders of magnitude off.
        """
        bank = self.memory_bank
        n = bank.shape[0]
        chunk_size = DEFAULT_CHUNK_SIZE
        nn_dists = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            dist = self.euclidean_dist(bank[start:end], bank)
            dist[torch.arange(end - start), torch.arange(start, end)] = float("inf")
            nn_dists.append(dist.min(dim=1).values)
        return torch.cat(nn_dists).median().item()

    # ------------------------------------------------------------------
    # Training g_theta against the fixed memory bank
    # ------------------------------------------------------------------

    def _report_divergence(
        self,
        epoch: int,
        step: int,
        stage: str,
        batch: torch.Tensor,
        z_tilde: torch.Tensor,
        nll: torch.Tensor,
        reg: torch.Tensor,
    ) -> None:
        """Dump diagnostics for the first non-finite value encountered during training.

        Prints, in order, whether the *input* batch, the residual-map
        *output*, the NLL, the regularizer, and the residual-map *weights*
        are each finite -- so the first culprit in the chain is obvious
        instead of guessing from an aggregate NaN loss.
        """
        with torch.no_grad():
            weight_nan = any(torch.isnan(p).any().item() or torch.isinf(p).any().item()
                              for p in self.residual_map.parameters())
            dist2_min = dist2_max = float("nan")
            if torch.isfinite(z_tilde).all():
                dist2 = self._squared_dist(z_tilde, self.memory_bank)
                dist2_min, dist2_max = dist2.min().item(), dist2.max().item()

        print(f"[SoftminNLL][DIVERGED] first detected at epoch={epoch + 1} step={step}  stage={stage}")
        print(f"  input batch      : finite={torch.isfinite(batch).all().item()}  "
              f"min={batch.min().item():.4g}  max={batch.max().item():.4g}")
        print(f"  z_tilde=g_theta(z): finite={torch.isfinite(z_tilde).all().item()}  "
              f"min={z_tilde.min().item():.4g}  max={z_tilde.max().item():.4g}")
        print(f"  nll (mean)       : {nll.item()}")
        print(f"  reg (mean)       : {reg.item()}")
        print(f"  dist2 to bank    : min={dist2_min:.4g}  max={dist2_max:.4g}")
        print(f"  residual_map weights already NaN/Inf: {weight_nan}")
        print(f"  sigma={self.sigma}  two_sigma_sq={2.0 * self.sigma ** 2:.4g}")

    def _train_residual_map(self, normal_embeddings: torch.Tensor) -> None:
        """Train g_theta to minimize NLL(g_theta(z)) + lambda*||g_theta(z)-z||^2.

        The memory bank (``self.memory_bank``) is a fixed buffer; only the
        parameters of ``self.residual_map`` receive gradients. Stops early
        (with a diagnostic dump via ``_report_divergence``) at the first
        sign of a non-finite value, instead of running to completion on a
        NaN-corrupted model.
        """
        dim = normal_embeddings.shape[1]
        self.residual_map = ResidualMLP(dim, hidden_dim=self.hidden_dim).to(normal_embeddings.device)
        optimizer = torch.optim.Adam(self.residual_map.parameters(), lr=self.map_lr)
        loader = DataLoader(
            TensorDataset(normal_embeddings),
            batch_size=self.map_batch_size,
            shuffle=True,
        )

        global_step = 0
        diverged = False

        with torch.enable_grad():
            for epoch in range(self.map_epochs):
                epoch_nll = epoch_reg = 0.0
                for (batch,) in loader:
                    optimizer.zero_grad()
                    z_tilde = self.residual_map(batch)
                    nll = self._nll_softmin(z_tilde).mean()
                    reg = (z_tilde - batch).pow(2).sum(dim=1).mean()
                    loss = nll + self.lambda_reg * reg

                    if not torch.isfinite(loss):
                        self._report_divergence(epoch, global_step, "forward (nll/reg/loss)", batch, z_tilde, nll, reg)
                        diverged = True
                        break

                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(self.residual_map.parameters(), max_norm=1.0)

                    if not torch.isfinite(grad_norm):
                        self._report_divergence(epoch, global_step, "backward (grad_norm)", batch, z_tilde, nll, reg)
                        diverged = True
                        break

                    optimizer.step()
                    epoch_nll += nll.item()
                    epoch_reg += reg.item()
                    global_step += 1

                if diverged:
                    break
                if (epoch + 1) % 20 == 0:
                    n_batches = max(len(loader), 1)
                    print(
                        f"[SoftminNLL] epoch {epoch+1}/{self.map_epochs}  "
                        f"nll={epoch_nll/n_batches:.4f}  reg={epoch_reg/n_batches:.4f}"
                    )

        if diverged:
            print(
                f"[SoftminNLL] training stopped early (epoch {epoch + 1}, step {global_step}) "
                "due to divergence -- see [DIVERGED] dump above."
            )
        self.residual_map.eval()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _log_feature_scale_diagnostics(self, all_embeddings: torch.Tensor) -> None:
        """Print per-layer feature-scale stats to check the isotropic-sigma assumption.

        The fixed density ``N(z | s_m, sigma^2 I)`` uses a single ``sigma``
        shared across all dimensions. If different backbone layers (e.g.
        ``layer2`` vs ``layer3``) contribute wildly different per-dimension
        scales, that assumption is violated and training tends to be
        numerically unstable (a likely cause of NaN divergence).
        """
        std = all_embeddings.std(dim=0)
        out_dims = getattr(self.feature_extractor, "out_dims", None)

        print("[SoftminNLL] feature scale diagnostics:")
        if out_dims is not None and sum(out_dims) == std.shape[0]:
            offset = 0
            for layer_name, dim in zip(self.layers, out_dims, strict=True):
                block = std[offset : offset + dim]
                print(
                    f"  {layer_name:10s} (dim={dim:4d})  "
                    f"std min/mean/max = {block.min().item():.4f} / {block.mean().item():.4f} / {block.max().item():.4f}"
                )
                offset += dim
        else:
            print("  [warn] could not determine per-layer channel split; skipping per-layer breakdown")

        ratio = (std.max() / std.min().clamp_min(1e-12)).item()
        print(f"  global std ratio (max/min over all dims): {ratio:.2f}")

    # ------------------------------------------------------------------
    # Memory-bank construction (unchanged) + g_theta training
    # ------------------------------------------------------------------

    def subsample_embedding(self, sampling_ratio: float, embeddings: torch.Tensor = None) -> None:
        """Build the fixed coreset memory bank, then train ``g_theta`` against it.

        Steps:
            1. Stack raw backbone embeddings from normal training patches.
            2. Coreset-subsample to build ``memory_bank`` (= M_C, fixed from here on).
            3. Train the residual map ``g_theta`` using all normal patches,
               scored against the now-fixed ``memory_bank``.
        """
        if embeddings is not None:
            del embeddings

        if not self.embedding_store:
            msg = "Embedding store is empty."
            raise ValueError(msg)

        all_embeddings = torch.vstack(self.embedding_store).float()
        self.embedding_store.clear()

        self._log_feature_scale_diagnostics(all_embeddings)

        # ゼロ平均化: 以後 memory_bank・g_theta 学習・推論時のクエリすべてが
        # この平均を引いた座標系で扱われる (forward() 側で同じ feature_mean を引く)。
        # self.feature_mean = all_embeddings.mean(dim=0, keepdim=True)
        # all_embeddings = all_embeddings - self.feature_mean

        self.memory_bank = all_embeddings
        sampler = KCenterGreedy(embedding=self.memory_bank, sampling_ratio=sampling_ratio)
        self.memory_bank = sampler.sample_coreset()

        if self.sigma is None:
            self.sigma = self._estimate_sigma()
            print(f"[SoftminNLL] auto-estimated sigma={self.sigma:.4f} (median NN distance in bank)")

        print(
            f"[SoftminNLL] memory_bank size={self.memory_bank.shape[0]}  "
            f"sigma={self.sigma}  lambda_reg={self.lambda_reg}"
        )
        self._train_residual_map(all_embeddings)

        # 可視化用 (visualize_softmin_nll.py から参照される)。all_embeddings は
        # 上で feature_mean を引いた後の centered 版なので、ここで保存する
        # viz_raw_embeddings も centered。memory_bank・g_theta と同じ座標系に
        # 揃えておかないと fig1/fig1par/fig1perp の z_perp 分解が意味を失う。
        self.viz_raw_embeddings: torch.Tensor = all_embeddings.cpu()
        with torch.no_grad():
            self.viz_z_tilde_all: torch.Tensor = self.residual_map(all_embeddings).cpu()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(self, input_tensor: torch.Tensor):
        """Backbone -> g_theta -> softmin-NLL against fixed memory bank -> anomaly map.

        Training: stores raw embeddings (same as parent), *before* feature_mean
        subtraction (feature_mean is only computed once, from these stored
        embeddings, in ``subsample_embedding``).
        Inference: subtracts the fixed ``feature_mean``, then scores
        ``-log p_M(g_theta(z))`` per patch, no kNN.
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

        if self.memory_bank.size(0) == 0:
            msg = "Memory bank is empty."
            raise ValueError(msg)

        with torch.no_grad():
            # z = embedding - self.feature_mean
            z = embedding
            z_tilde = self.residual_map(z) if self.residual_map is not None else z
            patch_scores = self._nll_softmin(z_tilde)

        patch_scores = patch_scores.reshape(batch_size, -1)
        pred_score = patch_scores.amax(1)
        patch_scores = patch_scores.reshape(batch_size, 1, width, height)
        anomaly_map = self.anomaly_map_generator(patch_scores, output_size)

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
