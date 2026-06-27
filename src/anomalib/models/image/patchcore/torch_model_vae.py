# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PatchCore with VAE feature normalization.

Extends PatchcoreModel by inserting a VAE between the backbone features and the
memory bank.  Only the VAE's posterior mean (μ) is stored, which maps normal-class
features to an approximately Gaussian latent space.  kNN scoring is then performed
in that latent space so the scoring metric benefits from a tighter, more isotropic
feature distribution.

Training pipeline:
    1. Epoch 1 (training_step): backbone features are collected into embedding_store
       as usual — VAE has not been trained yet.
    2. fit() / subsample_embedding(): VAE is trained on the collected features via
       ELBO, then every stored embedding is encoded to its μ.  The resulting μ-bank
       is coreset-subsampled and saved as memory_bank.

Inference:
    backbone features → VAE encoder → μ → kNN against μ-bank → anomaly score
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.models.components import KCenterGreedy

from .torch_model import PatchcoreModel


class PatchcoreVAE(nn.Module):
    """Maps patch features to a latent normal distribution (μ, log σ²).

    Args:
        input_dim: Dimensionality of incoming patch features.
        latent_dim: Dimensionality of the latent space (μ space).
    """

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        hidden_dim = max(input_dim // 2, latent_dim)
        self.encoder_mu = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (μ, log σ²) for input x."""
        return self.encoder_mu(x), self.encoder_logvar(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (reconstruction, μ, log σ²).

        Uses reparameterisation trick during training; returns μ directly at eval.
        """
        mu, logvar = self.encode(x)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return self.decoder(z), mu, logvar


class PatchcoreVAEModel(PatchcoreModel):
    """PatchCore model with VAE-based feature normalization.

    Inherits the full PatchcoreModel pipeline and overrides the memory-bank
    construction and the inference embedding step to operate in VAE latent space.

    Args:
        layers: Names of backbone layers to extract features from.
        backbone: Timm backbone name. Defaults to ``"wide_resnet50_2"``.
        pre_trained: Use pretrained backbone weights. Defaults to ``True``.
        num_neighbors: Number of nearest neighbours for scoring. Defaults to ``9``.
        latent_dim: VAE latent dimensionality.  Defaults to ``256``.
        vae_epochs: Number of epochs to train the VAE. Defaults to ``50``.
        vae_lr: Learning rate for VAE training. Defaults to ``1e-3``.
        vae_batch_size: Mini-batch size used inside VAE training. Defaults to ``512``.
    """

    def __init__(
        self,
        layers: Sequence[str],
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        num_neighbors: int = 9,
        latent_dim: int = 256,
        vae_epochs: int = 500,
        vae_lr: float = 1e-3,
        vae_batch_size: int = 512,
    ) -> None:
        super().__init__(
            layers=layers,
            backbone=backbone,
            pre_trained=pre_trained,
            num_neighbors=num_neighbors,
        )
        self.latent_dim = latent_dim
        self.vae_epochs = vae_epochs
        self.vae_lr = vae_lr
        self.vae_batch_size = vae_batch_size
        # VAE is built lazily once the embedding dimensionality is known.
        self.vae: PatchcoreVAE | None = None

    # ------------------------------------------------------------------
    # VAE training helper
    # ------------------------------------------------------------------

    def _train_vae(self, embeddings: torch.Tensor) -> None:
        """Train VAE on collected patch embeddings using ELBO loss.

        Args:
            embeddings: All training patch embeddings, shape (N, D).
        """
        assert self.vae is not None, "VAE must be initialised before training."
        self.vae.train()
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.vae_lr)
        dataset = torch.utils.data.TensorDataset(embeddings)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.vae_batch_size,
            shuffle=True,
            drop_last=False,
        )
        # torch.enable_grad() は Lightning の validation ループが張る
        # no_grad コンテキストを上書きする。fit() は on_validation_start()
        # から呼ばれるため、これがないと loss.backward() が失敗する。
        with torch.enable_grad():
            for epoch in range(self.vae_epochs):
                epoch_loss = 0.0
                for (batch,) in loader:
                    optimizer.zero_grad()
                    recon, mu, logvar = self.vae(batch)
                    recon_loss = F.mse_loss(recon, batch)
                    # KL divergence: -0.5 * mean(1 + log σ² - μ² - σ²)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + kl_loss
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                if (epoch + 1) % 10 == 0:
                    n_batches = max(len(loader), 1)
                    print(f"[PatchcoreVAE0.5] epoch {epoch + 1}/{self.vae_epochs}  loss={epoch_loss / n_batches:.4f}")

    # ------------------------------------------------------------------
    # Memory-bank construction  (overrides parent's subsample_embedding)
    # ------------------------------------------------------------------

    def subsample_embedding(self, sampling_ratio: float, embeddings: torch.Tensor = None) -> None:
        """Build the μ-based memory bank.

        Steps:
            1. Stack all stored raw embeddings.
            2. Initialise and train the VAE on those embeddings.
            3. Encode every embedding to its posterior mean μ.
            4. Apply coreset subsampling in μ-space and store as memory_bank.

        Args:
            sampling_ratio: Fraction of embeddings to keep after coreset selection.
            embeddings: **Unused** — kept for API compatibility with parent.
        """
        if embeddings is not None:
            del embeddings  # deprecated argument

        if len(self.embedding_store) == 0:
            msg = "Embedding store is empty. Cannot perform coreset selection."
            raise ValueError(msg)

        all_embeddings = torch.vstack(self.embedding_store).float()
        self.embedding_store.clear()

        # Build and train the VAE in float32 regardless of model precision.
        input_dim = all_embeddings.shape[1]
        self.vae = PatchcoreVAE(input_dim=input_dim, latent_dim=self.latent_dim)
        self.vae = self.vae.to(all_embeddings.device)
        print(
            f"[PatchcoreVAE] training VAE  input_dim={input_dim}  latent_dim={self.latent_dim}"
            f"  epochs={self.vae_epochs}"
        )
        self._train_vae(all_embeddings)

        # Encode all embeddings to μ and use as memory bank.
        self.vae.eval()
        with torch.no_grad():
            mu, _ = self.vae.encode(all_embeddings)

        # 可視化用に VAE 適用前の生特徴量を保持（メモリバンク構築後に参照可能）
        self.viz_raw_embeddings: torch.Tensor = all_embeddings.cpu()
        self.viz_mu_all: torch.Tensor = mu.cpu()

        self.memory_bank = mu
        sampler = KCenterGreedy(embedding=self.memory_bank, sampling_ratio=sampling_ratio)
        self.memory_bank = sampler.sample_coreset()

    # ------------------------------------------------------------------
    # Inference  (overrides parent's forward only for the embedding step)
    # ------------------------------------------------------------------

    def forward(self, input_tensor: torch.Tensor):
        """Process input through backbone → VAE encoder (μ) → kNN.

        During training, raw backbone embeddings are stored for later VAE training.
        During inference, embeddings are projected to μ before kNN search.
        """
        # dtype / tiling handled identically to the parent
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
            # Store raw features; VAE will be trained later in subsample_embedding.
            self.embedding_store.append(embedding)
            return embedding

        if self.memory_bank.size(0) == 0:
            msg = "Memory bank is empty. Cannot provide anomaly scores"
            raise ValueError(msg)

        # Project to μ-space before kNN.
        if self.vae is not None:
            self.vae.eval()
            with torch.no_grad():
                mu, _ = self.vae.encode(embedding.float())
            embedding = mu

        from anomalib.data import InferenceBatch

        patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))
        pred_score = self.compute_anomaly_score(patch_scores, locations, embedding)
        patch_scores = patch_scores.reshape((batch_size, 1, width, height))
        anomaly_map = self.anomaly_map_generator(patch_scores, output_size)

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
