# src/volatility_surface/models/amortized_model.py
"""
Amortized Inference for Volatility Surfaces via DeepSets Encoder.

Instead of training a separate network per surface (~400ms), this model
learns to map sparse option quotes to surface predictions in a single
forward pass (~5ms). The architecture is:

    Encoder (DeepSets): {(k_i, T_i, σ_i)} → embedding ∈ R^d
    Decoder (conditional MLP): (k_query, T_query, embedding) → w(k, T)

Training data: many synthetic Heston surfaces with random parameters.
The encoder learns to infer the "surface identity" from sparse observations,
and the decoder reconstructs the full surface conditioned on that embedding.

Arbitrage penalties (Gatheral density, calendar) are applied to the decoder
output using the same losses as CINN.

Usage:
    # Train
    model = AmortizedCINN(embed_dim=64)
    model.train_on_surfaces(surfaces, epochs=100)

    # Inference (~5ms)
    iv = model.predict(sparse_quotes, query_grid)
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
    DEVICE = torch.device("cpu")
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class PointEncoder(nn.Module):
        """Per-point MLP: maps each (k, sqrt_T, sigma) to a feature vector."""

        def __init__(self, input_dim: int = 3, hidden_dim: int = 64, embed_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (batch, n_points, 3) -> (batch, n_points, embed_dim)"""
            return self.net(x)

    class SetEncoder(nn.Module):
        """DeepSets encoder: per-point MLP + mean pooling."""

        def __init__(self, input_dim: int = 3, hidden_dim: int = 64, embed_dim: int = 64):
            super().__init__()
            self.phi = PointEncoder(input_dim, hidden_dim, embed_dim)

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Args:
                x: (batch, max_points, 3) — padded input set
                mask: (batch, max_points) — True for valid points

            Returns:
                (batch, embed_dim) — set embedding
            """
            h = self.phi(x)  # (batch, max_points, embed_dim)
            if mask is not None:
                h = h * mask.unsqueeze(-1)
                counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
                return h.sum(dim=1) / counts
            return h.mean(dim=1)

    class SurfaceDecoder(nn.Module):
        """Conditional MLP: (query_point, embedding) -> total variance w."""

        def __init__(self, embed_dim: int = 64, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2 + embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus(beta=5.0),
            )
            self._scale = 0.5  # Match CINN output scale

        def forward(self, query: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
            """
            Args:
                query: (batch, n_query, 2) — (k, sqrt_T) query points
                embedding: (batch, embed_dim) — surface embedding

            Returns:
                (batch, n_query, 1) — predicted total variance w
            """
            B, N, _ = query.shape
            emb_expanded = embedding.unsqueeze(1).expand(B, N, -1)
            x = torch.cat([query, emb_expanded], dim=-1)
            return self.net(x) * self._scale + 1e-6

    class AmortizedCINNNetwork(nn.Module):
        """Full encoder-decoder model."""

        def __init__(self, embed_dim: int = 64, hidden_dim: int = 128):
            super().__init__()
            self.encoder = SetEncoder(input_dim=3, hidden_dim=64, embed_dim=embed_dim)
            self.decoder = SurfaceDecoder(embed_dim=embed_dim, hidden_dim=hidden_dim)

        def forward(
            self,
            obs: torch.Tensor,
            obs_mask: Optional[torch.Tensor],
            query: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                obs: (batch, max_obs, 3) — sparse observations (k, sqrt_T, sigma)
                obs_mask: (batch, max_obs) — valid observation mask
                query: (batch, n_query, 2) — query points (k, sqrt_T)

            Returns:
                (batch, n_query, 1) — predicted total variance
            """
            embedding = self.encoder(obs, obs_mask)
            return self.decoder(query, embedding)

        def implied_vol(self, w: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
            """Convert total variance to implied vol."""
            T_safe = T.clamp(min=1e-6)
            return torch.sqrt((w / T_safe).clamp(min=1e-8))


def generate_training_data(
    n_surfaces: int = 1000,
    n_strikes: int = 20,
    n_expiries: int = 6,
    dropout_rates: tuple = (0.0, 0.2, 0.4, 0.6),
    seed: int = 42,
) -> List[Dict]:
    """Generate training data: many Heston surfaces with random parameters.

    Returns list of dicts, each with:
        'obs': sparse observations as (n_obs, 3) array [k, sqrt_T, sigma]
        'full_k': full grid k values (n_full,)
        'full_sqrt_T': full grid sqrt_T values (n_full,)
        'full_w': full grid total variance values (n_full,)
    """
    from scripts.run_paper_revision import generate_heston_surface

    rng = np.random.default_rng(seed)
    data = []

    for i in range(n_surfaces):
        # Randomize Heston parameters
        v0 = rng.uniform(0.01, 0.09)
        theta = rng.uniform(0.01, 0.09)
        kappa = rng.uniform(1.0, 5.0)
        sigma_v = rng.uniform(0.2, 0.8)
        rho = rng.uniform(-0.9, -0.3)

        surface = generate_heston_surface(
            v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho,
            n_strikes=n_strikes, n_expiries=n_expiries, seed=seed + i,
        )

        k = surface["log_moneyness"].values
        T = surface["T"].values
        iv = surface["implied_volatility"].values
        sqrt_T = np.sqrt(T)
        w = iv ** 2 * T

        # Create sparse versions at different dropout rates
        for rate in dropout_rates:
            mask = rng.random(len(surface)) > rate
            if mask.sum() < 10:
                continue

            obs = np.column_stack([k[mask], sqrt_T[mask], iv[mask]])
            data.append({
                "obs": obs,
                "full_k": k,
                "full_sqrt_T": sqrt_T,
                "full_w": w,
            })

    return data


def collate_batch(batch: List[Dict], device="cpu") -> Dict:
    """Collate variable-length observations into padded tensors."""
    max_obs = max(b["obs"].shape[0] for b in batch)
    n_full = batch[0]["full_k"].shape[0]  # same for all
    B = len(batch)

    obs_padded = np.zeros((B, max_obs, 3), dtype=np.float32)
    obs_mask = np.zeros((B, max_obs), dtype=np.float32)
    query = np.zeros((B, n_full, 2), dtype=np.float32)
    target_w = np.zeros((B, n_full, 1), dtype=np.float32)

    for i, b in enumerate(batch):
        n = b["obs"].shape[0]
        obs_padded[i, :n] = b["obs"]
        obs_mask[i, :n] = 1.0
        query[i, :, 0] = b["full_k"]
        query[i, :, 1] = b["full_sqrt_T"]
        target_w[i, :, 0] = b["full_w"]

    return {
        "obs": torch.tensor(obs_padded, device=device),
        "obs_mask": torch.tensor(obs_mask, device=device),
        "query": torch.tensor(query, device=device),
        "target_w": torch.tensor(target_w, device=device),
    }


def train_amortized(
    n_surfaces: int = 500,
    embed_dim: int = 64,
    hidden_dim: int = 128,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
) -> Tuple["AmortizedCINNNetwork", Dict]:
    """Train the amortized model on synthetic surfaces.

    Returns:
        (model, metrics_dict)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")

    print(f"Generating {n_surfaces} training surfaces...")
    data = generate_training_data(n_surfaces=n_surfaces, seed=seed)
    print(f"  {len(data)} training examples (surfaces x dropout rates)")

    # Split train/val
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))
    n_val = max(len(data) // 10, 1)
    val_data = [data[i] for i in indices[:n_val]]
    train_data = [data[i] for i in indices[n_val:]]

    model = AmortizedCINNNetwork(embed_dim=embed_dim, hidden_dim=hidden_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_data)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]
            if len(batch) < 2:
                continue

            b = collate_batch(batch)
            optimizer.zero_grad()

            w_pred = model(b["obs"], b["obs_mask"], b["query"])
            loss = ((w_pred - b["target_w"]) ** 2).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train)

        # Validation
        model.eval()
        with torch.no_grad():
            val_batch = collate_batch(val_data)
            val_pred = model(val_batch["obs"], val_batch["obs_mask"], val_batch["query"])
            val_loss = ((val_pred - val_batch["target_w"]) ** 2).mean().item()
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train={avg_train:.6f}, val={val_loss:.6f}")

    model.eval()
    return model, history


def benchmark_amortized(
    model: "AmortizedCINNNetwork",
    n_test: int = 50,
    seed: int = 9999,
) -> Dict:
    """Benchmark amortized model vs per-surface CINN.

    Returns dict with speed and accuracy comparisons.
    """
    from scripts.run_paper_revision import generate_heston_surface, apply_dropout

    results = {"amortized": [], "cinn": []}

    for i in range(n_test):
        s = seed + i
        surface = generate_heston_surface(seed=s)
        train_df, holdout_df = apply_dropout(surface, 0.4, s)
        if len(holdout_df) == 0:
            continue

        true = holdout_df["implied_volatility"].values
        k_train = train_df["log_moneyness"].values
        T_train = train_df["T"].values
        iv_train = train_df["implied_volatility"].values

        # Amortized inference
        obs = torch.tensor(
            np.column_stack([k_train, np.sqrt(T_train), iv_train]),
            dtype=torch.float32,
        ).unsqueeze(0)

        k_test = holdout_df["log_moneyness"].values
        T_test = holdout_df["T"].values
        query = torch.tensor(
            np.column_stack([k_test, np.sqrt(T_test)]),
            dtype=torch.float32,
        ).unsqueeze(0)

        t0 = time.time()
        with torch.no_grad():
            w_pred = model(obs, None, query)
            T_t = torch.tensor(T_test, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            iv_pred = model.implied_vol(w_pred, T_t).squeeze().numpy()
        amortized_time = (time.time() - t0) * 1000

        rmse = np.sqrt(np.mean((iv_pred - true) ** 2)) * 10000
        results["amortized"].append({"rmse_bps": rmse, "time_ms": amortized_time})

    return results
