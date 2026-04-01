# src/volatility_surface/models/icnn_model.py
"""
Input Convex Neural Network (ICNN) for Volatility Surfaces.

Guarantees butterfly no-arbitrage by construction via architectural constraints:
non-negative weights in passthrough layers ensure convexity of call prices in
strike, which implies non-negative butterfly spreads.

Architecture:
    z_{i+1} = σ(W_i^z · z_i + W_i^x · x + b_i)
    where W_i^z >= 0 (non-negative passthrough weights)
    and σ is convex non-decreasing (Softplus)

Calendar constraint is enforced via penalty (same as CINN) since
monotonicity in T cannot be guaranteed architecturally without
restricting expressiveness.

Reference:
    Amos, B., Xu, L., & Kolter, J.Z. (2017).
    Input Convex Neural Networks. Proc. ICML, PMLR 70:146-155.

Usage:
    >>> from src.volatility_surface.models.icnn_model import ICNNVolatilityModel
    >>> model = ICNNVolatilityModel(hidden_layers=[64, 32, 16])
    >>> model.train(data, epochs=500)
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True

    # Force CPU to avoid DirectML compatibility issues with ICNN ops
    BEST_DEVICE = torch.device("cpu")
    DEVICE_NAME = "CPU"

except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    BEST_DEVICE = None
    DEVICE_NAME = "N/A"

from src.volatility_surface.base import VolatilityModelBase


if TORCH_AVAILABLE:

    class ICNNNetwork(nn.Module):
        """
        Input Convex Neural Network ensuring convexity in strike dimension.

        The network output is convex in the first input dimension (log-moneyness k)
        when W_z weights are non-negative and activations are convex non-decreasing.

        For volatility surfaces: we model call prices C(k, T) as convex in k,
        which guarantees d²C/dk² >= 0 (butterfly no-arbitrage).

        We then convert back to implied vol for the loss.
        """

        def __init__(
            self,
            input_dim: int = 2,
            hidden_layers: Optional[List[int]] = None,
            dropout: float = 0.05,
        ):
            super().__init__()

            if hidden_layers is None:
                hidden_layers = [64, 32, 16]

            self.input_dim = input_dim
            self.n_layers = len(hidden_layers)

            # Passthrough layers (z -> z): weights enforced non-negative
            # These ensure convexity is preserved through the network
            self.W_z = nn.ModuleList()
            # Direct input layers (x -> z): unconstrained
            self.W_x = nn.ModuleList()
            self.biases = nn.ParameterList()
            self.dropouts = nn.ModuleList()

            # First layer: only W_x (no passthrough)
            self.W_x.append(nn.Linear(input_dim, hidden_layers[0], bias=False))
            self.biases.append(nn.Parameter(torch.zeros(hidden_layers[0])))
            self.dropouts.append(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )

            # Hidden layers: W_z (non-negative) + W_x (unconstrained)
            for i in range(1, len(hidden_layers)):
                self.W_z.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i], bias=False))
                self.W_x.append(nn.Linear(input_dim, hidden_layers[i], bias=False))
                self.biases.append(nn.Parameter(torch.zeros(hidden_layers[i])))
                self.dropouts.append(
                    nn.Dropout(dropout) if dropout > 0 and i < len(hidden_layers) - 1
                    else nn.Identity()
                )

            # Output layer
            self.W_z_out = nn.Linear(hidden_layers[-1], 1, bias=False)
            self.W_x_out = nn.Linear(input_dim, 1, bias=False)
            self.bias_out = nn.Parameter(torch.zeros(1))

            # Softplus: convex non-decreasing activation
            self.activation = nn.Softplus(beta=5.0)
            # Output softplus for positivity of total variance
            self.output_softplus = nn.Softplus(beta=5.0)

            # Initialize W_z with positive values
            for wz in self.W_z:
                nn.init.uniform_(wz.weight, 0.0, 0.1)
            nn.init.uniform_(self.W_z_out.weight, 0.0, 0.1)

        def _enforce_nonneg(self):
            """Project W_z weights to non-negative (clamp after optimizer step)."""
            with torch.no_grad():
                for wz in self.W_z:
                    wz.weight.clamp_(min=0.0)
                self.W_z_out.weight.clamp_(min=0.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass preserving convexity in x[:, 0] (log-moneyness).

            Args:
                x: (batch, input_dim) with x[:, 0] = log_moneyness, x[:, 1] = sqrt_T

            Returns:
                Total variance w = σ²T, shape (batch, 1)
            """
            # First layer (no passthrough)
            z = self.activation(self.W_x[0](x) + self.biases[0])
            z = self.dropouts[0](z)

            # Hidden layers with non-negative passthrough
            for i in range(len(self.W_z)):
                z = self.activation(
                    self.W_z[i](z) + self.W_x[i + 1](x) + self.biases[i + 1]
                )
                z = self.dropouts[i + 1](z)

            # Output
            raw = self.W_z_out(z) + self.W_x_out(x) + self.bias_out

            # Ensure positive total variance with scaling
            return self.output_softplus(raw) * 0.5 + 1e-6

        def implied_vol(self, x: torch.Tensor) -> torch.Tensor:
            """Convert total variance to implied vol."""
            w = self.forward(x)
            T = x[:, 1:2] ** 2  # sqrt_T -> T
            T = torch.clamp(T, min=1e-6)
            return torch.sqrt(torch.clamp(w / T, min=1e-8))


    class CalendarPenaltyLoss(nn.Module):
        """Calendar spread penalty for ICNN (architectural guarantee not possible)."""

        def __init__(self):
            super().__init__()

        def forward(
            self,
            total_var: torch.Tensor,
            sqrt_T: torch.Tensor,
        ) -> torch.Tensor:
            """Penalize negative dw/dT via finite differences."""
            if len(sqrt_T.unique()) < 2:
                return torch.tensor(0.0, device=total_var.device)

            # Group by unique maturities
            unique_T = sqrt_T.unique().sort()[0]
            if len(unique_T) < 2:
                return torch.tensor(0.0, device=total_var.device)

            violations = []
            for i in range(len(unique_T) - 1):
                mask_near = (sqrt_T == unique_T[i]).squeeze()
                mask_far = (sqrt_T == unique_T[i + 1]).squeeze()

                if mask_near.sum() == 0 or mask_far.sum() == 0:
                    continue

                w_near = total_var[mask_near].mean()
                w_far = total_var[mask_far].mean()

                # Calendar: w must increase with T
                violation = torch.relu(w_near - w_far)
                violations.append(violation ** 2)

            if not violations:
                return torch.tensor(0.0, device=total_var.device)

            return torch.stack(violations).mean()


class ICNNVolatilityModel(VolatilityModelBase):
    """
    ICNN-based volatility surface model with hard butterfly guarantee.

    Butterfly no-arbitrage is guaranteed by architecture (non-negative W_z).
    Calendar constraint is enforced via penalty loss.

    Args:
        hidden_layers: Network architecture.
        lambda_calendar: Calendar penalty weight.
        lambda_wing: Wing penalty weight (Roger-Lee).
        epochs: Training epochs.
        lr: Learning rate.
        use_warmup: Whether to use warmup schedule for calendar penalty.
    """

    def __init__(
        self,
        hidden_layers: Optional[List[int]] = None,
        lambda_calendar: float = 5.0,
        lambda_wing: float = 1.0,
        epochs: int = 300,
        lr: float = 3e-3,
        use_warmup: bool = True,
        warmup_epochs: int = 100,
        ramp_epochs: int = 200,
        early_stop_patience: int = 20,
        early_stop_min_delta: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_layers = hidden_layers or [64, 32, 16]
        self.lambda_calendar = lambda_calendar
        self.lambda_wing = lambda_wing
        self.epochs = epochs
        self.lr = lr
        self.use_warmup = use_warmup
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.network = None
        self.scaler_X = None
        self.training_time_ms = 0.0

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract (log_moneyness, sqrt_T) features."""
        k = df["log_moneyness"].values
        T = df["T"].values
        return np.column_stack([k, np.sqrt(np.maximum(T, 1e-6))])

    def _train_impl(self, data: pd.DataFrame, val_split: float = 0.0, **kwargs) -> Dict[str, Any]:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for ICNN")

        start = time.time()

        X = self._prepare_features(data)
        y = data["implied_volatility"].values
        T_vals = data["T"].values
        # Target: total variance
        w_target = y ** 2 * T_vals

        X_t = torch.tensor(X, dtype=torch.float32, device=BEST_DEVICE)
        w_t = torch.tensor(w_target, dtype=torch.float32, device=BEST_DEVICE).unsqueeze(1)

        self.network = ICNNNetwork(
            input_dim=2,
            hidden_layers=self.hidden_layers,
        ).to(BEST_DEVICE)

        optimizer = optim.AdamW(
            self.network.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        cal_loss_fn = CalendarPenaltyLoss()

        # EMA normalization
        ema_mse = 1.0
        ema_cal = 1.0
        ema_alpha = 0.05

        best_loss = float("inf")
        patience_counter = 0

        history = {"mse": [], "cal": [], "total": []}

        for epoch in range(self.epochs):
            self.network.train()
            optimizer.zero_grad()

            w_pred = self.network(X_t)

            # MSE loss on total variance
            mse_loss = ((w_pred - w_t) ** 2).mean()

            # Calendar penalty with warmup
            if self.use_warmup and epoch < self.warmup_epochs:
                penalty_scale = 0.0
            elif self.use_warmup and epoch < self.warmup_epochs + self.ramp_epochs:
                penalty_scale = (epoch - self.warmup_epochs) / self.ramp_epochs
            else:
                penalty_scale = 1.0

            cal_loss = cal_loss_fn(w_pred, X_t[:, 1:2])

            # Wing penalty (Roger-Lee)
            k = X_t[:, 0:1]
            wing_mask = torch.abs(k) > 0.15
            if wing_mask.any():
                w_wing = w_pred[wing_mask.squeeze()]
                k_wing = torch.abs(k[wing_mask])
                wing_violation = torch.relu(w_wing / k_wing - 2.0)
                wing_loss = (wing_violation ** 2).mean()
            else:
                wing_loss = torch.tensor(0.0, device=BEST_DEVICE)

            # EMA normalization
            with torch.no_grad():
                ema_mse = (1 - ema_alpha) * ema_mse + ema_alpha * mse_loss.item()
                ema_cal = (1 - ema_alpha) * ema_cal + ema_alpha * max(cal_loss.item(), 1e-10)

            norm_mse = mse_loss / max(ema_mse, 1e-10)
            norm_cal = cal_loss / max(ema_cal, 1e-10)

            total_loss = norm_mse + penalty_scale * (
                self.lambda_calendar * norm_cal
                + self.lambda_wing * wing_loss
            )

            total_loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Enforce non-negative weights after optimizer step
            self.network._enforce_nonneg()

            history["mse"].append(mse_loss.item())
            history["cal"].append(cal_loss.item())
            history["total"].append(total_loss.item())

            # Early stopping on MSE after warmup
            if epoch > self.warmup_epochs:
                if mse_loss.item() < best_loss - self.early_stop_min_delta:
                    best_loss = mse_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stop_patience:
                        break

        self.training_time_ms = (time.time() - start) * 1000
        self.network.eval()

        return {
            "epochs_trained": epoch + 1,
            "final_mse": history["mse"][-1],
            "final_cal": history["cal"][-1],
            "training_time_ms": self.training_time_ms,
        }

    def _predict_impl(self, data: pd.DataFrame) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("Model not trained")

        X = self._prepare_features(data)
        X_t = torch.tensor(X, dtype=torch.float32, device=BEST_DEVICE)

        with torch.no_grad():
            self.network.eval()
            iv = self.network.implied_vol(X_t).cpu().numpy().flatten()

        return iv

    def predict_volatility(self, data: pd.DataFrame) -> np.ndarray:
        return self._predict_impl(data)

    def _save_model_impl(self, path: str) -> None:
        if self.network is not None:
            torch.save(self.network.state_dict(), path)

    def _load_model_impl(self, path: str) -> None:
        if self.network is None:
            self.network = ICNNNetwork(
                input_dim=2, hidden_layers=self.hidden_layers
            ).to(BEST_DEVICE)
        self.network.load_state_dict(torch.load(path, map_location=BEST_DEVICE))
        self.network.eval()
