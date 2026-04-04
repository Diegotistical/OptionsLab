# src/volatility_surface/models/icnn_model.py
"""
Input Convex Neural Network (ICNN) for Volatility Surfaces.

Hybrid arbitrage enforcement:

1. **Architectural (hard)**: Non-negative passthrough weights W_z >= 0 guarantee
   the total variance w(k) is convex in log-moneyness k. This is a strong
   inductive bias that pushes solutions toward smooth, well-behaved surfaces.

2. **Gatheral density (soft)**: The true butterfly no-arbitrage condition is
   g(k) >= 0 (Gatheral's density condition), NOT simple convexity of w.
   We enforce g(k) >= 0 via autodiff penalty on the Breeden-Litzenberger
   density, using the same approach as the PINN model.

   g(k) = (1 - k·w'/2w)² - (w')²/4·(1/w + 1/4) + w''/2 >= 0

Together, the architectural convexity provides strong regularization while
the Gatheral penalty enforces the correct no-arbitrage condition.

Calendar constraint is enforced via penalty (same as CINN) since
monotonicity in T cannot be guaranteed architecturally.

Architecture:
    z_{i+1} = σ(W_i^z · z_i + W_i^x · x + b_i)
    where W_i^z >= 0 (non-negative passthrough weights)
    and σ is convex non-decreasing (Softplus)

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
        Input Convex Neural Network for total variance surfaces.

        Guarantees convexity of w(k) in log-moneyness k via non-negative
        passthrough weights — a strong inductive bias for volatility surfaces.

        Input:  (log_moneyness k, sqrt_T)
        Output: total variance w = σ²T, shape (batch, 1)
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
        """Calendar spread penalty: total variance must increase with maturity."""

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


    class GatheralDensityLoss(nn.Module):
        """
        Butterfly no-arbitrage via Gatheral's density condition g(k) >= 0.

        Uses automatic differentiation to compute dw/dk and d²w/dk² and
        evaluates the Breeden-Litzenberger density condition:

            g(k) = (1 - k·w'/(2w))² - (w')²/4·(1/w + 1/4) + w''/2 >= 0

        This is the necessary and sufficient condition for non-negative
        risk-neutral density (butterfly no-arbitrage).

        Reference: Gatheral, J. (2004). A parsimonious arbitrage-free
        implied volatility parameterization.
        """

        def __init__(self, epsilon: float = 1e-6):
            super().__init__()
            self.epsilon = epsilon

        def forward(
            self,
            model: nn.Module,
            x: torch.Tensor,
        ) -> torch.Tensor:
            """Compute Gatheral density violation penalty via autodiff."""
            x = x.detach().requires_grad_(True)

            w = model(x)
            k = x[:, 0:1]

            # First derivative dw/dk
            grad_w = torch.autograd.grad(
                outputs=w,
                inputs=x,
                grad_outputs=torch.ones_like(w),
                create_graph=True,
                retain_graph=True,
            )[0]
            dw_dk = grad_w[:, 0:1]

            # Second derivative d²w/dk²
            grad2_w = torch.autograd.grad(
                outputs=dw_dk,
                inputs=x,
                grad_outputs=torch.ones_like(dw_dk),
                create_graph=True,
                retain_graph=True,
            )[0]
            d2w_dk2 = grad2_w[:, 0:1]

            # Gatheral density: g(k) = (1 - k*w'/2w)^2 - (w')^2/4*(1/w + 1/4) + w''/2
            w_safe = w + self.epsilon
            term1 = (1 - k * dw_dk / (2 * w_safe)) ** 2
            term2 = (dw_dk ** 2) / 4 * (1 / w_safe + 0.25)
            term3 = d2w_dk2 / 2

            g_k = term1 - term2 + term3

            # Penalize negative g(k) (squared hinge)
            violations = torch.relu(-g_k)
            return (violations ** 2).mean()


class ICNNVolatilityModel(VolatilityModelBase):
    """
    ICNN-based volatility surface model with hybrid arbitrage enforcement.

    Architectural guarantee: total variance w(k) is convex in log-moneyness
    (strong inductive bias via non-negative W_z).

    Butterfly no-arbitrage: Gatheral density condition g(k) >= 0 enforced
    via autodiff penalty on the Breeden-Litzenberger density.

    Calendar no-arbitrage: total variance monotone in T via penalty.

    Args:
        hidden_layers: Network architecture.
        lambda_calendar: Calendar penalty weight.
        lambda_butterfly: Gatheral density penalty weight.
        lambda_wing: Wing penalty weight (Roger-Lee).
        epochs: Training epochs.
        lr: Learning rate.
        use_warmup: Whether to use warmup schedule for penalties.
    """

    def __init__(
        self,
        hidden_layers: Optional[List[int]] = None,
        lambda_calendar: float = 5.0,
        lambda_butterfly: float = 5.0,
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
        self.lambda_butterfly = lambda_butterfly
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
        butterfly_loss_fn = GatheralDensityLoss()

        # EMA normalization
        ema_mse = 1.0
        ema_cal = 1.0
        ema_but = 1.0
        ema_alpha = 0.05

        best_loss = float("inf")
        patience_counter = 0

        history = {"mse": [], "cal": [], "butterfly": [], "total": []}

        for epoch in range(self.epochs):
            self.network.train()
            optimizer.zero_grad()

            w_pred = self.network(X_t)

            # MSE loss on total variance
            mse_loss = ((w_pred - w_t) ** 2).mean()

            # Penalty warmup schedule
            if self.use_warmup and epoch < self.warmup_epochs:
                penalty_scale = 0.0
            elif self.use_warmup and epoch < self.warmup_epochs + self.ramp_epochs:
                penalty_scale = (epoch - self.warmup_epochs) / self.ramp_epochs
            else:
                penalty_scale = 1.0

            # Calendar penalty
            cal_loss = cal_loss_fn(w_pred, X_t[:, 1:2])

            # Gatheral density (butterfly) penalty via autodiff
            but_loss = butterfly_loss_fn(self.network, X_t)

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
                ema_but = (1 - ema_alpha) * ema_but + ema_alpha * max(but_loss.item(), 1e-10)

            norm_mse = mse_loss / max(ema_mse, 1e-10)
            norm_cal = cal_loss / max(ema_cal, 1e-10)
            norm_but = but_loss / max(ema_but, 1e-10)

            total_loss = norm_mse + penalty_scale * (
                self.lambda_calendar * norm_cal
                + self.lambda_butterfly * norm_but
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
            history["butterfly"].append(but_loss.item())
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
            "final_butterfly": history["butterfly"][-1],
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
