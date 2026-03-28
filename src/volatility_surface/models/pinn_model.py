# src/volatility_surface/models/pinn_model.py
"""
Hybrid Physics-Informed Neural Network for Volatility Surfaces.

This model combines neural network flexibility with physics-based arbitrage
constraints in the loss function, ensuring arbitrage-free volatility surfaces.

Mathematical Framework:
    Loss = MSE(σ̂, σ_market) + λ_cal·L_calendar + λ_but·L_butterfly + λ_wing·L_wing

where:
    L_calendar  = Σ max(0, -∂w/∂T)          # No calendar spread arbitrage
    L_butterfly = Σ max(0, -∂²C/∂K²)        # Positive butterfly spreads
    L_wing      = ||σ̂(k→±∞) - σ_asymptotic|| # Reasonable wing behavior

Reference:
    Novel approach combining:
    - Gatheral's SVI arbitrage conditions (2004)
    - Physics-Informed Neural Networks (Raissi et al., 2019)
    - Volatility surface arbitrage theory (Roper, 2010)

Usage:
    >>> from src.volatility_surface.models.pinn_model import PINNVolatilityModel
    >>> model = PINNVolatilityModel(lambda_calendar=1.0, lambda_butterfly=0.5)
    >>> model.train(data, epochs=500)
    >>> predictions = model.predict_volatility(test_data)
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True

    # AMD GPU Support: Try DirectML first (Windows AMD), then CUDA, then CPU
    def get_best_device():
        """Get the best available device with AMD GPU support."""
        # Try DirectML for AMD GPUs on Windows
        try:
            import torch_directml

            dml_device = torch_directml.device()
            # Test if it works
            test_tensor = torch.zeros(1, device=dml_device)
            del test_tensor
            return dml_device, "AMD (DirectML)"
        except Exception:
            pass

        # Try CUDA (NVIDIA)
        if torch.cuda.is_available():
            return torch.device("cuda"), "NVIDIA (CUDA)"

        # Try ROCm (AMD on Linux)
        try:
            if hasattr(torch, "hip") and torch.hip.is_available():
                return torch.device("cuda"), "AMD (ROCm/HIP)"
        except Exception:
            pass

        # Fallback to CPU
        return torch.device("cpu"), "CPU"

    BEST_DEVICE, DEVICE_NAME = get_best_device()

except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    BEST_DEVICE = None
    DEVICE_NAME = "N/A"

from src.volatility_surface.base import VolatilityModelBase

# =============================================================================
# Arbitrage Constraint Functions
# =============================================================================


@dataclass
class ArbitrageMetrics:
    """Container for arbitrage violation metrics."""

    calendar_violations: int = 0
    butterfly_violations: int = 0
    total_variance_violations: int = 0
    arbitrage_free_pct: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calendar_violations": self.calendar_violations,
            "butterfly_violations": self.butterfly_violations,
            "total_variance_violations": self.total_variance_violations,
            "arbitrage_free_pct": self.arbitrage_free_pct,
        }


def check_calendar_arbitrage(
    total_variance: np.ndarray,
    maturities: np.ndarray,
    tol: float = 1e-6,
) -> Tuple[bool, int]:
    """
    Check calendar spread arbitrage condition.

    No calendar arbitrage requires: ∂w/∂T ≥ 0
    i.e., total variance must be non-decreasing in maturity.

    Args:
        total_variance: Array of total variance w = σ²T at each point
        maturities: Corresponding maturities
        tol: Tolerance for numerical checks

    Returns:
        (is_arbitrage_free, num_violations)
    """
    # Sort by maturity
    sorted_idx = np.argsort(maturities)
    w_sorted = total_variance[sorted_idx]

    # Check if w is non-decreasing
    dw = np.diff(w_sorted)
    violations = np.sum(dw < -tol)

    return violations == 0, int(violations)


def check_butterfly_arbitrage(
    implied_vols: np.ndarray,
    log_strikes: np.ndarray,
    T: float,
    tol: float = 1e-6,
) -> Tuple[bool, int]:
    """
    Check butterfly spread (strike convexity) arbitrage condition.

    No butterfly arbitrage requires: ∂²C/∂K² ≥ 0
    This translates to constraints on the vol smile curvature.

    For the smile w(k), the condition becomes:
        g(k) = (1 - k·w'/2w)² - w'²/4·(1/w + 1/4) + w''/2 ≥ 0

    Args:
        implied_vols: Array of implied volatilities
        log_strikes: Log-moneyness values
        T: Time to maturity

    Returns:
        (is_arbitrage_free, num_violations)
    """
    # For simplicity, check second derivative of call prices
    # A sufficient condition is that smile is not too steep

    # Compute total variance
    w = implied_vols**2 * T

    # Numerical second derivative
    if len(log_strikes) < 3:
        return True, 0

    sorted_idx = np.argsort(log_strikes)
    k_sorted = log_strikes[sorted_idx]
    w_sorted = w[sorted_idx]

    violations = 0
    for i in range(1, len(k_sorted) - 1):
        dk = k_sorted[i + 1] - k_sorted[i - 1]
        if dk < 1e-10:
            continue
        # Second derivative of w
        d2w = (w_sorted[i + 1] - 2 * w_sorted[i] + w_sorted[i - 1]) / (dk / 2) ** 2

        # First derivative of w
        dw = (w_sorted[i + 1] - w_sorted[i - 1]) / dk

        # Simplified density positivity check
        # g(k) should be >= 0
        k = k_sorted[i]
        wi = w_sorted[i]

        if wi < 1e-10:
            continue

        term1 = (1 - k * dw / (2 * wi)) ** 2
        term2 = dw**2 / 4 * (1 / wi + 0.25)
        term3 = d2w / 2

        g_k = term1 - term2 + term3

        if g_k < -tol:
            violations += 1

    return violations == 0, violations


# =============================================================================
# PINN Neural Network Architecture
# =============================================================================


if TORCH_AVAILABLE:

    class PINNNetwork(nn.Module):
        """
        Physics-Informed Neural Network for Volatility Surfaces.

        Architecture includes:
        - Input: (log_moneyness, sqrt(T), additional features)
        - Input-concatenated residual connections at each hidden layer
        - Output: total variance w = σ²T (ensures positivity via softplus)
        """

        def __init__(
            self,
            input_dim: int = 2,
            hidden_layers: List[int] = None,
            dropout: float = 0.1,
            use_residual: bool = True,
            activation: str = "gelu",
        ):
            super().__init__()

            if hidden_layers is None:
                hidden_layers = [64, 32, 16]

            self.input_dim = input_dim
            self.use_residual = use_residual

            # Activation factory
            def make_activation(name: str):
                if name == "gelu":
                    return nn.GELU()
                elif name == "relu":
                    return nn.ReLU()
                elif name == "silu":
                    return nn.SiLU()
                else:
                    return nn.GELU()

            # Build layers with input-concatenated residual connections
            self.linears = nn.ModuleList()
            self.activations = nn.ModuleList()
            self.dropouts = nn.ModuleList()

            prev_dim = input_dim
            for i, hidden_dim in enumerate(hidden_layers):
                self.linears.append(nn.Linear(prev_dim, hidden_dim))
                self.activations.append(make_activation(activation))
                if dropout > 0 and i < len(hidden_layers) - 1:
                    self.dropouts.append(nn.Dropout(dropout))
                else:
                    self.dropouts.append(nn.Identity())
                # Next layer input: hidden_dim + input_dim (residual concat)
                prev_dim = (hidden_dim + input_dim) if use_residual else hidden_dim

            # Output layer: produces total variance
            self.output = nn.Linear(prev_dim, 1)

            # Softplus to ensure positivity
            self.softplus = nn.Softplus(beta=5.0)

            # Wing asymptote parameters (learnable)
            self.wing_slope = nn.Parameter(torch.tensor(0.1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass with input-concatenated residual connections.

            At each hidden layer, the original input is concatenated to the
            layer output before feeding into the next layer. This preserves
            gradient flow and allows deeper layers direct access to raw features.

            Args:
                x: Input tensor of shape (batch, input_dim)
                   Expected: [log_moneyness, sqrt_T, ...]

            Returns:
                Total variance w = σ²T
            """
            h = x
            for i, (linear, act, drop) in enumerate(
                zip(self.linears, self.activations, self.dropouts)
            ):
                h = drop(act(linear(h)))
                # Concatenate original input as residual connection
                if self.use_residual:
                    h = torch.cat([h, x], dim=-1)

            raw_output = self.output(h)

            # Ensure positive total variance with minimum
            w = self.softplus(raw_output) * 0.5 + 1e-6

            return w

        def implied_vol(self, x: torch.Tensor) -> torch.Tensor:
            """Compute implied volatility from total variance."""
            # x[:, 1] is sqrt_T, so T = x[:, 1]**2
            sqrt_T = x[:, 1:2]
            T = sqrt_T**2 + 1e-10

            w = self.forward(x)
            sigma = torch.sqrt(w / T)

            return sigma

    class CalendarLoss(nn.Module):
        """
        Calendar spread arbitrage penalty.

        Penalizes violations of ∂w/∂T ≥ 0.
        """

        def __init__(self, epsilon: float = 1e-6, squared: bool = True):
            super().__init__()
            self.epsilon = epsilon
            self.squared = squared

        def forward(
            self,
            model: nn.Module,
            x: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute calendar arbitrage penalty.

            Uses automatic differentiation to compute ∂w/∂T.
            """
            x.requires_grad_(True)

            w = model(x)

            # Compute gradient w.r.t. sqrt_T (index 1)
            grad_w = torch.autograd.grad(
                outputs=w,
                inputs=x,
                grad_outputs=torch.ones_like(w),
                create_graph=True,
                retain_graph=True,
            )[0]

            # dw/d(sqrt_T) = 2*sqrt_T * dw/dT
            # So dw/dT = dw/d(sqrt_T) / (2*sqrt_T)
            sqrt_T = x[:, 1:2]
            dw_dsqrtT = grad_w[:, 1:2]
            dw_dT = dw_dsqrtT / (2 * sqrt_T + self.epsilon)

            # Penalize negative derivatives (calendar arb)
            violations = torch.relu(-dw_dT)
            if self.squared:
                violations = violations.pow(2)

            return violations.mean()

    class ButterflyLoss(nn.Module):
        """
        Butterfly spread arbitrage penalty.

        Penalizes violations of Breeden-Litzenberger density positivity.
        """

        def __init__(self, epsilon: float = 1e-6, squared: bool = True):
            super().__init__()
            self.epsilon = epsilon
            self.squared = squared

        def forward(
            self,
            model: nn.Module,
            x: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute butterfly arbitrage penalty.

            Uses automatic differentiation to compute ∂²w/∂k².
            """
            x.requires_grad_(True)

            w = model(x)
            k = x[:, 0:1]  # log-moneyness
            sqrt_T = x[:, 1:2]
            T = sqrt_T**2 + self.epsilon

            # First derivative w.r.t k
            grad_w = torch.autograd.grad(
                outputs=w,
                inputs=x,
                grad_outputs=torch.ones_like(w),
                create_graph=True,
                retain_graph=True,
            )[0]

            dw_dk = grad_w[:, 0:1]

            # Second derivative
            grad2_w = torch.autograd.grad(
                outputs=dw_dk,
                inputs=x,
                grad_outputs=torch.ones_like(dw_dk),
                create_graph=True,
                retain_graph=True,
            )[0]

            d2w_dk2 = grad2_w[:, 0:1]

            # Gatheral's density condition (simplified)
            # g(k) = (1 - k*w'/2w)^2 - (w')^2/4*(1/w + 1/4) + w''/2 >= 0

            term1 = (1 - k * dw_dk / (2 * w + self.epsilon)) ** 2
            term2 = (dw_dk**2) / 4 * (1 / (w + self.epsilon) + 0.25)
            term3 = d2w_dk2 / 2

            g_k = term1 - term2 + term3

            # Penalize negative g(k) (butterfly arb)
            violations = torch.relu(-g_k)
            if self.squared:
                violations = violations.pow(2)

            return violations.mean()

    class WingLoss(nn.Module):
        """
        Wing behavior regularization.

        Encourages Rogers-Lee asymptotic bounds:
            lim(k→±∞) σ(k)/|k| ≤ √(2/T)
        """

        def __init__(self, epsilon: float = 1e-6, squared: bool = True):
            super().__init__()
            self.epsilon = epsilon
            self.squared = squared

        def forward(
            self,
            model: nn.Module,
            x: torch.Tensor,
            wing_threshold: float = 0.3,
        ) -> torch.Tensor:
            """Penalize unreasonable wing behavior."""
            k = x[:, 0:1]
            sqrt_T = x[:, 1:2]
            T = sqrt_T**2 + self.epsilon

            # Get predictions for wing points
            wing_mask = torch.abs(k) > wing_threshold

            if wing_mask.sum() == 0:
                return torch.tensor(0.0, device=x.device)

            w = model(x)
            sigma = torch.sqrt(w / T)

            # Rogers-Lee bound: σ(k) ≤ |k|√(2/T) for |k| → ∞
            max_sigma = torch.abs(k) * torch.sqrt(2 / T)

            # Penalty
            excess = torch.relu(sigma - max_sigma - 0.1)
            if self.squared:
                excess = excess.pow(2)

            return (
                excess[wing_mask].mean() if wing_mask.sum() > 0 else torch.tensor(0.0)
            )


# =============================================================================
# PINN Volatility Model
# =============================================================================


class PINNVolatilityModel(VolatilityModelBase):
    """
    Physics-Informed Neural Network for arbitrage-free volatility surfaces.

    Combines standard MSE loss with physics-based regularization terms
    to enforce no-arbitrage conditions during training.

    Attributes:
        lambda_calendar: Weight for calendar arbitrage penalty (default: 1.0)
        lambda_butterfly: Weight for butterfly arbitrage penalty (default: 0.5)
        lambda_wing: Weight for wing regularization (default: 0.1)
        hidden_layers: Neural network hidden layer sizes
        learning_rate: Optimizer learning rate
        epochs: Number of training epochs
    """

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        hidden_layers: Optional[List[int]] = None,
        lambda_calendar: float = 1.0,
        lambda_butterfly: float = 0.5,
        lambda_wing: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 500,
        batch_size: int = 64,
        dropout: float = 0.1,
        device: Optional[str] = None,
        enable_benchmark: bool = False,
        # Ablation flags
        use_residual: bool = True,
        activation: str = "gelu",
        use_warmup: bool = True,
        use_ema_norm: bool = True,
        squared_hinge: bool = True,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PINNVolatilityModel")

        if feature_columns is None:
            feature_columns = ["log_moneyness", "T"]

        super().__init__(
            feature_columns=feature_columns, enable_benchmark=enable_benchmark
        )

        self.hidden_layers = hidden_layers or [64, 32, 16]
        self.lambda_calendar = lambda_calendar
        self.lambda_butterfly = lambda_butterfly
        self.lambda_wing = lambda_wing
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        # Ablation flags
        self.use_residual = use_residual
        self.activation = activation
        self.use_warmup = use_warmup
        self.use_ema_norm = use_ema_norm
        self.squared_hinge = squared_hinge

        # Set device - prefer AMD DirectML if available
        if device is None:
            self.device = BEST_DEVICE
            self.device_name = DEVICE_NAME
        else:
            self.device = torch.device(device)
            self.device_name = str(device)

        # Model will be created during training
        self.model: Optional[PINNNetwork] = None
        self.optimizer = None
        self.training_history: List[Dict[str, float]] = []

    def _train_impl(self, df: pd.DataFrame, val_split: float = 0.2) -> Dict[str, float]:
        """Train the PINN model with physics-informed loss."""
        # Prepare data
        X = df[self.feature_columns].values.astype(np.float32)
        y = df["implied_volatility"].values.astype(np.float32)

        # Transform: use sqrt(T) for better numerical behavior
        X_transformed = X.copy()
        if "T" in self.feature_columns:
            t_idx = self.feature_columns.index("T")
            X_transformed[:, t_idx] = np.sqrt(X_transformed[:, t_idx])

        # Compute target as total variance w = σ²T
        T = X[:, t_idx] if "T" in self.feature_columns else 1.0
        w = y**2 * T

        # Split data
        n_val = int(len(X) * val_split) if val_split > 0 else 0
        indices = np.random.permutation(len(X))

        if n_val > 0:
            train_idx = indices[n_val:]
            val_idx = indices[:n_val]
        else:
            train_idx = indices
            val_idx = []

        X_train = torch.tensor(X_transformed[train_idx], device=self.device)
        w_train = torch.tensor(w[train_idx], device=self.device).unsqueeze(1)

        # Create model
        self.model = PINNNetwork(
            input_dim=len(self.feature_columns),
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            use_residual=self.use_residual,
            activation=self.activation,
        ).to(self.device)

        # Loss functions
        mse_loss = nn.MSELoss()
        calendar_loss = CalendarLoss(squared=self.squared_hinge)
        butterfly_loss = ButterflyLoss(squared=self.squared_hinge)
        wing_loss = WingLoss(squared=self.squared_hinge)

        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )

        # Training loop
        dataset = TensorDataset(X_train, w_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        import copy

        self.training_history = []
        best_loss = float("inf")
        best_feasible_mse = float("inf")
        best_state = None

        # Loss normalization: EMA variance trackers for each penalty term
        ema_var = {"calendar": 1.0, "butterfly": 1.0, "wing": 1.0}
        ema_alpha = 0.1  # EMA smoothing factor

        use_warmup = self.use_warmup
        use_ema_norm = self.use_ema_norm

        def lambda_schedule(epoch: int) -> float:
            """Warmup schedule: 100 epochs lambda=0, then linear ramp over 200."""
            if not use_warmup:
                return 1.0  # No warmup: full penalty from epoch 0
            if epoch < 100:
                return 0.0
            elif epoch < 300:
                return (epoch - 100) / 200.0
            else:
                return 1.0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_losses = {
                "mse": 0,
                "calendar": 0,
                "butterfly": 0,
                "wing": 0,
                "total": 0,
            }

            schedule_weight = lambda_schedule(epoch)

            for batch_X, batch_w in loader:
                self.optimizer.zero_grad()

                # Forward pass
                pred_w = self.model(batch_X)

                # MSE loss
                loss_mse = mse_loss(pred_w, batch_w)

                # Physics-informed losses
                loss_calendar = calendar_loss(self.model, batch_X)
                loss_butterfly = butterfly_loss(self.model, batch_X)
                loss_wing = wing_loss(self.model, batch_X)

                # Update EMA variance trackers for normalization
                if schedule_weight > 0:
                    for name, loss_val in [
                        ("calendar", loss_calendar),
                        ("butterfly", loss_butterfly),
                        ("wing", loss_wing),
                    ]:
                        val = loss_val.item()
                        ema_var[name] = (
                            (1 - ema_alpha) * ema_var[name] + ema_alpha * max(val, 1e-8)
                        )

                # Normalize penalties to unit variance (if enabled), apply warmup schedule
                if use_ema_norm:
                    norm_calendar = loss_calendar / max(ema_var["calendar"], 1e-8)
                    norm_butterfly = loss_butterfly / max(ema_var["butterfly"], 1e-8)
                    norm_wing = loss_wing / max(ema_var["wing"], 1e-8)
                else:
                    norm_calendar = loss_calendar
                    norm_butterfly = loss_butterfly
                    norm_wing = loss_wing

                # Total loss with warmup-scheduled penalty weights
                total_loss = (
                    loss_mse
                    + schedule_weight * self.lambda_calendar * norm_calendar
                    + schedule_weight * self.lambda_butterfly * norm_butterfly
                    + schedule_weight * self.lambda_wing * norm_wing
                )

                # Backward pass
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                # Accumulate losses
                epoch_losses["mse"] += loss_mse.item()
                epoch_losses["calendar"] += loss_calendar.item()
                epoch_losses["butterfly"] += loss_butterfly.item()
                epoch_losses["wing"] += loss_wing.item()
                epoch_losses["total"] += total_loss.item()

            scheduler.step()

            # Average losses
            n_batches = len(loader)
            epoch_record = {k: v / n_batches for k, v in epoch_losses.items()}
            epoch_record["epoch"] = epoch
            self.training_history.append(epoch_record)

            if epoch_losses["total"] < best_loss:
                best_loss = epoch_losses["total"]

            # Constraint-aware checkpoint: save best model that satisfies constraints
            checkpoint_start = 100 if not use_warmup else 300
            if epoch >= checkpoint_start and epoch % 50 == 0:
                self.model.eval()
                X_check = X_train.detach().clone().requires_grad_(True)
                check_cal = calendar_loss(self.model, X_check).item()
                check_but = butterfly_loss(self.model, X_check).item()
                self.model.train()

                is_feasible = (check_cal + check_but) < 1e-6
                current_mse = epoch_losses["mse"] / n_batches

                if is_feasible and current_mse < best_feasible_mse:
                    best_feasible_mse = current_mse
                    best_state = copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._is_trained = True

        # Compute final metrics
        return {
            "final_mse": self.training_history[-1]["mse"],
            "final_calendar_penalty": self.training_history[-1]["calendar"],
            "final_butterfly_penalty": self.training_history[-1]["butterfly"],
            "final_loss": self.training_history[-1]["total"],
            "epochs_trained": self.epochs,
        }

    def _predict_impl(self, df: pd.DataFrame) -> np.ndarray:
        """Predict implied volatility."""
        if self.model is None:
            raise RuntimeError("Model not trained")

        X = df[self.feature_columns].values.astype(np.float32)

        # Transform: use sqrt(T)
        X_transformed = X.copy()
        if "T" in self.feature_columns:
            t_idx = self.feature_columns.index("T")
            X_transformed[:, t_idx] = np.sqrt(X_transformed[:, t_idx])

        X_tensor = torch.tensor(X_transformed, device=self.device)

        self.model.eval()
        with torch.no_grad():
            sigma = self.model.implied_vol(X_tensor)

        return sigma.cpu().numpy().flatten()

    def check_arbitrage(self, df: pd.DataFrame) -> ArbitrageMetrics:
        """
        Check arbitrage conditions on predicted surface.

        Args:
            df: DataFrame with feature columns

        Returns:
            ArbitrageMetrics with violation counts
        """
        predictions = self.predict_volatility(df)

        log_strikes = df["log_moneyness"].values
        T = df["T"].values

        # Get unique maturities
        unique_T = np.unique(T)

        calendar_violations = 0
        butterfly_violations = 0

        # Check calendar arbitrage
        if len(unique_T) > 1:
            # Group by strike, check variance increasing in T
            for k in np.unique(log_strikes):
                mask = np.isclose(log_strikes, k)
                if mask.sum() > 1:
                    vols = predictions[mask]
                    mats = T[mask]
                    total_var = vols**2 * mats
                    _, cal_viol = check_calendar_arbitrage(total_var, mats)
                    calendar_violations += cal_viol

        # Check butterfly arbitrage per maturity
        for t in unique_T:
            mask = np.isclose(T, t)
            _, but_viol = check_butterfly_arbitrage(
                predictions[mask], log_strikes[mask], t
            )
            butterfly_violations += but_viol

        total_points = len(df)
        total_violations = calendar_violations + butterfly_violations
        arb_free_pct = 100 * (1 - total_violations / max(total_points, 1))

        return ArbitrageMetrics(
            calendar_violations=calendar_violations,
            butterfly_violations=butterfly_violations,
            arbitrage_free_pct=arb_free_pct,
        )

    def predict_with_uncertainty(
        self, df: pd.DataFrame, n_samples: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict implied volatility with uncertainty via MC dropout.

        Runs multiple forward passes with dropout enabled to estimate
        epistemic uncertainty. Higher uncertainty at wings and sparse regions.

        Args:
            df: DataFrame with feature columns
            n_samples: Number of MC forward passes

        Returns:
            (mean_predictions, std_predictions)
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        X = df[self.feature_columns].values.astype(np.float32)
        X_transformed = X.copy()
        if "T" in self.feature_columns:
            t_idx = self.feature_columns.index("T")
            X_transformed[:, t_idx] = np.sqrt(X_transformed[:, t_idx])

        X_tensor = torch.tensor(X_transformed, device=self.device)

        # MC dropout: enable training mode to activate dropout
        self.model.train()
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                sigma = self.model.implied_vol(X_tensor)
                predictions.append(sigma.cpu().numpy().flatten())

        self.model.eval()

        predictions = np.stack(predictions, axis=0)
        return predictions.mean(axis=0), predictions.std(axis=0)

    def _save_model_impl(self, model_path: str, scaler_path: str) -> None:
        """Save model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "hidden_layers": self.hidden_layers,
                "feature_columns": self.feature_columns,
                "lambda_calendar": self.lambda_calendar,
                "lambda_butterfly": self.lambda_butterfly,
                "lambda_wing": self.lambda_wing,
                "training_history": self.training_history,
                "dropout": self.dropout,
            },
            model_path,
        )

    def _load_model_impl(self, model_path: str, scaler_path: str) -> None:
        """Load model from disk."""
        checkpoint = torch.load(model_path, map_location=self.device)

        self.hidden_layers = checkpoint["hidden_layers"]
        self.feature_columns = checkpoint["feature_columns"]
        self.lambda_calendar = checkpoint["lambda_calendar"]
        self.lambda_butterfly = checkpoint["lambda_butterfly"]
        self.lambda_wing = checkpoint["lambda_wing"]
        self.training_history = checkpoint.get("training_history", [])

        self.model = PINNNetwork(
            input_dim=len(self.feature_columns),
            hidden_layers=self.hidden_layers,
            dropout=checkpoint.get("dropout", self.dropout),
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._is_trained = True


# =============================================================================
# Factory and Convenience
# =============================================================================


def create_pinn_model(
    constraint_strength: Literal["weak", "medium", "strong"] = "medium",
    **kwargs,
) -> PINNVolatilityModel:
    """
    Create PINN model with preset constraint strengths.

    Args:
        constraint_strength: Preset for lambda values
            - "weak": Prioritize fit (λ_cal=0.1, λ_but=0.05)
            - "medium": Balanced (λ_cal=1.0, λ_but=0.5)
            - "strong": Strict no-arb (λ_cal=5.0, λ_but=2.0)
        **kwargs: Additional arguments to PINNVolatilityModel

    Returns:
        Configured PINNVolatilityModel
    """
    presets = {
        "weak": {"lambda_calendar": 0.1, "lambda_butterfly": 0.05, "lambda_wing": 0.01},
        "medium": {"lambda_calendar": 1.0, "lambda_butterfly": 0.5, "lambda_wing": 0.1},
        "strong": {"lambda_calendar": 5.0, "lambda_butterfly": 2.0, "lambda_wing": 0.5},
    }

    params = presets.get(constraint_strength, presets["medium"])
    params.update(kwargs)

    return PINNVolatilityModel(**params)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PINNVolatilityModel",
    "PINNNetwork",
    "ArbitrageMetrics",
    "create_pinn_model",
    "check_calendar_arbitrage",
    "check_butterfly_arbitrage",
    "TORCH_AVAILABLE",
]
