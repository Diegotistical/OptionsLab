# src/volatility_surface/models/alm_model.py
"""
Augmented Lagrangian Method (ALM) for Arbitrage-Free Volatility Surfaces.

Replaces the fixed-penalty heuristic in pinn_model.py with a principled
constrained optimization framework using adaptive dual variables.

Mathematical Framework:
    Given the constrained problem:
        min_θ  L_MSE(θ)
        s.t.   g_j(θ) ≥ 0  for all j  (butterfly density positivity)
               h_j(θ) ≥ 0  for all j  (calendar monotonicity)
               r_j(θ) ≤ 2  for all j  (Roger-Lee wing bounds)

    We solve the augmented Lagrangian subproblem at iteration k:
        L_ALM(θ, μ^k, ρ^k) = L_MSE(θ)
            + Σ_j (ρ^k/2) · max(0, μ_j^k/ρ^k - g_j(θ))²   [butterfly]
            + Σ_j (ρ^k/2) · max(0, ν_j^k/ρ^k - h_j(θ))²   [calendar]
            + Σ_j (ρ^k/2) · max(0, ξ_j^k/ρ^k + r_j(θ)-2)²  [wings]

    Dual update:
        μ_j^{k+1} = max(0, μ_j^k - ρ^k · g_j(θ^k))
        ν_j^{k+1} = max(0, ν_j^k - ρ^k · h_j(θ^k))

    Penalty update:
        ρ^{k+1} = β · ρ^k  if constraint violation not decreasing sufficiently

    This is NOT a convergence proof. We implement ALM as an empirically
    superior alternative to fixed penalties and document its behavior.

Reference:
    - Bertsekas, D.P. (1999). Nonlinear Programming. Ch. 4.
    - Basir & Senocak (2023). Adaptive ALM for constrained NNs.
    - Lu et al. (2021). PINNs with hard constraints via ALM.
"""

import copy
import time
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
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from src.volatility_surface.base import VolatilityModelBase

# Import shared components from pinn_model
from src.volatility_surface.models.pinn_model import (
    BEST_DEVICE,
    DEVICE_NAME,
    ButterflyLoss,
    CalendarLoss,
    PINNNetwork,
    WingLoss,
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
)


@dataclass
class ALMState:
    """Tracks the state of the augmented Lagrangian optimization."""

    # Dual variables (Lagrange multipliers) — one per constraint evaluation point
    mu_butterfly: Optional[torch.Tensor] = None  # butterfly density >= 0
    mu_calendar: Optional[torch.Tensor] = None  # calendar monotonicity >= 0

    # Penalty parameter
    rho: float = 1.0

    # History for convergence monitoring
    constraint_violation_history: List[float] = field(default_factory=list)
    dual_norm_history: List[float] = field(default_factory=list)
    rho_history: List[float] = field(default_factory=list)
    mse_history: List[float] = field(default_factory=list)

    # Convergence diagnostics
    outer_iterations: int = 0


class ALMVolatilityModel(VolatilityModelBase):
    """
    Augmented Lagrangian volatility surface model.

    Replaces fixed penalty weights (λ_but, λ_cal) with adaptive dual variables
    that are updated after each outer ALM iteration. The penalty parameter ρ
    increases when constraint violations are not decreasing.

    Key differences from PINNVolatilityModel:
        1. Dual variables (μ, ν) adapt per-constraint-point, focusing
           enforcement where violations actually occur
        2. Penalty parameter ρ grows only when needed, avoiding over-smoothing
        3. No warmup heuristic needed — ALM's dual variables naturally
           produce small penalties early and large penalties where needed
        4. Constraint satisfaction is tracked and reported at each outer iteration

    This is still a heuristic in the non-convex setting (no convergence proof),
    but it is a more principled heuristic than fixed penalties.
    """

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        hidden_layers: Optional[List[int]] = None,
        # ALM hyperparameters
        rho_init: float = 1.0,
        rho_max: float = 1000.0,
        rho_growth: float = 2.0,
        outer_iterations: int = 10,
        inner_epochs: int = 50,
        constraint_tol: float = 1e-5,
        violation_reduction_target: float = 0.25,
        # Network hyperparameters
        learning_rate: float = 3e-3,
        batch_size: int = 64,
        dropout: float = 0.1,
        device: Optional[str] = None,
        enable_benchmark: bool = False,
        use_residual: bool = True,
        activation: str = "gelu",
        # Wing penalty (no dual variable — simple penalty suffices)
        lambda_wing: float = 1.0,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ALMVolatilityModel")

        if feature_columns is None:
            feature_columns = ["log_moneyness", "T"]

        super().__init__(
            feature_columns=feature_columns, enable_benchmark=enable_benchmark
        )

        self.hidden_layers = hidden_layers or [64, 32, 16]
        self.rho_init = rho_init
        self.rho_max = rho_max
        self.rho_growth = rho_growth
        self.outer_iterations = outer_iterations
        self.inner_epochs = inner_epochs
        self.constraint_tol = constraint_tol
        self.violation_reduction_target = violation_reduction_target
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        self.lambda_wing = lambda_wing
        self.use_residual = use_residual
        self.activation = activation

        if device is None:
            self.device = BEST_DEVICE
            self.device_name = DEVICE_NAME
        else:
            self.device = torch.device(device)
            self.device_name = str(device)

        self.model: Optional[PINNNetwork] = None
        self.alm_state: Optional[ALMState] = None
        self.training_history: List[Dict[str, Any]] = []

    def _compute_constraint_violations(
        self,
        model: PINNNetwork,
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute pointwise constraint violations.

        Returns:
            butterfly_violations: max(0, -g(k,T)) for each point
            calendar_violations: max(0, -∂w/∂T) for each point
        """
        X_eval = X.detach().clone().requires_grad_(True)

        # --- Butterfly (density positivity) ---
        w = model(X_eval)
        k = X_eval[:, 0:1]

        grad_w = torch.autograd.grad(
            w, X_eval, torch.ones_like(w),
            create_graph=True, retain_graph=True,
        )[0]
        dw_dk = grad_w[:, 0:1]

        grad2_w = torch.autograd.grad(
            dw_dk, X_eval, torch.ones_like(dw_dk),
            create_graph=True, retain_graph=True,
        )[0]
        d2w_dk2 = grad2_w[:, 0:1]

        eps = 1e-6
        term1 = (1 - k * dw_dk / (2 * w + eps)) ** 2
        term2 = (dw_dk ** 2) / 4 * (1 / (w + eps) + 0.25)
        term3 = d2w_dk2 / 2
        g_k = term1 - term2 + term3

        butterfly_violations = torch.relu(-g_k).squeeze(-1)

        # --- Calendar (total variance monotonicity) ---
        sqrt_T = X_eval[:, 1:2]
        dw_dsqrtT = grad_w[:, 1:2]
        dw_dT = dw_dsqrtT / (2 * sqrt_T + eps)

        calendar_violations = torch.relu(-dw_dT).squeeze(-1)

        return butterfly_violations, calendar_violations

    def _alm_loss(
        self,
        model: PINNNetwork,
        X: torch.Tensor,
        w_target: torch.Tensor,
        alm_state: ALMState,
        wing_loss_fn: WingLoss,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the augmented Lagrangian loss.

        L_ALM = L_MSE + Σ (ρ/2) · max(0, μ_j/ρ - g_j(θ))² + wing penalty

        This is the Powell-Hestenes-Rockafellar form of the ALM,
        which handles inequality constraints via the max(0, ...) operator.
        """
        pred_w = model(X)

        # MSE term
        loss_mse = nn.functional.mse_loss(pred_w, w_target)

        # Constraint violations
        but_viol, cal_viol = self._compute_constraint_violations(model, X)

        # ALM penalty: (ρ/2) · max(0, μ/ρ - g(θ))²
        # For violation v = max(0, -g(θ)), the ALM term becomes:
        # (ρ/2) · max(0, μ/ρ + v)² — which simplifies since v ≥ 0 and μ ≥ 0
        rho = alm_state.rho

        # Butterfly ALM term
        mu_but = alm_state.mu_butterfly
        alm_but = (rho / 2) * torch.mean(
            torch.max(
                torch.zeros_like(but_viol),
                mu_but / rho + but_viol,
            ) ** 2
        )

        # Calendar ALM term
        mu_cal = alm_state.mu_calendar
        alm_cal = (rho / 2) * torch.mean(
            torch.max(
                torch.zeros_like(cal_viol),
                mu_cal / rho + cal_viol,
            ) ** 2
        )

        # Wing penalty (simple penalty, no dual variable needed)
        loss_wing = wing_loss_fn(model, X)

        total = loss_mse + alm_but + alm_cal + self.lambda_wing * loss_wing

        diagnostics = {
            "mse": loss_mse.item(),
            "alm_butterfly": alm_but.item(),
            "alm_calendar": alm_cal.item(),
            "wing": loss_wing.item() if isinstance(loss_wing, torch.Tensor) else loss_wing,
            "max_but_violation": but_viol.max().item(),
            "max_cal_violation": cal_viol.max().item(),
            "mean_but_violation": but_viol.mean().item(),
            "mean_cal_violation": cal_viol.mean().item(),
            "rho": rho,
        }

        return total, diagnostics

    def _dual_update(
        self,
        model: PINNNetwork,
        X: torch.Tensor,
        alm_state: ALMState,
    ) -> float:
        """
        Update dual variables after an inner optimization.

        μ^{k+1} = max(0, μ^k + ρ^k · violation)

        Returns total constraint violation for convergence monitoring.
        """
        model.eval()
        with torch.no_grad():
            but_viol, cal_viol = self._compute_constraint_violations(model, X)

            # Dual update: μ ← max(0, μ + ρ · violation)
            alm_state.mu_butterfly = torch.relu(
                alm_state.mu_butterfly + alm_state.rho * but_viol
            )
            alm_state.mu_calendar = torch.relu(
                alm_state.mu_calendar + alm_state.rho * cal_viol
            )

            total_violation = but_viol.sum().item() + cal_viol.sum().item()

        return total_violation

    def _train_impl(self, df: pd.DataFrame, val_split: float = 0.2) -> Dict[str, Any]:
        """
        Train with augmented Lagrangian method.

        Outer loop: update dual variables μ, penalty ρ
        Inner loop: minimize L_ALM(θ, μ, ρ) w.r.t. θ
        """
        # ---- Data preparation (same as PINN) ----
        X = df[self.feature_columns].values.astype(np.float32)
        y = df["implied_volatility"].values.astype(np.float32)

        X_transformed = X.copy()
        if "T" in self.feature_columns:
            t_idx = self.feature_columns.index("T")
            X_transformed[:, t_idx] = np.sqrt(X_transformed[:, t_idx])

        T = X[:, t_idx] if "T" in self.feature_columns else 1.0
        w = y ** 2 * T

        X_train = torch.tensor(X_transformed, device=self.device)
        w_train = torch.tensor(w, device=self.device).unsqueeze(1)

        # ---- Model creation ----
        self.model = PINNNetwork(
            input_dim=len(self.feature_columns),
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            use_residual=self.use_residual,
            activation=self.activation,
        ).to(self.device)

        wing_loss_fn = WingLoss(squared=True)

        # ---- Initialize ALM state ----
        n_points = len(X_train)
        self.alm_state = ALMState(
            mu_butterfly=torch.zeros(n_points, device=self.device),
            mu_calendar=torch.zeros(n_points, device=self.device),
            rho=self.rho_init,
        )

        best_state = None
        best_feasible_mse = float("inf")
        prev_violation = float("inf")

        # ---- Outer ALM loop ----
        for outer_iter in range(self.outer_iterations):
            # Inner optimization: minimize L_ALM w.r.t. θ
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-4,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.inner_epochs
            )

            dataset = TensorDataset(X_train, w_train)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Also update dual variable indices for mini-batches
            # Since dual vars are per-point, we need to track indices
            for inner_epoch in range(self.inner_epochs):
                self.model.train()
                epoch_loss = 0
                epoch_diag = {}

                for batch_idx, (batch_X, batch_w) in enumerate(loader):
                    optimizer.zero_grad()

                    # Get indices for this batch's dual variables
                    start = batch_idx * self.batch_size
                    end = min(start + len(batch_X), n_points)

                    # Create a temporary ALM state for this batch
                    batch_state = ALMState(
                        mu_butterfly=self.alm_state.mu_butterfly[start:end],
                        mu_calendar=self.alm_state.mu_calendar[start:end],
                        rho=self.alm_state.rho,
                    )

                    loss, diag = self._alm_loss(
                        self.model, batch_X, batch_w, batch_state, wing_loss_fn,
                    )

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_diag = diag  # Keep last batch diagnostics

                scheduler.step()

            # ---- Dual variable update ----
            total_violation = self._dual_update(
                self.model, X_train, self.alm_state,
            )

            # ---- Penalty parameter update ----
            violation_ratio = total_violation / max(prev_violation, 1e-10)
            if violation_ratio > self.violation_reduction_target:
                # Constraints not decreasing fast enough — increase ρ
                self.alm_state.rho = min(
                    self.alm_state.rho * self.rho_growth,
                    self.rho_max,
                )

            # ---- Record history ----
            self.alm_state.constraint_violation_history.append(total_violation)
            self.alm_state.dual_norm_history.append(
                self.alm_state.mu_butterfly.norm().item()
                + self.alm_state.mu_calendar.norm().item()
            )
            self.alm_state.rho_history.append(self.alm_state.rho)
            self.alm_state.mse_history.append(epoch_diag.get("mse", 0))
            self.alm_state.outer_iterations = outer_iter + 1

            outer_record = {
                "outer_iter": outer_iter,
                "total_violation": total_violation,
                "rho": self.alm_state.rho,
                "dual_norm": self.alm_state.dual_norm_history[-1],
                **epoch_diag,
            }
            self.training_history.append(outer_record)

            # ---- Checkpoint if feasible ----
            if total_violation < self.constraint_tol:
                current_mse = epoch_diag.get("mse", float("inf"))
                if current_mse < best_feasible_mse:
                    best_feasible_mse = current_mse
                    best_state = copy.deepcopy(self.model.state_dict())

            prev_violation = total_violation

            # ---- Early termination ----
            if total_violation < self.constraint_tol:
                # Check if dual variables have stabilized
                if outer_iter >= 2:
                    dual_change = abs(
                        self.alm_state.dual_norm_history[-1]
                        - self.alm_state.dual_norm_history[-2]
                    )
                    if dual_change < 1e-4:
                        break

        # Load best feasible model if found
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._is_trained = True

        return {
            "final_mse": self.training_history[-1].get("mse", 0),
            "final_violation": self.training_history[-1].get("total_violation", 0),
            "final_rho": self.alm_state.rho,
            "outer_iterations": self.alm_state.outer_iterations,
            "converged": total_violation < self.constraint_tol,
            "device": self.device_name,
        }

    def _predict_impl(self, df: pd.DataFrame) -> np.ndarray:
        """Predict implied volatility."""
        X = df[self.feature_columns].values.astype(np.float32)

        X_transformed = X.copy()
        if "T" in self.feature_columns:
            t_idx = self.feature_columns.index("T")
            T = X_transformed[:, t_idx].copy()
            X_transformed[:, t_idx] = np.sqrt(X_transformed[:, t_idx])

        X_tensor = torch.tensor(X_transformed, device=self.device)

        self.model.eval()
        with torch.no_grad():
            w_pred = self.model(X_tensor).cpu().numpy().flatten()

        sigma = np.sqrt(w_pred / T)
        return sigma

    def _save_model_impl(self, path: str) -> None:
        """Save model state."""
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "alm_state": {
                    "rho": self.alm_state.rho,
                    "outer_iterations": self.alm_state.outer_iterations,
                    "violation_history": self.alm_state.constraint_violation_history,
                    "rho_history": self.alm_state.rho_history,
                },
                "config": {
                    "hidden_layers": self.hidden_layers,
                    "rho_init": self.rho_init,
                    "rho_max": self.rho_max,
                    "rho_growth": self.rho_growth,
                    "outer_iterations": self.outer_iterations,
                    "inner_epochs": self.inner_epochs,
                },
            },
            path,
        )

    def _load_model_impl(self, path: str) -> None:
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model = PINNNetwork(
            input_dim=len(self.feature_columns),
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            use_residual=self.use_residual,
            activation=self.activation,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self._is_trained = True

    def get_convergence_report(self) -> Dict[str, Any]:
        """
        Return ALM convergence diagnostics.

        This is the key output for the paper: it shows how constraint
        violations decrease over outer iterations, how ρ adapts, and
        whether dual variables stabilize.
        """
        if not self.alm_state:
            return {}

        return {
            "outer_iterations": self.alm_state.outer_iterations,
            "final_rho": self.alm_state.rho,
            "final_violation": (
                self.alm_state.constraint_violation_history[-1]
                if self.alm_state.constraint_violation_history
                else None
            ),
            "converged": (
                self.alm_state.constraint_violation_history[-1] < self.constraint_tol
                if self.alm_state.constraint_violation_history
                else False
            ),
            "violation_trajectory": self.alm_state.constraint_violation_history,
            "rho_trajectory": self.alm_state.rho_history,
            "dual_norm_trajectory": self.alm_state.dual_norm_history,
            "mse_trajectory": self.alm_state.mse_history,
        }
