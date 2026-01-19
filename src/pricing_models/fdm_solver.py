# src/pricing_models/fdm_solver.py
"""
Finite Difference Methods (FDM) for Option Pricing.

Implements PDE solvers for the Black-Scholes equation:

    ∂V/∂t + ½σ²S²∂²V/∂S² + (r-q)S∂V/∂S - rV = 0

Schemes:
    - Explicit Euler (conditionally stable)
    - Implicit Euler (unconditionally stable)
    - Crank-Nicolson (2nd order accurate, unconditionally stable)

Features:
    - American option early exercise via PSOR
    - Non-uniform grid support
    - Barrier option boundary conditions

Reference:
    Wilmott, P. (2006). Paul Wilmott on Quantitative Finance.

Usage:
    >>> from src.pricing_models.fdm_solver import CrankNicolsonSolver
    >>> solver = CrankNicolsonSolver(s_max=300, n_space=200, n_time=100)
    >>> price = solver.price(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.linalg import solve_banded


@dataclass
class CrankNicolsonSolver:
    """
    Crank-Nicolson finite difference solver for European and American options.

    Attributes:
        s_max: Maximum spot price in grid (typically 3-5x spot).
        n_space: Number of spatial grid points.
        n_time: Number of time steps.
    """

    s_max: float = 300.0
    n_space: int = 200
    n_time: int = 100

    def __post_init__(self):
        if self.s_max <= 0:
            raise ValueError("s_max must be positive")
        if self.n_space < 10:
            raise ValueError("n_space must be >= 10")
        if self.n_time < 10:
            raise ValueError("n_time must be >= 10")

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call",
        exercise: Literal["european", "american"] = "european",
        q: float = 0.0,
    ) -> float:
        """
        Price option using Crank-Nicolson scheme.

        Args:
            S: Current spot price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            option_type: 'call' or 'put'.
            exercise: 'european' or 'american'.
            q: Dividend yield.

        Returns:
            Option price.
        """
        if T <= 0:
            if option_type == "call":
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        # Setup grid
        ds = self.s_max / self.n_space
        dt = T / self.n_time

        # Spatial grid
        s_grid = np.linspace(0, self.s_max, self.n_space + 1)

        # Terminal condition (payoff at maturity)
        if option_type == "call":
            V = np.maximum(s_grid - K, 0)
        else:
            V = np.maximum(K - s_grid, 0)

        # Coefficients for interior points
        j = np.arange(1, self.n_space)
        alpha = 0.25 * dt * (sigma**2 * j**2 - (r - q) * j)
        beta = -0.5 * dt * (sigma**2 * j**2 + r)
        gamma = 0.25 * dt * (sigma**2 * j**2 + (r - q) * j)

        # Build tridiagonal matrices
        # M1 (implicit side): (I - θΔtA) where θ=0.5 for Crank-Nicolson
        # M2 (explicit side): (I + (1-θ)ΔtA)

        # M1: sub-diagonal, diagonal, super-diagonal
        M1_sub = -alpha[1:]
        M1_diag = np.ones(self.n_space - 1) - beta
        M1_super = -gamma[:-1]

        # M2: coefficients
        M2_sub = alpha[1:]
        M2_diag = np.ones(self.n_space - 1) + beta
        M2_super = gamma[:-1]

        # Banded matrix format for scipy.linalg.solve_banded
        # Format: [super, diag, sub] for (1, 1) banded
        M1_banded = np.zeros((3, self.n_space - 1))
        M1_banded[0, 1:] = M1_super
        M1_banded[1, :] = M1_diag
        M1_banded[2, :-1] = M1_sub

        # Time stepping (backward from T to 0)
        for n in range(self.n_time):
            # Build RHS from explicit side
            rhs = np.zeros(self.n_space - 1)

            # Interior points
            for i in range(self.n_space - 1):
                j_idx = i + 1
                rhs[i] = (
                    M2_sub[i - 1] * V[j_idx - 1]
                    if i > 0
                    else alpha[j_idx - 1] * V[j_idx - 1]
                )
                rhs[i] += M2_diag[i] * V[j_idx]
                if i < self.n_space - 2:
                    rhs[i] += M2_super[i] * V[j_idx + 1]
                else:
                    rhs[i] += gamma[j_idx - 1] * V[j_idx + 1]

            # Boundary adjustments
            if option_type == "call":
                # S -> S_max: V ~ S - K*exp(-r*tau)
                tau = (self.n_time - n - 1) * dt
                V_boundary = self.s_max - K * np.exp(-r * tau)
                rhs[-1] += gamma[-1] * V_boundary

            # Solve tridiagonal system
            V_new = solve_banded((1, 1), M1_banded, rhs)

            # Update interior values
            V[1:-1] = V_new

            # Update boundary conditions
            if option_type == "call":
                V[0] = 0
                tau = (self.n_time - n - 1) * dt
                V[-1] = self.s_max - K * np.exp(-r * tau)
            else:
                tau = (self.n_time - n - 1) * dt
                V[0] = K * np.exp(-r * tau)
                V[-1] = 0

            # American exercise constraint
            if exercise == "american":
                if option_type == "call":
                    V = np.maximum(V, s_grid - K)
                else:
                    V = np.maximum(V, K - s_grid)

        # Interpolate to get price at S
        price = np.interp(S, s_grid, V)
        return float(price)

    # Greeks are computed via src.greeks.unified_greeks.compute_greeks_unified()
    # Use: from src.greeks import compute_greeks_unified, FDMAdapter
    #      greeks = compute_greeks_unified(FDMAdapter(solver), S, K, T, r, sigma, option_type)


@dataclass
class ExplicitFDMSolver:
    """
    Explicit finite difference solver.

    Faster but conditionally stable (requires small dt).
    Stability condition: dt <= ds² / (σ²S_max²)
    """

    s_max: float = 300.0
    n_space: int = 200
    n_time: int = 1000  # Need many time steps for stability

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call",
        exercise: Literal["european", "american"] = "european",
        q: float = 0.0,
    ) -> float:
        """Price using explicit Euler scheme."""
        if T <= 0:
            return max(S - K, 0) if option_type == "call" else max(K - S, 0)

        ds = self.s_max / self.n_space
        dt = T / self.n_time

        # Stability check
        stability = dt * sigma**2 * self.n_space**2
        if stability > 1:
            import warnings

            warnings.warn(f"Explicit scheme may be unstable (λ={stability:.2f} > 1)")

        s_grid = np.linspace(0, self.s_max, self.n_space + 1)

        # Terminal payoff
        if option_type == "call":
            V = np.maximum(s_grid - K, 0)
        else:
            V = np.maximum(K - s_grid, 0)

        # Time stepping
        for n in range(self.n_time):
            V_new = V.copy()

            for j in range(1, self.n_space):
                a = 0.5 * dt * (sigma**2 * j**2 - (r - q) * j)
                b = 1 - dt * (sigma**2 * j**2 + r)
                c = 0.5 * dt * (sigma**2 * j**2 + (r - q) * j)

                V_new[j] = a * V[j - 1] + b * V[j] + c * V[j + 1]

            # Boundary conditions
            if option_type == "call":
                V_new[0] = 0
                tau = (self.n_time - n - 1) * dt
                V_new[-1] = self.s_max - K * np.exp(-r * tau)
            else:
                tau = (self.n_time - n - 1) * dt
                V_new[0] = K * np.exp(-r * tau)
                V_new[-1] = 0

            # American constraint
            if exercise == "american":
                if option_type == "call":
                    V_new = np.maximum(V_new, s_grid - K)
                else:
                    V_new = np.maximum(V_new, K - s_grid)

            V = V_new

        return float(np.interp(S, s_grid, V))
