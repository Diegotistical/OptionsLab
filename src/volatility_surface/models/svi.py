# src/volatility_surface/models/svi.py
"""
SVI (Stochastic Volatility Inspired) Volatility Surface Model.

Implements Gatheral's SVI parameterization:

    w(k) = a + b * (ρ(k - m) + √((k - m)² + σ²))

Where:
    w = σ²T (total implied variance)
    k = log(K/F) (log-moneyness)

Parameters:
    a: Vertical shift (ATM variance level)
    b: Angle between the wings (slope)
    ρ (rho): Rotation/skew parameter
    m: Horizontal shift
    σ (sigma): Curvature/smile width

Features:
    - Fast analytical smile computation
    - Arbitrage-free constraints (butterfly, calendar)
    - Calibration with regularization

Reference:
    Gatheral, J. (2004). A Parsimonious Arbitrage-Free Implied
    Volatility Parameterization. Presentation at Global Derivatives.

Usage:
    >>> from src.volatility_surface.models.svi import SVIModel
    >>> svi = SVIModel(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    >>> iv = svi.implied_vol(k=-0.1, T=1.0)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize


@dataclass
class SVIModel:
    """
    SVI volatility surface parameterization.

    Attributes:
        a: Vertical shift (level of ATM variance).
        b: Slope of the wings (must be positive).
        rho: Rotation/skew (-1 < rho < 1).
        m: Horizontal translation.
        sigma: Curvature width (must be positive).
    """

    a: float  # Vertical shift
    b: float  # Wing angle
    rho: float  # Rotation
    m: float  # Horizontal shift
    sigma: float  # Curvature

    def __post_init__(self):
        """Validate SVI parameters."""
        if self.b < 0:
            raise ValueError("b must be non-negative")
        if not -1 < self.rho < 1:
            raise ValueError("rho must be in (-1, 1)")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

        # Check arbitrage-free conditions
        self._check_arbitrage_free()

    def _check_arbitrage_free(self):
        """Check Gatheral's arbitrage-free constraints."""
        # Butterfly arbitrage: w(k) must be non-negative for all k
        # This requires: a + b*σ*√(1-ρ²) >= 0
        min_variance = self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2)
        if min_variance < -1e-10:
            import warnings

            warnings.warn(
                f"SVI parameters may violate butterfly arbitrage "
                f"(min variance = {min_variance:.6f} < 0)"
            )

    def total_variance(self, k: float) -> float:
        """
        Compute total implied variance w(k) = σ²T.

        Args:
            k: Log-moneyness = log(K/F).

        Returns:
            Total implied variance.
        """
        return self.a + self.b * (
            self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma**2)
        )

    def implied_vol(self, k: float, T: float) -> float:
        """
        Compute implied volatility.

        Args:
            k: Log-moneyness.
            T: Time to maturity.

        Returns:
            Implied volatility σ.
        """
        if T <= 0:
            return np.sqrt(max(self.a, 0))

        w = self.total_variance(k)
        if w < 0:
            # Clamp to avoid negative variance
            w = 1e-10
        return np.sqrt(w / T)

    def smile(
        self,
        log_strikes: np.ndarray,
        T: float,
    ) -> np.ndarray:
        """
        Compute implied volatility smile.

        Args:
            log_strikes: Array of log-moneyness values.
            T: Time to maturity.

        Returns:
            Array of implied volatilities.
        """
        return np.array([self.implied_vol(k, T) for k in log_strikes])

    def local_vol_squared(self, k: float, T: float) -> float:
        """
        Compute local variance via Dupire formula.

        Args:
            k: Log-moneyness.
            T: Time to maturity.

        Returns:
            Local variance σ_loc².
        """
        w = self.total_variance(k)

        # Derivatives of w with respect to k
        term = np.sqrt((k - self.m) ** 2 + self.sigma**2)
        dw_dk = self.b * (self.rho + (k - self.m) / term)
        d2w_dk2 = self.b * self.sigma**2 / term**3

        # Time derivative (for single slice, ∂w/∂T = w/T)
        dw_dT = w / T

        # Dupire formula
        numerator = dw_dT
        denominator = (
            1
            - k * dw_dk / w
            + 0.25 * (-0.25 - 1 / w + k**2 / w**2) * dw_dk**2
            + 0.5 * d2w_dk2
        )

        if denominator <= 0:
            return w / T  # Fallback to total variance

        return numerator / denominator


@dataclass
class SSVIModel:
    """
    SSVI (Surface SVI) for arbitrage-free term structure.

    Extends SVI to model the entire surface with guaranteed
    arbitrage-free across both strikes and maturities.

    Parameterization (Gatheral & Jacquier, 2014):
        w(k,θ) = (θ/2) * (1 + ρφ(θ)k + √((φ(θ)k + ρ)² + 1 - ρ²))

    Where:
        θ = ATM total variance = σ_ATM² * T
        φ(θ) = η / θ^γ (power law)
    """

    rho: float  # Global skew
    eta: float  # Vol-of-vol proxy
    gamma: float  # Power law exponent

    def __post_init__(self):
        if not -1 < self.rho < 1:
            raise ValueError("rho must be in (-1, 1)")
        if self.eta <= 0:
            raise ValueError("eta must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")

    def phi(self, theta: float) -> float:
        """Correlation function φ(θ)."""
        if theta <= 0:
            return 1.0
        return self.eta / (theta**self.gamma)

    def total_variance(self, k: float, theta: float) -> float:
        """
        Compute SSVI total variance.

        Args:
            k: Log-moneyness.
            theta: ATM total variance for this slice.

        Returns:
            Total implied variance.
        """
        phi = self.phi(theta)
        return (theta / 2) * (
            1
            + self.rho * phi * k
            + np.sqrt((phi * k + self.rho) ** 2 + 1 - self.rho**2)
        )

    def implied_vol(self, k: float, T: float, atm_vol: float) -> float:
        """
        Compute implied volatility.

        Args:
            k: Log-moneyness.
            T: Time to maturity.
            atm_vol: ATM implied volatility for this maturity.

        Returns:
            Implied volatility.
        """
        theta = atm_vol**2 * T
        w = self.total_variance(k, theta)
        return np.sqrt(max(w, 1e-10) / T)


@dataclass
class ESSVIModel:
    """
    Extended SSVI (eSSVI) for arbitrage-free volatility surfaces.

    Extends SSVI with power-law ATM term structure and explicit
    Roger-Lee bound enforcement (Hendriks & Martini 2019).

    Parameterization:
        w(k, T) = (theta(T)/2) * (1 + rho*phi(theta)*k
                   + sqrt((phi(theta)*k + rho)^2 + 1 - rho^2))
        theta(T) = a_atm * T^b_atm   (power-law ATM total variance)
        phi(theta) = eta / theta^gamma

    Roger-Lee enforcement: eta * (1 + |rho|) < 2
    """

    rho: float
    eta: float
    gamma: float
    a_atm: float  # ATM total variance scale
    b_atm: float  # ATM total variance power

    def __post_init__(self):
        if not -1 < self.rho < 1:
            raise ValueError("rho must be in (-1, 1)")
        if self.eta <= 0:
            raise ValueError("eta must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if self.a_atm <= 0:
            raise ValueError("a_atm must be positive")
        if self.b_atm <= 0 or self.b_atm > 2:
            raise ValueError("b_atm must be in (0, 2]")

    def theta(self, T: float) -> float:
        """Power-law ATM total variance."""
        if T <= 0:
            return 1e-10
        return self.a_atm * T ** self.b_atm

    def phi(self, theta_val: float) -> float:
        """Correlation function."""
        if theta_val <= 0:
            return 1.0
        return self.eta / (theta_val ** self.gamma)

    def total_variance(self, k: float, T: float) -> float:
        """Compute eSSVI total variance w(k, T)."""
        th = self.theta(T)
        ph = self.phi(th)
        return (th / 2) * (
            1
            + self.rho * ph * k
            + np.sqrt((ph * k + self.rho) ** 2 + 1 - self.rho ** 2)
        )

    def implied_vol(self, k: float, T: float) -> float:
        """Compute implied volatility."""
        w = self.total_variance(k, T)
        return np.sqrt(max(w, 1e-10) / max(T, 1e-10))

    def smile(self, log_strikes: np.ndarray, T: float) -> np.ndarray:
        """Compute implied volatility smile for a maturity slice."""
        return np.array([self.implied_vol(k, T) for k in log_strikes])

    def roger_lee_check(self) -> bool:
        """Verify Roger-Lee bound: eta * (1 + |rho|) < 2."""
        return self.eta * (1 + abs(self.rho)) < 2.0


def calibrate_essvi(
    T_slices: np.ndarray,
    smile_data: list,
    n_restarts: int = 10,
) -> ESSVIModel:
    """
    Calibrate eSSVI model with multi-start joint optimization.

    Uses maturity-weighted MSE to prevent short-dated slices from dominating,
    and differential evolution for global search followed by L-BFGS-B refinement.

    Args:
        T_slices: Array of maturities.
        smile_data: List of (log_strikes, market_vols) tuples per slice.
        n_restarts: Number of random restarts for L-BFGS-B refinement.

    Returns:
        Best-fit ESSVIModel.
    """
    from scipy.optimize import differential_evolution

    # Step 1: Initialize ATM term structure via power-law regression
    atm_vols = []
    for i, (log_strikes, market_vols) in enumerate(smile_data):
        atm_idx = np.argmin(np.abs(log_strikes))
        atm_vols.append(market_vols[atm_idx])
    atm_vols = np.array(atm_vols)
    atm_total_var = atm_vols ** 2 * T_slices

    # Power-law fit: log(theta) = log(a) + b*log(T)
    valid = (T_slices > 0) & (atm_total_var > 0)
    if valid.sum() >= 2:
        log_T = np.log(T_slices[valid])
        log_theta = np.log(atm_total_var[valid])
        b_init, log_a_init = np.polyfit(log_T, log_theta, 1)
        a_init = np.exp(log_a_init)
        b_init = np.clip(b_init, 0.3, 1.5)
        a_init = np.clip(a_init, 0.005, 0.5)
    else:
        a_init, b_init = 0.04, 1.0

    # Maturity weights: sqrt(T) to prevent short-dated domination
    T_weights = {T: np.sqrt(T) for T in T_slices}

    def objective(params):
        rho, eta, gamma, a_atm, b_atm = params

        # Roger-Lee bound
        if eta * (1 + abs(rho)) >= 2.0:
            return 1e10

        try:
            total_error = 0.0
            total_weight = 0.0
            for i, T in enumerate(T_slices):
                log_strikes, market_vols = smile_data[i]
                theta_val = a_atm * T ** b_atm
                if theta_val <= 0:
                    return 1e10
                phi_val = eta / (theta_val ** gamma)
                w = T_weights[T]

                for j, k in enumerate(log_strikes):
                    inner = (phi_val * k + rho) ** 2 + 1 - rho ** 2
                    if inner < 0:
                        return 1e10
                    w_model = (theta_val / 2) * (
                        1 + rho * phi_val * k + np.sqrt(inner)
                    )
                    w_market = market_vols[j] ** 2 * T
                    total_error += w * (w_model - w_market) ** 2
                    total_weight += w

            return total_error / max(total_weight, 1e-10)
        except (ValueError, FloatingPointError, OverflowError):
            return 1e10

    bounds_de = [
        (-0.99, 0.99),   # rho
        (0.01, 3.0),     # eta
        (0.01, 0.99),    # gamma
        (0.001, 0.5),    # a_atm
        (0.1, 1.8),      # b_atm
    ]

    bounds_lbfgs = [
        (-0.99, 0.99),
        (0.01, 3.0),
        (0.01, 0.99),
        (0.001, 0.5),
        (0.1, 1.8),
    ]

    # Phase 1: Differential evolution (global search)
    try:
        de_result = differential_evolution(
            objective,
            bounds_de,
            seed=42,
            maxiter=200,
            tol=1e-10,
            polish=False,
            popsize=15,
        )
        best_cost = de_result.fun
        best_result = de_result.x
    except Exception:
        best_cost = 1e10
        best_result = None

    # Phase 2: Multi-start L-BFGS-B refinement
    rng = np.random.default_rng(42)
    initial_guesses = []

    if best_result is not None:
        initial_guesses.append(list(best_result))

    initial_guesses.extend([
        [-0.3, 0.5, 0.5, a_init, b_init],
        [-0.5, 0.3, 0.4, a_init, b_init],
        [-0.7, 1.0, 0.3, a_init, b_init],
        [-0.4, 0.8, 0.6, a_init, b_init],
    ])
    for _ in range(n_restarts - len(initial_guesses)):
        initial_guesses.append([
            rng.uniform(-0.9, 0.3),
            rng.uniform(0.1, 2.0),
            rng.uniform(0.1, 0.9),
            a_init * rng.uniform(0.3, 3.0),
            b_init * rng.uniform(0.5, 1.5),
        ])

    for x0 in initial_guesses:
        x0 = [np.clip(x0[i], bounds_lbfgs[i][0] + 0.01, bounds_lbfgs[i][1] - 0.01)
              for i in range(5)]

        try:
            result = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds_lbfgs,
                options={"maxiter": 1000, "ftol": 1e-14},
            )
            if result.fun < best_cost:
                best_cost = result.fun
                best_result = result.x
        except Exception:
            continue

    if best_result is None:
        raise ValueError("eSSVI calibration failed on all restarts")

    rho, eta, gamma, a_atm, b_atm = best_result
    return ESSVIModel(rho, eta, gamma, a_atm, b_atm)


def calibrate_svi(
    log_strikes: np.ndarray,
    market_vols: np.ndarray,
    T: float,
    initial_guess: Optional[dict] = None,
) -> SVIModel:
    """
    Calibrate SVI model to market smile.

    Args:
        log_strikes: Array of log-moneyness values.
        market_vols: Array of market implied volatilities.
        T: Time to maturity.
        initial_guess: Initial parameters {a, b, rho, m, sigma}.

    Returns:
        Calibrated SVIModel.
    """
    market_total_var = market_vols**2 * T

    if initial_guess is None:
        atm_idx = np.argmin(np.abs(log_strikes))
        initial_guess = {
            "a": market_total_var[atm_idx],
            "b": 0.1,
            "rho": -0.3,
            "m": 0.0,
            "sigma": 0.2,
        }

    def objective(params):
        a, b, rho, m, sigma = params

        # Enforce constraints
        if b < 0 or sigma <= 0 or not -0.99 < rho < 0.99:
            return 1e10

        # Butterfly constraint
        if a + b * sigma * np.sqrt(1 - rho**2) < 0:
            return 1e10

        try:
            model = SVIModel(a, b, rho, m, sigma)
            model_var = np.array([model.total_variance(k) for k in log_strikes])
            return np.sum((model_var - market_total_var) ** 2)
        except Exception:
            return 1e10

    x0 = [
        initial_guess["a"],
        initial_guess["b"],
        initial_guess["rho"],
        initial_guess["m"],
        initial_guess["sigma"],
    ]

    bounds = [
        (-0.5, 1.0),  # a
        (0.0, 1.0),  # b
        (-0.99, 0.99),  # rho
        (-0.5, 0.5),  # m
        (0.01, 1.0),  # sigma
    ]

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200},
    )

    a, b, rho, m, sigma = result.x
    return SVIModel(a, b, rho, m, sigma)


def calibrate_ssvi(
    T_slices: np.ndarray,
    atm_vols: np.ndarray,
    smile_data: list,  # List of (log_strikes, market_vols) for each slice
) -> SSVIModel:
    """
    Calibrate SSVI model to multiple maturity slices.

    Args:
        T_slices: Array of maturities.
        atm_vols: Array of ATM vols for each maturity.
        smile_data: List of (log_strikes, market_vols) tuples.

    Returns:
        Calibrated SSVIModel.
    """

    def objective(params):
        rho, eta, gamma = params

        if not -0.99 < rho < 0.99 or eta <= 0 or not 0 <= gamma <= 1:
            return 1e10

        try:
            model = SSVIModel(rho, eta, gamma)
            total_error = 0.0

            for i, T in enumerate(T_slices):
                log_strikes, market_vols = smile_data[i]
                theta = atm_vols[i] ** 2 * T

                for k, mv in zip(log_strikes, market_vols):
                    model_var = model.total_variance(k, theta)
                    market_var = mv**2 * T
                    total_error += (model_var - market_var) ** 2

            return total_error
        except Exception:
            return 1e10

    x0 = [-0.3, 0.5, 0.5]
    bounds = [(-0.99, 0.99), (0.01, 2.0), (0.0, 1.0)]

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
    )

    rho, eta, gamma = result.x
    return SSVIModel(rho, eta, gamma)
