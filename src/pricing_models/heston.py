# src/pricing_models/heston.py
"""
Heston Stochastic Volatility Model.

Implements the Heston (1993) model where variance follows a CIR process:

    dS = (r - q) S dt + √v S dW₁
    dv = κ(θ - v) dt + σᵥ √v dW₂
    dW₁ dW₂ = ρ dt

Parameters:
    κ (kappa): Mean reversion speed
    θ (theta): Long-run variance
    σᵥ (sigma_v): Volatility of variance (vol-of-vol)
    ρ (rho): Correlation between spot and variance
    v₀ (v0): Initial variance

Features:
    - Semi-analytical pricing via characteristic function (Lewis/Gatheral)
    - Monte Carlo with full truncation Euler scheme
    - Greeks via finite differences

Reference:
    Heston, S. L. (1993). A Closed-Form Solution for Options with
    Stochastic Volatility with Applications to Bond and Currency Options.
    Review of Financial Studies, 6(2), 327-343.

Usage:
    >>> from src.pricing_models.heston import HestonPricer
    >>> pricer = HestonPricer(kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04)
    >>> price = pricer.price_european(S=100, K=100, T=1.0, r=0.05)
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


@dataclass
class HestonPricer:
    """
    Heston stochastic volatility model pricer.

    Attributes:
        kappa: Mean reversion speed of variance.
        theta: Long-run variance level.
        sigma_v: Volatility of variance (vol-of-vol).
        rho: Correlation between spot and variance Brownian motions.
        v0: Initial variance.
    """

    kappa: float  # Mean reversion speed
    theta: float  # Long-run variance
    sigma_v: float  # Vol of vol
    rho: float  # Correlation
    v0: float  # Initial variance

    def __post_init__(self):
        """Validate Heston parameters."""
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.theta <= 0:
            raise ValueError("theta must be positive")
        if self.sigma_v <= 0:
            raise ValueError("sigma_v must be positive")
        if not -1 <= self.rho <= 1:
            raise ValueError("rho must be in [-1, 1]")
        if self.v0 <= 0:
            raise ValueError("v0 must be positive")

        # Feller condition check (for positivity of variance)
        feller = 2 * self.kappa * self.theta - self.sigma_v**2
        if feller < 0:
            import warnings

            warnings.warn(
                f"Feller condition not satisfied (2κθ - σᵥ² = {feller:.4f} < 0). "
                "Variance may hit zero in simulations."
            )

    def _characteristic_function(
        self,
        u: complex,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
    ) -> complex:
        """
        Heston characteristic function (Gatheral formulation).

        Uses the "good" formulation that avoids branch cut issues.
        """
        kappa, theta, sigma_v, rho, v0 = (
            self.kappa,
            self.theta,
            self.sigma_v,
            self.rho,
            self.v0,
        )

        # Log-forward moneyness
        x = np.log(S / K) + (r - q) * T

        # Gatheral's formulation
        alpha = -0.5 * u * (u + 1j)
        beta = kappa - rho * sigma_v * 1j * u

        gamma = 0.5 * sigma_v**2
        d = np.sqrt(beta**2 - 4 * alpha * gamma)

        # Select branch to ensure continuity
        r_plus = (beta + d) / (sigma_v**2)
        r_minus = (beta - d) / (sigma_v**2)

        g = r_minus / r_plus

        # Time-dependent coefficients
        exp_dT = np.exp(-d * T)
        C = kappa * (
            r_minus * T - (2 / sigma_v**2) * np.log((1 - g * exp_dT) / (1 - g))
        )
        D = r_minus * (1 - exp_dT) / (1 - g * exp_dT)

        return np.exp(C * theta + D * v0 + 1j * u * x)

    def price_european(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: Literal["call", "put"] = "call",
    ) -> float:
        """
        Price European option using semi-analytical characteristic function.

        Uses numerical integration of the Heston characteristic function
        following the Lewis (2000) / Gatheral approach.

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            q: Dividend yield.
            option_type: 'call' or 'put'.

        Returns:
            Option price.
        """
        if T <= 0:
            intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
            return intrinsic

        # Forward price
        F = S * np.exp((r - q) * T)

        def integrand(u):
            """Real part of the integrand for Lewis formula."""
            cf = self._characteristic_function(u - 0.5j, S, K, T, r, q)
            return np.real(np.exp(-1j * u * np.log(K / F)) * cf / (u**2 + 0.25))

        # Numerical integration
        integral, _ = quad(integrand, 0, 100, limit=100)

        # Call price via Lewis formula
        call_price = (
            S * np.exp(-q * T) - (np.sqrt(K * F) / np.pi) * np.exp(-r * T) * integral
        )

        if option_type == "call":
            return max(call_price, 0.0)
        else:
            # Put-call parity
            put_price = call_price - S * np.exp(-q * T) + K * np.exp(-r * T)
            return max(put_price, 0.0)

    def price_monte_carlo(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: Literal["call", "put"] = "call",
        n_paths: int = 100000,
        n_steps: int = 252,
        seed: Optional[int] = None,
    ) -> float:
        """
        Price European option via Monte Carlo simulation.

        Uses the Full Truncation Euler scheme for variance positivity.

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            q: Dividend yield.
            option_type: 'call' or 'put'.
            n_paths: Number of simulation paths.
            n_steps: Number of time steps.
            seed: Random seed.

        Returns:
            Option price.
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        kappa, theta, sigma_v, rho = self.kappa, self.theta, self.sigma_v, self.rho

        # Initialize
        log_S = np.full(n_paths, np.log(S))
        v = np.full(n_paths, self.v0)

        # Correlation structure
        rho_sqrt = np.sqrt(1 - rho**2)

        for _ in range(n_steps):
            # Correlated Brownian motions
            Z1 = np.random.standard_normal(n_paths)
            Z2 = rho * Z1 + rho_sqrt * np.random.standard_normal(n_paths)

            # Full truncation scheme for variance
            v_pos = np.maximum(v, 0)
            sqrt_v = np.sqrt(v_pos)

            # Update log-spot
            log_S += (r - q - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * Z1

            # Update variance (truncated Euler)
            v += kappa * (theta - v_pos) * dt + sigma_v * sqrt_v * sqrt_dt * Z2
            v = np.maximum(v, 0)  # Enforce positivity

        # Terminal prices
        S_T = np.exp(log_S)

        # Payoffs
        if option_type == "call":
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)

        return np.exp(-r * T) * np.mean(payoffs)

    def simulate_paths(
        self,
        S: float,
        T: float,
        r: float,
        q: float = 0.0,
        n_paths: int = 1000,
        n_steps: int = 252,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate spot and variance paths.

        Returns:
            Tuple of (spot_paths, variance_paths) with shape (n_paths, n_steps + 1).
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        kappa, theta, sigma_v, rho = self.kappa, self.theta, self.sigma_v, self.rho
        rho_sqrt = np.sqrt(1 - rho**2)

        # Initialize paths
        spot_paths = np.zeros((n_paths, n_steps + 1))
        var_paths = np.zeros((n_paths, n_steps + 1))
        spot_paths[:, 0] = S
        var_paths[:, 0] = self.v0

        log_S = np.log(S) * np.ones(n_paths)
        v = self.v0 * np.ones(n_paths)

        for t in range(1, n_steps + 1):
            Z1 = np.random.standard_normal(n_paths)
            Z2 = rho * Z1 + rho_sqrt * np.random.standard_normal(n_paths)

            v_pos = np.maximum(v, 0)
            sqrt_v = np.sqrt(v_pos)

            log_S += (r - q - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * Z1
            v += kappa * (theta - v_pos) * dt + sigma_v * sqrt_v * sqrt_dt * Z2
            v = np.maximum(v, 0)

            spot_paths[:, t] = np.exp(log_S)
            var_paths[:, t] = v

        return spot_paths, var_paths

    # Greeks are computed via src.greeks.unified_greeks.compute_greeks_unified()
    # Use: from src.greeks import compute_greeks_unified, HestonAdapter
    #      greeks = compute_greeks_unified(HestonAdapter(heston), S, K, T, r, sigma, option_type)


def calibrate_heston(
    market_data: dict,
    initial_params: Optional[dict] = None,
) -> HestonPricer:
    """
    Calibrate Heston model to market implied volatility surface.

    Args:
        market_data: Dict with keys:
            - spot: Current spot price
            - strikes: Array of strikes
            - maturities: Array of maturities
            - market_ivs: 2D array of implied volatilities
            - r: Risk-free rate
            - q: Dividend yield (optional)
        initial_params: Initial guess for (kappa, theta, sigma_v, rho, v0).

    Returns:
        Calibrated HestonPricer instance.
    """
    from scipy.optimize import minimize

    spot = market_data["spot"]
    strikes = np.array(market_data["strikes"])
    maturities = np.array(market_data["maturities"])
    market_ivs = np.array(market_data["market_ivs"])
    r_rate = market_data["r"]
    q_yield = market_data.get("q", 0.0)

    # Default initial params
    if initial_params is None:
        initial_params = {
            "kappa": 2.0,
            "theta": 0.04,
            "sigma_v": 0.3,
            "rho": -0.5,
            "v0": 0.04,
        }

    def objective(params):
        kappa, theta, sigma_v, rho, v0 = params

        # Bounds enforcement
        if kappa <= 0 or theta <= 0 or sigma_v <= 0 or v0 <= 0:
            return 1e10
        if not -0.99 <= rho <= 0.99:
            return 1e10

        try:
            pricer = HestonPricer(kappa, theta, sigma_v, rho, v0)
        except ValueError:
            return 1e10

        total_error = 0.0
        count = 0

        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                if np.isnan(market_ivs[i, j]):
                    continue

                try:
                    model_price = pricer.price_european(spot, K, T, r_rate, q_yield)
                    # Convert to IV for comparison
                    from src.pricing_models.iv_solver import implied_volatility

                    model_iv = implied_volatility(
                        model_price, spot, K, T, r_rate, "call", q_yield
                    )
                    total_error += (model_iv - market_ivs[i, j]) ** 2
                    count += 1
                except Exception:
                    total_error += 1.0  # Penalty for failures
                    count += 1

        return total_error / max(count, 1)

    x0 = [
        initial_params["kappa"],
        initial_params["theta"],
        initial_params["sigma_v"],
        initial_params["rho"],
        initial_params["v0"],
    ]

    bounds = [
        (0.01, 10.0),  # kappa
        (0.001, 1.0),  # theta
        (0.01, 2.0),  # sigma_v
        (-0.99, 0.99),  # rho
        (0.001, 1.0),  # v0
    ]

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200},
    )

    kappa, theta, sigma_v, rho, v0 = result.x
    return HestonPricer(kappa, theta, sigma_v, rho, v0)
