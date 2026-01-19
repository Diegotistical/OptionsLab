# src/greeks/unified_greeks.py
"""
Unified Greeks Computation for All Pricing Models.

Provides a protocol-based adapter system that works with ANY pricer,
optimized for performance by reusing price calculations.

Features:
    - Works with BinomialTree, MonteCarlo, Heston, SABR, FDM, etc.
    - Computes all Greeks in a single optimized pass
    - Caches intermediate results to avoid redundant calculations
    - Supports first-order (Δ, Γ, ν, Θ, ρ) and second-order (Vanna, Charm, Vomma)
    - Adaptive step sizes for numerical stability

Usage:
    >>> from src.greeks.unified_greeks import compute_greeks_unified, HestonAdapter
    >>> from src.pricing_models import HestonPricer
    >>> heston = HestonPricer(kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04)
    >>> greeks = compute_greeks_unified(HestonAdapter(heston), S=100, K=100, T=1.0, r=0.05, sigma=0.2)
"""

from collections import OrderedDict
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from src.exceptions.greek_exceptions import GreeksError

__all__ = [
    "PricerProtocol",
    "compute_greeks_unified",
    "HestonAdapter",
    "SABRAdapter",
    "FDMAdapter",
    "JumpDiffusionAdapter",
    "ExoticAdapter",
]


# =============================================================================
# Protocol Definition
# =============================================================================


@runtime_checkable
class PricerProtocol(Protocol):
    """
    Protocol that defines the interface any pricer must implement.

    Models that naturally conform: MonteCarloPricer, BinomialTree
    Models needing adapters: HestonPricer, SABRModel, FDM solvers
    """

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
        **kwargs,
    ) -> float:
        """Compute option price."""
        ...


# =============================================================================
# Adapter Classes for Non-Conforming Models
# =============================================================================


class HestonAdapter:
    """
    Adapter to make HestonPricer conform to PricerProtocol.

    Maps sigma parameter to v0 (initial variance = sigma²).
    """

    def __init__(self, heston_pricer):
        self.heston = heston_pricer
        self._original_v0 = heston_pricer.v0

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
        **kwargs,
    ) -> float:
        # Map sigma to v0 = sigma²
        self.heston.v0 = sigma**2
        try:
            return self.heston.price_european(S, K, T, r, q, option_type)
        finally:
            self.heston.v0 = self._original_v0

    def __repr__(self):
        return f"HestonAdapter({self.heston})"


class SABRAdapter:
    """
    Adapter to make SABRModel conform to PricerProtocol.

    Uses SABR implied vol in Black formula.
    """

    def __init__(self, sabr_model):
        self.sabr = sabr_model

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,  # Ignored, SABR computes its own vol
        option_type: Literal["call", "put"],
        q: float = 0.0,
        **kwargs,
    ) -> float:
        F = S * np.exp((r - q) * T)
        return self.sabr.price(F, K, T, r, option_type)


class FDMAdapter:
    """
    Adapter for CrankNicolsonSolver / ExplicitFDMSolver.
    """

    def __init__(self, fdm_solver, exercise: str = "european"):
        self.fdm = fdm_solver
        self.exercise = exercise

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
        **kwargs,
    ) -> float:
        return self.fdm.price(S, K, T, r, sigma, option_type, self.exercise, q)


class JumpDiffusionAdapter:
    """
    Adapter for MertonJumpDiffusion / KouJumpDiffusion.
    """

    def __init__(self, jd_model):
        self.jd = jd_model

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
        **kwargs,
    ) -> float:
        return self.jd.price(S, K, T, r, sigma, option_type, q)


class ExoticAdapter:
    """
    Adapter for exotic options (Asian, Barrier, American, Lookback, etc.).

    Maps the unified signature to the exotic option's dataclass-based interface.
    """

    def __init__(
        self, exotic_option, n_paths: int = 50000, n_steps: int = 252, **exotic_kwargs
    ):
        """
        Args:
            exotic_option: An exotic option instance (AsianOption, BarrierOption, etc.)
            n_paths: Number of Monte Carlo paths.
            n_steps: Number of time steps.
            **exotic_kwargs: Additional kwargs passed to exotic.price().
        """
        self.exotic = exotic_option
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.exotic_kwargs = exotic_kwargs

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
        **kwargs,
    ) -> float:
        # Update exotic option parameters
        self.exotic.S = S
        self.exotic.K = K
        self.exotic.T = T
        self.exotic.r = r
        self.exotic.sigma = sigma
        self.exotic.q = q

        # Merge kwargs
        price_kwargs = {**self.exotic_kwargs, **kwargs}
        if "option_type" not in price_kwargs:
            price_kwargs["option_type"] = option_type

        return self.exotic.price(
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            **price_kwargs,
        )


# =============================================================================
# Optimized Greeks Computation
# =============================================================================


def compute_greeks_unified(
    pricer: PricerProtocol,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
    include_second_order: bool = True,
    **pricer_kwargs,
) -> OrderedDict[str, float]:
    """
    Compute all Greeks for any pricer using optimized finite differences.

    Performance optimizations:
        - Reuses price calculations across Greeks (e.g., P(S+h) used for both Delta and Gamma)
        - Computes Delta and Gamma together in 3 price calls instead of 5
        - Uses adaptive step sizes based on parameter magnitudes
        - Caches intermediate results

    Args:
        pricer: Any object implementing PricerProtocol (or adapted via wrapper).
        S: Spot price.
        K: Strike price.
        T: Time to maturity.
        r: Risk-free rate.
        sigma: Volatility.
        option_type: 'call' or 'put'.
        q: Dividend yield.
        include_second_order: Whether to compute Vanna, Charm, Vomma.
        **pricer_kwargs: Additional kwargs passed to pricer.price().

    Returns:
        OrderedDict with: price, delta, gamma, vega, theta, rho,
        and optionally vanna, charm, vomma.
    """
    try:
        # Adaptive step sizes for numerical stability
        h_S = max(1e-4, 0.01 * S)  # 1% of spot
        h_sigma = max(1e-4, 0.01)  # 1% vol bump
        h_r = 1e-4  # 1bp rate bump
        h_T = 1 / 365.0  # 1 day time bump

        # Helper: price with modified params (using cache for efficiency)
        _cache = {}

        def get_price(S_=S, K_=K, T_=T, r_=r, sigma_=sigma, q_=q):
            key = (S_, K_, T_, r_, sigma_, q_)
            if key not in _cache:
                _cache[key] = pricer.price(
                    S_, K_, T_, r_, sigma_, option_type, q_, **pricer_kwargs
                )
            return _cache[key]

        # =====================================================================
        # First-Order Greeks (Optimized: reuse calculations)
        # =====================================================================

        # Base price
        p_mid = get_price()

        # Delta & Gamma (3 prices for both, not 5)
        p_S_up = get_price(S_=S + h_S)
        p_S_down = get_price(S_=S - h_S)

        delta = (p_S_up - p_S_down) / (2 * h_S)
        gamma = (p_S_up - 2 * p_mid + p_S_down) / (h_S**2)

        # Vega
        p_sigma_up = get_price(sigma_=sigma + h_sigma)
        p_sigma_down = get_price(sigma_=sigma - h_sigma)
        vega = (p_sigma_up - p_sigma_down) / (2 * h_sigma)

        # Theta
        if T > h_T:
            p_T_down = get_price(T_=T - h_T)
            theta = (p_T_down - p_mid) / h_T
        else:
            theta = -p_mid / max(T, 1e-6)  # Decay to 0 at expiry

        # Rho
        p_r_up = get_price(r_=r + h_r)
        p_r_down = get_price(r_=r - h_r)
        rho = (p_r_up - p_r_down) / (2 * h_r)

        greeks = OrderedDict(
            [
                ("price", p_mid),
                ("delta", delta),
                ("gamma", gamma),
                ("vega", vega),
                ("theta", theta),
                ("rho", rho),
            ]
        )

        # =====================================================================
        # Second-Order Greeks (Optional)
        # =====================================================================

        if include_second_order:
            # Vanna: ∂Δ/∂σ (cross-gamma)
            # Requires: P(S+h, σ+h), P(S+h, σ-h), P(S-h, σ+h), P(S-h, σ-h)
            p_Su_sigmaU = get_price(S_=S + h_S, sigma_=sigma + h_sigma)
            p_Su_sigmaD = get_price(S_=S + h_S, sigma_=sigma - h_sigma)
            p_Sd_sigmaU = get_price(S_=S - h_S, sigma_=sigma + h_sigma)
            p_Sd_sigmaD = get_price(S_=S - h_S, sigma_=sigma - h_sigma)

            vanna = (p_Su_sigmaU - p_Su_sigmaD - p_Sd_sigmaU + p_Sd_sigmaD) / (
                4 * h_S * h_sigma
            )

            # Charm: ∂Δ/∂T (delta decay)
            if T > h_T:
                p_Su_Td = get_price(S_=S + h_S, T_=T - h_T)
                p_Sd_Td = get_price(S_=S - h_S, T_=T - h_T)
                delta_T_down = (p_Su_Td - p_Sd_Td) / (2 * h_S)
                charm = (delta_T_down - delta) / h_T
            else:
                charm = 0.0

            # Vomma: ∂ν/∂σ (vol convexity)
            vomma = (p_sigma_up - 2 * p_mid + p_sigma_down) / (h_sigma**2)

            greeks["vanna"] = vanna
            greeks["charm"] = charm
            greeks["vomma"] = vomma

        return greeks

    except Exception as e:
        raise GreeksError(f"Failed to compute unified Greeks: {str(e)}") from e


# =============================================================================
# Convenience: All Greeks for a specific model type
# =============================================================================


def greeks_heston(
    heston_pricer,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> OrderedDict[str, float]:
    """Convenience function for Heston Greeks."""
    return compute_greeks_unified(
        HestonAdapter(heston_pricer), S, K, T, r, sigma, option_type, q
    )


def greeks_sabr(
    sabr_model,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
) -> OrderedDict[str, float]:
    """Convenience function for SABR Greeks."""
    return compute_greeks_unified(
        SABRAdapter(sabr_model),
        S,
        K,
        T,
        r,
        0.2,
        option_type,
        q,  # sigma ignored by SABR
    )


def greeks_fdm(
    fdm_solver,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    exercise: str = "european",
    q: float = 0.0,
) -> OrderedDict[str, float]:
    """Convenience function for FDM Greeks."""
    return compute_greeks_unified(
        FDMAdapter(fdm_solver, exercise), S, K, T, r, sigma, option_type, q
    )
