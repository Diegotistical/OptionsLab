# src/pricing_models/monte_carlo.py
"""
Ultra-Fast Monte Carlo Option Pricing Engine.

Optimizations:
    - Single-step mode for European options (analytically equivalent, 100x faster)
    - Antithetic variates built into all backends (2x variance reduction)
    - Pluggable simulation backends (NumPy/Numba/QMC)
    - Minimal overhead orchestration

Usage:
    >>> from src.pricing_models.monte_carlo import MonteCarloPricer, MCMethod
    >>> pricer = MonteCarloPricer(n_paths=100000)  # Single-step default = fast
    >>> price = pricer.price(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call')
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Union

import numpy as np

from src.simulation.gbm_numba import NUMBA_AVAILABLE, simulate_gbm_numba
from src.simulation.gbm_numpy import simulate_gbm_numpy, simulate_gbm_numpy_fast
from src.simulation.gbm_qmc import simulate_gbm_qmc


class MCMethod(Enum):
    """Monte Carlo simulation backend."""

    NUMPY = "numpy"
    NUMBA = "numba"
    QMC = "qmc"
    FAST = "fast"  # Single-step, maximum speed


@dataclass
class MCResult:
    """Monte Carlo result with statistics."""

    price: float
    std_error: float = 0.0
    n_paths: int = 0


class MonteCarloPricer:
    """
    Production Monte Carlo pricer.

    Defaults to single-step mode which is analytically exact for
    European options and ~100x faster than multi-step.
    """

    __slots__ = ("num_simulations", "num_steps", "seed", "method", "_use_numba")

    def __init__(
        self,
        num_simulations: int = 100000,
        num_steps: int = 1,  # Default to single-step = fast
        seed: Optional[int] = None,
        method: MCMethod = MCMethod.NUMPY,
    ):
        if num_simulations < 1:
            raise ValueError("num_simulations must be >= 1")

        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.seed = (
            seed if seed is not None else np.random.default_rng().integers(0, 2**31)
        )
        self.method = method
        self._use_numba = NUMBA_AVAILABLE and method == MCMethod.NUMBA

    def _simulate(
        self,
        S: float,
        T: float,
        r: float,
        sigma: float,
        q: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Dispatch to simulation backend."""
        actual_seed = seed if seed is not None else self.seed

        # Fast single-step mode
        if self.method == MCMethod.FAST or (
            self.num_steps == 1 and self.method == MCMethod.NUMPY
        ):
            return simulate_gbm_numpy_fast(
                S, T, r, sigma, q, self.num_simulations, actual_seed
            )

        if self.method == MCMethod.QMC:
            return simulate_gbm_qmc(
                S, T, r, sigma, q, self.num_simulations, self.num_steps, actual_seed
            )

        if self._use_numba:
            return simulate_gbm_numba(
                S, T, r, sigma, q, self.num_simulations, self.num_steps, actual_seed
            )

        return simulate_gbm_numpy(
            S, T, r, sigma, q, self.num_simulations, self.num_steps, actual_seed
        )

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
        seed: Optional[int] = None,
        return_error: bool = False,
    ) -> Union[float, MCResult]:
        """
        Price European option.

        Args:
            S, K, T, r, sigma: Standard option params.
            option_type: 'call' or 'put'.
            q: Dividend yield.
            seed: Optional seed override.
            return_error: Return MCResult with std error.

        Returns:
            Price (float) or MCResult.
        """
        if T <= 0:
            intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
            return MCResult(intrinsic, 0.0, 0) if return_error else intrinsic

        terminal = self._simulate(S, T, r, sigma, q, seed)

        # Vectorized payoff
        if option_type == "call":
            payoffs = np.maximum(terminal - K, 0.0)
        else:
            payoffs = np.maximum(K - terminal, 0.0)

        discount = np.exp(-r * T)
        price = float(discount * np.mean(payoffs))

        if return_error:
            std_error = float(discount * np.std(payoffs) / np.sqrt(len(payoffs)))
            return MCResult(price, std_error, len(payoffs))

        return price

    def price_with_control_variate(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
        seed: Optional[int] = None,
    ) -> float:
        """
        Price with terminal spot as control variate.

        Control: E[S_T] = S * exp((r-q)*T)
        """
        terminal = self._simulate(S, T, r, sigma, q, seed)

        if option_type == "call":
            payoffs = np.maximum(terminal - K, 0.0)
        else:
            payoffs = np.maximum(K - terminal, 0.0)

        discounted = np.exp(-r * T) * payoffs

        # Control variate adjustment
        control_mean = np.mean(terminal)
        forward = S * np.exp((r - q) * T)

        cov = np.cov(discounted, terminal)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 1e-10 else 0.0

        return float(np.mean(discounted) - beta * (control_mean - forward))


# Keep NUMBA_AVAILABLE export for backward compatibility
__all__ = ["MonteCarloPricer", "MCMethod", "MCResult", "NUMBA_AVAILABLE"]
