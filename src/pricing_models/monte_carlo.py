# src/pricing_models/monte_carlo.py

from typing import Callable, Literal, Optional

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from ..exceptions.montecarlo_exceptions import InputValidationError, MonteCarloError

__all__ = ["MonteCarloPricer"]


class MonteCarloPricer:
    """
    Monte Carlo option pricer with Greeks computation and optional Numba acceleration.

    Features:
    - European option pricing
    - Delta, Gamma, Vega, Theta, Rho
    - Antithetic variance reduction
    - Vectorized or Numba-accelerated simulation
    - Input validation for parameters
    - Customizable number of simulations and steps
    """

    def __init__(
        self,
        num_simulations: int = 10000,
        num_steps: int = 100,
        seed: Optional[int] = None,
        use_numba: bool = False,
    ):
        if num_simulations <= 0 or num_steps <= 0:
            raise InputValidationError(
                "num_simulations and num_steps must be positive integers"
            )
        if use_numba and not NUMBA_AVAILABLE:
            raise MonteCarloError("Numba not installed; cannot enable acceleration")

        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.rng = np.random.default_rng(seed)
        self.use_numba = use_numba

    def _validate_inputs(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        q: float,
    ):
        if option_type not in {"call", "put"}:
            raise InputValidationError("option_type must be 'call' or 'put'")
        if S <= 0 or K <= 0 or T <= 0 or sigma < 0 or q < 0:
            raise InputValidationError(
                "Spot, strike, T, sigma, and q must be non-negative and T > 0"
            )

    def _simulate_terminal_prices_vectorized(
        self, S: float, T: float, r: float, sigma: float, q: float
    ) -> np.ndarray:
        dt = T / self.num_steps
        drift = (r - q - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)

        rand_normals = self.rng.normal(size=(self.num_simulations, self.num_steps))
        antithetic = -rand_normals

        log_paths_pos = np.log(S) + np.cumsum(drift + vol * rand_normals, axis=1)
        log_paths_neg = np.log(S) + np.cumsum(drift + vol * antithetic, axis=1)

        terminal_prices_pos = np.exp(log_paths_pos[:, -1])
        terminal_prices_neg = np.exp(log_paths_neg[:, -1])

        return np.concatenate([terminal_prices_pos, terminal_prices_neg])

    def _simulate_terminal_prices_numba(
        self, S: float, T: float, r: float, sigma: float, q: float
    ) -> np.ndarray:
        from numba import njit

        @njit
        def simulate(num_simulations, num_steps, S, drift, vol, seed=None):
            rng = np.random.RandomState(seed)
            terminal = np.empty(num_simulations * 2, dtype=np.float64)
            for i in range(num_simulations):
                logS_pos = np.log(S)
                logS_neg = np.log(S)
                for t in range(num_steps):
                    z = rng.normal()
                    logS_pos += drift + vol * z
                    logS_neg += drift - vol * z
                terminal[i] = np.exp(logS_pos)
                terminal[i + num_simulations] = np.exp(logS_neg)
            return terminal

        dt = T / self.num_steps
        drift = (r - q - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)
        return simulate(self.num_simulations, self.num_steps, S, drift, vol)

    def _simulate_terminal_prices(
        self, S: float, T: float, r: float, sigma: float, q: float
    ) -> np.ndarray:
        if self.use_numba:
            return self._simulate_terminal_prices_numba(S, T, r, sigma, q)
        return self._simulate_terminal_prices_vectorized(S, T, r, sigma, q)

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
    ) -> float:
        """
        Compute European option price using Monte Carlo simulation.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            q: Dividend yield (default: 0.0)

        Returns:
            Option price as float
        """
        self._validate_inputs(S, K, T, r, sigma, option_type, q)

        terminal_prices = self._simulate_terminal_prices(S, T, r, sigma, q)

        if option_type == "call":
            payoffs = np.maximum(terminal_prices - K, 0.0)
        else:
            payoffs = np.maximum(K - terminal_prices, 0.0)

        return np.exp(-r * T) * np.mean(payoffs)

    def delta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
        h: float = 1e-4,
    ) -> float:
        """
        Delta: derivative of price with respect to spot price.
        """
        return (
            self.price(S + h, K, T, r, sigma, option_type, q)
            - self.price(S - h, K, T, r, sigma, option_type, q)
        ) / (2 * h)

    def gamma(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
        h: float = 1e-4,
    ) -> float:
        """
        Gamma: second derivative of price with respect to spot price.
        """
        price_up = self.price(S + h, K, T, r, sigma, option_type, q)
        price_mid = self.price(S, K, T, r, sigma, option_type, q)
        price_down = self.price(S - h, K, T, r, sigma, option_type, q)
        return (price_up - 2 * price_mid + price_down) / (h * h)

    def vega(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
        h: float = 1e-4,
    ) -> float:
        """
        Vega: derivative of price with respect to volatility.
        """
        return (
            self.price(S, K, T, r, sigma + h, option_type, q)
            - self.price(S, K, T, r, sigma - h, option_type, q)
        ) / (2 * h)

    def theta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
        dt: float = 1 / 365,
    ) -> float:
        """
        Theta: derivative of price with respect to time (negative decay).
        """
        if T > dt:
            return (
                self.price(S, K, T - dt, r, sigma, option_type, q)
                - self.price(S, K, T, r, sigma, option_type, q)
            ) / dt
        return -self.price(S, K, T, r, sigma, option_type, q) / dt

    def rho(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
        h: float = 1e-4,
    ) -> float:
        """
        Rho: derivative of price with respect to interest rate.
        """
        return (
            self.price(S, K, T, r + h, sigma, option_type, q)
            - self.price(S, K, T, r, sigma, option_type, q)
        ) / h
