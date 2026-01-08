# src/pricing_models/monte_carlo.py
"""
High-performance Monte Carlo option pricer with Greeks computation.

This module provides an optimized Monte Carlo simulation engine for pricing
European options and computing Greeks (Delta, Gamma, Vega, Theta, Rho).

Features:
    - Vectorized NumPy operations for base performance
    - Optional Numba JIT acceleration with parallel execution
    - Antithetic variance reduction
    - Reproducible results via seeded RNG
    - Thread-safe implementation

Example:
    >>> from src.pricing_models.monte_carlo import MonteCarloPricer
    >>> pricer = MonteCarloPricer(num_simulations=100000, use_numba=True)
    >>> price = pricer.price(S=100, K=100, T=1.0, r=0.05, sigma=0.2,
    ...                      option_type='call')
    >>> print(f"Option price: {price:.4f}")
"""

from typing import Literal, Optional, Tuple

import numpy as np

# Check for Numba availability
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Create dummy decorators for when Numba is not available
    def njit(*args, **kwargs):
        """Dummy njit decorator when Numba is not installed."""

        def decorator(func):
            return func

        return decorator if not args else decorator(args[0])

    def prange(*args, **kwargs):
        """Dummy prange that falls back to range."""
        return range(*args)


from src.exceptions.montecarlo_exceptions import InputValidationError, MonteCarloError

__all__ = ["MonteCarloPricer", "NUMBA_AVAILABLE"]


# =============================================================================
# Module-level Numba-compiled functions (compiled once at import time)
# =============================================================================

if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _simulate_terminal_prices_numba_single(
        S: float,
        T: float,
        r: float,
        sigma: float,
        q: float,
        num_simulations: int,
        num_steps: int,
        seed: int,
    ) -> np.ndarray:
        """
        Simulate terminal prices using Numba JIT with antithetic variates.

        This function is compiled to machine code at module import time,
        eliminating JIT compilation overhead during runtime.

        Args:
            S: Spot price.
            T: Time to maturity in years.
            r: Risk-free interest rate.
            sigma: Volatility.
            q: Dividend yield.
            num_simulations: Number of Monte Carlo paths.
            num_steps: Number of time discretization steps.
            seed: Random seed for reproducibility.

        Returns:
            Array of terminal prices with shape (2 * num_simulations,).
        """
        np.random.seed(seed)
        dt = T / num_steps
        drift = (r - q - 0.5 * sigma * sigma) * dt
        vol = sigma * np.sqrt(dt)

        # Pre-allocate output array
        terminal = np.empty(num_simulations * 2, dtype=np.float64)

        for i in range(num_simulations):
            log_S_pos = np.log(S)
            log_S_neg = np.log(S)

            for t in range(num_steps):
                z = np.random.randn()
                log_S_pos += drift + vol * z
                log_S_neg += drift - vol * z  # Antithetic variate

            terminal[i] = np.exp(log_S_pos)
            terminal[i + num_simulations] = np.exp(log_S_neg)

        return terminal

    @njit(parallel=True, cache=True, fastmath=True)
    def _simulate_terminal_prices_numba_parallel(
        S: float,
        T: float,
        r: float,
        sigma: float,
        q: float,
        num_simulations: int,
        num_steps: int,
        seed: int,
    ) -> np.ndarray:
        """
        Parallel Numba simulation with antithetic variates.

        Uses parallel range (prange) for multi-core execution.
        Each thread gets a unique seed derived from the base seed.

        Args:
            S: Spot price.
            T: Time to maturity in years.
            r: Risk-free interest rate.
            sigma: Volatility.
            q: Dividend yield.
            num_simulations: Number of Monte Carlo paths.
            num_steps: Number of time discretization steps.
            seed: Base random seed.

        Returns:
            Array of terminal prices with shape (2 * num_simulations,).
        """
        dt = T / num_steps
        drift = (r - q - 0.5 * sigma * sigma) * dt
        vol = sigma * np.sqrt(dt)

        terminal = np.empty(num_simulations * 2, dtype=np.float64)

        for i in prange(num_simulations):
            # Unique seed per simulation for parallel safety
            np.random.seed(seed + i)
            log_S_pos = np.log(S)
            log_S_neg = np.log(S)

            for t in range(num_steps):
                z = np.random.randn()
                log_S_pos += drift + vol * z
                log_S_neg += drift - vol * z

            terminal[i] = np.exp(log_S_pos)
            terminal[i + num_simulations] = np.exp(log_S_neg)

        return terminal


class MonteCarloPricer:
    """
    Monte Carlo option pricer with Greeks computation and optional acceleration.

    This class provides European option pricing using Monte Carlo simulation
    with antithetic variance reduction. It supports both pure NumPy
    vectorization and Numba JIT compilation for high performance.

    Attributes:
        num_simulations (int): Number of Monte Carlo paths.
        num_steps (int): Number of time discretization steps.
        use_numba (bool): Whether to use Numba acceleration.
        use_parallel (bool): Whether to use parallel execution (requires Numba).
        rng (np.random.Generator): Random number generator instance.

    Example:
        >>> pricer = MonteCarloPricer(num_simulations=50000, use_numba=True)
        >>> price = pricer.price(100, 100, 1.0, 0.05, 0.2, 'call')
        >>> delta = pricer.delta(100, 100, 1.0, 0.05, 0.2, 'call')
    """

    def __init__(
        self,
        num_simulations: int = 10000,
        num_steps: int = 100,
        seed: Optional[int] = None,
        use_numba: bool = False,
        use_parallel: bool = False,
    ) -> None:
        """
        Initialize Monte Carlo pricer.

        Args:
            num_simulations: Number of Monte Carlo paths. Must be positive.
                Higher values reduce variance but increase computation time.
            num_steps: Number of time discretization steps. Must be positive.
                More steps improve accuracy for path-dependent options.
            seed: Random seed for reproducibility. If None, uses system entropy.
            use_numba: Enable Numba JIT acceleration. Requires numba package.
            use_parallel: Enable parallel execution. Requires use_numba=True.

        Raises:
            InputValidationError: If num_simulations or num_steps <= 0.
            MonteCarloError: If use_numba=True but Numba is not installed.
        """
        if num_simulations <= 0 or num_steps <= 0:
            raise InputValidationError(
                "num_simulations and num_steps must be positive integers"
            )

        if use_numba and not NUMBA_AVAILABLE:
            raise MonteCarloError(
                "Numba is not installed. Install with: pip install numba"
            )

        if use_parallel and not use_numba:
            raise MonteCarloError("Parallel execution requires use_numba=True")

        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.seed = (
            seed if seed is not None else np.random.default_rng().integers(0, 2**31)
        )
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.use_parallel = use_parallel and self.use_numba
        self.rng = np.random.default_rng(seed)

    def _validate_inputs(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        q: float,
    ) -> None:
        """
        Validate option parameters.

        Args:
            S: Spot price (must be > 0).
            K: Strike price (must be > 0).
            T: Time to maturity (must be > 0).
            r: Risk-free rate.
            sigma: Volatility (must be >= 0).
            option_type: 'call' or 'put'.
            q: Dividend yield (must be >= 0).

        Raises:
            InputValidationError: If any parameter is invalid.
        """
        if option_type not in {"call", "put"}:
            raise InputValidationError("option_type must be 'call' or 'put'")
        if S <= 0:
            raise InputValidationError("Spot price S must be positive")
        if K <= 0:
            raise InputValidationError("Strike price K must be positive")
        if T <= 0:
            raise InputValidationError("Time to maturity T must be positive")
        if sigma < 0:
            raise InputValidationError("Volatility sigma must be non-negative")
        if q < 0:
            raise InputValidationError("Dividend yield q must be non-negative")

    def _simulate_terminal_prices_vectorized(
        self,
        S: float,
        T: float,
        r: float,
        sigma: float,
        q: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate terminal prices using vectorized NumPy operations.

        Uses antithetic variance reduction by simulating paths with both
        positive and negative random innovations.

        Args:
            S: Spot price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            q: Dividend yield.
            seed: Optional seed for this specific simulation.

        Returns:
            Array of terminal prices with shape (2 * num_simulations,).
        """
        rng = np.random.default_rng(seed if seed is not None else self.seed)

        dt = T / self.num_steps
        drift = (r - q - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)

        # Generate random normals for all paths and steps
        rand_normals = rng.standard_normal((self.num_simulations, self.num_steps))

        # Compute log price increments
        increments = drift + vol * rand_normals
        antithetic_increments = drift - vol * rand_normals  # Antithetic paths

        # Cumulative sum for path simulation
        log_paths_pos = np.log(S) + np.cumsum(increments, axis=1)
        log_paths_neg = np.log(S) + np.cumsum(antithetic_increments, axis=1)

        # Extract terminal prices
        terminal_pos = np.exp(log_paths_pos[:, -1])
        terminal_neg = np.exp(log_paths_neg[:, -1])

        return np.concatenate([terminal_pos, terminal_neg])

    def _simulate_terminal_prices(
        self,
        S: float,
        T: float,
        r: float,
        sigma: float,
        q: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate terminal prices using the best available method.

        Dispatches to Numba-accelerated or vectorized implementation
        based on configuration.

        Args:
            S: Spot price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            q: Dividend yield.
            seed: Optional seed override.

        Returns:
            Array of terminal prices.
        """
        actual_seed = seed if seed is not None else self.seed

        if self.use_numba:
            if self.use_parallel:
                return _simulate_terminal_prices_numba_parallel(
                    S, T, r, sigma, q, self.num_simulations, self.num_steps, actual_seed
                )
            else:
                return _simulate_terminal_prices_numba_single(
                    S, T, r, sigma, q, self.num_simulations, self.num_steps, actual_seed
                )
        return self._simulate_terminal_prices_vectorized(S, T, r, sigma, q, actual_seed)

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
    ) -> float:
        """
        Compute European option price using Monte Carlo simulation.

        Args:
            S: Current spot price of the underlying asset.
            K: Strike price of the option.
            T: Time to maturity in years (e.g., 0.5 for 6 months).
            r: Annualized risk-free interest rate (e.g., 0.05 for 5%).
            sigma: Annualized volatility (e.g., 0.2 for 20%).
            option_type: Type of option - 'call' or 'put'.
            q: Continuous dividend yield (default 0.0).
            seed: Optional seed for this specific pricing call.

        Returns:
            The estimated option price as a float.

        Raises:
            InputValidationError: If any input parameter is invalid.

        Example:
            >>> pricer = MonteCarloPricer(num_simulations=100000)
            >>> call_price = pricer.price(100, 100, 1.0, 0.05, 0.2, 'call')
            >>> put_price = pricer.price(100, 100, 1.0, 0.05, 0.2, 'put')
        """
        self._validate_inputs(S, K, T, r, sigma, option_type, q)

        terminal_prices = self._simulate_terminal_prices(S, T, r, sigma, q, seed)

        if option_type == "call":
            payoffs = np.maximum(terminal_prices - K, 0.0)
        else:
            payoffs = np.maximum(K - terminal_prices, 0.0)

        return float(np.exp(-r * T) * np.mean(payoffs))

    def price_with_std_error(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
        seed: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Compute option price with standard error estimate.

        Returns both the price and its Monte Carlo standard error,
        useful for assessing convergence and confidence intervals.

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            option_type: 'call' or 'put'.
            q: Dividend yield.
            seed: Optional random seed.

        Returns:
            Tuple of (price, standard_error).

        Example:
            >>> pricer = MonteCarloPricer(num_simulations=100000)
            >>> price, std_err = pricer.price_with_std_error(100, 100, 1.0, 0.05, 0.2, 'call')
            >>> print(f"Price: {price:.4f} ± {1.96 * std_err:.4f} (95% CI)")
        """
        self._validate_inputs(S, K, T, r, sigma, option_type, q)

        terminal_prices = self._simulate_terminal_prices(S, T, r, sigma, q, seed)

        if option_type == "call":
            payoffs = np.maximum(terminal_prices - K, 0.0)
        else:
            payoffs = np.maximum(K - terminal_prices, 0.0)

        discounted = np.exp(-r * T) * payoffs
        price = float(np.mean(discounted))
        std_error = float(np.std(discounted) / np.sqrt(len(discounted)))

        return price, std_error

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
        seed: Optional[int] = None,
    ) -> float:
        """
        Compute Delta using central finite difference.

        Delta measures the option's sensitivity to changes in the
        underlying asset price. Uses common random numbers (CRN)
        for variance reduction.

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            option_type: 'call' or 'put'.
            q: Dividend yield.
            h: Finite difference step size (default 1e-4).
            seed: Random seed for CRN.

        Returns:
            Estimated Delta value.
        """
        actual_seed = seed if seed is not None else self.seed

        # Use same seed for variance reduction (CRN)
        price_up = self.price(S + h, K, T, r, sigma, option_type, q, actual_seed)
        price_down = self.price(S - h, K, T, r, sigma, option_type, q, actual_seed)

        return (price_up - price_down) / (2 * h)

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
        seed: Optional[int] = None,
    ) -> float:
        """
        Compute Gamma using central finite difference.

        Gamma measures the rate of change of Delta with respect to
        the underlying price. It indicates convexity of the option value.

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            option_type: 'call' or 'put'.
            q: Dividend yield.
            h: Finite difference step size.
            seed: Random seed for CRN.

        Returns:
            Estimated Gamma value.
        """
        actual_seed = seed if seed is not None else self.seed

        price_up = self.price(S + h, K, T, r, sigma, option_type, q, actual_seed)
        price_mid = self.price(S, K, T, r, sigma, option_type, q, actual_seed)
        price_down = self.price(S - h, K, T, r, sigma, option_type, q, actual_seed)

        return (price_up - 2 * price_mid + price_down) / (h * h)

    def delta_gamma(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
        h: float = 1e-4,
        seed: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Compute Delta and Gamma efficiently in a single call.

        More efficient than calling delta() and gamma() separately
        as it reuses price calculations.

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            option_type: 'call' or 'put'.
            q: Dividend yield.
            h: Finite difference step size.
            seed: Random seed for CRN.

        Returns:
            Tuple of (delta, gamma).
        """
        actual_seed = seed if seed is not None else self.seed

        price_up = self.price(S + h, K, T, r, sigma, option_type, q, actual_seed)
        price_mid = self.price(S, K, T, r, sigma, option_type, q, actual_seed)
        price_down = self.price(S - h, K, T, r, sigma, option_type, q, actual_seed)

        delta = (price_up - price_down) / (2 * h)
        gamma = (price_up - 2 * price_mid + price_down) / (h * h)

        return delta, gamma

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
        seed: Optional[int] = None,
    ) -> float:
        """
        Compute Vega using central finite difference.

        Vega measures the option's sensitivity to changes in volatility.
        Result is per 1% change in volatility (divide by 100 for per 1pp).

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            option_type: 'call' or 'put'.
            q: Dividend yield.
            h: Finite difference step size for sigma.
            seed: Random seed for CRN.

        Returns:
            Estimated Vega value.
        """
        actual_seed = seed if seed is not None else self.seed

        price_up = self.price(S, K, T, r, sigma + h, option_type, q, actual_seed)
        price_down = self.price(S, K, T, r, sigma - h, option_type, q, actual_seed)

        return (price_up - price_down) / (2 * h)

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
        seed: Optional[int] = None,
    ) -> float:
        """
        Compute Theta using forward finite difference.

        Theta measures the time decay of the option value, typically
        expressed as value change per day.

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            option_type: 'call' or 'put'.
            q: Dividend yield.
            dt: Time step (default 1 day = 1/365).
            seed: Random seed for CRN.

        Returns:
            Estimated Theta (negative for long positions).
        """
        actual_seed = seed if seed is not None else self.seed

        if T > dt:
            price_now = self.price(S, K, T, r, sigma, option_type, q, actual_seed)
            price_later = self.price(
                S, K, T - dt, r, sigma, option_type, q, actual_seed
            )
            return (price_later - price_now) / dt

        return -self.price(S, K, T, r, sigma, option_type, q, actual_seed) / dt

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
        seed: Optional[int] = None,
    ) -> float:
        """
        Compute Rho using forward finite difference.

        Rho measures the option's sensitivity to changes in the
        risk-free interest rate.

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            option_type: 'call' or 'put'.
            q: Dividend yield.
            h: Finite difference step size.
            seed: Random seed.

        Returns:
            Estimated Rho value.
        """
        actual_seed = seed if seed is not None else self.seed

        price_up = self.price(S, K, T, r + h, sigma, option_type, q, actual_seed)
        price_base = self.price(S, K, T, r, sigma, option_type, q, actual_seed)

        return (price_up - price_base) / h

    def all_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Compute all Greeks in a single call.

        Returns a dictionary with Delta, Gamma, Vega, Theta, and Rho.
        More efficient than calling each Greek method separately
        when all are needed.

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            option_type: 'call' or 'put'.
            q: Dividend yield.
            seed: Random seed for CRN.

        Returns:
            Dictionary with keys: 'delta', 'gamma', 'vega', 'theta', 'rho'.

        Example:
            >>> pricer = MonteCarloPricer(num_simulations=50000)
            >>> greeks = pricer.all_greeks(100, 100, 1.0, 0.05, 0.2, 'call')
            >>> print(f"Delta: {greeks['delta']:.4f}")
        """
        actual_seed = seed if seed is not None else self.seed

        delta, gamma = self.delta_gamma(
            S, K, T, r, sigma, option_type, q, seed=actual_seed
        )
        vega = self.vega(S, K, T, r, sigma, option_type, q, seed=actual_seed)
        theta = self.theta(S, K, T, r, sigma, option_type, q, seed=actual_seed)
        rho = self.rho(S, K, T, r, sigma, option_type, q, seed=actual_seed)

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }
