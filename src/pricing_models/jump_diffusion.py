# src/pricing_models/jump_diffusion.py
"""
Merton Jump-Diffusion Model.

Implements the Merton (1976) jump-diffusion model:

    dS/S = (μ - λκ) dt + σ dW + dJ

Where J is a compound Poisson process with:
    - λ: Jump intensity (expected jumps per year)
    - κ: Mean jump size (E[e^Y - 1] where Y ~ N(μ_J, σ_J²))
    - Y: Log-jump size, normally distributed

Parameters:
    λ (lambda_j): Jump intensity
    μ_J (mu_j): Mean of log-jump size
    σ_J (sigma_j): Volatility of log-jump size

Features:
    - Analytical pricing via series expansion (Merton formula)
    - Monte Carlo simulation with jumps
    - Captures fat tails and crash risk

Reference:
    Merton, R. C. (1976). Option Pricing When Underlying Stock Returns
    Are Discontinuous. Journal of Financial Economics, 3, 125-144.

Usage:
    >>> from src.pricing_models.jump_diffusion import MertonJumpDiffusion
    >>> jd = MertonJumpDiffusion(lambda_j=0.1, mu_j=-0.1, sigma_j=0.2)
    >>> price = jd.price(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.special import factorial
from scipy.stats import norm


@dataclass
class MertonJumpDiffusion:
    """
    Merton jump-diffusion model.

    Attributes:
        lambda_j: Jump intensity (expected number of jumps per year).
        mu_j: Mean of log-jump size.
        sigma_j: Standard deviation of log-jump size.
    """

    lambda_j: float  # Jump intensity
    mu_j: float  # Mean log-jump
    sigma_j: float  # Vol of log-jump

    def __post_init__(self):
        """Validate parameters."""
        if self.lambda_j < 0:
            raise ValueError("lambda_j must be non-negative")
        if self.sigma_j < 0:
            raise ValueError("sigma_j must be non-negative")

    @property
    def kappa(self) -> float:
        """Mean jump size: E[e^Y - 1]."""
        return np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call",
        q: float = 0.0,
        n_terms: int = 50,
    ) -> float:
        """
        Analytical price using Merton's series expansion.

        The price is a Poisson-weighted sum of Black-Scholes prices
        with adjusted parameters.

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Diffusion volatility.
            option_type: 'call' or 'put'.
            q: Dividend yield.
            n_terms: Number of terms in series expansion.

        Returns:
            Option price.
        """
        if T <= 0:
            intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
            return intrinsic

        lambda_j, mu_j, sigma_j = self.lambda_j, self.mu_j, self.sigma_j
        kappa = self.kappa

        # Adjusted drift to make S risk-neutral
        lambda_prime = lambda_j * (1 + kappa)

        price = 0.0
        for n in range(n_terms):
            # Poisson probability of n jumps
            poisson_weight = (
                np.exp(-lambda_prime * T) * (lambda_prime * T) ** n / factorial(n)
            )

            # Adjusted volatility for n jumps
            sigma_n = np.sqrt(sigma**2 + n * sigma_j**2 / T)

            # Adjusted drift for n jumps
            r_n = r - lambda_j * kappa + n * np.log(1 + kappa) / T

            # Black-Scholes price with adjusted params
            bs_price = self._black_scholes(S, K, T, r_n, sigma_n, option_type, q)

            price += poisson_weight * bs_price

            # Convergence check
            if poisson_weight < 1e-12:
                break

        return price

    @staticmethod
    def _black_scholes(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        q: float,
    ) -> float:
        """Standard Black-Scholes formula."""
        if sigma <= 0 or T <= 0:
            if option_type == "call":
                return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
            else:
                return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(
                -d1
            )

    def price_monte_carlo(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call",
        q: float = 0.0,
        n_paths: int = 100000,
        n_steps: int = 252,
        seed: Optional[int] = None,
    ) -> float:
        """
        Price via Monte Carlo with jump simulation.

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Diffusion volatility.
            option_type: 'call' or 'put'.
            q: Dividend yield.
            n_paths: Number of paths.
            n_steps: Number of time steps.
            seed: Random seed.

        Returns:
            Option price.
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        lambda_j, mu_j, sigma_j = self.lambda_j, self.mu_j, self.sigma_j
        kappa = self.kappa

        # Compensated drift
        drift = (r - q - lambda_j * kappa - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)

        log_S = np.full(n_paths, np.log(S))

        for _ in range(n_steps):
            # Diffusion component
            dW = np.random.standard_normal(n_paths)
            log_S += drift + vol * dW

            # Jump component (Poisson arrivals)
            n_jumps = np.random.poisson(lambda_j * dt, n_paths)

            # Apply jumps
            for i in range(n_paths):
                if n_jumps[i] > 0:
                    jump_sizes = np.random.normal(mu_j, sigma_j, n_jumps[i])
                    log_S[i] += np.sum(jump_sizes)

        S_T = np.exp(log_S)

        if option_type == "call":
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)

        return np.exp(-r * T) * np.mean(payoffs)

    def simulate_path(
        self,
        S: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        n_steps: int = 252,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate a single price path with jumps.

        Returns:
            Array of prices with shape (n_steps + 1,).
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        lambda_j, mu_j, sigma_j = self.lambda_j, self.mu_j, self.sigma_j
        kappa = self.kappa

        drift = (r - q - lambda_j * kappa - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)

        path = np.zeros(n_steps + 1)
        path[0] = S
        log_S = np.log(S)

        for t in range(1, n_steps + 1):
            # Diffusion
            dW = np.random.standard_normal()
            log_S += drift + vol * dW

            # Jumps
            n_jumps = np.random.poisson(lambda_j * dt)
            if n_jumps > 0:
                jump_sizes = np.random.normal(mu_j, sigma_j, n_jumps)
                log_S += np.sum(jump_sizes)

            path[t] = np.exp(log_S)

        return path


@dataclass
class KouJumpDiffusion:
    """
    Kou Double-Exponential Jump-Diffusion Model.

    Jump sizes follow a double-exponential (asymmetric) distribution,
    better capturing the asymmetry of equity returns.

    Attributes:
        lambda_j: Jump intensity.
        p: Probability of upward jump.
        eta1: Rate of upward exponential.
        eta2: Rate of downward exponential.
    """

    lambda_j: float  # Jump intensity
    p: float  # Prob of up jump
    eta1: float  # Up exponential rate
    eta2: float  # Down exponential rate

    def __post_init__(self):
        if not 0 <= self.p <= 1:
            raise ValueError("p must be in [0, 1]")
        if self.eta1 <= 1:
            raise ValueError("eta1 must be > 1 for finite mean")
        if self.eta2 <= 0:
            raise ValueError("eta2 must be positive")

    @property
    def kappa(self) -> float:
        """Mean jump size E[e^Y - 1]."""
        return (
            self.p * self.eta1 / (self.eta1 - 1)
            + (1 - self.p) * self.eta2 / (self.eta2 + 1)
            - 1
        )

    def simulate_jump(self, n: int = 1) -> np.ndarray:
        """Simulate n double-exponential jumps."""
        jumps = np.zeros(n)
        u = np.random.uniform(0, 1, n)

        # Up jumps (positive exponential)
        up_mask = u < self.p
        jumps[up_mask] = np.random.exponential(1 / self.eta1, np.sum(up_mask))

        # Down jumps (negative exponential)
        down_mask = ~up_mask
        jumps[down_mask] = -np.random.exponential(1 / self.eta2, np.sum(down_mask))

        return jumps

    def price_monte_carlo(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call",
        q: float = 0.0,
        n_paths: int = 100000,
        n_steps: int = 252,
        seed: Optional[int] = None,
    ) -> float:
        """Monte Carlo pricing with double-exponential jumps."""
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        kappa = self.kappa
        drift = (r - q - self.lambda_j * kappa - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)

        log_S = np.full(n_paths, np.log(S))

        for _ in range(n_steps):
            dW = np.random.standard_normal(n_paths)
            log_S += drift + vol * dW

            # Jump arrivals
            n_jumps = np.random.poisson(self.lambda_j * dt, n_paths)
            total_jumps = np.sum(n_jumps)

            if total_jumps > 0:
                all_jumps = self.simulate_jump(total_jumps)
                idx = 0
                for i in range(n_paths):
                    if n_jumps[i] > 0:
                        log_S[i] += np.sum(all_jumps[idx : idx + n_jumps[i]])
                        idx += n_jumps[i]

        S_T = np.exp(log_S)

        if option_type == "call":
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)

        return np.exp(-r * T) * np.mean(payoffs)
