# src/pricing_models/exotic_options.py
"""
Exotic Options Pricing Module.

Implements pricing for path-dependent and early-exercise options:
- Asian Options (arithmetic/geometric average)
- Barrier Options (knock-in/knock-out, up/down)
- American Options (Longstaff-Schwartz least squares Monte Carlo)

These are the use cases where ML surrogates actually outperform
vectorized Monte Carlo, since path-dependent options cannot be
fully vectorized across different option configurations.

Usage:
    from src.pricing_models.exotic_options import AsianOption, BarrierOption, AmericanOption

    asian = AsianOption(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    price = asian.price(n_paths=100000, avg_type="arithmetic")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


@dataclass
class ExoticOptionBase(ABC):
    """Base class for exotic options."""

    S: float  # Spot price
    K: float  # Strike price
    T: float  # Time to maturity
    r: float  # Risk-free rate
    sigma: float  # Volatility
    q: float = 0.0  # Dividend yield
    seed: Optional[int] = None

    def _generate_paths(
        self,
        n_paths: int,
        n_steps: int,
    ) -> np.ndarray:
        """
        Generate GBM price paths.

        Returns:
            Array of shape (n_paths, n_steps + 1) with price paths.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        dt = self.T / n_steps
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        # Generate random increments
        Z = np.random.standard_normal((n_paths, n_steps))

        # Build paths
        log_returns = drift + diffusion * Z
        log_S = np.zeros((n_paths, n_steps + 1))
        log_S[:, 0] = np.log(self.S)
        log_S[:, 1:] = np.log(self.S) + np.cumsum(log_returns, axis=1)

        return np.exp(log_S)

    @abstractmethod
    def price(self, n_paths: int, n_steps: int, **kwargs) -> float:
        """Compute option price via Monte Carlo."""
        pass

    # Greeks are computed via src.greeks.unified_greeks.compute_greeks_unified()
    # Exotic options need a custom adapter or wrapper.
    # Example usage:
    #   from src.greeks import compute_greeks_unified
    #   # Create a wrapper that conforms to PricerProtocol
    #   class ExoticAdapter:
    #       def __init__(self, exotic): self.exotic = exotic
    #       def price(self, S, K, T, r, sigma, option_type, q=0.0, **kw):
    #           self.exotic.S, self.exotic.K, self.exotic.T = S, K, T
    #           self.exotic.r, self.exotic.sigma = r, sigma
    #           return self.exotic.price(**kw)
    #   greeks = compute_greeks_unified(ExoticAdapter(asian_opt), ...)


@dataclass
class AsianOption(ExoticOptionBase):
    """
    Asian Option pricing via Monte Carlo.

    Asian options pay based on the average price over the life of the option,
    reducing sensitivity to manipulation at maturity.
    """

    def price(
        self,
        n_paths: int = 100000,
        n_steps: int = 252,
        avg_type: Literal["arithmetic", "geometric"] = "arithmetic",
        option_type: Literal["call", "put"] = "call",
    ) -> float:
        """
        Price Asian option.

        Args:
            n_paths: Number of simulation paths.
            n_steps: Number of time steps (252 = daily for 1 year).
            avg_type: "arithmetic" or "geometric" average.
            option_type: "call" or "put".

        Returns:
            Option price.
        """
        paths = self._generate_paths(n_paths, n_steps)

        # Compute average (excluding initial price for standard Asian)
        if avg_type == "arithmetic":
            avg_price = np.mean(paths[:, 1:], axis=1)
        else:  # geometric
            avg_price = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))

        # Compute payoffs
        if option_type == "call":
            payoffs = np.maximum(avg_price - self.K, 0)
        else:
            payoffs = np.maximum(self.K - avg_price, 0)

        # Discount to present value
        return np.exp(-self.r * self.T) * np.mean(payoffs)

    def price_geometric_closed_form(
        self,
        option_type: Literal["call", "put"] = "call",
    ) -> float:
        """
        Closed-form price for geometric Asian option.

        Uses the fact that geometric average of lognormal is lognormal.
        """
        from scipy.stats import norm

        # Adjusted parameters for geometric average
        sigma_adj = self.sigma / np.sqrt(3)
        r_adj = 0.5 * (self.r - self.q - self.sigma**2 / 6)

        d1 = (np.log(self.S / self.K) + (r_adj + 0.5 * sigma_adj**2) * self.T) / (
            sigma_adj * np.sqrt(self.T)
        )
        d2 = d1 - sigma_adj * np.sqrt(self.T)

        if option_type == "call":
            return self.S * np.exp((r_adj - self.r) * self.T) * norm.cdf(
                d1
            ) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(
                (r_adj - self.r) * self.T
            ) * norm.cdf(-d1)


@dataclass
class BarrierOption(ExoticOptionBase):
    """
    Barrier Option pricing via Monte Carlo.

    Barrier options become active (knock-in) or inactive (knock-out)
    when the underlying crosses a barrier level.
    """

    barrier: float = 0.0  # Barrier level

    def price(
        self,
        n_paths: int = 100000,
        n_steps: int = 252,
        barrier_type: Literal[
            "up-and-out", "up-and-in", "down-and-out", "down-and-in"
        ] = "up-and-out",
        option_type: Literal["call", "put"] = "call",
    ) -> float:
        """
        Price barrier option.

        Args:
            n_paths: Number of simulation paths.
            n_steps: Number of time steps.
            barrier_type: Type of barrier.
            option_type: "call" or "put".

        Returns:
            Option price.
        """
        if self.barrier <= 0:
            raise ValueError("Barrier must be positive")

        paths = self._generate_paths(n_paths, n_steps)

        # Check barrier crossing
        if barrier_type.startswith("up"):
            crossed = np.any(paths >= self.barrier, axis=1)
        else:  # down
            crossed = np.any(paths <= self.barrier, axis=1)

        # Determine which paths pay off
        if barrier_type.endswith("out"):
            # Knock-out: pays only if NOT crossed
            active = ~crossed
        else:  # knock-in
            # Knock-in: pays only if crossed
            active = crossed

        # Terminal payoffs
        S_T = paths[:, -1]
        if option_type == "call":
            payoffs = np.maximum(S_T - self.K, 0)
        else:
            payoffs = np.maximum(self.K - S_T, 0)

        # Zero out inactive paths
        payoffs = payoffs * active

        return np.exp(-self.r * self.T) * np.mean(payoffs)


@dataclass
class AmericanOption(ExoticOptionBase):
    """
    American Option pricing via Longstaff-Schwartz LSM.

    American options can be exercised at any time before maturity.
    Uses least-squares Monte Carlo for optimal exercise boundary.
    """

    def price(
        self,
        n_paths: int = 50000,
        n_steps: int = 50,
        option_type: Literal["call", "put"] = "put",
        poly_degree: int = 3,
    ) -> float:
        """
        Price American option using Longstaff-Schwartz algorithm.

        Args:
            n_paths: Number of simulation paths.
            n_steps: Number of exercise opportunities.
            option_type: "call" or "put".
            poly_degree: Degree of polynomial for continuation value.

        Returns:
            Option price.

        Note:
            American calls on non-dividend paying stocks should equal
            European calls (never optimal to exercise early).
        """
        paths = self._generate_paths(n_paths, n_steps)
        dt = self.T / n_steps
        discount = np.exp(-self.r * dt)

        # Compute intrinsic values at each step
        if option_type == "call":
            intrinsic = np.maximum(paths - self.K, 0)
        else:
            intrinsic = np.maximum(self.K - paths, 0)

        # Initialize cash flows with terminal payoff
        cash_flows = intrinsic[:, -1].copy()
        exercise_time = np.full(n_paths, n_steps)

        # Backward induction
        for t in range(n_steps - 1, 0, -1):
            # Discount cash flows
            cash_flows = cash_flows * discount

            # Find paths that are in-the-money
            itm = intrinsic[:, t] > 0

            if np.sum(itm) > poly_degree + 1:
                # Fit regression for continuation value
                X = paths[itm, t]
                Y = cash_flows[itm]

                # Polynomial basis
                X_poly = np.column_stack([X**i for i in range(poly_degree + 1)])

                # Least squares regression
                try:
                    coeffs = np.linalg.lstsq(X_poly, Y, rcond=None)[0]
                    continuation = X_poly @ coeffs
                except np.linalg.LinAlgError:
                    continuation = Y  # Fallback if regression fails

                # Exercise if intrinsic > continuation
                exercise_now = intrinsic[itm, t] > continuation

                # Update cash flows and exercise times
                exercise_indices = np.where(itm)[0][exercise_now]
                cash_flows[exercise_indices] = intrinsic[exercise_indices, t]
                exercise_time[exercise_indices] = t

        # Final discount to time 0
        cash_flows = cash_flows * discount

        return np.mean(cash_flows)

    def early_exercise_boundary(
        self,
        n_paths: int = 10000,
        n_steps: int = 50,
        option_type: Literal["call", "put"] = "put",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate early exercise boundary.

        Returns:
            Tuple of (times, boundary_prices) where exercising is optimal.
        """
        paths = self._generate_paths(n_paths, n_steps)
        times = np.linspace(0, self.T, n_steps + 1)

        # Simplified boundary estimation
        if option_type == "call":
            intrinsic = np.maximum(paths - self.K, 0)
        else:
            intrinsic = np.maximum(self.K - paths, 0)

        # Find approximate boundary at each time step
        boundary = np.zeros(n_steps + 1)

        for t in range(n_steps + 1):
            itm = intrinsic[:, t] > 0
            if np.sum(itm) > 0:
                if option_type == "put":
                    # For put, boundary is where exercise becomes optimal
                    boundary[t] = np.percentile(paths[itm, t], 10)
                else:
                    boundary[t] = np.percentile(paths[itm, t], 90)
            else:
                boundary[t] = np.nan

        return times, boundary


@dataclass
class LookbackOption(ExoticOptionBase):
    """
    Lookback Option pricing via Monte Carlo.

    Lookback options have payoffs depending on the extremum (max or min)
    of the underlying price over the option's life.

    Types:
        - Floating strike: K is set to the min (call) or max (put) observed
        - Fixed strike: Payoff based on max (call) or min (put) observed
    """

    def price(
        self,
        n_paths: int = 100000,
        n_steps: int = 252,
        lookback_type: Literal["floating", "fixed"] = "floating",
        option_type: Literal["call", "put"] = "call",
    ) -> float:
        """
        Price lookback option.

        Args:
            n_paths: Number of simulation paths.
            n_steps: Number of time steps.
            lookback_type: 'floating' or 'fixed' strike.
            option_type: 'call' or 'put'.

        Returns:
            Option price.
        """
        paths = self._generate_paths(n_paths, n_steps)
        S_T = paths[:, -1]
        S_max = np.max(paths, axis=1)
        S_min = np.min(paths, axis=1)

        if lookback_type == "floating":
            # Floating strike: strike is path-dependent
            if option_type == "call":
                # Call: pay S_T - S_min (buy at minimum)
                payoffs = S_T - S_min
            else:
                # Put: pay S_max - S_T (sell at maximum)
                payoffs = S_max - S_T
        else:
            # Fixed strike
            if option_type == "call":
                # Call: pay max(S_max - K, 0)
                payoffs = np.maximum(S_max - self.K, 0)
            else:
                # Put: pay max(K - S_min, 0)
                payoffs = np.maximum(self.K - S_min, 0)

        return np.exp(-self.r * self.T) * np.mean(payoffs)


@dataclass
class AutocallableOption(ExoticOptionBase):
    """
    Autocallable (Snowball) structured product pricing.

    Autocallables pay a coupon if the underlying is above a barrier
    on observation dates, and knock out (redeem early) if it exceeds
    an autocall barrier.

    Popular in Asian and European structured products markets.
    """

    autocall_barrier: float = 1.0  # Autocall level (relative to spot)
    coupon_barrier: float = 0.8  # Coupon payment barrier
    coupon_rate: float = 0.10  # Annual coupon rate
    ki_barrier: float = 0.6  # Knock-in (put) barrier

    def price(
        self,
        n_paths: int = 100000,
        n_steps: int = 252,
        observation_freq: int = 21,  # ~monthly
        **kwargs,
    ) -> float:
        """
        Price autocallable product.

        Args:
            n_paths: Number of simulation paths.
            n_steps: Number of time steps.
            observation_freq: Steps between observations.

        Returns:
            Product price as percentage of notional.
        """
        paths = self._generate_paths(n_paths, n_steps)
        dt = self.T / n_steps

        # Observation times (indices)
        obs_times = list(range(observation_freq, n_steps + 1, observation_freq))
        n_obs = len(obs_times)

        # Initialize payoffs
        payoffs = np.zeros(n_paths)
        redeemed = np.zeros(n_paths, dtype=bool)

        # Track knock-in event
        min_path = np.min(paths / self.S, axis=1)
        knocked_in = min_path <= self.ki_barrier

        for i, t in enumerate(obs_times):
            active = ~redeemed
            S_t = paths[active, t]
            S_rel = S_t / self.S

            # Check autocall
            autocall_mask = S_rel >= self.autocall_barrier
            autocall_indices = np.where(active)[0][autocall_mask]

            # Coupon + principal for autocalled paths
            time_fraction = (i + 1) / n_obs
            coupon = self.coupon_rate * time_fraction * self.T
            payoffs[autocall_indices] = (1 + coupon) * np.exp(-self.r * t * dt)
            redeemed[autocall_indices] = True

        # Final observation for non-redeemed paths
        still_active = ~redeemed
        S_final = paths[still_active, -1]
        S_rel_final = S_final / self.S

        # If knocked in, act like a put
        ki_active = knocked_in[still_active]

        # Payoffs at maturity
        final_payoffs = np.ones(np.sum(still_active))

        # Add coupon if above coupon barrier
        above_coupon = S_rel_final >= self.coupon_barrier
        final_payoffs[above_coupon] += self.coupon_rate * self.T

        # Apply knock-in loss if breached and below strike
        below_strike = S_rel_final < 1.0
        loss_mask = ki_active & below_strike
        final_payoffs[loss_mask] = S_rel_final[loss_mask]

        payoffs[still_active] = final_payoffs * np.exp(-self.r * self.T)

        return np.mean(payoffs)


@dataclass
class CliquetOption(ExoticOptionBase):
    """
    Cliquet (Ratchet) Option pricing.

    Cliquet options pay based on the sum of capped/floored local returns
    over a series of reset periods.

    Popular in equity-linked structured products.
    """

    local_cap: float = 0.05  # Cap on each period's return
    local_floor: float = -0.05  # Floor on each period's return
    global_cap: float = 0.30  # Cap on total return
    global_floor: float = 0.0  # Floor on total return

    def price(
        self,
        n_paths: int = 100000,
        n_steps: int = 252,
        n_periods: int = 12,  # Number of reset periods
        **kwargs,
    ) -> float:
        """
        Price cliquet option.

        Args:
            n_paths: Number of simulation paths.
            n_steps: Number of time steps.
            n_periods: Number of cliquet periods.

        Returns:
            Option price.
        """
        paths = self._generate_paths(n_paths, n_steps)

        # Period boundaries
        steps_per_period = n_steps // n_periods
        total_return = np.zeros(n_paths)

        for p in range(n_periods):
            start_idx = p * steps_per_period
            end_idx = (p + 1) * steps_per_period

            S_start = paths[:, start_idx]
            S_end = paths[:, end_idx]

            # Local return
            local_return = (S_end - S_start) / S_start

            # Apply local cap/floor
            capped_return = np.clip(local_return, self.local_floor, self.local_cap)
            total_return += capped_return

        # Apply global cap/floor
        total_return = np.clip(total_return, self.global_floor, self.global_cap)

        # Payoff is notional * total_return (minimum 0)
        payoffs = np.maximum(total_return, 0) * self.S

        return np.exp(-self.r * self.T) * np.mean(payoffs)


# Convenience functions
def price_asian(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    avg_type: str = "arithmetic",
    option_type: str = "call",
    n_paths: int = 100000,
    seed: int = None,
) -> float:
    """Quick Asian option pricing."""
    return AsianOption(S=S, K=K, T=T, r=r, sigma=sigma, seed=seed).price(
        n_paths=n_paths, avg_type=avg_type, option_type=option_type
    )


def price_barrier(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    barrier: float,
    barrier_type: str = "up-and-out",
    option_type: str = "call",
    n_paths: int = 100000,
    seed: int = None,
) -> float:
    """Quick barrier option pricing."""
    return BarrierOption(
        S=S, K=K, T=T, r=r, sigma=sigma, barrier=barrier, seed=seed
    ).price(n_paths=n_paths, barrier_type=barrier_type, option_type=option_type)


def price_american(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "put",
    n_paths: int = 50000,
    seed: int = None,
) -> float:
    """Quick American option pricing."""
    return AmericanOption(S=S, K=K, T=T, r=r, sigma=sigma, seed=seed).price(
        n_paths=n_paths, option_type=option_type
    )
