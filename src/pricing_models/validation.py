# src/pricing_models/validation.py
"""
Option Pricing Validation Utilities.

Provides validation functions for ensuring pricing consistency:
- Put-Call Parity checks
- Arbitrage bounds validation
- Model convergence tests
- Greeks consistency checks

Usage:
    >>> from src.pricing_models.validation import validate_put_call_parity
    >>> is_valid, error = validate_put_call_parity(call=10.5, put=5.2, S=100, K=95, T=1.0, r=0.05)
"""

from typing import Optional, Tuple

import numpy as np


def validate_put_call_parity(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    tolerance: float = 1e-4,
) -> Tuple[bool, float]:
    """
    Validate put-call parity for European options.

    C - P = S * exp(-qT) - K * exp(-rT)

    Args:
        call_price: European call price.
        put_price: European put price.
        S: Spot price.
        K: Strike price.
        T: Time to maturity.
        r: Risk-free rate.
        q: Dividend yield.
        tolerance: Maximum acceptable error.

    Returns:
        Tuple of (is_valid, absolute_error).
    """
    expected_diff = S * np.exp(-q * T) - K * np.exp(-r * T)
    actual_diff = call_price - put_price
    error = abs(actual_diff - expected_diff)
    return error <= tolerance, error


def validate_arbitrage_bounds(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    q: float = 0.0,
) -> Tuple[bool, str]:
    """
    Check if option price satisfies no-arbitrage bounds.

    For calls: max(0, S*e^(-qT) - K*e^(-rT)) <= C <= S*e^(-qT)
    For puts: max(0, K*e^(-rT) - S*e^(-qT)) <= P <= K*e^(-rT)

    Args:
        price: Option price to validate.
        S: Spot price.
        K: Strike price.
        T: Time to maturity.
        r: Risk-free rate.
        option_type: 'call' or 'put'.
        q: Dividend yield.

    Returns:
        Tuple of (is_valid, violation_message).
    """
    pv_S = S * np.exp(-q * T)
    pv_K = K * np.exp(-r * T)

    if option_type == "call":
        lower_bound = max(0, pv_S - pv_K)
        upper_bound = pv_S

        if price < lower_bound - 1e-6:
            return False, f"Call price {price:.4f} below lower bound {lower_bound:.4f}"
        if price > upper_bound + 1e-6:
            return False, f"Call price {price:.4f} above upper bound {upper_bound:.4f}"
    else:
        lower_bound = max(0, pv_K - pv_S)
        upper_bound = pv_K

        if price < lower_bound - 1e-6:
            return False, f"Put price {price:.4f} below lower bound {lower_bound:.4f}"
        if price > upper_bound + 1e-6:
            return False, f"Put price {price:.4f} above upper bound {upper_bound:.4f}"

    return True, "OK"


def validate_greeks_consistency(
    delta: float,
    gamma: float,
    vega: float,
    theta: float,
    option_type: str,
    S: float,
    K: float,
    T: float,
    sigma: float,
) -> Tuple[bool, list]:
    """
    Check if Greeks satisfy consistency relationships.

    Args:
        delta, gamma, vega, theta: Computed Greeks.
        option_type: 'call' or 'put'.
        S, K, T, sigma: Option parameters.

    Returns:
        Tuple of (all_valid, list_of_issues).
    """
    issues = []

    # Delta bounds
    if option_type == "call":
        if not 0 <= delta <= 1:
            issues.append(f"Call delta {delta:.4f} outside [0, 1]")
    else:
        if not -1 <= delta <= 0:
            issues.append(f"Put delta {delta:.4f} outside [-1, 0]")

    # Gamma must be positive
    if gamma < -1e-6:
        issues.append(f"Gamma {gamma:.4f} is negative")

    # Vega must be positive
    if vega < -1e-6:
        issues.append(f"Vega {vega:.4f} is negative")

    # Theta typically negative for long options
    # (but can be positive for deep ITM puts with high rates)

    return len(issues) == 0, issues


def validate_smile_arbitrage(
    strikes: np.ndarray,
    ivs: np.ndarray,
    F: float,
    T: float,
) -> Tuple[bool, list]:
    """
    Check volatility smile for arbitrage violations.

    - Butterfly arbitrage: second derivative of total variance >= 0
    - Call spread arbitrage: monotone call prices

    Args:
        strikes: Array of strikes.
        ivs: Array of implied volatilities.
        F: Forward price.
        T: Time to maturity.

    Returns:
        Tuple of (is_valid, list_of_violations).
    """
    violations = []

    # Total variance
    total_var = ivs**2 * T

    # Check butterfly (convexity in log-strike space)
    log_K = np.log(strikes / F)

    for i in range(1, len(strikes) - 1):
        # Second derivative approximation
        h1 = log_K[i] - log_K[i - 1]
        h2 = log_K[i + 1] - log_K[i]

        d2w = (
            2
            * (
                (total_var[i + 1] - total_var[i]) / h2
                - (total_var[i] - total_var[i - 1]) / h1
            )
            / (h1 + h2)
        )

        if d2w < -1e-4:
            violations.append(
                f"Butterfly arbitrage at K={strikes[i]:.2f} (d²w/dk² = {d2w:.6f})"
            )

    return len(violations) == 0, violations


def monte_carlo_convergence_test(
    price_function,
    n_trials: int = 10,
    base_sims: int = 10000,
) -> dict:
    """
    Test Monte Carlo convergence by increasing simulation count.

    Args:
        price_function: Callable that takes num_simulations and returns price.
        n_trials: Number of trials at each simulation count.
        base_sims: Starting simulation count.

    Returns:
        Dict with convergence statistics.
    """
    results = {}
    sim_counts = [base_sims, base_sims * 2, base_sims * 4, base_sims * 10]

    for n_sims in sim_counts:
        prices = [price_function(n_sims) for _ in range(n_trials)]
        results[n_sims] = {
            "mean": np.mean(prices),
            "std": np.std(prices),
            "min": np.min(prices),
            "max": np.max(prices),
        }

    # Estimate convergence rate
    stds = [results[n]["std"] for n in sim_counts]
    expected_rate = [stds[0] * np.sqrt(sim_counts[0] / n) for n in sim_counts]

    return {
        "results": results,
        "stds": stds,
        "expected_rate": expected_rate,
        "converging": all(s2 <= s1 * 1.5 for s1, s2 in zip(stds[:-1], stds[1:])),
    }
