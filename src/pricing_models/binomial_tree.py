# src/pricing_models/binomial_tree.py

import warnings
from enum import IntEnum
from typing import Callable, Literal, Optional, Tuple

import numpy as np
from numba import njit


class OptionType(IntEnum):
    """Enumeration for option type"""
    CALL = 0
    PUT = 1


class ExerciseStyle(IntEnum):
    """Enumeration for exercise style"""
    EUROPEAN = 0
    AMERICAN = 1


class BinomialTreeError(Exception):
    """Base exception class for binomial tree operations"""


class InputValidationError(BinomialTreeError):
    """Raised for invalid input parameters"""


def error_handler(func: Callable) -> Callable:
    """
    Decorator to wrap public methods with consistent error handling.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BinomialTreeError:
            raise
        except Exception as e:
            raise BinomialTreeError(
                f"Unexpected error in {func.__name__}: {str(e)}"
            ) from e
    return wrapper


@njit(cache=True, fastmath=True)
def _solve_binomial_tree(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    n_steps: int,
    option_type: int,
    exercise_style: int,
) -> Tuple[float, float, float]:
    """
    Core Numba kernel for Binomial Tree.
    
    Uses O(N) memory by storing only the current layer of option values.
    Computes Price, Delta, and Gamma in a single backward pass.
    """
    # 1. Precompute parameters
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    df = np.exp(-r * dt)
    
    # Risk-neutral probability
    drift = np.exp((r - q) * dt)
    p = (drift - d) / (u - d)
    
    # Clamp probabilities to ensure stability
    if p < 0.0:
        p = 0.0
    elif p > 1.0:
        p = 1.0
    
    # 2. Initialize terminal payoffs (Step N)
    # We allocate ONE array of size N+1. This is O(N) memory.
    values = np.empty(n_steps + 1, dtype=np.float64)
    
    u_over_d = u / d
    s_d_N = S * (d ** n_steps)
    
    for j in range(n_steps + 1):
        spot_price = s_d_N * (u_over_d ** j)
        if option_type == 0:  # CALL
            values[j] = max(spot_price - K, 0.0)
        else:  # PUT
            values[j] = max(K - spot_price, 0.0)

    # Variables to store option values at Step 2 and Step 1 for Greeks
    val_2_2, val_2_1, val_2_0 = 0.0, 0.0, 0.0
    val_1_1, val_1_0 = 0.0, 0.0

    # 3. Backward Induction
    for i in range(n_steps - 1, -1, -1):
        current_s_d_i = S * (d ** i)
        
        for j in range(i + 1):
            # Continuation Value
            continuation = df * (p * values[j + 1] + (1 - p) * values[j])
            
            # Exercise Value (if American)
            if exercise_style == 1:  # AMERICAN
                spot_price = current_s_d_i * (u_over_d ** j)
                if option_type == 0: # CALL
                    intrinsic = max(spot_price - K, 0.0)
                else: # PUT
                    intrinsic = max(K - spot_price, 0.0)
                values[j] = max(continuation, intrinsic)
            else:
                values[j] = continuation
        
        # Capture values for Greeks
        if i == 2:
            val_2_0 = values[0] # dd
            val_2_1 = values[1] # ud
            val_2_2 = values[2] # uu
        elif i == 1:
            val_1_0 = values[0] # d
            val_1_1 = values[1] # u

    # 4. Extract Results
    price = values[0]
    
    # Delta
    s_u = S * u
    s_d = S * d
    delta = (val_1_1 - val_1_0) / (s_u - s_d)
    
    # Gamma
    s_uu = S * u * u
    s_ud = S # u*d = 1
    s_dd = S * d * d
    
    numerator_1 = (val_2_2 - val_2_1) / (s_uu - s_ud)
    numerator_2 = (val_2_1 - val_2_0) / (s_ud - s_dd)
    gamma = (numerator_1 - numerator_2) / (0.5 * (s_uu - s_dd))
    
    return price, delta, gamma


class BinomialTree:
    """
    Production-grade CRR Binomial Tree pricer.
    
    Optimizations:
    - O(N) memory complexity (reusing 1D array)
    - Single-pass calculation for Price, Delta, and Gamma
    - Numba JIT compilation
    - Analytical Greeks (no finite difference re-runs)
    """

    def __init__(self, num_steps: int = 200):
        if num_steps <= 0:
            raise InputValidationError("num_steps must be positive integer")
        self.num_steps = num_steps

    def _validate_inputs(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        exercise_style: Literal["european", "american"],
        q: float,
    ) -> Tuple[int, int]:
        if option_type not in {"call", "put"}:
            raise InputValidationError("option_type must be 'call' or 'put'")
        if exercise_style not in {"european", "american"}:
            raise InputValidationError(
                "exercise_style must be 'european' or 'american'"
            )
        if S <= 0 or K <= 0:
            raise InputValidationError("Spot/strike must be positive")
        if T < 0 or sigma < 0 or q < 0:
            raise InputValidationError("T/sigma/q must be non-negative")
            
        ot_enum = OptionType.CALL.value if option_type == "call" else OptionType.PUT.value
        es_enum = ExerciseStyle.EUROPEAN.value if exercise_style == "european" else ExerciseStyle.AMERICAN.value
        return ot_enum, es_enum

    @error_handler
    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        exercise_style: Literal["european", "american"] = "european",
        q: float = 0.0,
    ) -> float:
        """Computes option price."""
        if T <= 1e-6:
            return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

        ot, es = self._validate_inputs(S, K, T, r, sigma, option_type, exercise_style, q)
        price, _, _ = _solve_binomial_tree(
            S, K, T, r, sigma, q, self.num_steps, ot, es
        )
        return price

    @error_handler
    def delta(
        self,
        S: float, K: float, T: float, r: float, sigma: float,
        option_type: Literal["call", "put"],
        exercise_style: Literal["european", "american"] = "european",
        q: float = 0.0,
    ) -> float:
        """Computes Delta analytically from the tree."""
        if T <= 1e-6: return 0.0
        
        ot, es = self._validate_inputs(S, K, T, r, sigma, option_type, exercise_style, q)
        _, delta, _ = _solve_binomial_tree(S, K, T, r, sigma, q, self.num_steps, ot, es)
        return delta

    @error_handler
    def gamma(
        self,
        S: float, K: float, T: float, r: float, sigma: float,
        option_type: Literal["call", "put"],
        exercise_style: Literal["european", "american"] = "european",
        q: float = 0.0,
    ) -> float:
        """Computes Gamma analytically from the tree."""
        if T <= 1e-6: return 0.0
        
        ot, es = self._validate_inputs(S, K, T, r, sigma, option_type, exercise_style, q)
        _, _, gamma = _solve_binomial_tree(S, K, T, r, sigma, q, self.num_steps, ot, es)
        return gamma

    @error_handler
    def calculate_all(
        self,
        S: float, K: float, T: float, r: float, sigma: float,
        option_type: Literal["call", "put"],
        exercise_style: Literal["european", "american"] = "european",
        q: float = 0.0,
    ) -> dict:
        """
        Compute Price, Delta, and Gamma efficiently in a single run.
        """
        if T <= 1e-6:
            p = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
            return {"price": p, "delta": 0.0, "gamma": 0.0}

        ot, es = self._validate_inputs(S, K, T, r, sigma, option_type, exercise_style, q)
        price, delta, gamma = _solve_binomial_tree(S, K, T, r, sigma, q, self.num_steps, ot, es)
        return {"price": price, "delta": delta, "gamma": gamma}