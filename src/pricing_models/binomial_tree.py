import numpy as np
from typing import Literal, Optional, Tuple, Callable
from numba import njit, types
from enum import IntEnum
import warnings

class OptionType(IntEnum):
    CALL = 0
    PUT = 1

class ExerciseStyle(IntEnum):
    EUROPEAN = 0
    AMERICAN = 1

class BinomialTreeError(Exception):
    """Base exception class for binomial tree operations"""

class InputValidationError(BinomialTreeError):
    """Raised for invalid input parameters"""

class NumericalWarning(UserWarning):
    """Warnings about numerical precision issues"""

def error_handler(func: Callable) -> Callable:
    """Decorator to wrap all public methods with consistent error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BinomialTreeError:
            raise
        except Exception as e:
            raise BinomialTreeError(f"Unexpected error in {func.__name__}: {str(e)}") from e
    return wrapper

@njit(cache=True, fastmath=True)
def _compute_asset_prices(S: float, u: float, d: float, n_steps: int) -> np.ndarray:
    """Vectorized CRR asset price tree construction"""
    asset_prices = np.empty((n_steps + 1, n_steps + 1), dtype=np.float64)
    for i in range(n_steps + 1):
        j = np.arange(i + 1)
        asset_prices[i, :i+1] = S * (u ** j) * (d ** (i - j))
    return asset_prices

@njit(cache=True, fastmath=True)
def _backward_induction(
    asset_prices: np.ndarray,
    K: float,
    r: float,
    dt: float,
    p: float,
    option_type: int,
    exercise_style: int
) -> np.ndarray:
    n = asset_prices.shape[0] - 1
    disc = np.exp(-r * dt)
    option_values = np.empty_like(asset_prices)
    
    # Terminal payoffs
    if option_type == OptionType.CALL:
        option_values[-1] = np.maximum(asset_prices[-1] - K, 0)
    else:
        option_values[-1] = np.maximum(K - asset_prices[-1], 0)
    
    for step in range(n-1, -1, -1):
        option_values[step] = disc * (
            p * option_values[step+1, 1:step+2] + 
            (1 - p) * option_values[step+1, :step+1]
        )
        
        if exercise_style == ExerciseStyle.AMERICAN:
            if option_type == OptionType.CALL:
                intrinsic = np.maximum(asset_prices[step, :step+1] - K, 0)
            else:
                intrinsic = np.maximum(K - asset_prices[step, :step+1], 0)
            option_values[step] = np.maximum(option_values[step], intrinsic)
    
    return option_values

class BinomialTree:
    """
    Production-grade CRR binomial tree pricer with:
    - Comprehensive error handling
    - Type enforcement
    - Numerical stability safeguards
    - Full Numba acceleration
    - Explicit documentation
    """
    
    def __init__(self, num_steps: int = 500):
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
        q: float
    ) -> Tuple[int, int]:
        # Type enforcement
        for param, name in [
            (S, "S"), (K, "K"), (T, "T"),
            (r, "r"), (sigma, "sigma"), (q, "q")
        ]:
            if not isinstance(param, (int, float, np.floating)):
                raise InputValidationError(f"{name} must be numeric, got {type(param)}")
        
        if option_type not in {"call", "put"}:
            raise InputValidationError("option_type must be 'call' or 'put'")
        if exercise_style not in {"european", "american"}:
            raise InputValidationError("exercise_style must be 'european' or 'american'")
        if S <= 0 or K <= 0:
            raise InputValidationError("Spot/strike must be positive")
        if T < 0 or sigma < 0 or q < 0:
            raise InputValidationError("T/sigma/q must be non-negative")
            
        return (
            OptionType[option_type.upper()].value,
            ExerciseStyle[exercise_style.upper()].value
        )

    def _compute_params(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float
    ) -> Tuple[float, float, float, float]:
        """Centralized parameter calculation with numerical safeguards"""
        dt = T / self.num_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p_num = np.exp((r - q) * dt) - d
        p_den = u - d
        
        if abs(p_den) < 1e-10:
            raise InputValidationError("u and d are equal (dt too small?)")
            
        p = p_num / p_den
        
        # Handle floating point errors in probability
        if p < -1e-8 or p > 1 + 1e-8:
            raise InputValidationError(f"Invalid probability: {p:.6f}")
        
        if p < 0.0:
            warnings.warn(f"Clamped p={p:.6f} to 0.0", NumericalWarning)
            p = 0.0
        elif p > 1.0:
            warnings.warn(f"Clamped p={p:.6f} to 1.0", NumericalWarning)
            p = 1.0
        
        return u, d, p, dt

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
        q: float = 0.0
    ) -> float:
        if T == 0:
            return max(S-K, 0.0) if option_type == "call" else max(K-S, 0.0)
            
        if sigma == 0:
            df = np.exp(-r * T)
            fwd = S * np.exp((r - q) * T)
            intrinsic = max(fwd - K, 0.0) if option_type == "call" else max(K - fwd, 0.0)
            return intrinsic * df
        
        ot, es = self._validate_inputs(
            S, K, T, r, sigma, option_type, exercise_style, q
        )
        u, d, p, dt = self._compute_params(S, K, T, r, sigma, q)
        
        asset_prices = _compute_asset_prices(S, u, d, self.num_steps)
        option_values = _backward_induction(
            asset_prices, K, r, dt, p, ot, es
        )
        
        return option_values[0, 0]

    @error_handler
    def delta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        exercise_style: Literal["european", "american"] = "european",
        q: float = 0.0,
        h: Optional[float] = None
    ) -> float:
        ot, es = self._validate_inputs(
            S, K, T, r, sigma, option_type, exercise_style, q
        )
        h = h or max(S * 1e-4, 1e-6)
        
        if es == ExerciseStyle.AMERICAN:
            u, d, p, dt = self._compute_params(S, K, T, r, sigma, q)
            asset_prices = _compute_asset_prices(S, u, d, self.num_steps)
            option_values = _backward_induction(
                asset_prices, K, r, dt, p, ot, es
            )
            return (option_values[1,1] - option_values[1,0]) / (S*(u - d))
        
        price_up = self.price(S + h, K, T, r, sigma, option_type, exercise_style, q)
        price_down = self.price(S - h, K, T, r, sigma, option_type, exercise_style, q)
        return (price_up - price_down) / (2*h)

    @error_handler
    def gamma(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        exercise_style: Literal["european", "american"] = "european",
        q: float = 0.0,
        h: Optional[float] = None
    ) -> float:
        ot, es = self._validate_inputs(
            S, K, T, r, sigma, option_type, exercise_style, q
        )
        h = h or max(S * 1e-4, 1e-6)
        
        if es == ExerciseStyle.AMERICAN:
            if self.num_steps < 2:
                raise InputValidationError("num_steps must be >=2 for American gamma")
            u, d, p, dt = self._compute_params(S, K, T, r, sigma, q)
            asset_prices = _compute_asset_prices(S, u, d, self.num_steps)
            option_values = _backward_induction(
                asset_prices, K, r, dt, p, ot, es
            )
            delta_up = (option_values[2,2] - option_values[2,1]) / (S*u*(u - d))
            delta_down = (option_values[2,1] - option_values[2,0]) / (S*d*(u - d))
            return (delta_up - delta_down) / (0.5 * S * (u - d))
        
        price_up = self.price(S + h, K, T, r, sigma, option_type, exercise_style, q)
        price_mid = self.price(S, K, T, r, sigma, option_type, exercise_style, q)
        price_down = self.price(S - h, K, T, r, sigma, option_type, exercise_style, q)
        return (price_up - 2*price_mid + price_down) / (h*h)