# src/pricing_models/binomial_tree.py

import numpy as np
from typing import Literal, Optional, Tuple, Callable
from numba import njit
from enum import IntEnum
import warnings

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

class NumericalWarning(UserWarning):
    """Warnings about numerical precision issues"""

def error_handler(func: Callable) -> Callable:
    """
    Decorator to wrap public methods with consistent error handling.
    
    Catches exceptions and wraps them in BinomialTreeError.
    """
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
    """
    Construct the binomial tree of underlying asset prices.

    Args:
        S: Spot price
        u: Up factor
        d: Down factor
        n_steps: Number of steps in the tree

    Returns:
        2D array of shape (n_steps+1, n_steps+1) with asset prices at each node
    """
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
    """
    Compute option values via backward induction on a binomial tree.

    Args:
        asset_prices: Asset price tree
        K: Strike price
        r: Risk-free rate
        dt: Time step size
        p: Risk-neutral probability
        option_type: OptionType.CALL or OptionType.PUT
        exercise_style: ExerciseStyle.EUROPEAN or ExerciseStyle.AMERICAN

    Returns:
        Option value tree of same shape as asset_prices
    """
    n = asset_prices.shape[0] - 1
    disc = np.exp(-r * dt)
    option_values = np.empty_like(asset_prices)
    
    # Terminal payoffs
    if option_type == OptionType.CALL:
        option_values[-1, :n+1] = np.maximum(asset_prices[-1, :n+1] - K, 0)
    else:
        option_values[-1, :n+1] = np.maximum(K - asset_prices[-1, :n+1], 0)
    
    # Backward induction
    for step in range(n-1, -1, -1):
        option_values[step, :step+1] = disc * (
            p * option_values[step+1, 1:step+2] + 
            (1 - p) * option_values[step+1, :step+1]
        )
        # American early exercise
        if exercise_style == ExerciseStyle.AMERICAN:
            if option_type == OptionType.CALL:
                intrinsic = np.maximum(asset_prices[step, :step+1] - K, 0)
            else:
                intrinsic = np.maximum(K - asset_prices[step, :step+1], 0)
            option_values[step, :step+1] = np.maximum(option_values[step, :step+1], intrinsic)
    
    return option_values

class BinomialTree:
    """
    CRR Binomial Tree option pricer.
    
    Features:
    - European & American options
    - Vectorized Numba acceleration
    - Numerical safeguards
    - Delta & Gamma calculation
    - Error handling
    """
    
    def __init__(self, num_steps: int = 500):
        """
        Initialize the pricer.

        Args:
            num_steps: Number of steps in the binomial tree
        """
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
        """
        Validate input parameters and convert to enums.

        Returns:
            Tuple of integers: (OptionType, ExerciseStyle)
        """
        for param, name in [(S,"S"),(K,"K"),(T,"T"),(r,"r"),(sigma,"sigma"),(q,"q")]:
            if not isinstance(param, (int,float,np.floating)):
                raise InputValidationError(f"{name} must be numeric, got {type(param)}")
        if option_type not in {"call","put"}:
            raise InputValidationError("option_type must be 'call' or 'put'")
        if exercise_style not in {"european","american"}:
            raise InputValidationError("exercise_style must be 'european' or 'american'")
        if S <= 0 or K <= 0:
            raise InputValidationError("Spot/strike must be positive")
        if T < 0 or sigma < 0 or q < 0:
            raise InputValidationError("T/sigma/q must be non-negative")
        return OptionType[option_type.upper()].value, ExerciseStyle[exercise_style.upper()].value

    def _compute_params(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float
    ) -> Tuple[float, float, float, float]:
        """
        Compute the tree parameters: u, d, p, dt.

        Returns:
            Tuple: (u, d, p, dt)
        """
        dt = T / self.num_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r-q)*dt) - d) / (u-d)
        p = min(max(p,0.0),1.0)  # Clamp probability
        return u,d,p,dt

    @error_handler
    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call","put"],
        exercise_style: Literal["european","american"]="european",
        q: float=0.0
    ) -> float:
        """
        Compute option price.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity in years
            r: Risk-free rate
            sigma: Volatility
            option_type: "call" or "put"
            exercise_style: "european" or "american"
            q: Dividend yield

        Returns:
            Option price
        """
        if T==0:
            return max(S-K,0.0) if option_type=="call" else max(K-S,0.0)
        if sigma==0:
            df = np.exp(-r*T)
            fwd = S*np.exp((r-q)*T)
            intrinsic = max(fwd-K,0.0) if option_type=="call" else max(K-fwd,0.0)
            return intrinsic*df
        ot, es = self._validate_inputs(S,K,T,r,sigma,option_type,exercise_style,q)
        u,d,p,dt = self._compute_params(S,K,T,r,sigma,q)
        asset_prices = _compute_asset_prices(S,u,d,self.num_steps)
        option_values = _backward_induction(asset_prices,K,r,dt,p,ot,es)
        return option_values[0,0]

    @error_handler
    def delta(self, S,K,T,r,sigma,option_type,exercise_style="european",q=0.0,h:Optional[float]=None) -> float:
        """
        Compute option delta using central finite difference.

        Args:
            h: Price bump for numerical derivative
        """
        h = h or max(S*1e-4,1e-6)
        price_up = self.price(S+h,K,T,r,sigma,option_type,exercise_style,q)
        price_down = self.price(S-h,K,T,r,sigma,option_type,exercise_style,q)
        return (price_up-price_down)/(2*h)

    @error_handler
    def gamma(self, S,K,T,r,sigma,option_type,exercise_style="european",q:float=0.0,h:Optional[float]=None) -> float:
        """
        Compute option gamma using central finite difference.

        Args:
            h: Price bump for numerical derivative
        """
        h = h or max(S*1e-4,1e-6)
        price_up = self.price(S+h,K,T,r,sigma,option_type,exercise_style,q)
        price_mid = self.price(S,K,T,r,sigma,option_type,exercise_style,q)
        price_down = self.price(S-h,K,T,r,sigma,option_type,exercise_style,q)
        return (price_up-2*price_mid+price_down)/(h*h)
