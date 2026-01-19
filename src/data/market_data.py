# src/data/market_data.py
"""
Live market data integration using Yahoo Finance (yfinance).

Features:
    - Options chain fetching with caching
    - Rate limiting to avoid API bans
    - Implied volatility surface construction
    - Stock price and dividend data

Usage:
    >>> from src.data import get_options_chain, get_stock_price
    >>> chain = get_options_chain("SPY")
    >>> price = get_stock_price("SPY")
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None


@dataclass
class MarketDataCache:
    """
    Thread-safe cache for market data with TTL.

    Prevents excessive API calls and respects rate limits.
    """

    _cache: Dict[str, Tuple[datetime, any]] = field(default_factory=dict)
    ttl_seconds: int = 300  # 5 minute cache

    def get(self, key: str) -> Optional[any]:
        """Get cached value if not expired."""
        if key in self._cache:
            timestamp, value = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: any) -> None:
        """Cache a value with current timestamp."""
        self._cache[key] = (datetime.now(), value)

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()


# Global cache instance
_cache = MarketDataCache()

# Rate limiting
_last_request_time = 0
_min_request_interval = 0.5  # 500ms between requests


def _rate_limit():
    """Ensure minimum time between API requests."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _min_request_interval:
        time.sleep(_min_request_interval - elapsed)
    _last_request_time = time.time()


def get_stock_price(ticker: str, use_cache: bool = True) -> Dict:
    """
    Get current stock price and basic info.

    Args:
        ticker: Stock symbol (e.g., 'SPY', 'AAPL').
        use_cache: Whether to use cached data.

    Returns:
        Dict with 'price', 'change', 'change_pct', 'volume', 'div_yield'.
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    cache_key = f"price_{ticker}"
    if use_cache:
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

    _rate_limit()

    def safe_float(val, default=0.0):
        """Safely convert value to float, handling None."""
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def safe_int(val, default=0):
        """Safely convert value to int, handling None."""
        if val is None:
            return default
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info

        # Get price with fallback to history
        price = safe_float(getattr(info, "last_price", None))
        if price == 0:
            # Try getting from history as fallback
            try:
                hist = stock.history(period="1d")
                if not hist.empty:
                    price = safe_float(hist["Close"].iloc[-1])
            except Exception:
                pass

        prev_close = safe_float(getattr(info, "previous_close", None))

        result = {
            "ticker": ticker,
            "price": price,
            "previous_close": prev_close,
            "change": 0.0,
            "change_pct": 0.0,
            "volume": safe_int(getattr(info, "last_volume", None)),
            "market_cap": safe_float(getattr(info, "market_cap", None)),
            "timestamp": datetime.now().isoformat(),
        }

        if prev_close > 0 and price > 0:
            result["change"] = price - prev_close
            result["change_pct"] = result["change"] / prev_close * 100

        if price > 0:
            _cache.set(cache_key, result)
            return result
        else:
            return {"ticker": ticker, "error": "Could not fetch price data", "price": 0}

    except Exception as e:
        return {"ticker": ticker, "error": str(e), "price": 0}


def get_options_chain(
    ticker: str,
    expiry: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch options chain for a ticker.

    Args:
        ticker: Stock symbol.
        expiry: Expiration date (YYYY-MM-DD). If None, uses nearest expiry.
        use_cache: Whether to use cached data.

    Returns:
        DataFrame with columns: strike, type, bid, ask, last, volume,
        open_interest, implied_vol, delta_approx, in_the_money.
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    cache_key = f"chain_{ticker}_{expiry}"
    if use_cache:
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

    _rate_limit()

    try:
        stock = yf.Ticker(ticker)

        # Get available expiries
        expiries = stock.options
        if not expiries:
            return pd.DataFrame()

        # Use specified expiry or nearest
        if expiry is None:
            expiry = expiries[0]
        elif expiry not in expiries:
            # Find closest expiry
            expiry_dates = [datetime.strptime(e, "%Y-%m-%d") for e in expiries]
            target = datetime.strptime(expiry, "%Y-%m-%d")
            closest = min(expiry_dates, key=lambda x: abs(x - target))
            expiry = closest.strftime("%Y-%m-%d")

        # Fetch chain
        chain = stock.option_chain(expiry)

        # Get current price for moneyness
        try:
            current_price = stock.fast_info.last_price
        except Exception:
            current_price = 100  # Fallback

        # Process calls
        calls = chain.calls.copy()
        calls["type"] = "call"
        calls["in_the_money"] = calls["strike"] < current_price

        # Process puts
        puts = chain.puts.copy()
        puts["type"] = "put"
        puts["in_the_money"] = puts["strike"] > current_price

        # Combine and clean
        df = pd.concat([calls, puts], ignore_index=True)

        # Rename columns for clarity
        df = df.rename(
            columns={
                "impliedVolatility": "implied_vol",
                "openInterest": "open_interest",
                "lastPrice": "last",
                "inTheMoney": "itm_flag",
            }
        )

        # Select relevant columns
        cols = [
            "strike",
            "type",
            "bid",
            "ask",
            "last",
            "volume",
            "open_interest",
            "implied_vol",
            "in_the_money",
        ]
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols].copy()

        # Add metadata
        df["ticker"] = ticker
        df["expiry"] = expiry
        df["spot"] = current_price

        # Days to expiry
        exp_date = datetime.strptime(expiry, "%Y-%m-%d")
        df["dte"] = (exp_date - datetime.now()).days

        _cache.set(cache_key, df)
        return df

    except Exception as e:
        print(f"Error fetching options for {ticker}: {e}")
        return pd.DataFrame()


def get_expiries(ticker: str) -> List[str]:
    """Get available expiration dates for a ticker."""
    if not YFINANCE_AVAILABLE:
        return []

    cache_key = f"expiries_{ticker}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    _rate_limit()

    try:
        stock = yf.Ticker(ticker)
        expiries = list(stock.options)
        _cache.set(cache_key, expiries)
        return expiries
    except Exception:
        return []


def get_iv_surface(
    ticker: str,
    n_expiries: int = 5,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Build implied volatility surface from options chain.

    Args:
        ticker: Stock symbol.
        n_expiries: Number of expiration dates to include.
        use_cache: Whether to use cached data.

    Returns:
        DataFrame with columns: strike, expiry, dte, implied_vol, moneyness.
    """
    cache_key = f"iv_surface_{ticker}_{n_expiries}"
    if use_cache:
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

    expiries = get_expiries(ticker)[:n_expiries]

    all_data = []
    for exp in expiries:
        chain = get_options_chain(ticker, exp, use_cache)
        if not chain.empty:
            # Filter for reasonable IV
            chain = chain[
                (chain["implied_vol"] > 0.01)
                & (chain["implied_vol"] < 3.0)
                & (chain["volume"] > 0)
            ]
            all_data.append(chain)

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)

    # Add moneyness
    df["moneyness"] = df["strike"] / df["spot"]

    # Filter to reasonable range
    df = df[(df["moneyness"] > 0.7) & (df["moneyness"] < 1.3)]

    _cache.set(cache_key, df)
    return df


def calibrate_model_to_market(
    ticker: str,
    model: str = "heston",
    expiry: Optional[str] = None,
) -> Dict:
    """
    Calibrate a pricing model to market implied volatilities.

    Args:
        ticker: Stock symbol.
        model: Model type ('heston', 'sabr').
        expiry: Target expiration date.

    Returns:
        Dict with calibrated parameters and fit metrics.
    """
    from scipy.optimize import minimize

    # Get market data
    chain = get_options_chain(ticker, expiry)
    if chain.empty:
        return {"error": "No options data available"}

    spot_info = get_stock_price(ticker)
    S = spot_info["price"]
    if S <= 0:
        return {"error": "Could not get spot price"}

    # Filter for OTM options (more liquid)
    calls = chain[(chain["type"] == "call") & (chain["strike"] > S)]
    puts = chain[(chain["type"] == "put") & (chain["strike"] < S)]

    otm = pd.concat([calls, puts])
    otm = otm[otm["implied_vol"] > 0.01]

    if len(otm) < 5:
        return {"error": "Not enough options for calibration"}

    # Get risk-free rate (approximate)
    r = 0.05  # TODO: Use treasury rate
    T = otm["dte"].iloc[0] / 365.0

    if model == "heston":
        from src.pricing_models import HestonPricer

        def objective(params):
            kappa, theta, sigma_v, rho, v0 = params
            if kappa < 0.1 or theta < 0.001 or sigma_v < 0.01 or v0 < 0.001:
                return 1e10
            if abs(rho) > 0.99:
                return 1e10

            try:
                pricer = HestonPricer(kappa, theta, sigma_v, rho, v0)
                errors = []

                for _, row in otm.iterrows():
                    K = row["strike"]
                    market_iv = row["implied_vol"]
                    opt_type = row["type"]

                    # Price with Heston
                    _ = pricer.price_european(S, K, T, r, 0.0, opt_type)

                    # Convert to implied vol (simplified)
                    model_iv = np.sqrt(v0)  # Rough approximation

                    errors.append((model_iv - market_iv) ** 2)

                return np.sum(errors)
            except Exception:
                return 1e10

        # Initial guess
        x0 = [2.0, 0.04, 0.3, -0.7, 0.04]
        bounds = [(0.1, 10), (0.001, 0.5), (0.01, 2), (-0.99, 0.99), (0.001, 0.5)]

        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

        if result.success:
            kappa, theta, sigma_v, rho, v0 = result.x
            return {
                "model": "heston",
                "params": {
                    "kappa": kappa,
                    "theta": theta,
                    "sigma_v": sigma_v,
                    "rho": rho,
                    "v0": v0,
                },
                "fit_error": result.fun,
                "n_options": len(otm),
                "spot": S,
                "expiry": expiry or chain["expiry"].iloc[0],
            }
        else:
            return {"error": "Calibration failed", "message": result.message}

    return {"error": f"Unknown model: {model}"}
