# src/utils/market_data.py
"""
Market Data Pipeline Module.

Provides utilities for fetching and processing market data:
- Yahoo Finance stock and option chain fetching
- Option chain parsing and normalization
- Data storage (parquet/CSV)

Usage:
    from src.utils.market_data import YahooFinanceFetcher

    fetcher = YahooFinanceFetcher()
    stock_data = fetcher.get_stock_data("AAPL")
    option_chain = fetcher.get_option_chain("AAPL")
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Check for yfinance availability
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn("yfinance not installed. Run: pip install yfinance")


@dataclass
class StockData:
    """Container for stock data."""

    ticker: str
    current_price: float
    history: pd.DataFrame
    info: Dict

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "current_price": self.current_price,
            "history": self.history.to_dict(),
            "info": self.info,
        }


@dataclass
class OptionChainData:
    """Container for option chain data."""

    ticker: str
    spot_price: float
    expiration_dates: List[str]
    calls: pd.DataFrame
    puts: pd.DataFrame
    fetched_at: datetime

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "spot_price": self.spot_price,
            "expiration_dates": self.expiration_dates,
            "calls": self.calls.to_dict(),
            "puts": self.puts.to_dict(),
            "fetched_at": self.fetched_at.isoformat(),
        }


class YahooFinanceFetcher:
    """
    Fetch stock and options data from Yahoo Finance.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize fetcher.

        Args:
            cache_dir: Directory to cache fetched data.
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance required: pip install yfinance")

        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def get_stock_data(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> StockData:
        """
        Fetch stock price history.

        Args:
            ticker: Stock symbol (e.g., "AAPL").
            period: Time period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max").
            interval: Data interval ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo").

        Returns:
            StockData object with history and current price.
        """
        stock = yf.Ticker(ticker)

        # Get historical data
        history = stock.history(period=period, interval=interval)

        # Get current price
        try:
            current_price = (
                stock.info.get("currentPrice")
                or stock.info.get("regularMarketPrice")
                or history["Close"].iloc[-1]
            )
        except (IndexError, KeyError):
            current_price = np.nan

        return StockData(
            ticker=ticker,
            current_price=current_price,
            history=history,
            info=stock.info if hasattr(stock, "info") else {},
        )

    def get_option_chain(
        self,
        ticker: str,
        expiration: Optional[str] = None,
    ) -> OptionChainData:
        """
        Fetch option chain data.

        Args:
            ticker: Stock symbol.
            expiration: Specific expiration date (YYYY-MM-DD) or None for all.

        Returns:
            OptionChainData with calls and puts DataFrames.
        """
        stock = yf.Ticker(ticker)

        # Get available expiration dates
        expiration_dates = list(stock.options)

        if not expiration_dates:
            raise ValueError(f"No options available for {ticker}")

        # Get spot price
        try:
            spot = stock.info.get("currentPrice") or stock.info.get(
                "regularMarketPrice"
            )
        except Exception:
            spot = np.nan

        # Fetch option chains
        all_calls = []
        all_puts = []

        dates_to_fetch = [expiration] if expiration else expiration_dates

        for exp_date in dates_to_fetch:
            try:
                opt = stock.option_chain(exp_date)

                # Add expiration column
                calls = opt.calls.copy()
                puts = opt.puts.copy()
                calls["expiration"] = exp_date
                puts["expiration"] = exp_date

                all_calls.append(calls)
                all_puts.append(puts)
            except Exception as e:
                logger.warning(f"Failed to fetch options for {exp_date}: {e}")

        calls_df = (
            pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
        )
        puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

        return OptionChainData(
            ticker=ticker,
            spot_price=spot,
            expiration_dates=expiration_dates,
            calls=calls_df,
            puts=puts_df,
            fetched_at=datetime.now(),
        )

    def get_risk_free_rate(self) -> float:
        """
        Get approximate risk-free rate from Treasury yields.

        Returns:
            10-year Treasury yield as decimal.
        """
        try:
            # Use 10-year Treasury ETF as proxy
            tlt = yf.Ticker("^TNX")
            rate = tlt.info.get("regularMarketPrice", 4.0) / 100
            return rate
        except Exception:
            return 0.05  # Default fallback


class OptionChainParser:
    """
    Parse and normalize option chain data.
    """

    @staticmethod
    def parse_yahoo_chain(chain: OptionChainData) -> pd.DataFrame:
        """
        Parse Yahoo Finance option chain into standardized format.

        Returns:
            DataFrame with columns: strike, expiration, type, bid, ask, mid, volume,
            open_interest, implied_volatility, delta, gamma, theta, vega
        """
        # Combine calls and puts
        calls = chain.calls.copy()
        puts = chain.puts.copy()

        calls["option_type"] = "call"
        puts["option_type"] = "put"

        combined = pd.concat([calls, puts], ignore_index=True)

        # Standardize column names
        column_map = {
            "strike": "strike",
            "lastPrice": "last_price",
            "bid": "bid",
            "ask": "ask",
            "volume": "volume",
            "openInterest": "open_interest",
            "impliedVolatility": "implied_volatility",
            "expiration": "expiration",
            "option_type": "option_type",
        }

        # Rename existing columns
        combined = combined.rename(
            columns={k: v for k, v in column_map.items() if k in combined.columns}
        )

        # Calculate mid price
        if "bid" in combined.columns and "ask" in combined.columns:
            combined["mid_price"] = (combined["bid"] + combined["ask"]) / 2

        # Calculate time to maturity
        if "expiration" in combined.columns:
            today = datetime.now()
            combined["T"] = combined["expiration"].apply(
                lambda x: (datetime.strptime(x, "%Y-%m-%d") - today).days / 365
            )

        # Add moneyness
        if chain.spot_price and "strike" in combined.columns:
            combined["moneyness"] = combined["strike"] / chain.spot_price

        return combined

    @staticmethod
    def filter_liquid_options(
        df: pd.DataFrame,
        min_volume: int = 10,
        min_open_interest: int = 100,
        max_bid_ask_spread: float = 0.5,
    ) -> pd.DataFrame:
        """
        Filter for liquid options only.
        """
        mask = pd.Series(True, index=df.index)

        if "volume" in df.columns:
            mask &= df["volume"] >= min_volume

        if "open_interest" in df.columns:
            mask &= df["open_interest"] >= min_open_interest

        if "bid" in df.columns and "ask" in df.columns:
            spread = (df["ask"] - df["bid"]) / df["mid_price"]
            mask &= spread <= max_bid_ask_spread

        return df[mask].copy()


def save_to_parquet(
    data: Union[pd.DataFrame, OptionChainData, StockData],
    path: Union[str, Path],
) -> None:
    """Save data to parquet format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        data.to_parquet(path)
    elif isinstance(data, OptionChainData):
        # Save as combined DataFrame
        combined = pd.concat([data.calls, data.puts], ignore_index=True)
        combined.to_parquet(path)
    elif isinstance(data, StockData):
        data.history.to_parquet(path)
    else:
        raise TypeError(f"Cannot save {type(data)} to parquet")


def load_from_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Load data from parquet format."""
    return pd.read_parquet(path)


# Convenience functions
def fetch_option_chain(ticker: str) -> OptionChainData:
    """Quick option chain fetch."""
    return YahooFinanceFetcher().get_option_chain(ticker)


def fetch_stock_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Quick stock history fetch."""
    return YahooFinanceFetcher().get_stock_data(ticker, period=period).history
