# src/data/data_loader.py
"""
Unified data loading for option chain and volatility surface data.

Supports:
- CSV files (CBOE, OptionMetrics, custom formats)
- Parquet files
- Synthetic data generation (for testing without rate limits)

Usage:
    >>> from src.data.data_loader import OptionChainLoader
    >>> dataset = OptionChainLoader.from_csv("options.csv")
    >>> surface = dataset.to_model_input()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm


# =============================================================================
# Data Container
# =============================================================================


@dataclass
class OptionChainDataset:
    """
    Standardized option chain container with preprocessing utilities.

    Holds raw option chain data and provides methods for filtering,
    IV computation, and conversion to model-ready format.
    """

    data: pd.DataFrame
    underlying_price: float = 100.0
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    source: str = "unknown"

    def __post_init__(self):
        """Validate and standardize column names."""
        self._standardize_columns()

    def _standardize_columns(self) -> None:
        """Map common column name variations to standard names."""
        column_map = {
            # Strike variations
            "Strike": "strike",
            "strike_price": "strike",
            "K": "strike",
            # Expiry variations
            "Expiry": "expiry",
            "expiration": "expiry",
            "expiration_date": "expiry",
            "exp_date": "expiry",
            # Type variations
            "Type": "type",
            "option_type": "type",
            "cp_flag": "type",
            "call_put": "type",
            # IV variations
            "IV": "implied_vol",
            "impl_vol": "implied_vol",
            "impl_volatility": "implied_vol",
            "implied_volatility": "implied_vol",
            "mid_iv": "implied_vol",
            # Price variations
            "Mid": "mid",
            "mid_price": "mid",
            "Last": "last",
            "last_price": "last",
            # Greeks
            "Delta": "delta",
            "Gamma": "gamma",
            "Vega": "vega",
            "Theta": "theta",
        }

        for old_name, new_name in column_map.items():
            if old_name in self.data.columns and new_name not in self.data.columns:
                self.data = self.data.rename(columns={old_name: new_name})

        # Standardize type values
        if "type" in self.data.columns:
            self.data["type"] = self.data["type"].astype(str).str.lower()
            self.data["type"] = self.data["type"].replace(
                {"c": "call", "p": "put", "C": "call", "P": "put"}
            )

    @property
    def n_options(self) -> int:
        return len(self.data)

    @property
    def n_expiries(self) -> int:
        if "expiry" in self.data.columns:
            return self.data["expiry"].nunique()
        return 1

    @property
    def strikes(self) -> np.ndarray:
        return self.data["strike"].unique() if "strike" in self.data.columns else np.array([])

    def filter_liquid(
        self,
        min_volume: int = 100,
        min_oi: int = 50,
    ) -> "OptionChainDataset":
        """
        Filter to liquid options only.

        Args:
            min_volume: Minimum trading volume
            min_oi: Minimum open interest

        Returns:
            Filtered OptionChainDataset
        """
        mask = pd.Series([True] * len(self.data))

        if "volume" in self.data.columns:
            mask &= self.data["volume"] >= min_volume
        if "open_interest" in self.data.columns:
            mask &= self.data["open_interest"] >= min_oi

        return OptionChainDataset(
            data=self.data[mask].copy(),
            underlying_price=self.underlying_price,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            source=self.source,
        )

    def filter_moneyness(
        self,
        min_moneyness: float = 0.8,
        max_moneyness: float = 1.2,
    ) -> "OptionChainDataset":
        """Filter to options within moneyness range."""
        if "strike" not in self.data.columns:
            return self

        moneyness = self.data["strike"] / self.underlying_price
        mask = (moneyness >= min_moneyness) & (moneyness <= max_moneyness)

        return OptionChainDataset(
            data=self.data[mask].copy(),
            underlying_price=self.underlying_price,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            source=self.source,
        )

    def compute_iv_from_prices(self) -> "OptionChainDataset":
        """
        Compute implied volatility from option prices if not present.
        Uses Newton-Raphson on Black-Scholes.
        """
        if "implied_vol" in self.data.columns:
            return self

        if "mid" not in self.data.columns and "last" not in self.data.columns:
            raise ValueError("Need 'mid' or 'last' price to compute IV")

        price_col = "mid" if "mid" in self.data.columns else "last"

        # Compute IV for each option
        ivs = []
        for _, row in self.data.iterrows():
            try:
                iv = self._newton_iv(
                    price=row[price_col],
                    S=self.underlying_price,
                    K=row["strike"],
                    T=row.get("dte", 30) / 365,
                    r=self.risk_free_rate,
                    q=self.dividend_yield,
                    option_type=row.get("type", "call"),
                )
                ivs.append(iv)
            except Exception:
                ivs.append(np.nan)

        self.data = self.data.copy()
        self.data["implied_vol"] = ivs
        return self

    def _newton_iv(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        option_type: str,
        tol: float = 1e-6,
        max_iter: int = 50,
    ) -> float:
        """Newton-Raphson IV solver."""
        if T <= 0 or price <= 0:
            return np.nan

        sigma = 0.3  # Initial guess

        for _ in range(max_iter):
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if option_type == "call":
                bs_price = (
                    S * np.exp(-q * T) * norm.cdf(d1)
                    - K * np.exp(-r * T) * norm.cdf(d2)
                )
            else:
                bs_price = (
                    K * np.exp(-r * T) * norm.cdf(-d2)
                    - S * np.exp(-q * T) * norm.cdf(-d1)
                )

            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

            if vega < 1e-10:
                return np.nan

            diff = bs_price - price
            if abs(diff) < tol:
                return sigma

            sigma = sigma - diff / vega
            sigma = max(0.01, min(sigma, 5.0))

        return sigma

    def compute_log_moneyness(self, forward: Optional[float] = None) -> pd.DataFrame:
        """
        Compute log-moneyness k = log(K/F) for each option.

        Args:
            forward: Forward price. If None, computed from spot and rates.

        Returns:
            DataFrame with log_moneyness column added
        """
        if "strike" not in self.data.columns:
            raise ValueError("Need 'strike' column")

        df = self.data.copy()

        # Get time to expiry
        if "T" in df.columns:
            T = df["T"]
        elif "dte" in df.columns:
            T = df["dte"] / 365
        else:
            T = 1.0  # Default 1 year

        if forward is None:
            F = self.underlying_price * np.exp(
                (self.risk_free_rate - self.dividend_yield) * T
            )
        else:
            F = forward

        df["log_moneyness"] = np.log(df["strike"] / F)
        df["T"] = T

        return df

    def to_model_input(self) -> pd.DataFrame:
        """
        Convert to format ready for volatility model training.

        Returns:
            DataFrame with columns: log_moneyness, T, implied_volatility
        """
        df = self.compute_log_moneyness()

        if "implied_vol" not in df.columns:
            raise ValueError("No implied_vol column. Run compute_iv_from_prices() first.")

        result = df[["log_moneyness", "T", "implied_vol"]].copy()
        result = result.rename(columns={"implied_vol": "implied_volatility"})
        result = result.dropna()

        return result


# =============================================================================
# Loaders
# =============================================================================


class OptionChainLoader:
    """
    Unified interface for loading option chain data from various sources.
    """

    @staticmethod
    def from_csv(
        path: Union[str, Path],
        format: Literal["auto", "cboe", "optionmetrics", "custom"] = "auto",
        underlying_price: Optional[float] = None,
        **kwargs,
    ) -> OptionChainDataset:
        """
        Load option chain from CSV file.

        Args:
            path: Path to CSV file
            format: Data format (auto-detected if 'auto')
            underlying_price: Spot price (auto-detected if present in file)
            **kwargs: Additional pandas read_csv arguments

        Returns:
            OptionChainDataset
        """
        df = pd.read_csv(path, **kwargs)

        # Auto-detect format
        if format == "auto":
            format = OptionChainLoader._detect_format(df)

        # Apply format-specific parsing
        if format == "cboe":
            df = OptionChainLoader._parse_cboe(df)
        elif format == "optionmetrics":
            df = OptionChainLoader._parse_optionmetrics(df)

        # Get underlying price
        if underlying_price is None:
            underlying_price = OptionChainLoader._infer_underlying(df)

        return OptionChainDataset(
            data=df,
            underlying_price=underlying_price,
            source=f"csv:{Path(path).name}",
        )

    @staticmethod
    def from_parquet(
        path: Union[str, Path],
        underlying_price: Optional[float] = None,
    ) -> OptionChainDataset:
        """Load option chain from Parquet file."""
        df = pd.read_parquet(path)

        if underlying_price is None:
            underlying_price = OptionChainLoader._infer_underlying(df)

        return OptionChainDataset(
            data=df,
            underlying_price=underlying_price,
            source=f"parquet:{Path(path).name}",
        )

    @staticmethod
    def from_synthetic(
        n_strikes: int = 50,
        maturities: Optional[List[float]] = None,
        atm_vol: float = 0.2,
        skew: float = -0.3,
        smile: float = 0.05,
        spot: float = 100.0,
        seed: Optional[int] = None,
    ) -> OptionChainDataset:
        """
        Generate synthetic option chain with realistic smile.

        This is the recommended approach for development/testing
        to avoid API rate limiting issues.

        Args:
            n_strikes: Number of strikes per maturity
            maturities: List of maturities in years
            atm_vol: ATM implied volatility
            skew: Skew coefficient
            smile: Smile curvature
            spot: Underlying spot price
            seed: Random seed

        Returns:
            OptionChainDataset with synthetic data
        """
        if maturities is None:
            maturities = [0.083, 0.25, 0.5, 1.0, 2.0]  # 1m, 3m, 6m, 1y, 2y

        if seed is not None:
            np.random.seed(seed)

        records = []
        for T in maturities:
            # Generate strikes around ATM
            moneyness_range = np.linspace(0.8, 1.2, n_strikes)
            strikes = spot * moneyness_range

            for K in strikes:
                log_m = np.log(K / spot)

                # Adjust smile for maturity (flattens for longer dates)
                T_adj = np.sqrt(T)
                iv = atm_vol + skew / T_adj * log_m + smile * log_m**2
                iv += np.random.normal(0, 0.003)  # Add noise
                iv = max(iv, 0.02)  # Floor

                records.append(
                    {
                        "strike": K,
                        "T": T,
                        "dte": int(T * 365),
                        "type": "call",
                        "implied_vol": iv,
                        "volume": np.random.randint(100, 5000),
                        "open_interest": np.random.randint(500, 20000),
                    }
                )

        df = pd.DataFrame(records)
        return OptionChainDataset(
            data=df,
            underlying_price=spot,
            source="synthetic",
        )

    @staticmethod
    def from_yfinance(
        ticker: str,
        n_expiries: int = 3,
        use_cache: bool = True,
    ) -> OptionChainDataset:
        """
        Load option chain from Yahoo Finance with rate limit handling.
        
        Uses exponential backoff and caching to avoid rate limits.
        
        Args:
            ticker: Stock symbol (e.g., 'SPY', 'AAPL', 'AMD')
            n_expiries: Number of expiration dates to fetch
            use_cache: Use cached data if available
            
        Returns:
            OptionChainDataset with live market data
        """
        try:
            from src.data.market_data import (
                get_options_chain,
                get_stock_price,
                get_expiries,
                safe_yfinance_call,
                YFINANCE_AVAILABLE,
            )
        except ImportError:
            raise ImportError("market_data module required for yfinance loading")
        
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
        
        # Get spot price with retry
        spot_data = get_stock_price(ticker, use_cache=use_cache)
        spot = spot_data.get("price", 100.0)
        
        if spot <= 0:
            raise ValueError(f"Could not get spot price for {ticker}")
        
        # Get expiries
        expiries = get_expiries(ticker)[:n_expiries]
        
        if not expiries:
            raise ValueError(f"No options available for {ticker}")
        
        # Fetch chains for each expiry
        all_data = []
        for exp in expiries:
            try:
                chain = get_options_chain(ticker, exp, use_cache=use_cache)
                if not chain.empty:
                    all_data.append(chain)
            except Exception as e:
                print(f"Warning: Failed to fetch {ticker} {exp}: {e}")
                continue
        
        if not all_data:
            raise ValueError(f"Could not fetch any option data for {ticker}")
        
        df = pd.concat(all_data, ignore_index=True)
        
        # Filter valid IV
        df = df[df["implied_vol"] > 0.01]
        df = df[df["implied_vol"] < 3.0]
        
        # Add T column from dte
        if "dte" in df.columns:
            df["T"] = df["dte"] / 365.0
        
        return OptionChainDataset(
            data=df,
            underlying_price=spot,
            source=f"yfinance:{ticker}",
        )

    @staticmethod
    def _detect_format(df: pd.DataFrame) -> str:
        """Auto-detect CSV format from column names."""
        cols = set(df.columns.str.lower())

        if "underlying_symbol" in cols or "root" in cols:
            return "cboe"
        if "optionid" in cols or "securityid" in cols:
            return "optionmetrics"
        return "custom"

    @staticmethod
    def _parse_cboe(df: pd.DataFrame) -> pd.DataFrame:
        """Parse CBOE format."""
        # CBOE-specific column mappings
        return df

    @staticmethod
    def _parse_optionmetrics(df: pd.DataFrame) -> pd.DataFrame:
        """Parse OptionMetrics format."""
        # OptionMetrics-specific column mappings
        return df

    @staticmethod
    def _infer_underlying(df: pd.DataFrame) -> float:
        """Infer underlying price from data."""
        if "underlying_price" in df.columns:
            return float(df["underlying_price"].iloc[0])
        if "spot" in df.columns:
            return float(df["spot"].iloc[0])
        if "strike" in df.columns:
            return float(df["strike"].median())
        return 100.0


# =============================================================================
# Convenience Functions
# =============================================================================


def load_option_data(
    source: Union[str, Path],
    **kwargs,
) -> OptionChainDataset:
    """
    Load option data from any supported source.

    Automatically detects format based on file extension.

    Args:
        source: File path or 'synthetic' for generated data
        **kwargs: Additional arguments passed to loader

    Returns:
        OptionChainDataset
    """
    if source == "synthetic":
        return OptionChainLoader.from_synthetic(**kwargs)

    path = Path(source)

    if path.suffix.lower() == ".parquet":
        return OptionChainLoader.from_parquet(path, **kwargs)
    elif path.suffix.lower() == ".csv":
        return OptionChainLoader.from_csv(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


# =============================================================================
# Main for Testing
# =============================================================================


if __name__ == "__main__":
    # Generate synthetic data
    print("Generating synthetic option chain...")
    dataset = OptionChainLoader.from_synthetic(
        n_strikes=30,
        maturities=[0.25, 0.5, 1.0],
        seed=42,
    )

    print(f"Generated {dataset.n_options} options across {dataset.n_expiries} expiries")
    print(f"Underlying: ${dataset.underlying_price:.2f}")
    print(f"\nSample data:\n{dataset.data.head()}")

    # Convert to model input
    model_input = dataset.to_model_input()
    print(f"\nModel input shape: {model_input.shape}")
    print(model_input.head())
