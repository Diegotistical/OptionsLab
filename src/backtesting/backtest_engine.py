# src/backtesting/backtest_engine.py
"""
Historical backtesting engine for options strategies.

Simulates P&L using historical price data and compares model predictions
to realized outcomes. Supports delta hedging analysis.

Usage:
    >>> from src.backtesting import BacktestEngine
    >>> engine = BacktestEngine()
    >>> result = engine.run_delta_hedge("SPY", "2024-01-01", "2024-06-01", 100, 0.25)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    strategy: str
    ticker: str
    start_date: str
    end_date: str
    initial_value: float
    final_value: float
    total_pnl: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    daily_pnl: List[float] = field(default_factory=list)
    cumulative_pnl: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    model_greeks: Dict = field(default_factory=dict)


class BacktestEngine:
    """
    Backtesting engine for options strategies.
    """

    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate

    def _fetch_historical_data(
        self,
        ticker: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch historical price data."""
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance required: pip install yfinance")

        stock = yf.Ticker(ticker)
        hist = stock.history(start=start, end=end)

        if hist.empty:
            raise ValueError(f"No data for {ticker} from {start} to {end}")

        return hist

    def _bs_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
    ) -> float:
        """Black-Scholes price."""
        if T <= 0:
            if option_type == "call":
                return max(S - K, 0)
            return max(K - S, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def _bs_delta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
    ) -> float:
        """Black-Scholes delta."""
        if T <= 0:
            if option_type == "call":
                return 1.0 if S > K else 0.0
            return -1.0 if S < K else 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        if option_type == "call":
            return norm.cdf(d1)
        return norm.cdf(d1) - 1

    def _bs_gamma(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Black-Scholes gamma."""
        if T <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def run_delta_hedge(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        strike: float,
        maturity_years: float,
        sigma: float = 0.20,
        option_type: Literal["call", "put"] = "call",
        position_size: int = 100,
        hedge_frequency: Literal["daily", "weekly"] = "daily",
    ) -> BacktestResult:
        """
        Run delta hedging backtest.

        Simulates selling an option and delta hedging with stock.
        Shows P&L from gamma exposure.

        Returns:
            BacktestResult with P&L analysis.
        """
        # Fetch data
        hist = self._fetch_historical_data(ticker, start_date, end_date)
        prices = hist["Close"].values
        dates = hist.index.strftime("%Y-%m-%d").tolist()
        n_days = len(prices)

        if n_days < 2:
            raise ValueError("Need at least 2 days of data")

        # Initialize
        S0 = prices[0]
        T0 = maturity_years
        dt = 1 / 252  # Trading days

        # Option premium received
        option_price = self._bs_price(S0, strike, T0, self.r, sigma, option_type)
        premium_received = option_price * position_size * 100

        # Initial delta hedge
        delta = self._bs_delta(S0, strike, T0, self.r, sigma, option_type)
        shares_held = -delta * position_size * 100  # Short delta to hedge
        cash = premium_received - shares_held * S0

        # Track P&L
        daily_pnl = []
        cumulative_pnl = []
        hedge_deltas = []

        running_pnl = 0
        hedge_step = 1 if hedge_frequency == "daily" else 5

        for i in range(1, n_days):
            S_prev = prices[i - 1]
            S_curr = prices[i]
            T_remaining = max(T0 - i * dt, 0.001)

            # P&L from stock position
            stock_pnl = shares_held * (S_curr - S_prev)

            # Option value change
            opt_prev = self._bs_price(
                S_prev, strike, T_remaining + dt, self.r, sigma, option_type
            )
            opt_curr = self._bs_price(
                S_curr, strike, T_remaining, self.r, sigma, option_type
            )
            option_pnl = -(opt_curr - opt_prev) * position_size * 100  # Short option

            day_pnl = stock_pnl + option_pnl
            running_pnl += day_pnl

            daily_pnl.append(day_pnl)
            cumulative_pnl.append(running_pnl)

            # Rebalance hedge
            if i % hedge_step == 0:
                new_delta = self._bs_delta(
                    S_curr, strike, T_remaining, self.r, sigma, option_type
                )
                target_shares = -new_delta * position_size * 100
                shares_change = target_shares - shares_held
                cash -= shares_change * S_curr
                shares_held = target_shares
                hedge_deltas.append(new_delta)

        # Final settlement
        S_final = prices[-1]

        if option_type == "call":
            payoff = max(S_final - strike, 0)
        else:
            payoff = max(strike - S_final, 0)

        # Final P&L
        final_stock_value = shares_held * S_final
        final_option_liability = payoff * position_size * 100
        final_value = cash + final_stock_value - final_option_liability
        total_pnl = final_value

        # Metrics
        daily_returns = np.array(daily_pnl) / (premium_received + abs(shares_held * S0))
        sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)

        cum = np.array(cumulative_pnl)
        peaks = np.maximum.accumulate(cum)
        drawdowns = (peaks - cum) / (np.abs(peaks) + 1e-10)
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

        win_rate = np.mean(np.array(daily_pnl) > 0) if daily_pnl else 0

        return BacktestResult(
            strategy=f"Delta Hedge ({hedge_frequency})",
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_value=premium_received,
            final_value=final_value,
            total_pnl=total_pnl,
            total_return_pct=(
                total_pnl / premium_received * 100 if premium_received > 0 else 0
            ),
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            n_trades=len(hedge_deltas),
            daily_pnl=daily_pnl,
            cumulative_pnl=cumulative_pnl,
            dates=dates[1:],
            model_greeks={"initial_delta": delta, "sigma": sigma},
        )

    def compare_model_to_realized(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Dict:
        """
        Compare model-predicted volatility to realized volatility.

        Returns:
            Dict with implied vs realized comparison.
        """
        hist = self._fetch_historical_data(ticker, start_date, end_date)

        # Realized volatility (20-day rolling)
        log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        realized_vol = log_returns.rolling(20).std() * np.sqrt(252)

        return {
            "ticker": ticker,
            "period": f"{start_date} to {end_date}",
            "realized_vol_mean": float(realized_vol.mean()),
            "realized_vol_min": float(realized_vol.min()),
            "realized_vol_max": float(realized_vol.max()),
            "realized_vol_series": realized_vol.values.tolist(),
            "dates": hist.index[20:].strftime("%Y-%m-%d").tolist(),
        }


def run_delta_hedge_backtest(
    ticker: str,
    start: str,
    end: str,
    strike: float,
    maturity: float = 0.25,
    sigma: float = 0.20,
) -> BacktestResult:
    """Convenience function for delta hedge backtest."""
    engine = BacktestEngine()
    return engine.run_delta_hedge(ticker, start, end, strike, maturity, sigma)
