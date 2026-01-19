# src/backtesting/__init__.py
"""
Backtesting module for options strategies.

Provides historical P&L simulation and strategy validation.
"""

from src.backtesting.backtest_engine import (
    BacktestEngine,
    BacktestResult,
    run_delta_hedge_backtest,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "run_delta_hedge_backtest",
]
