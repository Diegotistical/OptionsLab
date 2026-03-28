# src/experiments/delta_hedging.py
"""
Delta Hedging Experiment Engine.

Compares hedging performance across different volatility surface models.
The key insight: arbitrage-free surfaces (CINN) produce smoother, more
consistent Greeks, leading to lower hedging P&L variance and tail risk.

Experiment design:
    1. Generate synthetic option portfolios (ATM/OTM/ITM mix)
    2. Simulate underlying price paths (GBM)
    3. Delta-hedge each option using Greeks from each vol surface model
    4. Measure hedging P&L variance, tail risk, and Greek accuracy

Usage:
    >>> from src.experiments.delta_hedging import DeltaHedgingExperiment
    >>> exp = DeltaHedgingExperiment(vol_models={"CINN": cinn_model, "SVI": svi_model})
    >>> results = exp.run(n_paths=10000, rebalance_freq="daily")
    >>> print(results.summary())
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class HedgingMetrics:
    """Hedging performance metrics for a single model."""

    model_name: str
    pnl_mean: float = 0.0
    pnl_std: float = 0.0
    pnl_p1: float = 0.0  # 1st percentile (worst loss)
    pnl_p99: float = 0.0  # 99th percentile
    pnl_skew: float = 0.0
    pnl_kurtosis: float = 0.0
    mean_abs_delta_error: float = 0.0  # vs analytical BS delta
    mean_abs_gamma_error: float = 0.0
    hedge_efficiency: float = 0.0  # 1 - var(hedged) / var(unhedged)
    rebalance_freq: str = ""
    n_options: int = 0
    n_paths: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "Model": self.model_name,
            "P&L Mean": self.pnl_mean,
            "P&L Std": self.pnl_std,
            "P&L 1%": self.pnl_p1,
            "P&L 99%": self.pnl_p99,
            "P&L Skew": self.pnl_skew,
            "P&L Kurtosis": self.pnl_kurtosis,
            "|Delta Error|": self.mean_abs_delta_error,
            "|Gamma Error|": self.mean_abs_gamma_error,
            "Hedge Eff.": self.hedge_efficiency,
        }


@dataclass
class HedgingResults:
    """Container for all hedging experiment results."""

    models: List[HedgingMetrics] = field(default_factory=list)
    pnl_distributions: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.models:
            return pd.DataFrame()
        return pd.DataFrame([m.to_dict() for m in self.models]).set_index("Model")

    def summary(self) -> str:
        df = self.to_dataframe()
        return f"Delta Hedging Results\n\n{df.to_string()}"


def _bs_delta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Analytical Black-Scholes delta for a call."""
    if T <= 1e-10:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.cdf(d1)


def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Analytical Black-Scholes gamma for a call."""
    if T <= 1e-10:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes call price."""
    if T <= 1e-10:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


class DeltaHedgingExperiment:
    """
    Delta hedging experiment comparing vol surface models.

    Each model provides delta estimates via its vol surface. We hedge
    a portfolio of options and measure residual P&L.
    """

    def __init__(
        self,
        vol_models: Dict[str, object],
        S0: float = 100.0,
        r: float = 0.05,
        q: float = 0.0,
        true_vol: float = 0.2,
    ):
        """
        Args:
            vol_models: Dict mapping model name -> trained vol surface model.
                Each must have predict_volatility(df) returning implied vols.
            S0: Initial spot price.
            r: Risk-free rate.
            q: Dividend yield.
            true_vol: True volatility used to simulate the underlying.
        """
        self.vol_models = vol_models
        self.S0 = S0
        self.r = r
        self.q = q
        self.true_vol = true_vol

    def generate_portfolio(
        self,
        n_options: int = 100,
        strike_range: Tuple[float, float] = (0.85, 1.15),
        maturities: Optional[List[float]] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate a synthetic option portfolio.

        Returns DataFrame with columns: strike, maturity, moneyness_type
        """
        if maturities is None:
            maturities = [1 / 12, 3 / 12, 6 / 12]  # 1M, 3M, 6M

        rng = np.random.default_rng(seed)

        strikes = self.S0 * rng.uniform(strike_range[0], strike_range[1], n_options)
        mat_choices = rng.choice(maturities, n_options)

        moneyness = strikes / self.S0
        moneyness_type = np.where(
            moneyness < 0.95, "ITM", np.where(moneyness > 1.05, "OTM", "ATM")
        )

        return pd.DataFrame({
            "strike": strikes,
            "maturity": mat_choices,
            "moneyness_type": moneyness_type,
        })

    def _get_model_delta(
        self,
        model,
        S: float,
        K: float,
        T: float,
    ) -> float:
        """Get delta from a vol surface model."""
        if T <= 1e-10:
            return 1.0 if S > K else 0.0

        log_m = np.log(K / S)
        df = pd.DataFrame({"log_moneyness": [log_m], "T": [T]})
        try:
            sigma = float(model.predict_volatility(df)[0])
            sigma = np.clip(sigma, 0.01, 5.0)
        except Exception:
            sigma = self.true_vol

        return _bs_delta(S, K, T, self.r, sigma, self.q)

    def _hedge_single_option(
        self,
        model,
        price_path: np.ndarray,
        K: float,
        T: float,
        dt: float,
    ) -> float:
        """
        Delta-hedge a single call option along a price path.

        Returns the hedging P&L (ideally near zero for perfect hedging).
        """
        n_steps = len(price_path) - 1

        # Initial option value
        S0 = price_path[0]
        sigma0 = self.true_vol
        option_value_0 = _bs_price(S0, K, T, self.r, sigma0, self.q)

        # Track cash and stock positions
        delta_prev = self._get_model_delta(model, S0, K, T)
        cash = option_value_0 - delta_prev * S0  # Sell option, buy delta shares

        for i in range(1, n_steps + 1):
            S_i = price_path[i]
            T_remaining = T - i * dt

            if T_remaining <= 0:
                break

            # Rebalance delta
            delta_new = self._get_model_delta(model, S_i, K, T_remaining)
            cash -= (delta_new - delta_prev) * S_i  # Buy/sell shares
            cash *= np.exp(self.r * dt)  # Interest on cash
            delta_prev = delta_new

        # Final P&L: portfolio value - option payoff
        S_T = price_path[-1]
        option_payoff = max(S_T - K, 0.0)
        portfolio_value = delta_prev * S_T + cash
        hedging_pnl = portfolio_value - option_payoff

        return hedging_pnl

    def run(
        self,
        n_paths: int = 5000,
        n_options: int = 50,
        rebalance_freq: Literal["daily", "4h", "hourly"] = "daily",
        seed: int = 42,
    ) -> HedgingResults:
        """
        Run the full delta hedging experiment.

        Args:
            n_paths: Number of Monte Carlo price paths per option.
            n_options: Number of options in portfolio.
            rebalance_freq: Hedging rebalancing frequency.
            seed: Random seed for reproducibility.

        Returns:
            HedgingResults with metrics for each model.
        """
        from src.simulation.gbm_numpy import simulate_gbm_paths

        # Rebalancing frequency -> steps per year
        freq_map = {"daily": 252, "4h": 252 * 6, "hourly": 252 * 8}
        steps_per_year = freq_map.get(rebalance_freq, 252)

        portfolio = self.generate_portfolio(n_options=n_options, seed=seed)
        results = HedgingResults()

        # Also compute unhedged P&L for hedge efficiency
        unhedged_pnls = []

        for model_name, model in self.vol_models.items():
            all_pnls = []
            all_delta_errors = []
            all_gamma_errors = []

            for _, opt in portfolio.iterrows():
                K = opt["strike"]
                T = opt["maturity"]
                n_steps = max(int(T * steps_per_year), 1)
                dt = T / n_steps

                # Simulate paths for this option
                paths = simulate_gbm_paths(
                    S=self.S0,
                    T=T,
                    r=self.r,
                    sigma=self.true_vol,
                    q=self.q,
                    n_paths=n_paths,
                    n_steps=n_steps,
                    seed=seed + hash(str(K)) % 10000,
                )

                for p in range(min(n_paths, paths.shape[0])):
                    pnl = self._hedge_single_option(model, paths[p], K, T, dt)
                    all_pnls.append(pnl)

                # Greek accuracy at initial point
                log_m = np.log(K / self.S0)
                df_query = pd.DataFrame({"log_moneyness": [log_m], "T": [T]})
                try:
                    model_vol = float(model.predict_volatility(df_query)[0])
                    model_vol = np.clip(model_vol, 0.01, 5.0)
                except Exception:
                    model_vol = self.true_vol

                true_delta = _bs_delta(self.S0, K, T, self.r, self.true_vol, self.q)
                model_delta = _bs_delta(self.S0, K, T, self.r, model_vol, self.q)
                all_delta_errors.append(abs(model_delta - true_delta))

                true_gamma = _bs_gamma(self.S0, K, T, self.r, self.true_vol, self.q)
                model_gamma = _bs_gamma(self.S0, K, T, self.r, model_vol, self.q)
                all_gamma_errors.append(abs(model_gamma - true_gamma))

            pnl_arr = np.array(all_pnls)
            results.pnl_distributions[model_name] = pnl_arr

            # Compute unhedged variance (only once)
            if not unhedged_pnls:
                # Unhedged = just option payoff variance
                for _, opt in portfolio.iterrows():
                    K = opt["strike"]
                    T = opt["maturity"]
                    n_steps = max(int(T * steps_per_year), 1)
                    paths = simulate_gbm_paths(
                        S=self.S0, T=T, r=self.r, sigma=self.true_vol, q=self.q,
                        n_paths=n_paths, n_steps=n_steps,
                        seed=seed + hash(str(K)) % 10000,
                    )
                    initial_price = _bs_price(self.S0, K, T, self.r, self.true_vol, self.q)
                    for p in range(min(n_paths, paths.shape[0])):
                        payoff = max(paths[p, -1] - K, 0.0)
                        unhedged_pnls.append(payoff - initial_price)

            unhedged_var = np.var(unhedged_pnls) if unhedged_pnls else 1.0

            metrics = HedgingMetrics(
                model_name=model_name,
                pnl_mean=float(np.mean(pnl_arr)),
                pnl_std=float(np.std(pnl_arr)),
                pnl_p1=float(np.percentile(pnl_arr, 1)),
                pnl_p99=float(np.percentile(pnl_arr, 99)),
                pnl_skew=float(_skewness(pnl_arr)),
                pnl_kurtosis=float(_kurtosis(pnl_arr)),
                mean_abs_delta_error=float(np.mean(all_delta_errors)),
                mean_abs_gamma_error=float(np.mean(all_gamma_errors)),
                hedge_efficiency=float(1 - np.var(pnl_arr) / max(unhedged_var, 1e-10)),
                rebalance_freq=rebalance_freq,
                n_options=n_options,
                n_paths=n_paths,
            )
            results.models.append(metrics)

        return results


def _skewness(x: np.ndarray) -> float:
    """Compute skewness."""
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-10:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis."""
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-10:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)
