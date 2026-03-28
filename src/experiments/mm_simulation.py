# src/experiments/mm_simulation.py
"""
Market Making Simulation with Vol-Surface-Aware Quoting.

Integrates OptionsLab's volatility surface models with Liquidity-Arena's
Avellaneda-Stoikov market maker. The key insight: the A-S model's sigma^2
directly drives both reservation price and spread width. A better sigma
estimate from the vol surface should yield:

  - Tighter spreads in calm markets (more competitive quoting)
  - Wider spreads in volatile markets (better risk control)
  - Lower adverse selection (quotes adjust to true vol faster)

Architecture:
  - Self-contained Python simulator (no C++ engine dependency)
  - Imports Liquidity-Arena agents and price processes via sys.path
  - Adds a lightweight Python order book for matching
  - VolSurfaceAwareMM subclasses AvellanedaStoikovMM, overriding sigma estimation

Usage:
    >>> from src.experiments.mm_simulation import run_mm_comparison
    >>> results = run_mm_comparison(vol_models={"CINN": cinn, "SVI": svi}, n_runs=50)
    >>> print(results.to_dataframe())
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add Liquidity-Arena to path for imports
_LA_PATH = str(Path("D:/Coding/C++/Liquidity-Arena"))
if _LA_PATH not in sys.path:
    sys.path.insert(0, _LA_PATH)

from simulation.agents.base_agent import AgentOrder, BaseAgent
from simulation.agents.market_maker import AvellanedaStoikovMM
from simulation.agents.noise_trader import NoiseTrader
from simulation.market.tcp_client import BookUpdateMsg


# =============================================================================
# Vol-Surface-Aware Market Maker
# =============================================================================


class VolSurfaceAwareMM(AvellanedaStoikovMM):
    """
    A-S market maker that uses a vol surface model for sigma estimation.

    Overrides _estimate_sigma_sq to blend surface-derived vol with rolling
    window estimates. The surface provides a structural estimate of volatility,
    while the rolling window captures high-frequency dynamics.
    """

    def __init__(
        self,
        vol_surface_model,
        option_T: float = 0.25,
        surface_weight: float = 0.7,
        **kwargs,
    ):
        """
        Args:
            vol_surface_model: Trained model with predict_volatility(df).
            option_T: Maturity of the option whose vol we query (ATM).
            surface_weight: Blending weight for surface vol vs rolling (0-1).
        """
        super().__init__(**kwargs)
        self.vol_model = vol_surface_model
        self.option_T = option_T
        self.surface_weight = surface_weight
        self._surface_sigma_sq_cache: Optional[float] = None
        self._cache_step: int = -1

    def _estimate_sigma_sq(self, mid: float) -> float:
        """Override: blend vol surface estimate with rolling window."""
        rolling_sigma_sq = super()._estimate_sigma_sq(mid)

        try:
            # Query surface at ATM (log_moneyness = 0)
            df = pd.DataFrame({"log_moneyness": [0.0], "T": [self.option_T]})
            surface_vol = float(self.vol_model.predict_volatility(df)[0])
            surface_vol = np.clip(surface_vol, 0.01, 5.0)

            # Scale: surface vol is annualized, but we need per-tick variance.
            # The rolling estimate is in tick-space, so scale surface to match.
            # Approximate: surface_sigma^2 * (mid^2) gives price-space variance.
            surface_sigma_sq = (surface_vol * mid) ** 2 / 252.0 + 1e-6

            # Blend
            alpha = self.surface_weight
            return alpha * surface_sigma_sq + (1 - alpha) * rolling_sigma_sq
        except Exception:
            return rolling_sigma_sq


# =============================================================================
# Lightweight Python Order Book (for standalone simulation)
# =============================================================================


class SimpleOrderBook:
    """
    Minimal order book for simulation. Price-time priority, FIFO matching.
    Not performance-critical — this runs at Python speed for research.
    """

    def __init__(self):
        self.bids: Dict[int, List[Tuple[int, int, str]]] = {}  # price -> [(qty, oid, agent_id)]
        self.asks: Dict[int, List[Tuple[int, int, str]]] = {}  # price -> [(qty, oid, agent_id)]
        self.best_bid: int = 0
        self.best_ask: int = 999999

    def add_order(
        self, order: AgentOrder, agent_id: str
    ) -> List[Tuple[str, str, int, int]]:
        """
        Add order to book. Returns list of fills: (maker_agent, taker_agent, price, qty).
        """
        fills = []

        if order.order_type == 1:  # MARKET
            if order.side == 0:  # BUY market
                order.price = 999999
            else:
                order.price = 0

        if order.side == 0:  # BID - match against asks
            while order.quantity > 0 and self.asks and order.price >= self.best_ask:
                best = self.best_ask
                if best not in self.asks or not self.asks[best]:
                    self._update_best_ask()
                    continue

                maker_qty, maker_oid, maker_agent = self.asks[best][0]
                fill_qty = min(order.quantity, maker_qty)

                fills.append((maker_agent, agent_id, best, fill_qty))
                order.quantity -= fill_qty

                if fill_qty >= maker_qty:
                    self.asks[best].pop(0)
                    if not self.asks[best]:
                        del self.asks[best]
                        self._update_best_ask()
                else:
                    self.asks[best][0] = (maker_qty - fill_qty, maker_oid, maker_agent)

            # Rest on book
            if order.quantity > 0 and order.order_type == 0:
                self.bids.setdefault(order.price, []).append(
                    (order.quantity, order.order_id, agent_id)
                )
                self.best_bid = max(self.best_bid, order.price)

        else:  # ASK - match against bids
            while order.quantity > 0 and self.bids and order.price <= self.best_bid:
                best = self.best_bid
                if best not in self.bids or not self.bids[best]:
                    self._update_best_bid()
                    continue

                maker_qty, maker_oid, maker_agent = self.bids[best][0]
                fill_qty = min(order.quantity, maker_qty)

                fills.append((maker_agent, agent_id, best, fill_qty))
                order.quantity -= fill_qty

                if fill_qty >= maker_qty:
                    self.bids[best].pop(0)
                    if not self.bids[best]:
                        del self.bids[best]
                        self._update_best_bid()
                else:
                    self.bids[best][0] = (maker_qty - fill_qty, maker_oid, maker_agent)

            if order.quantity > 0 and order.order_type == 0:
                self.asks.setdefault(order.price, []).append(
                    (order.quantity, order.order_id, agent_id)
                )
                self.best_ask = min(self.best_ask, order.price)

        return fills

    def cancel_orders(self, agent_id: str):
        """Cancel all orders from a given agent."""
        for price in list(self.bids.keys()):
            self.bids[price] = [o for o in self.bids[price] if o[2] != agent_id]
            if not self.bids[price]:
                del self.bids[price]
        for price in list(self.asks.keys()):
            self.asks[price] = [o for o in self.asks[price] if o[2] != agent_id]
            if not self.asks[price]:
                del self.asks[price]
        self._update_best_bid()
        self._update_best_ask()

    def _update_best_bid(self):
        self.best_bid = max(self.bids.keys()) if self.bids else 0

    def _update_best_ask(self):
        self.best_ask = min(self.asks.keys()) if self.asks else 999999

    def get_update(self) -> BookUpdateMsg:
        """Build a BookUpdateMsg from current state."""
        return BookUpdateMsg(
            best_bid=self.best_bid,
            best_ask=self.best_ask,
            num_levels=min(5, max(len(self.bids), len(self.asks))),
            bid_prices=[],
            bid_quantities=[],
            ask_prices=[],
            ask_quantities=[],
        )


# =============================================================================
# Simulation Runner
# =============================================================================


@dataclass
class MMSimulationMetrics:
    """Metrics from a single MM simulation run."""

    model_name: str
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    avg_spread: float = 0.0
    spread_std: float = 0.0
    max_inventory: int = 0
    avg_abs_inventory: float = 0.0
    total_fills: int = 0
    adverse_selection: float = 0.0


@dataclass
class MMComparisonResults:
    """Results comparing multiple MM configurations."""

    runs: Dict[str, List[MMSimulationMetrics]] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Aggregate across runs into summary table."""
        rows = []
        for model_name, run_list in self.runs.items():
            pnls = [r.total_pnl for r in run_list]
            sharpes = [r.sharpe for r in run_list]
            spreads = [r.avg_spread for r in run_list]
            spread_stds = [r.spread_std for r in run_list]
            inventories = [r.avg_abs_inventory for r in run_list]
            max_invs = [r.max_inventory for r in run_list]
            fills = [r.total_fills for r in run_list]
            adv_sel = [r.adverse_selection for r in run_list]

            rows.append({
                "Model": model_name,
                "P&L Mean": np.mean(pnls),
                "P&L Std": np.std(pnls),
                "Sharpe Mean": np.mean(sharpes),
                "Sharpe Std": np.std(sharpes),
                "Avg Spread": np.mean(spreads),
                "Spread Volatility": np.mean(spread_stds),
                "Avg |Inventory|": np.mean(inventories),
                "Max Inventory": np.mean(max_invs),
                "Fills/Run": np.mean(fills),
                "Adverse Selection": np.mean(adv_sel),
            })

        return pd.DataFrame(rows).set_index("Model")


def run_single_simulation(
    mm_agent: BaseAgent,
    n_steps: int = 10000,
    initial_price: int = 10000,
    seed: int = 42,
    tick_size: float = 0.01,
) -> MMSimulationMetrics:
    """
    Run a single MM simulation with noise traders providing liquidity.

    Uses Liquidity-Arena's RegimeSwitchingProcess for realistic price dynamics.
    """
    from simulation.market.price_process import RegimeSwitchingProcess

    # Price process
    price_proc = RegimeSwitchingProcess(
        initial_price=initial_price,
        kappa=0.3,
        theta=initial_price,
        seed=seed,
    )

    # Noise traders
    noise_agents = [
        NoiseTrader(
            agent_id=f"NOISE_{i}",
            order_prob=0.3,
            spread_range=20,
            min_qty=10,
            max_qty=50,
            market_order_prob=0.05,
            seed=seed + 100 + i,
        )
        for i in range(3)
    ]

    book = SimpleOrderBook()

    # Tracking
    pnl_history = []
    spread_history = []
    inventory_history = []
    max_inv = 0
    fill_count = 0
    adverse_sel_cost = 0.0
    peak_pnl = 0.0
    max_dd = 0.0

    dt = 1.0  # abstract time step

    for step in range(n_steps):
        # Update true price
        true_mid = price_proc.step(dt)
        true_mid_int = int(round(true_mid))

        # Seed the book with noise trader orders
        for nt in noise_agents:
            update = book.get_update()
            # Override with true mid so noise traders quote around it
            update.best_bid = max(update.best_bid, true_mid_int - 5)
            update.best_ask = min(update.best_ask, true_mid_int + 5)

            orders = nt.on_book_update(update, step)
            for order in orders:
                fills = book.add_order(order, nt.agent_id)
                for maker_agent, taker_agent, price, qty in fills:
                    fill_count += 1

        # MM cancels stale quotes then places new ones
        book.cancel_orders(mm_agent.agent_id)

        update = book.get_update()
        if update.best_bid <= 0:
            update.best_bid = true_mid_int - 3
        if update.best_ask >= 999999:
            update.best_ask = true_mid_int + 3

        orders = mm_agent.on_book_update(update, step)
        mm_fills_this_step = []

        for order in orders:
            fills = book.add_order(order, mm_agent.agent_id)
            for maker_agent, taker_agent, price, qty in fills:
                fill_count += 1

                # Process MM fills
                if maker_agent == mm_agent.agent_id:
                    # MM was maker
                    mm_order = order
                    signed_qty = qty if mm_order.side == 0 else -qty
                    mm_agent.stats.inventory += signed_qty
                    cash_flow = -price * signed_qty
                    mm_agent.stats.realized_pnl += cash_flow * tick_size
                    mm_fills_this_step.append((price, signed_qty))

                elif taker_agent == mm_agent.agent_id:
                    # MM was taker
                    signed_qty = qty if order.side == 0 else -qty
                    mm_agent.stats.inventory += signed_qty
                    cash_flow = -price * signed_qty
                    mm_agent.stats.realized_pnl += cash_flow * tick_size
                    mm_fills_this_step.append((price, signed_qty))

        # Adverse selection: how much did the price move against us after fills?
        for fill_price, signed_qty in mm_fills_this_step:
            # Use true mid as the "where price went"
            price_move = true_mid_int - fill_price
            if signed_qty > 0:  # We bought
                adverse_sel_cost += max(0, -price_move * abs(signed_qty)) * tick_size
            else:  # We sold
                adverse_sel_cost += max(0, price_move * abs(signed_qty)) * tick_size

        # Update PnL
        mm_agent.stats.unrealized_pnl = mm_agent.stats.inventory * true_mid_int * tick_size
        mm_agent.stats.total_pnl = mm_agent.stats.realized_pnl + mm_agent.stats.unrealized_pnl

        pnl_history.append(mm_agent.stats.total_pnl)
        inventory_history.append(abs(mm_agent.stats.inventory))
        max_inv = max(max_inv, abs(mm_agent.stats.inventory))

        # Track spread
        if hasattr(mm_agent, 'quote_history') and mm_agent.quote_history:
            last_quote = mm_agent.quote_history[-1]
            spread_history.append(last_quote.get("spread", 0))

        # Drawdown
        peak_pnl = max(peak_pnl, mm_agent.stats.total_pnl)
        dd = peak_pnl - mm_agent.stats.total_pnl
        max_dd = max(max_dd, dd)

    # Compute Sharpe
    if len(pnl_history) > 1:
        returns = np.diff(pnl_history)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
    else:
        sharpe = 0.0

    return MMSimulationMetrics(
        model_name=mm_agent.agent_id,
        total_pnl=mm_agent.stats.total_pnl,
        realized_pnl=mm_agent.stats.realized_pnl,
        sharpe=sharpe,
        max_drawdown=max_dd,
        avg_spread=np.mean(spread_history) if spread_history else 0.0,
        spread_std=np.std(spread_history) if spread_history else 0.0,
        max_inventory=max_inv,
        avg_abs_inventory=np.mean(inventory_history) if inventory_history else 0.0,
        total_fills=fill_count,
        adverse_selection=adverse_sel_cost,
    )


def run_mm_comparison(
    vol_models: Dict[str, object],
    n_runs: int = 50,
    n_steps: int = 10000,
    initial_price: int = 10000,
    base_seed: int = 42,
    mm_kwargs: Optional[Dict] = None,
) -> MMComparisonResults:
    """
    Compare MM performance with different vol surface models.

    Args:
        vol_models: Dict mapping model name -> trained vol surface model.
            Use None for the baseline (rolling sigma only).
        n_runs: Number of simulation runs per model.
        n_steps: Steps per simulation.
        initial_price: Starting price in ticks.
        base_seed: Base random seed (offset per run).
        mm_kwargs: Additional kwargs for AvellanedaStoikovMM.

    Returns:
        MMComparisonResults with aggregated metrics.
    """
    if mm_kwargs is None:
        mm_kwargs = {
            "gamma": 0.01,
            "kappa": 1.5,
            "max_inventory": 100,
            "num_levels": 3,
            "level_spacing": 2,
            "base_quantity": 50,
            "total_steps": n_steps,
        }

    results = MMComparisonResults()

    for model_name, model in vol_models.items():
        runs = []

        for run_idx in range(n_runs):
            seed = base_seed + run_idx * 1000

            if model is None:
                # Baseline: standard A-S with rolling sigma
                mm = AvellanedaStoikovMM(
                    agent_id=f"MM_baseline",
                    seed=seed,
                    **mm_kwargs,
                )
            else:
                # Vol-surface-aware MM
                mm = VolSurfaceAwareMM(
                    vol_surface_model=model,
                    agent_id=f"MM_{model_name}",
                    seed=seed,
                    **mm_kwargs,
                )

            metrics = run_single_simulation(
                mm_agent=mm,
                n_steps=n_steps,
                initial_price=initial_price,
                seed=seed,
            )
            metrics.model_name = model_name
            runs.append(metrics)

        results.runs[model_name] = runs

    return results
