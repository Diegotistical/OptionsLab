# src/pricing_models/binomial_tree.py

import numpy as np
from typing import Optional

class BinomialTree:
    """Binomial tree option pricer for European options"""

    def __init__(self, num_steps: int = 100, seed: Optional[int] = None):
        if num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")
        self.num_steps = num_steps
        self.rng = np.random.default_rng(seed)

    def price(self, S: float, K: float, T: float, r: float, sigma: float, 
              option_type: str, q: float = 0.0) -> float:
        """Calculate European option price using the binomial tree method"""

        # Input validation
        if option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        if S <= 0 or K <= 0 or T <= 0 or sigma < 0:
            raise ValueError("Inputs must be positive and T > 0")

        dt = T / self.num_steps
        u = np.exp(sigma * np.sqrt(dt))  # up factor
        d = np.exp(-sigma * np.sqrt(dt))  # down factor
        p = (np.exp((r - q) * dt) - d) / (u - d)  # risk-neutral probability

        # Initialize asset prices at maturity
        asset_prices = np.zeros(self.num_steps + 1)
        asset_prices[0] = S * (d ** self.num_steps)
        for i in range(1, self.num_steps + 1):
            asset_prices[i] = asset_prices[i - 1] * (u / d)

        # Initialize option values at maturity
        if option_type == "call":
            option_values = np.maximum(asset_prices - K, 0.0)
        else:  # put
            option_values = np.maximum(K - asset_prices, 0.0)

        # Backward induction to calculate option price at time t=0
        for step in range(self.num_steps - 1, -1, -1):
            option_values[:-1] = (p * option_values[1:] + (1 - p) * option_values[:-1]) * np.exp(-r * dt)

        return option_values[0]