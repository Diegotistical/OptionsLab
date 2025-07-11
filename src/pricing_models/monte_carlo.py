# src/pricing_models/monte_carlo.py

import numpy as np
from typing import Optional

class MonteCarloPricer:
    """Monte Carlo option pricer using antithetic variance reduction and geometric Brownian motion"""

    def __init__(self, num_simulations: int = 10000, num_steps: int = 100, seed: Optional[int] = None):
        if num_simulations <= 0 or num_steps <= 0:
            raise ValueError("num_simulations and num_steps must be positive integers")
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.rng = np.random.default_rng(seed)

    def _simulate_terminal_prices(self, S: float, T: float, r: float, sigma: float, q: float) -> np.ndarray:
        dt = T / self.num_steps
        drift = (r - q - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)

        # Generate normal random variables
        rand_normals = self.rng.normal(size=(self.num_simulations, self.num_steps))
        antithetic = -rand_normals

        increments_pos = drift + vol * rand_normals
        increments_neg = drift + vol * antithetic

        log_paths_pos = np.log(S) + np.cumsum(increments_pos, axis=1)
        log_paths_neg = np.log(S) + np.cumsum(increments_neg, axis=1)

        terminal_prices_pos = np.exp(log_paths_pos[:, -1])
        terminal_prices_neg = np.exp(log_paths_neg[:, -1])

        # Concatenate both
        return np.concatenate([terminal_prices_pos, terminal_prices_neg])

    def price(self, S: float, K: float, T: float, r: float, sigma: float, 
              option_type: str, q: float = 0.0) -> float:
        """Calculate European option price using Monte Carlo simulation with variance reduction"""

        # Input validation
        if option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        if S <= 0 or K <= 0 or T <= 0 or sigma < 0:
            raise ValueError("Inputs must be positive and T > 0")

        terminal_prices = self._simulate_terminal_prices(S, T, r, sigma, q)

        if option_type == "call":
            payoffs = np.maximum(terminal_prices - K, 0.0)
        else:  # put
            payoffs = np.maximum(K - terminal_prices, 0.0)

        discounted_payoff = np.exp(-r * T) * np.mean(payoffs)
        return discounted_payoff
