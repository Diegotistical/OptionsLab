# src/pricing_models/monte_carlo.py

import numpy as np
from typing import Optional, Literal, Callable
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from ..exceptions.montecarlo_exceptions import MonteCarloError, InputValidationError

__all__ = ['MonteCarloPricer']

class MonteCarloPricer:
    """
    Monte Carlo option pricer adapted for Exotic Options and High Volatility.
    """

    def __init__(
        self,
        num_simulations: int = 250000,
        num_steps: int = 84,
        seed: Optional[int] = None,
        use_numba: bool = False
    ):
        if num_simulations <= 0 or num_steps <= 0:
            raise InputValidationError("num_simulations and num_steps must be positive integers")
        if use_numba and not NUMBA_AVAILABLE:
            raise MonteCarloError("Numba not installed; cannot enable acceleration")

        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.rng = np.random.default_rng(seed)
        self.use_numba = use_numba

    def _validate_inputs(
        self, S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float
    ):
        # 1. ACTUALIZADO: Permite los nombres de las opciones exóticas del reto
        valid_options = {"call", "put", "ko_put", "chooser", "binary_put"}
        if option_type not in valid_options:
            raise InputValidationError(f"option_type must be one of {valid_options}")
        if S <= 0 or K <= 0 or T <= 0 or sigma < 0 or q < 0:
            raise InputValidationError("Spot, strike, T, sigma, and q must be non-negative and T > 0")

    def _clean_sigma(self, sigma: float) -> float:
        # 2. NUEVO: Auto-corrector de porcentajes. 
        # Si pones 300, lo convierte a 3.0 automáticamente para evitar el colapso matemático.
        if sigma >= 10.0:
            return sigma / 100.0
        return sigma

    def _simulate_terminal_prices_vectorized(
        self, S: float, T: float, r: float, sigma: float, q: float
    ) -> np.ndarray:
        
        # Aplicamos el filtro de porcentaje aquí
        sigma = self._clean_sigma(sigma)

        dt = T / self.num_steps
        drift = (r - q - 0.5 * sigma ** 2) * dt
        vol = sigma * np.sqrt(dt)

        rand_normals = self.rng.normal(size=(self.num_simulations, self.num_steps))
        antithetic = -rand_normals

        log_paths_pos = np.log(S) + np.cumsum(drift + vol * rand_normals, axis=1)
        log_paths_neg = np.log(S) + np.cumsum(drift + vol * antithetic, axis=1)

        # 3. ACTUALIZADO: Devuelve la matriz COMPLETA de caminos, no solo el final
        paths_pos = np.exp(log_paths_pos)
        paths_neg = np.exp(log_paths_neg)

        return np.concatenate([paths_pos, paths_neg], axis=0)

    def _simulate_terminal_prices_numba(
        self, S: float, T: float, r: float, sigma: float, q: float
    ) -> np.ndarray:
        # Nota: Numba no se actualizó para devolver el camino completo en esta versión.
        # Usa use_numba=False para evaluar las opciones exóticas.
        pass

    def _simulate_terminal_prices(
        self, S: float, T: float, r: float, sigma: float, q: float
    ) -> np.ndarray:
        if self.use_numba:
            return self._simulate_terminal_prices_numba(S, T, r, sigma, q)
        return self._simulate_terminal_prices_vectorized(S, T, r, sigma, q)

    def price(
        self, S: float, K: float, T: float, r: float, sigma: float,
        option_type: str, q: float = 0.0
    ) -> float:
        
        self._validate_inputs(S, K, T, r, sigma, option_type, q)

        # Genera los caminos completos
        paths = self._simulate_terminal_prices(S, T, r, sigma, q)
        
        # El precio final está en la última columna
        terminal_prices = paths[:, -1]

        # 4. ACTUALIZADO: Lógica de Exóticas integrada
        if option_type == "call":
            payoffs = np.maximum(terminal_prices - K, 0.0)
            
        elif option_type == "put":
            payoffs = np.maximum(K - terminal_prices, 0.0)
            
        elif option_type == "ko_put":
            # Lógica de la barrera (Lava) en 45
            hit_barrier = np.any(paths <= 45.0, axis=1)
            payoffs = np.maximum(K - terminal_prices, 0.0)
            payoffs[hit_barrier] = 0.0
            
        elif option_type == "chooser":
            # Elige entre Call y Put en el paso 56 (Día 14)
            precio_dia_14 = paths[:, 55] 
            es_call = precio_dia_14 > K
            pago_call = np.maximum(terminal_prices - K, 0.0)
            pago_put = np.maximum(K - terminal_prices, 0.0)
            payoffs = np.where(es_call, pago_call, pago_put)
            
        elif option_type == "binary_put":
            # Paga 100 fijo si termina por debajo del Strike
            payoffs = np.where(terminal_prices < K, 100.0, 0.0)

        # Descuenta el valor al presente (Aunque con r=0 no afecta)
        return np.exp(-r * T) * np.mean(payoffs)

    # (Puedes mantener los métodos de las letras Griegas igual, 
    #  solo asegúrate de que usen el sigma corregido si los llamas directamente).
    def delta(self, S, K, T, r, sigma, option_type, q=0.0, h=1e-4):
        return (self.price(S + h, K, T, r, sigma, option_type, q) -
                self.price(S - h, K, T, r, sigma, option_type, q)) / (2 * h)

    def gamma(self, S, K, T, r, sigma, option_type, q=0.0, h=1e-4):
        price_up = self.price(S + h, K, T, r, sigma, option_type, q)
        price_mid = self.price(S, K, T, r, sigma, option_type, q)
        price_down = self.price(S - h, K, T, r, sigma, option_type, q)
        return (price_up - 2 * price_mid + price_down) / (h * h)

    def vega(self, S, K, T, r, sigma, option_type, q=0.0, h=1e-4):
        sigma = self._clean_sigma(sigma) # Limpiamos antes de mutar
        return (self.price(S, K, T, r, sigma + h, option_type, q) -
                self.price(S, K, T, r, sigma - h, option_type, q)) / (2 * h)

    def theta(self, S, K, T, r, sigma, option_type, q=0.0, dt=1/365):
        if T > dt:
            return (self.price(S, K, T - dt, r, sigma, option_type, q) -
                    self.price(S, K, T, r, sigma, option_type, q)) / dt
        return -self.price(S, K, T, r, sigma, option_type, q) / dt

    def rho(self, S, K, T, r, sigma, option_type, q=0.0, h=1e-4):
        return (self.price(S, K, T, r + h, sigma, option_type, q) -
                self.price(S, K, T, r, sigma, option_type, q)) / h
