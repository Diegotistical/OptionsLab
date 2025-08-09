# src / volatility_surface / surface_generator.py

import threading
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import logging
import time

# Configure logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VolatilitySurfaceGenerator:
    """
    Masterclass volatility surface generator with modular interpolation, thread safety,
    benchmarking hooks, and validation.

    Input:
        strikes, maturities, ivs: 1D np.ndarrays of raw option data
    """

    def __init__(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        ivs: np.ndarray,
        *,
        strike_points: int = 50,
        maturity_points: int = 50,
        interp_method: str = "cubic",
        allow_extrapolation: bool = False,
        benchmark: bool = True,
    ):
        assert len(strikes) == len(maturities) == len(ivs), "Input arrays must be same length"
        self._lock = threading.RLock()

        self.strikes = strikes
        self.maturities = maturities
        self.ivs = ivs

        self.strike_points = strike_points
        self.maturity_points = maturity_points
        self.interp_method = interp_method
        self.allow_extrapolation = allow_extrapolation
        self.benchmark = benchmark

        self.grid_strikes: Optional[np.ndarray] = None
        self.grid_maturities: Optional[np.ndarray] = None
        self.grid_ivs: Optional[np.ndarray] = None

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.interp_method not in {"linear", "cubic", "nearest"}:
            raise ValueError(f"Interpolation method '{self.interp_method}' not supported")
        if np.any(self.ivs < 0):
            raise ValueError("Implied volatilities must be non-negative")
        if np.any(self.strikes <= 0):
            raise ValueError("Strikes must be strictly positive")
        if np.any(self.maturities < 0):
            raise ValueError("Maturities must be non-negative (time to expiry)")

    def _start_benchmark(self) -> float:
        return time.perf_counter() if self.benchmark else 0.0

    def _end_benchmark(self, start: float, action: str) -> None:
        if self.benchmark:
            duration = time.perf_counter() - start
            logger.info(f"[VolSurfaceGen] {action} took {duration:.4f} seconds")

    def generate_surface(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Produces a meshgrid of strikes and maturities with interpolated implied volatilities.

        Returns:
            grid_strikes, grid_maturities, grid_ivs -- all 2D np.ndarrays shaped (maturity_points, strike_points)
        """

        with self._lock:
            start = self._start_benchmark()

            strike_min, strike_max = self.strikes.min(), self.strikes.max()
            maturity_min, maturity_max = self.maturities.min(), self.maturities.max()

            self.grid_strikes = np.linspace(strike_min, strike_max, self.strike_points)
            self.grid_maturities = np.linspace(maturity_min, maturity_max, self.maturity_points)
            grid_strike_mesh, grid_maturity_mesh = np.meshgrid(self.grid_strikes, self.grid_maturities)

            points = np.vstack((self.strikes, self.maturities)).T
            values = self.ivs

            # Perform interpolation
            grid_ivs = griddata(
                points,
                values,
                (grid_strike_mesh, grid_maturity_mesh),
                method=self.interp_method,
                fill_value=np.nan if not self.allow_extrapolation else None,
            )

            # Optionally handle extrapolation or mask invalid points
            if not self.allow_extrapolation:
                # Mask points outside convex hull of input data
                mask = np.isnan(grid_ivs)
                if np.any(mask):
                    logger.warning(f"Extrapolation detected: masking {np.sum(mask)} points in grid")

            self.grid_ivs = grid_ivs

            self._end_benchmark(start, "Surface generation")

            return grid_strike_mesh, grid_maturity_mesh, self.grid_ivs

    def get_surface_point(self, strike: float, maturity: float) -> float:
        """
        Interpolates IV at a single (strike, maturity) coordinate.

        Thread-safe and fast.

        Raises ValueError if point outside data range and extrapolation disabled.
        """

        with self._lock:
            points = np.vstack((self.strikes, self.maturities)).T
            val = griddata(points, self.ivs, [(strike, maturity)], method=self.interp_method)
            if val is None or np.isnan(val):
                if self.allow_extrapolation:
                    # fallback to nearest neighbor
                    val = griddata(points, self.ivs, [(strike, maturity)], method="nearest")
                else:
                    raise ValueError(f"Requested point (strike={strike}, maturity={maturity}) is outside interpolation domain")
            return float(val[0])

    def update_data(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        ivs: np.ndarray,
    ) -> None:
        """
        Thread-safe update of underlying raw data points.

        Requires input arrays of equal length.
        """

        assert len(strikes) == len(maturities) == len(ivs), "Input arrays must be same length"

        with self._lock:
            self.strikes = strikes
            self.maturities = maturities
            self.ivs = ivs
            self._validate_inputs()

            # Reset cached surface
            self.grid_strikes = None
            self.grid_maturities = None
            self.grid_ivs = None
            logger.info("VolatilitySurfaceGenerator data updated, cached surface cleared.")

