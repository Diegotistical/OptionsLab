# src/volatility_surface/surface_generator.py

import threading
import time
import logging
from typing import Optional, Tuple, Dict
import numpy as np
from scipy.interpolate import griddata
from numba import njit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Try to import CuPy for optional GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def benchmark(enabled: bool):
    """Decorator to benchmark method execution time."""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if enabled:
                start = time.perf_counter()
                result = func(self, *args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info(f"[Benchmark] {func.__name__} took {elapsed:.4f} sec")
                return result
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

class VolatilitySurfaceGenerator:
    """
    Hybrid Numba & GPU-accelerated volatility surface generator.

    Features
    --------
    - Thread-safe single-point and batch queries
    - Hybrid LRU-style caching for repeated queries
    - Numba acceleration for nearest-neighbor fallback
    - Optional GPU acceleration via CuPy
    - Benchmark logging for performance measurement
    - Flexible interpolation: 'linear', 'cubic', 'nearest'
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
        allow_extrapolation: bool = True,
        benchmark: bool = True,
        use_numba: bool = True,
        use_gpu: bool = False
    ):
        assert len(strikes) == len(maturities) == len(ivs), "Input arrays must have same length"
        self._lock = threading.RLock()
        self.strikes = strikes
        self.maturities = maturities
        self.ivs = ivs
        self.strike_points = strike_points
        self.maturity_points = maturity_points
        self.interp_method = interp_method
        self.allow_extrapolation = allow_extrapolation
        self.benchmark = benchmark
        self.use_numba = use_numba
        self.use_gpu = use_gpu and GPU_AVAILABLE

        if use_gpu and not GPU_AVAILABLE:
            logger.warning("CuPy not available; GPU acceleration disabled")

        self.grid_strikes: Optional[np.ndarray] = None
        self.grid_maturities: Optional[np.ndarray] = None
        self.grid_ivs: Optional[np.ndarray] = None
        self._batch_cache: Dict[Tuple[float, float], float] = {}

        self._validate_inputs()

        # Pre-compile Numba function for speed
        if self.use_numba:
            dummy_query = np.array([[self.strikes[0], self.maturities[0]]])
            _numba_nearest(self.strikes, self.maturities, self.ivs, dummy_query)

    def _validate_inputs(self) -> None:
        if self.interp_method not in {"linear", "cubic", "nearest"}:
            raise ValueError(f"Interpolation method '{self.interp_method}' not supported")
        if np.any(self.ivs < 0):
            raise ValueError("Implied volatilities must be non-negative")
        if np.any(self.strikes <= 0):
            raise ValueError("Strikes must be strictly positive")
        if np.any(self.maturities < 0):
            raise ValueError("Maturities must be non-negative")

    @benchmark(enabled=True)
    def generate_surface(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate full strike-maturity IV surface as 2D grids.

        Returns
        -------
        grid_strikes : np.ndarray
            2D array of strike grid values.
        grid_maturities : np.ndarray
            2D array of maturity grid values.
        grid_ivs : np.ndarray
            2D array of interpolated implied volatilities.
        """
        with self._lock:
            self.grid_strikes = np.linspace(self.strikes.min(), self.strikes.max(), self.strike_points)
            self.grid_maturities = np.linspace(self.maturities.min(), self.maturities.max(), self.maturity_points)
            grid_strike_mesh, grid_maturity_mesh = np.meshgrid(self.grid_strikes, self.grid_maturities)
            points = np.vstack((self.strikes, self.maturities)).T
            self.grid_ivs = griddata(
                points,
                self.ivs,
                (grid_strike_mesh, grid_maturity_mesh),
                method=self.interp_method,
                fill_value=np.nan if not self.allow_extrapolation else None
            )
            return grid_strike_mesh, grid_maturity_mesh, self.grid_ivs

    def get_surface_point(self, strike: float, maturity: float) -> float:
        """Interpolate IV at a single (strike, maturity) point."""
        with self._lock:
            val = griddata(
                np.vstack((self.strikes, self.maturities)).T,
                self.ivs,
                [(strike, maturity)],
                method=self.interp_method
            )
            if val is None or np.isnan(val):
                if self.allow_extrapolation:
                    val = _numba_nearest(self.strikes, self.maturities, self.ivs, np.array([[strike, maturity]]))
                else:
                    raise ValueError(f"Point (strike={strike}, maturity={maturity}) outside interpolation domain")
            return float(val[0])

    def get_surface_batch(self, strikes: np.ndarray, maturities: np.ndarray) -> np.ndarray:
        """Batch interpolation with optional Numba/GPU acceleration."""
        with self._lock:
            queries = np.vstack((strikes, maturities)).T
            results = np.empty(len(queries), dtype=np.float64)
            new_points = []

            for i, (s, m) in enumerate(queries):
                key = (s, m)
                if key in self._batch_cache:
                    results[i] = self._batch_cache[key]
                else:
                    results[i] = np.nan
                    new_points.append(i)

            if new_points:
                new_coords = queries[new_points]
                if self.use_gpu:
                    strikes_gpu = cp.asarray(self.strikes)
                    maturities_gpu = cp.asarray(self.maturities)
                    ivs_gpu = cp.asarray(self.ivs)
                    coords_gpu = cp.asarray(new_coords)
                    new_values = _gpu_nearest(strikes_gpu, maturities_gpu, ivs_gpu, coords_gpu)
                    new_values = cp.asnumpy(new_values)
                elif self.use_numba and self.interp_method == "nearest":
                    new_values = _numba_nearest(self.strikes, self.maturities, self.ivs, new_coords)
                else:
                    new_values = griddata(
                        np.vstack((self.strikes, self.maturities)).T,
                        self.ivs,
                        new_coords,
                        method=self.interp_method
                    )
                for idx, val in zip(new_points, new_values):
                    results[idx] = val
                    self._batch_cache[(queries[idx,0], queries[idx,1])] = val

            return results

    def update_data(self, strikes: np.ndarray, maturities: np.ndarray, ivs: np.ndarray) -> None:
        """Update raw data and clear cached surface and batch cache."""
        assert len(strikes) == len(maturities) == len(ivs), "Input arrays must be same length"
        with self._lock:
            self.strikes = strikes
            self.maturities = maturities
            self.ivs = ivs
            self._validate_inputs()
            self.grid_strikes = None
            self.grid_maturities = None
            self.grid_ivs = None
            self._batch_cache.clear()
            logger.info("Data updated, cached surface and batch cache cleared.")

@njit
def _numba_nearest(strikes: np.ndarray, maturities: np.ndarray, ivs: np.ndarray, queries: np.ndarray) -> np.ndarray:
    n_queries = queries.shape[0]
    result = np.empty(n_queries, dtype=np.float64)
    for i in range(n_queries):
        q_strike, q_maturity = queries[i]
        min_dist = np.inf
        min_idx = -1
        for j in range(len(strikes)):
            dist = (strikes[j] - q_strike)**2 + (maturities[j] - q_maturity)**2
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        result[i] = ivs[min_idx]
    return result

def _gpu_nearest(strikes_gpu, maturities_gpu, ivs_gpu, coords_gpu):
    n_queries = coords_gpu.shape[0]
    result = cp.empty(n_queries, dtype=cp.float64)
    for i in range(n_queries):
        dist = (strikes_gpu - coords_gpu[i,0])**2 + (maturities_gpu - coords_gpu[i,1])**2
        min_idx = cp.argmin(dist)
        result[i] = ivs_gpu[min_idx]
    return result

# Demo benchmarking + visualization

if __name__ == "__main__":
    np.random.seed(42)
    N = 5000
    strikes = np.random.uniform(50, 150, N)
    maturities = np.random.uniform(0.01, 2.0, N)
    ivs = np.random.uniform(0.1, 0.5, N)

    gen = VolatilitySurfaceGenerator(
        strikes, maturities, ivs,
        strike_points=100, maturity_points=100,
        interp_method='nearest', use_numba=True, use_gpu=False
    )

    queries = np.column_stack((
        np.random.uniform(50, 150, 10000),
        np.random.uniform(0.01, 2.0, 10000)
    ))

    print("Benchmarking 10,000 nearest-neighbor queries...")
    t0 = time.time()
    results_cpu = gen.get_surface_batch(queries[:,0], queries[:,1])
    print("CPU Numba:", time.time() - t0)

    if GPU_AVAILABLE:
        gen.use_gpu = True
        t0 = time.time()
        results_gpu = gen.get_surface_batch(queries[:,0], queries[:,1])
        print("GPU CuPy:", time.time() - t0)

    # Generate full surface
    X, Y, Z = gen.generate_surface()

    # 3D visualization
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity")
    ax.set_zlabel("Implied Volatility")
    ax.set_title("Volatility Surface (3D)")

    # 2D heatmap visualization
    ax2 = fig.add_subplot(122)
    c = ax2.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
    fig.colorbar(c, ax=ax2, label='Implied Volatility')
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Maturity")
    ax2.set_title("Volatility Surface Heatmap (2D)")

    plt.tight_layout()
    plt.show()

