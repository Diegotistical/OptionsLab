# src/optimization/reproducibility.py
"""
Reproducibility infrastructure for deterministic optimization.

Provides:
- Global seed management across NumPy, PyTorch, Python, LightGBM
- Per-trial deterministic seeding
- Thread/resource control to prevent contention
- GPU determinism controls for CUDA

Usage:
    from src.optimization.reproducibility import set_global_seed, set_thread_limits

    set_global_seed(42)
    set_thread_limits(n_jobs=4, omp_threads=4, torch_threads=4)
"""

import hashlib
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Lazy imports for optional dependencies
_torch_available = None
_lgb_available = None


def _check_torch() -> bool:
    global _torch_available
    if _torch_available is None:
        try:
            import torch

            _torch_available = True
        except ImportError:
            _torch_available = False
    return _torch_available


def _check_lightgbm() -> bool:
    global _lgb_available
    if _lgb_available is None:
        try:
            import lightgbm

            _lgb_available = True
        except ImportError:
            _lgb_available = False
    return _lgb_available


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducibility settings."""

    seed: int = 42
    n_jobs: int = 1
    omp_threads: int = 4
    torch_threads: int = 4
    torch_deterministic: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False

    # LightGBM specific
    lgb_deterministic: bool = True
    lgb_force_col_wise: bool = True

    def apply(self) -> None:
        """Apply all reproducibility settings."""
        set_global_seed(self.seed)
        set_thread_limits(
            n_jobs=self.n_jobs,
            omp_threads=self.omp_threads,
            torch_threads=self.torch_threads,
        )
        if _check_torch():
            set_torch_determinism(
                strict=self.torch_deterministic,
                cudnn_deterministic=self.cudnn_deterministic,
                cudnn_benchmark=self.cudnn_benchmark,
            )


def set_global_seed(seed: int) -> None:
    """
    Set global random seed across all relevant libraries.

    Affects:
        - Python's random module
        - NumPy's random generator
        - PyTorch (CPU and CUDA if available)
        - LightGBM (via environment variable)

    Args:
        seed: Integer seed value for reproducibility.
    """
    # Python stdlib
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    if _check_torch():
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # Multi-GPU

    # LightGBM seed is set per-model, but we set env for safety
    os.environ["LIGHTGBM_SEED"] = str(seed)

    # For hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_trial_seed(study_seed: int, trial_number: int) -> int:
    """
    Generate deterministic seed for a specific trial.

    Ensures trial N always gets the same seed regardless of execution order.
    Uses cryptographic hashing for uniform distribution.

    Args:
        study_seed: Base seed for the study.
        trial_number: Trial index (0-based).

    Returns:
        Deterministic seed for this trial.
    """
    # Use SHA256 for uniform distribution
    hash_input = f"{study_seed}:{trial_number}".encode()
    hash_digest = hashlib.sha256(hash_input).hexdigest()

    # Take first 8 hex chars (32 bits) for seed
    return int(hash_digest[:8], 16)


def set_thread_limits(
    n_jobs: int = 1,
    omp_threads: Optional[int] = None,
    torch_threads: Optional[int] = None,
    mkl_threads: Optional[int] = None,
) -> None:
    """
    Set thread limits to prevent resource contention.

    Critical for parallel Optuna trials - without this, threads will
    oversubscribe and performance collapses.

    Args:
        n_jobs: Number of parallel jobs (for joblib/sklearn).
        omp_threads: OpenMP thread count (affects NumPy, LightGBM).
        torch_threads: PyTorch thread count (intra-op parallelism).
        mkl_threads: Intel MKL thread count (affects NumPy on Intel).
    """
    omp_threads = omp_threads or n_jobs
    mkl_threads = mkl_threads or n_jobs

    # OpenMP (affects NumPy BLAS, LightGBM, XGBoost)
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)

    # Intel MKL (NumPy on Intel CPUs)
    os.environ["MKL_NUM_THREADS"] = str(mkl_threads)

    # OpenBLAS
    os.environ["OPENBLAS_NUM_THREADS"] = str(omp_threads)

    # Numexpr
    os.environ["NUMEXPR_NUM_THREADS"] = str(omp_threads)

    # PyTorch
    if _check_torch() and torch_threads is not None:
        import torch

        torch.set_num_threads(torch_threads)
        torch.set_num_interop_threads(max(1, torch_threads // 2))


def set_torch_determinism(
    strict: bool = True,
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False,
) -> None:
    """
    Configure PyTorch for deterministic operations.

    Warning:
        Enabling strict determinism has performance implications:
        - Some operations become slower
        - Some operations may error if no deterministic implementation exists

    Args:
        strict: If True, use torch.use_deterministic_algorithms(True).
        cudnn_deterministic: If True, set cudnn.deterministic = True.
        cudnn_benchmark: If False, disable cudnn autotuning (recommended for reproducibility).
    """
    if not _check_torch():
        return

    import torch

    if strict:
        # This will error on non-deterministic ops
        torch.use_deterministic_algorithms(True, warn_only=False)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


def get_lightgbm_deterministic_params(seed: int) -> dict:
    """
    Get LightGBM parameters for deterministic training.

    Args:
        seed: Random seed.

    Returns:
        Dict of parameters to pass to LightGBM.
    """
    return {
        "seed": seed,
        "deterministic": True,
        "force_col_wise": True,
        "force_row_wise": False,
        "num_threads": 1,  # Single-threaded for determinism
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "drop_seed": seed,
    }


def get_cv_split_generator(seed: int):
    """
    Get a seeded CV split generator.

    Use this to ensure CV splits are deterministic across runs.

    Args:
        seed: Random seed for splits.

    Returns:
        Numpy RandomGenerator for CV splits.
    """
    return np.random.default_rng(seed)


def compute_data_hash(data: np.ndarray, truncate: int = 8) -> str:
    """
    Compute a hash of data for tracking/versioning.

    Args:
        data: NumPy array to hash.
        truncate: Number of hex characters to return.

    Returns:
        Truncated SHA256 hash of data.
    """
    # Use array bytes for consistent hashing
    data_bytes = data.tobytes()
    hash_digest = hashlib.sha256(data_bytes).hexdigest()
    return hash_digest[:truncate]
