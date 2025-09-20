import logging
import numpy as np
from typing import Optional

try:
    from scipy.stats import qmc
    HAS_SOBOL = True
except ImportError:
    qmc = None
    HAS_SOBOL = False

logger = logging.getLogger(__name__)

def uniform_to_normal_np(u: np.ndarray) -> np.ndarray:
    """Convert uniform [0,1] to standard normal using erfinv or Box-Muller fallback."""
    try:
        return np.sqrt(2.0) * np.erfinv(2 * u - 1)
    except Exception:
        b, m = u.shape
        if m % 2 != 0:
            u = np.concatenate([u, np.random.random((b, 1))], axis=1)
            b, m = u.shape
        u1 = u[:, 0::2]
        u2 = u[:, 1::2]
        R = np.sqrt(-2.0 * np.log(np.clip(u1, 1e-16, 1 - 1e-16)))
        Theta = 2 * np.pi * u2
        z1 = R * np.cos(Theta)
        z2 = R * np.sin(Theta)
        return np.concatenate([z1, z2], axis=1)[:, :m]

def create_sobol_sampler(dims: int, seed: Optional[int], scramble: bool = True) -> Optional[qmc.Sobol]:
    """Create Sobol sampler if available; warns for high dims (>40)."""
    if HAS_SOBOL:
        if dims > 40:
            logger.warning(f"Sobol dims={dims} high; quality may degrade")
        return qmc.Sobol(d=dims, scramble=scramble, seed=seed)
    return None
