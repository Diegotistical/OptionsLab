import numpy as np
import logging
logger = logging.getLogger(__name__)

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    njit = lambda f: f
    HAS_NUMBA = False

def generate_paths_cpu(S_b, T_b, r_b, sigma_b, q_b, BM, times):
    @njit if HAS_NUMBA else lambda f: f
    def inner():
        b, sims, steps = BM.shape
        S_paths = np.empty((b, sims, steps))
        for i in range(b):
            drift_base = (r_b[i] - q_b[i] - 0.5*sigma_b[i]**2) * T_b[i]
            vol_base = sigma_b[i]
            S0 = S_b[i]
            for j in range(sims):
                for k in range(steps):
                    drift = drift_base * times[k]
                    vol = vol_base * BM[i,j,k]
                    S_paths[i,j,k] = S0*np.exp(drift+vol)
        return S_paths
    return inner()

def brownian_bridge_recursive(Z, times, xp):
    if len(times) > 1024:
        logger.warning("High steps in BB; perf may degrade")
    b,sims,steps = Z.shape
    BM = xp.zeros((b,sims,steps), dtype=xp.float64)
    dt = xp.diff(times, prepend=0)
    BM[..., -1] = xp.cumsum(xp.sqrt(dt)*Z, axis=-1)[..., -1]
    level = 1
    while level < steps:
        for start in range(0, steps, 2*level):
            left = start
            mid = start + level
            right = start + 2*level
            if right >= steps: break
            t_l, t_m, t_r = times[left], times[mid], times[right]
            a = (t_r - t_m)/(t_r - t_l)
            b_ = (t_m - t_l)/(t_r - t_l)
            std = xp.sqrt((t_m - t_l)*(t_r - t_m)/(t_r - t_l))
            BM[...,mid] = a*BM[...,left] + b_*BM[...,right] + std*Z[...,mid]
        level *= 2
    return BM
