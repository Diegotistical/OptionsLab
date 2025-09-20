

import numpy as np
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False
try:
    import psutil
except ImportError:
    psutil = None

from .rng import uniform_to_normal_np, create_sobol_sampler
from .paths import generate_paths_cpu, brownian_bridge_recursive
from .payoffs import european_call_payoff, european_put_payoff, arithmetic_asian_call_payoff, arithmetic_asian_put_payoff

def determine_batch_size_for_mc(total_rows, num_simulations, path_steps, max_mem_fraction, use_gpu=False):
    dtype_size = np.float64().itemsize
    overhead = 1.5
    bytes_per_sim = dtype_size*path_steps*3*overhead
    if psutil is None:
        max_mem = 4*1024**3
    elif use_gpu and HAS_CUPY:
        max_mem = cp.get_default_memory_pool().total_bytes() or 4*1024**3
    else:
        max_mem = psutil.virtual_memory().available
    mem_per_row = num_simulations*bytes_per_sim
    batch_size = int(max_mem*max_mem_fraction/mem_per_row)
    return max(1,min(batch_size,total_rows))

def apply_control_variate(payoffs, geom_samples, geom_expect, reg, xp):
    mean_p = xp.mean(payoffs, axis=1, keepdims=True)
    mean_g = xp.mean(geom_samples, axis=1, keepdims=True)
    dev_p = payoffs - mean_p
    dev_g = geom_samples - mean_g
    cov = xp.einsum('ij,ij->i', dev_p, dev_g)/(payoffs.shape[1]-1)
    var_g = xp.var(geom_samples, axis=1)+reg
    beta = cov/var_g
    adjusted = payoffs - beta[:,None]*(geom_samples - xp.asarray(geom_expect)[:,None])
    return adjusted
