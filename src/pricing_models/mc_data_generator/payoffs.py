import numpy as np

def european_call_payoff(S_T: np.ndarray, K: np.ndarray) -> np.ndarray:
    ST = S_T[:, :, -1] if S_T.ndim == 3 else S_T
    return np.maximum(ST - K[:, None], 0.0)

def european_put_payoff(S_T: np.ndarray, K: np.ndarray) -> np.ndarray:
    ST = S_T[:, :, -1] if S_T.ndim == 3 else S_T
    return np.maximum(K[:, None] - ST, 0.0)

def arithmetic_asian_call_payoff(S_paths: np.ndarray, K: np.ndarray) -> np.ndarray:
    avg = S_paths.mean(axis=2)
    return np.maximum(avg - K[:, None], 0.0)

def arithmetic_asian_put_payoff(S_paths: np.ndarray, K: np.ndarray) -> np.ndarray:
    avg = S_paths.mean(axis=2)
    return np.maximum(K[:, None] - avg, 0.0)
