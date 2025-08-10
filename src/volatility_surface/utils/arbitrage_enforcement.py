# src / volatility_surface / utils / arbitrage_enforcement.py

import numpy as np
import logging
from typing import Dict, Any


logger = logging.getLogger(__name__)

class ArbitrageEnforcementError(Exception):
    """Custom exception for arbitrage enforcement issues."""
    pass

def check_monotonicity(y: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Check if `y` is monotone non-decreasing along specified axis.
    
    Args:
        y (np.ndarray): Input array (surface).
        axis (int): Axis along which to check monotonicity.
        
    Returns:
        np.ndarray: Boolean mask of shape same as `y` where True indicates monotonicity violation.
    
    Raises:
        ValueError: If input is not a numpy ndarray or axis is invalid.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError(f"Input must be np.ndarray, got {type(y)}")
    if axis < 0 or axis >= y.ndim:
        raise ValueError(f"Invalid axis {axis} for array with ndim {y.ndim}")
    
    diffs = np.diff(y, axis=axis)
    violations = diffs < 0  # decreasing points
    
    # Pad violations to original shape
    pad_shape = list(violations.shape)
    pad_shape[axis] = 1
    violations_padded = np.concatenate([violations, np.zeros(pad_shape, dtype=bool)], axis=axis)
    return violations_padded

def _project_to_monotone(seq: np.ndarray) -> np.ndarray:
    """
    Pool Adjacent Violators Algorithm (PAVA) to enforce isotonic regression (monotone non-decreasing).
    
    Args:
        seq (np.ndarray): 1D numeric array.
    
    Returns:
        np.ndarray: Monotonically non-decreasing array closest to `seq` under L2 norm.
    """
    n = len(seq)
    y = seq.copy()
    weights = np.ones(n)
    i = 0
    while i < n - 1:
        if y[i] > y[i + 1]:
            total_weight = weights[i] + weights[i + 1]
            avg = (weights[i] * y[i] + weights[i + 1] * y[i + 1]) / total_weight
            y[i] = avg
            y[i + 1] = avg
            weights[i] = total_weight
            
            j = i - 1
            while j >= 0 and y[j] > y[j + 1]:
                total_weight = weights[j] + weights[j + 1]
                avg = (weights[j] * y[j] + weights[j + 1] * y[j + 1]) / total_weight
                y[j] = avg
                y[j + 1] = avg
                weights[j] = total_weight
                j -= 1
            i = max(j, 0)
        else:
            i += 1
    return y

def enforce_monotonicity(y: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Enforce monotonicity along a given axis by projecting each slice onto monotone sequences.
    
    Args:
        y (np.ndarray): Input array (surface).
        axis (int): Axis along which to enforce monotonicity.
    
    Returns:
        np.ndarray: Corrected array with enforced monotonicity.
    
    Raises:
        ArbitrageEnforcementError: If input dimensions are unsupported.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError(f"Input must be np.ndarray, got {type(y)}")
    if y.ndim < 1:
        raise ArbitrageEnforcementError("Input array must be at least 1-dimensional")
    
    y_corrected = y.copy()
    
    # Apply PAVA along specified axis for each slice
    # For now, only support 2D arrays with axis=1 (strikes) for simplicity and performance
    if y.ndim == 2 and axis == 1:
        for i in range(y.shape[0]):
            y_corrected[i, :] = _project_to_monotone(y_corrected[i, :])
    else:
        raise NotImplementedError("Only 2D arrays with axis=1 supported currently")
    
    return y_corrected

def _project_to_convex(seq: np.ndarray, max_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
    """
    Project sequence onto convex set by enforcing non-negative second discrete differences.
    
    Args:
        seq (np.ndarray): 1D numeric array.
        max_iter (int): Maximum iterations for correction.
        tol (float): Tolerance for convergence.
    
    Returns:
        np.ndarray: Convex sequence closest to `seq`.
    """
    y = seq.copy()
    for iteration in range(max_iter):
        diff2 = np.diff(y, n=2)
        violations = diff2 < 0
        if not np.any(violations):
            break
        
        # Fix violating points
        for idx in np.where(violations)[0]:
            y[idx + 1] = (y[idx] + y[idx + 2]) / 2
        
        # Early exit if corrections below tolerance
        if np.max(np.abs(np.diff(y, n=2))) < tol:
            break
    else:
        logger.warning("Max iterations reached in convexity projection without full convergence")
    return y

def enforce_convexity(y: np.ndarray, axis: int = 1, max_iter: int = 50) -> np.ndarray:
    """
    Enforce convexity along given axis by projecting each slice onto convex sequences.
    
    Args:
        y (np.ndarray): Input array (surface).
        axis (int): Axis along which to enforce convexity.
        max_iter (int): Maximum iterations per projection.
    
    Returns:
        np.ndarray: Corrected array with enforced convexity.
    
    Raises:
        ArbitrageEnforcementError: If input dimensions are unsupported.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError(f"Input must be np.ndarray, got {type(y)}")
    if y.ndim < 1:
        raise ArbitrageEnforcementError("Input array must be at least 1-dimensional")
    
    y_corrected = y.copy()
    
    if y.ndim == 2 and axis == 1:
        for i in range(y.shape[0]):
            y_corrected[i, :] = _project_to_convex(y_corrected[i, :], max_iter=max_iter)
    else:
        raise NotImplementedError("Only 2D arrays with axis=1 supported currently")
    
    return y_corrected


def detect_arbitrage_violations(surface: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Detect arbitrage violations on a 2D volatility surface.
    
    Args:
        surface (np.ndarray): 2D volatility surface array (maturities × strikes).
    
    Returns:
        Dict[str, np.ndarray]: Dictionary with boolean masks indicating violations.
            Keys: 'monotonicity_strike', 'convexity_strike'
    """
    if not isinstance(surface, np.ndarray):
        raise TypeError(f"Input must be np.ndarray, got {type(surface)}")
    if surface.ndim != 2:
        raise ValueError("Surface must be 2D array (maturities × strikes)")
    
    monotonicity_violations = check_monotonicity(surface, axis=1)
    convexity_violations = np.diff(surface, n=2, axis=1) < 0
    
    return {
        'monotonicity_strike': monotonicity_violations,
        'convexity_strike': convexity_violations,
    }


def correct_arbitrage(surface: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """
    Corrects arbitrage violations by enforcing monotonicity and convexity on volatility surface.
    
    Args:
        surface (np.ndarray): 2D volatility surface array.
        max_iter (int): Max iterations for convexity enforcement.
    
    Returns:
        np.ndarray: Corrected surface array.
    """
    logger.info("Starting arbitrage correction...")
    corrected = enforce_monotonicity(surface, axis=1)
    corrected = enforce_convexity(corrected, axis=1, max_iter=max_iter)
    logger.info("Arbitrage correction completed.")
    return corrected

# Example standalone test block
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(0)
    maturities = 10
    strikes = 20
    base_surface = np.linspace(0.15, 0.3, strikes)
    surface = np.tile(base_surface, (maturities, 1))
    noise = np.random.normal(scale=0.01, size=surface.shape)
    noisy_surface = surface + noise

    violations = detect_arbitrage_violations(noisy_surface)
    print(f"Monotonicity violations: {np.sum(violations['monotonicity_strike'])}")
    print(f"Convexity violations: {np.sum(violations['convexity_strike'])}")

    corrected_surface = correct_arbitrage(noisy_surface)

    # Optional visualization to verify
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(noisy_surface, aspect='auto', cmap='viridis')
    axs[0].set_title("Noisy Surface")
    axs[1].imshow(corrected_surface, aspect='auto', cmap='viridis')
    axs[1].set_title("Corrected Surface")
    plt.show()
        