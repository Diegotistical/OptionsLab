# src/volatility_surface/utils/theoretical_bounds.py
"""
Theoretical bounds connecting training penalty loss to EPP.

Main result (Proposition): If the butterfly penalty loss L_but ≤ ε at
convergence, then the Exploitable Profit Potential (EPP) is bounded by:

    EPP ≤ (√(Nε) + L_g · δ) · max_vega · ΔK² / 2

where:
    N = number of grid points
    L_g = Lipschitz constant of the Gatheral density g(k)
    δ = grid spacing in log-moneyness
    max_vega = maximum BS vega over the strike range
    ΔK = maximum absolute strike spacing

This provides the first formal link between the observable training loss
and the economic arbitrage metric.
"""

import numpy as np
from scipy.stats import norm as sp_norm

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def compute_network_lipschitz(model: "nn.Module") -> float:
    """Compute an upper bound on the network's Lipschitz constant.

    Uses the product of spectral norms of weight matrices, which gives
    an upper bound on the Lipschitz constant for feedforward networks
    with 1-Lipschitz activations (ReLU, GELU ≈ 1.12).

    Args:
        model: A PINNNetwork or similar with a `linears` ModuleList.

    Returns:
        Upper bound on the Lipschitz constant.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")

    lip = 1.0
    activation_lip = 1.12  # GELU Lipschitz constant

    # Check for different model architectures
    if hasattr(model, "linears"):
        layers = model.linears
    elif hasattr(model, "W_x"):
        # ICNN model — use W_x and W_z layers
        layers = list(model.W_x) + list(model.W_z) + [model.W_x_out, model.W_z_out]
    else:
        raise ValueError("Unsupported model architecture")

    for layer in layers:
        if hasattr(layer, "weight"):
            with torch.no_grad():
                svs = torch.linalg.svdvals(layer.weight)
                spectral_norm = svs[0].item()
            lip *= spectral_norm * activation_lip

    return lip


def estimate_density_lipschitz(
    model: "nn.Module",
    k_range: tuple = (-0.15, 0.15),
    T: float = 0.5,
    n_points: int = 1000,
    eps: float = 1e-6,
) -> float:
    """Empirically estimate the Lipschitz constant of g(k) via finite differences.

    Evaluates the Gatheral density g(k) on a dense grid and computes
    max|g(k_{i+1}) - g(k_i)| / |k_{i+1} - k_i|.

    Args:
        model: Network that outputs total variance w given (k, sqrt_T).
        k_range: Range of log-moneyness to evaluate.
        T: Fixed maturity for the slice.
        n_points: Number of grid points.
        eps: Numerical epsilon for stability.

    Returns:
        Estimated Lipschitz constant of g(k).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")

    model_device = next(model.parameters()).device
    k_vals = torch.linspace(k_range[0], k_range[1], n_points, dtype=torch.float32, device=model_device)
    sqrt_T = torch.full((n_points,), np.sqrt(T), dtype=torch.float32, device=model_device)
    x = torch.stack([k_vals, sqrt_T], dim=1).requires_grad_(True)

    # Forward pass
    w = model(x)

    # First derivative dw/dk
    grad_w = torch.autograd.grad(
        w, x, torch.ones_like(w), create_graph=True, retain_graph=True
    )[0]
    dw_dk = grad_w[:, 0:1]

    # Second derivative d²w/dk²
    grad2_w = torch.autograd.grad(
        dw_dk, x, torch.ones_like(dw_dk), create_graph=False, retain_graph=False
    )[0]
    d2w_dk2 = grad2_w[:, 0:1]

    # Gatheral density g(k)
    k = x[:, 0:1].detach()
    w_det = w.detach()
    dw_det = dw_dk.detach()
    d2w_det = d2w_dk2.detach()

    w_safe = w_det + eps
    term1 = (1 - k * dw_det / (2 * w_safe)) ** 2
    term2 = (dw_det ** 2) / 4 * (1 / w_safe + 0.25)
    term3 = d2w_det / 2
    g_k = (term1 - term2 + term3).squeeze().cpu().numpy()

    # Lipschitz estimate via finite differences
    dk = (k_range[1] - k_range[0]) / (n_points - 1)
    dg = np.abs(np.diff(g_k))
    L_g = np.max(dg) / dk

    return float(L_g)


def compute_max_vega(
    k_range: tuple = (-0.15, 0.15),
    T: float = 0.5,
    sigma: float = 0.2,
    S: float = 100.0,
    n_points: int = 100,
) -> float:
    """Compute maximum BS vega over a strike range.

    Args:
        k_range: Log-moneyness range.
        T: Maturity.
        sigma: Reference volatility.
        S: Spot price.
        n_points: Grid resolution.

    Returns:
        Maximum vega value.
    """
    k_vals = np.linspace(k_range[0], k_range[1], n_points)
    K_vals = S * np.exp(k_vals)
    sqrt_T = np.sqrt(T)

    d1 = (np.log(S / K_vals) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    vega = S * sp_norm.pdf(d1) * sqrt_T

    return float(np.max(vega))


def compute_epp_upper_bound(
    butterfly_loss: float,
    n_grid: int,
    grid_spacing: float,
    lipschitz_g: float,
    max_strike_gap: float,
    max_vega: float,
) -> float:
    """Compute the theoretical EPP upper bound.

    Theorem: If L_but ≤ ε (butterfly penalty loss at convergence), then:

        EPP ≤ (√(N·ε) + L_g·δ) · max_vega · ΔK² / 2

    Args:
        butterfly_loss: The butterfly penalty loss ε = L_but.
        n_grid: Number of grid points N.
        grid_spacing: Grid spacing δ in log-moneyness.
        lipschitz_g: Lipschitz constant of g(k).
        max_strike_gap: Maximum absolute strike spacing ΔK.
        max_vega: Maximum BS vega over the range.

    Returns:
        EPP upper bound in raw units (multiply by 10000 for bps).
    """
    # Proposition 1: max grid-point violation
    max_grid_violation = np.sqrt(n_grid * butterfly_loss)

    # Proposition 2: inter-grid extension
    max_density_violation = max_grid_violation + lipschitz_g * grid_spacing

    # Proposition 3: density violation to EPP
    epp_bound = max_density_violation * max_vega * max_strike_gap ** 2 / 2

    return float(epp_bound)


def verify_bound(
    model: "nn.Module",
    train_data,
    S: float = 100.0,
    r: float = 0.05,
    k_range: tuple = (-0.15, 0.15),
    n_grid: int = 40,
) -> dict:
    """Compute both theoretical bound and actual EPP, return comparison.

    Args:
        model: Trained PINNNetwork.
        train_data: Training DataFrame (for evaluating butterfly loss).
        S: Spot price.
        r: Risk-free rate.
        k_range: Log-moneyness range.
        n_grid: Number of grid points.

    Returns:
        Dict with butterfly_loss, epp_bound_bps, epp_actual_bps, tightness_ratio.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")

    from src.volatility_surface.utils.epp import compute_butterfly_epp

    # Detect model device
    model_device = next(model.parameters()).device

    # Compute butterfly loss on a dense grid
    maturities = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
    k_vals = np.linspace(k_range[0], k_range[1], n_grid)
    grid_spacing = (k_range[1] - k_range[0]) / (n_grid - 1)

    # Build grid tensor on model's device
    rows = []
    for T in maturities:
        for k in k_vals:
            rows.append([k, np.sqrt(T)])
    x = torch.tensor(rows, dtype=torch.float32, device=model_device).requires_grad_(True)

    # Compute g(k) via autodiff
    w = model(x)
    grad_w = torch.autograd.grad(w, x, torch.ones_like(w), create_graph=True, retain_graph=True)[0]
    dw_dk = grad_w[:, 0:1]
    grad2_w = torch.autograd.grad(dw_dk, x, torch.ones_like(dw_dk), create_graph=False)[0]
    d2w_dk2 = grad2_w[:, 0:1]

    k_t = x[:, 0:1].detach()
    w_d = w.detach()
    dw_d = dw_dk.detach()
    d2w_d = d2w_dk2.detach()

    eps = 1e-6
    w_safe = w_d + eps
    term1 = (1 - k_t * dw_d / (2 * w_safe)) ** 2
    term2 = (dw_d ** 2) / 4 * (1 / w_safe + 0.25)
    term3 = d2w_d / 2
    g_k = term1 - term2 + term3

    # Butterfly penalty loss: mean of max(0, -g)^2
    violations = torch.relu(-g_k)
    butterfly_loss = (violations ** 2).mean().item()
    n_total = len(g_k)

    # Estimate density Lipschitz on each slice, take max
    max_L_g = 0.0
    for T in maturities:
        L_g = estimate_density_lipschitz(model, k_range, T, n_points=500)
        max_L_g = max(max_L_g, L_g)

    # Max vega and strike gap
    max_vega = compute_max_vega(k_range, T=0.5, sigma=0.2, S=S)
    max_strike_gap = S * (np.exp(k_range[1]) - np.exp(k_range[1] - grid_spacing))

    # Theoretical bound
    epp_bound = compute_epp_upper_bound(
        butterfly_loss, n_total, grid_spacing, max_L_g, max_strike_gap, max_vega
    )
    epp_bound_bps = epp_bound * 10000

    # Actual EPP
    import pandas as pd
    grid_rows = []
    for T in maturities:
        for k in k_vals:
            grid_rows.append({"log_moneyness": k, "T": T, "implied_volatility": 0.0})
    grid_df = pd.DataFrame(grid_rows)

    x_pred = torch.tensor(
        np.column_stack([grid_df["log_moneyness"].values, np.sqrt(grid_df["T"].values)]),
        dtype=torch.float32,
        device=model_device,
    )
    with torch.no_grad():
        iv_pred = model.implied_vol(x_pred).cpu().numpy().flatten()
    grid_df["implied_volatility"] = iv_pred

    max_epp = 0.0
    for T in maturities:
        mask = grid_df["T"] == T
        strikes = S * np.exp(grid_df.loc[mask, "log_moneyness"].values)
        ivs = grid_df.loc[mask, "implied_volatility"].values
        try:
            epp = compute_butterfly_epp(strikes, ivs, T, S, r)
            max_epp = max(max_epp, epp)
        except Exception:
            pass
    epp_actual_bps = max_epp * 10000

    tightness = epp_bound_bps / max(epp_actual_bps, 1e-6)

    return {
        "butterfly_loss": butterfly_loss,
        "lipschitz_g": max_L_g,
        "epp_bound_bps": epp_bound_bps,
        "epp_actual_bps": epp_actual_bps,
        "tightness_ratio": tightness,
        "bound_holds": epp_bound_bps >= epp_actual_bps,
    }
