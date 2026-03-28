"""
Generate publication-quality figures for the CINN volatility surface paper.

Produces:
  Figure 1: Fitted volatility surface comparison (CINN vs MLP vs ground truth)
  Figure 2: Training convergence with constraint violations (dual-axis)
  Figure 3: Arb-free percentage evolution across training epochs (ablation)

All figures saved as PDF to docs/research/figures/ for LaTeX inclusion.

Usage:
    python scripts/generate_figures.py            # all figures
    python scripts/generate_figures.py --fig 1    # single figure
    python scripts/generate_figures.py --quick     # faster (fewer epochs)
"""

import os
import sys
import copy
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.vol_surface_benchmark import generate_synthetic_surface
from src.volatility_surface.models.pinn_model import PINNVolatilityModel

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

# Colorblind-friendly palette (Okabe-Ito)
COLORS = {
    "cinn": "#0072B2",       # blue
    "mlp": "#D55E00",        # vermilion
    "truth": "#009E73",      # bluish green
    "no_warmup": "#CC79A7",  # reddish purple
    "no_ema": "#E69F00",     # orange
    "mse": "#0072B2",
    "calendar": "#D55E00",
    "butterfly": "#009E73",
    "total": "#222222",
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "docs", "research", "figures")


def setup_style():
    """Configure matplotlib for publication-quality output."""
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        try:
            plt.style.use("seaborn-paper")
        except OSError:
            pass  # fall back to default

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "text.usetex": False,
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_eval_grid(df):
    """Build a dense evaluation grid spanning the training data range."""
    k_min, k_max = df["log_moneyness"].min(), df["log_moneyness"].max()
    T_vals = sorted(df["T"].unique())
    n_k = 60
    k_grid = np.linspace(k_min, k_max, n_k)
    rows = []
    for T in T_vals:
        for k in k_grid:
            rows.append({"log_moneyness": k, "T": T})
    return pd.DataFrame(rows), k_grid, T_vals


def ground_truth_iv(k, T, base_vol=0.2, term_slope=-0.02):
    """Reproduce the analytic ground truth from generate_synthetic_surface (no noise)."""
    atm_vol = max(base_vol + term_slope * np.log(T), 0.05)
    skew = -0.25 / np.sqrt(T)
    smile_coeff = 0.04
    sigma = atm_vol + skew * k + smile_coeff * k ** 2
    return np.maximum(sigma, 0.01)


def train_model(epochs, quick=False, **kwargs):
    """Train a PINNVolatilityModel and return (model, df)."""
    n_strikes = 30
    df = generate_synthetic_surface(n_strikes=n_strikes, seed=42)
    model = PINNVolatilityModel(
        epochs=epochs,
        hidden_layers=[64, 32, 16],
        **kwargs,
    )
    model.train(df)
    return model, df


# ---------------------------------------------------------------------------
# Figure 1 -- Fitted Volatility Surfaces
# ---------------------------------------------------------------------------

def figure1(quick=False):
    """
    Three heatmap subplots (ground truth, CINN, MLP) + error subplots.
    """
    print("[Figure 1] Training CINN and MLP models ...")
    epochs = 200 if not quick else 80

    # --- CINN (constrained) ---
    cinn, df = train_model(
        epochs=epochs,
        lambda_calendar=1.0, lambda_butterfly=0.5, lambda_wing=0.1,
        use_warmup=True, squared_hinge=True, use_residual=True, activation="gelu",
        use_ema_norm=True,
    )

    # --- MLP (unconstrained, same architecture) ---
    mlp, _ = train_model(
        epochs=epochs,
        lambda_calendar=0.0, lambda_butterfly=0.0, lambda_wing=0.0,
        use_warmup=False, use_residual=True, activation="gelu",
        use_ema_norm=False,
    )

    # Build evaluation grid
    grid_df, k_grid, T_vals = build_eval_grid(df)
    n_k = len(k_grid)
    n_T = len(T_vals)

    # Ground truth (noise-free)
    gt_iv = np.array([ground_truth_iv(row["log_moneyness"], row["T"])
                       for _, row in grid_df.iterrows()])
    cinn_iv = cinn.predict_volatility(grid_df)
    mlp_iv = mlp.predict_volatility(grid_df)

    # Reshape for contourf: shape (n_T, n_k)
    GT = gt_iv.reshape(n_T, n_k)
    CINN = cinn_iv.reshape(n_T, n_k)
    MLP = mlp_iv.reshape(n_T, n_k)

    K, T = np.meshgrid(k_grid, T_vals)

    # Shared colour limits for surface plots
    vmin = min(GT.min(), CINN.min(), MLP.min())
    vmax = max(GT.max(), CINN.max(), MLP.max())

    # Error maps
    err_cinn = np.abs(CINN - GT)
    err_mlp = np.abs(MLP - GT)
    err_vmax = max(err_cinn.max(), err_mlp.max())

    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.0, 0.85], hspace=0.35, wspace=0.30)

    titles_top = ["(a) Ground Truth", "(b) CINN Fit", "(c) MLP Fit"]
    data_top = [GT, CINN, MLP]
    for idx in range(3):
        ax = fig.add_subplot(gs[0, idx])
        cf = ax.contourf(K, T, data_top[idx], levels=25, cmap="viridis",
                         vmin=vmin, vmax=vmax)
        ax.set_xlabel("Log-moneyness $k$")
        if idx == 0:
            ax.set_ylabel("Maturity $T$ (years)")
        ax.set_title(titles_top[idx])
        plt.colorbar(cf, ax=ax, label="$\\sigma_{\\mathrm{iv}}$" if idx == 2 else "")

    titles_bot = ["(d) |CINN Error|", "(e) |MLP Error|"]
    data_bot = [err_cinn, err_mlp]
    for idx in range(2):
        ax = fig.add_subplot(gs[1, idx])
        cf = ax.contourf(K, T, data_bot[idx], levels=20, cmap="Reds",
                         vmin=0, vmax=err_vmax)
        ax.set_xlabel("Log-moneyness $k$")
        if idx == 0:
            ax.set_ylabel("Maturity $T$ (years)")
        ax.set_title(titles_bot[idx])
        plt.colorbar(cf, ax=ax, label="|Error|" if idx == 1 else "")

    # Summary stats text box
    ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.axis("off")
    cinn_rmse = np.sqrt(np.mean((CINN - GT) ** 2))
    mlp_rmse = np.sqrt(np.mean((MLP - GT) ** 2))
    cinn_max = err_cinn.max()
    mlp_max = err_mlp.max()
    txt = (
        f"RMSE\n"
        f"  CINN: {cinn_rmse:.4f}\n"
        f"  MLP:  {mlp_rmse:.4f}\n\n"
        f"Max |Error|\n"
        f"  CINN: {cinn_max:.4f}\n"
        f"  MLP:  {mlp_max:.4f}\n\n"
        f"Epochs: {epochs}\n"
        f"Train pts: {len(df)}"
    )
    ax_txt.text(0.1, 0.9, txt, transform=ax_txt.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax_txt.set_title("(f) Summary")

    path = os.path.join(FIG_DIR, "fig1_surface_fit.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 2 -- Training Convergence with Constraint Violations
# ---------------------------------------------------------------------------

def figure2(quick=False):
    """
    Dual-axis plot: MSE loss (log scale) vs constraint violation magnitude.
    Vertical shading for warmup/ramp regions.
    """
    print("[Figure 2] Training CINN for convergence plot ...")
    epochs = 500 if not quick else 200

    model, df = train_model(
        epochs=epochs,
        lambda_calendar=1.0, lambda_butterfly=0.5, lambda_wing=0.1,
        use_warmup=True, squared_hinge=True, use_residual=True,
        activation="gelu", use_ema_norm=True,
    )

    hist = model.training_history
    ep = np.array([h["epoch"] for h in hist])
    mse = np.array([h["mse"] for h in hist])
    cal = np.array([h["calendar"] for h in hist])
    but = np.array([h["butterfly"] for h in hist])
    total = np.array([h["total"] for h in hist])

    constraint_mag = cal + but  # combined constraint violation

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # Left axis -- losses (log scale)
    l_mse, = ax1.semilogy(ep, mse, color=COLORS["mse"], linewidth=1.5, label="MSE loss")
    l_tot, = ax1.semilogy(ep, total, color=COLORS["total"], linewidth=1.0,
                           linestyle="--", alpha=0.7, label="Total loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (log scale)", color=COLORS["mse"])
    ax1.tick_params(axis="y", labelcolor=COLORS["mse"])

    # Right axis -- constraint violations
    ax2 = ax1.twinx()
    l_cal, = ax2.plot(ep, cal, color=COLORS["calendar"], linewidth=1.2,
                       alpha=0.85, label="Calendar penalty")
    l_but, = ax2.plot(ep, but, color=COLORS["butterfly"], linewidth=1.2,
                       alpha=0.85, label="Butterfly penalty")
    ax2.set_ylabel("Constraint penalty magnitude")
    ax2.tick_params(axis="y")

    # Warmup shading
    warmup_end = min(100, epochs)
    ramp_end = min(300, epochs)
    ax1.axvspan(0, warmup_end, alpha=0.08, color="gray", label="_nolegend_")
    ax1.axvspan(warmup_end, ramp_end, alpha=0.05, color="blue", label="_nolegend_")
    ax1.axvline(warmup_end, color="gray", linestyle=":", linewidth=0.8)
    ax1.axvline(ramp_end, color="gray", linestyle=":", linewidth=0.8)

    # Annotations
    y_top = ax1.get_ylim()[1]
    ax1.annotate("Warmup\n($\\lambda=0$)", xy=(warmup_end / 2, y_top),
                 fontsize=8, ha="center", va="top", color="gray")
    if ramp_end > warmup_end + 10:
        ax1.annotate("Ramp", xy=((warmup_end + ramp_end) / 2, y_top),
                     fontsize=8, ha="center", va="top", color="steelblue")

    # Checkpoint marker -- find epoch where best feasible checkpoint was saved
    # Approximate: first epoch >= checkpoint_start where constraint_mag drops below 1e-6
    checkpoint_start = 300 if epochs > 300 else epochs
    ckpt_epoch = None
    for i, e in enumerate(ep):
        if e >= checkpoint_start and constraint_mag[i] < 1e-4:
            ckpt_epoch = int(e)
            break
    if ckpt_epoch is not None:
        idx = list(ep).index(ckpt_epoch)
        ax1.axvline(ckpt_epoch, color="green", linestyle="-.", linewidth=0.9, alpha=0.7)
        ax1.annotate(f"Checkpoint\n(ep {ckpt_epoch})",
                     xy=(ckpt_epoch, mse[idx]), fontsize=7,
                     xytext=(ckpt_epoch + epochs * 0.05, mse[idx]),
                     arrowprops=dict(arrowstyle="->", color="green", lw=0.8),
                     color="green")

    # Combined legend
    lines = [l_mse, l_tot, l_cal, l_but]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=8, framealpha=0.9)

    ax1.set_xlim(0, epochs)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_convergence.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 3 -- Arb-Free % Across Training Epochs (Ablation Variants)
# ---------------------------------------------------------------------------

def _compute_arb_free_pct(model, df):
    """Compute % of expiry slices that are calendar-arbitrage-free."""
    import torch
    if model.model is None:
        return 0.0

    maturities = sorted(df["T"].unique())
    n_slices = len(maturities)
    arb_free_count = 0

    model.model.eval()
    for T_val in maturities:
        slice_df = df[df["T"] == T_val]
        if len(slice_df) < 3:
            arb_free_count += 1
            continue

        X = slice_df[model.feature_columns].values.astype(np.float32)
        X_scaled = X.copy()
        X_scaled[:, 1] = np.sqrt(np.abs(X_scaled[:, 1]))

        X_t = torch.tensor(X_scaled, device=model.device, requires_grad=True)
        w = model.model(X_t)

        grad_w = torch.autograd.grad(
            outputs=w.sum(), inputs=X_t, create_graph=False
        )[0]
        dw_dsqrtT = grad_w[:, 1]
        sqrt_T = X_t[:, 1]
        dw_dT = dw_dsqrtT / (2 * sqrt_T + 1e-8)

        if (dw_dT < -1e-6).any().item():
            continue
        arb_free_count += 1

    return (arb_free_count / n_slices) * 100 if n_slices > 0 else 0.0


def figure3(quick=False):
    """
    Train three ablation variants, evaluate arb-free % every N epochs.
    """
    import torch
    from src.volatility_surface.models.pinn_model import (
        PINNNetwork, CalendarLoss, ButterflyLoss, WingLoss,
    )

    print("[Figure 3] Training ablation variants for arb-free evolution ...")
    total_epochs = 500 if not quick else 200
    eval_every = 50 if not quick else 25

    df = generate_synthetic_surface(n_strikes=30, seed=42)

    variants = {
        "CINN (full)": dict(
            use_residual=True, activation="gelu",
            use_warmup=True, use_ema_norm=True, squared_hinge=True,
        ),
        "No warmup": dict(
            use_residual=True, activation="gelu",
            use_warmup=False, use_ema_norm=True, squared_hinge=True,
        ),
        "No EMA norm": dict(
            use_residual=True, activation="gelu",
            use_warmup=True, use_ema_norm=False, squared_hinge=True,
        ),
    }

    variant_colors = {
        "CINN (full)": COLORS["cinn"],
        "No warmup": COLORS["no_warmup"],
        "No EMA norm": COLORS["no_ema"],
    }
    variant_styles = {
        "CINN (full)": "-",
        "No warmup": "--",
        "No EMA norm": "-.",
    }

    results = {}  # name -> {epoch: arb_free_pct}

    for name, flags in variants.items():
        print(f"  Variant: {name}")
        results[name] = {}

        # We need to train step by step, evaluating periodically.
        # Re-implement the training loop inline so we can pause and evaluate.

        model = PINNVolatilityModel(
            epochs=total_epochs,
            hidden_layers=[64, 32, 16],
            lambda_calendar=1.0, lambda_butterfly=0.5, lambda_wing=0.1,
            **flags,
        )

        # Prepare data (mimicking _train_impl)
        X = df[model.feature_columns].values.astype(np.float32)
        y = df["implied_volatility"].values.astype(np.float32)
        X_transformed = X.copy()
        t_idx = model.feature_columns.index("T")
        X_transformed[:, t_idx] = np.sqrt(X_transformed[:, t_idx])
        T_vals = X[:, t_idx]
        w = y ** 2 * T_vals

        # No validation split for simplicity here
        X_train = torch.tensor(X_transformed, device=model.device)
        w_train = torch.tensor(w, device=model.device).unsqueeze(1)

        model.model = PINNNetwork(
            input_dim=len(model.feature_columns),
            hidden_layers=model.hidden_layers,
            dropout=model.dropout,
            use_residual=model.use_residual,
            activation=model.activation,
        ).to(model.device)

        mse_loss = torch.nn.MSELoss()
        calendar_loss = CalendarLoss(squared=model.squared_hinge)
        butterfly_loss = ButterflyLoss(squared=model.squared_hinge)
        wing_loss = WingLoss(squared=model.squared_hinge)

        optimizer = torch.optim.AdamW(
            model.model.parameters(), lr=model.learning_rate, weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

        dataset = torch.utils.data.TensorDataset(X_train, w_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=model.batch_size, shuffle=True)

        ema_var = {"calendar": 1.0, "butterfly": 1.0, "wing": 1.0}
        ema_alpha = 0.1
        use_warmup = model.use_warmup
        use_ema_norm = model.use_ema_norm

        def lambda_schedule(epoch):
            if not use_warmup:
                return 1.0
            if epoch < 100:
                return 0.0
            elif epoch < 300:
                return (epoch - 100) / 200.0
            else:
                return 1.0

        # Evaluate at epoch 0
        model._is_trained = True
        results[name][0] = _compute_arb_free_pct(model, df)

        for epoch in range(total_epochs):
            model.model.train()
            schedule_weight = lambda_schedule(epoch)

            for batch_X, batch_w in loader:
                optimizer.zero_grad()
                pred_w = model.model(batch_X)
                loss_mse = mse_loss(pred_w, batch_w)
                loss_cal = calendar_loss(model.model, batch_X)
                loss_but = butterfly_loss(model.model, batch_X)
                loss_wing = wing_loss(model.model, batch_X)

                if schedule_weight > 0:
                    for n_, lv_ in [("calendar", loss_cal), ("butterfly", loss_but), ("wing", loss_wing)]:
                        val = lv_.item()
                        ema_var[n_] = (1 - ema_alpha) * ema_var[n_] + ema_alpha * max(val, 1e-8)

                if use_ema_norm:
                    norm_cal = loss_cal / max(ema_var["calendar"], 1e-8)
                    norm_but = loss_but / max(ema_var["butterfly"], 1e-8)
                    norm_wing = loss_wing / max(ema_var["wing"], 1e-8)
                else:
                    norm_cal, norm_but, norm_wing = loss_cal, loss_but, loss_wing

                total_loss = (
                    loss_mse
                    + schedule_weight * model.lambda_calendar * norm_cal
                    + schedule_weight * model.lambda_butterfly * norm_but
                    + schedule_weight * model.lambda_wing * norm_wing
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            # Periodic evaluation
            if (epoch + 1) % eval_every == 0:
                pct = _compute_arb_free_pct(model, df)
                results[name][epoch + 1] = pct
                print(f"    epoch {epoch + 1:4d}: arb-free = {pct:.1f}%")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for name, epoch_dict in results.items():
        epochs_arr = sorted(epoch_dict.keys())
        pcts = [epoch_dict[e] for e in epochs_arr]
        ax.plot(epochs_arr, pcts,
                color=variant_colors[name],
                linestyle=variant_styles[name],
                linewidth=1.8, marker="o", markersize=4, label=name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Arbitrage-free slices (%)")
    ax.set_ylim(-5, 105)
    ax.set_xlim(0, total_epochs)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title("Arbitrage-Free Percentage During Training")

    # Shading for warmup region
    warmup_end = min(100, total_epochs)
    ramp_end = min(300, total_epochs)
    ax.axvspan(0, warmup_end, alpha=0.06, color="gray")
    ax.axvspan(warmup_end, ramp_end, alpha=0.04, color="blue")
    ax.axvline(warmup_end, color="gray", linestyle=":", linewidth=0.7)
    ax.axvline(ramp_end, color="gray", linestyle=":", linewidth=0.7)

    # Annotate regions
    ax.annotate("Warmup", xy=(warmup_end / 2, 102), fontsize=8, ha="center", color="gray")
    if ramp_end > warmup_end + 20:
        ax.annotate("Ramp", xy=((warmup_end + ramp_end) / 2, 102),
                     fontsize=8, ha="center", color="steelblue")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_arb_free_evolution.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for the CINN vol-surface paper."
    )
    parser.add_argument("--fig", type=int, default=None,
                        help="Generate only a specific figure (1, 2, or 3)")
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer epochs for faster generation")
    args = parser.parse_args()

    setup_style()
    ensure_fig_dir()

    fig_funcs = {1: figure1, 2: figure2, 3: figure3}

    if args.fig is not None:
        if args.fig not in fig_funcs:
            print(f"Unknown figure number: {args.fig}. Choose from 1, 2, 3.")
            sys.exit(1)
        fig_funcs[args.fig](quick=args.quick)
    else:
        for num, func in fig_funcs.items():
            func(quick=args.quick)

    print("\nDone. Figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
