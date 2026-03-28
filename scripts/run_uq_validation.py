"""
Uncertainty Quantification validation: MC dropout coverage and uncertainty-vs-distance.
Produces quantitative UQ results for the paper.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_uq_validation(quick: bool = False):
    """Run MC dropout UQ validation."""
    from src.volatility_surface.models.pinn_model import PINNVolatilityModel
    from src.benchmarks.vol_surface_benchmark import generate_synthetic_surface

    print("=" * 60)
    print("UQ Validation: MC Dropout Coverage & Uncertainty-vs-Distance")
    print("=" * 60)

    # Generate full surface
    df_full = generate_synthetic_surface(n_strikes=30, seed=42)
    n_total = len(df_full)

    # Create sparsity levels
    sparsity_levels = [0.2, 0.4, 0.6] if not quick else [0.4]
    n_mc = 50 if not quick else 20
    results = []

    for dropout_frac in sparsity_levels:
        print(f"\n--- Dropout: {dropout_frac*100:.0f}% ---")

        # Split into train/test
        np.random.seed(42)
        n_drop = int(n_total * dropout_frac)
        perm = np.random.permutation(n_total)
        test_idx = perm[:n_drop]
        train_idx = perm[n_drop:]

        df_train = df_full.iloc[train_idx].copy()
        df_test = df_full.iloc[test_idx].copy()

        print(f"  Train: {len(df_train)}, Test: {len(df_test)}")

        # Train CINN
        model = PINNVolatilityModel(
            epochs=200 if quick else 500,
            hidden_layers=[64, 32, 16],
            dropout=0.1,
        )
        model.train(df_train)

        # MC Dropout predictions on test set
        if not hasattr(model, 'predict_with_uncertainty'):
            print("  WARNING: predict_with_uncertainty not available, using manual MC dropout")
            # Manual MC dropout
            import torch
            model.model.train()  # Enable dropout
            X_test = df_test[model.feature_columns].values.astype(np.float32)

            # Scale features the same way as training
            X_scaled = X_test.copy()
            X_scaled[:, 1] = np.sqrt(np.abs(X_scaled[:, 1]))  # sqrt(T)

            X_tensor = torch.tensor(X_scaled, device=model.device)

            preds = []
            with torch.no_grad():
                for _ in range(n_mc):
                    iv = model.model.implied_vol(X_tensor).cpu().numpy().flatten()
                    preds.append(iv)

            preds = np.array(preds)  # (n_mc, n_test)
            mean_pred = preds.mean(axis=0)
            std_pred = preds.std(axis=0)
            model.model.eval()
        else:
            # Use built-in method - returns (mean, std) tuple of implied vols
            mean_pred, std_pred = model.predict_with_uncertainty(df_test, n_samples=n_mc)

        # Ground truth: implied volatility
        true_iv = df_test['implied_volatility'].values

        # Coverage: what fraction of true IVs fall within 95% CI?
        lower = mean_pred - 2 * std_pred
        upper = mean_pred + 2 * std_pred
        covered = ((true_iv >= lower) & (true_iv <= upper)).mean()

        # Uncertainty vs distance to nearest training point
        train_features = df_train[model.feature_columns].values
        test_features = df_test[model.feature_columns].values
        distances = []
        for tf in test_features:
            dists = np.sqrt(((train_features - tf)**2).sum(axis=1))
            distances.append(dists.min())
        distances = np.array(distances)

        # Spearman correlation
        from scipy.stats import spearmanr
        corr, pval = spearmanr(distances, std_pred)

        print(f"  MC samples: {n_mc}")
        print(f"  95% CI coverage: {covered:.3f} (target: 0.95)")
        print(f"  Mean uncertainty: {std_pred.mean():.4f}")
        print(f"  Uncertainty-distance Spearman r: {corr:.3f} (p={pval:.4f})")

        results.append({
            "Dropout (%)": int(dropout_frac * 100),
            "Coverage (95% CI)": round(covered, 3),
            "Mean Uncertainty": round(std_pred.mean(), 4),
            "Spearman r (unc vs dist)": round(corr, 3),
            "Spearman p-value": round(pval, 4),
        })

    # Output
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("UQ VALIDATION RESULTS")
    print("=" * 60)
    print(df_results.to_string(index=False))

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results"
    )
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "uq_validation.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    return df_results


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run_uq_validation(quick=quick)
