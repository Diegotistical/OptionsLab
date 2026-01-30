# tests/test_vol_surface_benchmark.py
"""
Unit tests for volatility surface benchmark framework.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmarks.vol_surface_benchmark import (
    BenchmarkResults,
    ErrorMetrics,
    ModelBenchmark,
    SABRWrapper,
    SpeedMetrics,
    StabilityMetrics,
    SVIWrapper,
    VolSurfaceBenchmark,
    generate_synthetic_smile,
    generate_synthetic_surface,
)

# =============================================================================
# Test Metrics Dataclasses
# =============================================================================


class TestErrorMetrics:
    """Tests for ErrorMetrics dataclass."""

    def test_default_values(self):
        metrics = ErrorMetrics()
        assert metrics.rmse == 0.0
        assert metrics.mae == 0.0
        assert metrics.mape == 0.0

    def test_to_dict(self):
        metrics = ErrorMetrics(rmse=0.01, mae=0.008, mape=2.5)
        d = metrics.to_dict()
        assert d["RMSE"] == 0.01
        assert d["MAE"] == 0.008
        assert d["MAPE (%)"] == 2.5

    def test_custom_values(self):
        metrics = ErrorMetrics(
            rmse=0.015,
            mae=0.012,
            mape=3.2,
            max_error=0.05,
            atm_error=0.008,
            wing_error=0.025,
        )
        assert metrics.max_error == 0.05
        assert metrics.wing_error == 0.025


class TestSpeedMetrics:
    """Tests for SpeedMetrics dataclass."""

    def test_default_values(self):
        metrics = SpeedMetrics()
        assert metrics.calibration_time_ms == 0.0
        assert metrics.prediction_time_ms == 0.0
        assert metrics.throughput == 0.0

    def test_to_dict(self):
        metrics = SpeedMetrics(
            calibration_time_ms=100.0,
            prediction_time_ms=5.0,
            throughput=200.0,
        )
        d = metrics.to_dict()
        assert d["Calibration (ms)"] == 100.0
        assert d["Throughput (smiles/s)"] == 200.0


class TestStabilityMetrics:
    """Tests for StabilityMetrics dataclass."""

    def test_default_values(self):
        metrics = StabilityMetrics()
        assert metrics.param_cv == 0.0
        assert metrics.arbitrage_free_pct == 0.0

    def test_to_dict(self):
        metrics = StabilityMetrics(
            param_cv=0.15,
            arbitrage_free_pct=98.5,
            convergence_rate=95.0,
        )
        d = metrics.to_dict()
        assert d["Param CV"] == 0.15
        assert d["Arb-Free (%)"] == 98.5


# =============================================================================
# Test Benchmark Results
# =============================================================================


class TestBenchmarkResults:
    """Tests for BenchmarkResults container."""

    def test_empty_results(self):
        results = BenchmarkResults()
        assert len(results.models) == 0
        df = results.to_dataframe()
        assert len(df) == 0 or df.empty

    def test_with_models(self):
        model1 = ModelBenchmark(
            model_name="SVI",
            error=ErrorMetrics(rmse=0.01),
            speed=SpeedMetrics(calibration_time_ms=10.0),
        )
        model2 = ModelBenchmark(
            model_name="SABR",
            error=ErrorMetrics(rmse=0.02),
            speed=SpeedMetrics(calibration_time_ms=8.0),
        )
        results = BenchmarkResults(models=[model1, model2], n_trials=5)

        df = results.to_dataframe()
        assert len(df) == 2
        assert "SVI" in df.index
        assert "SABR" in df.index

    def test_best_model(self):
        model1 = ModelBenchmark(
            model_name="SVI",
            error=ErrorMetrics(rmse=0.02),
        )
        model2 = ModelBenchmark(
            model_name="MLP",
            error=ErrorMetrics(rmse=0.01),
        )
        results = BenchmarkResults(models=[model1, model2])

        assert results.best_model("RMSE") == "MLP"

    def test_summary(self):
        model = ModelBenchmark(model_name="SVI")
        results = BenchmarkResults(models=[model], n_trials=10)
        summary = results.summary()
        assert "10 trials" in summary
        assert "SVI" in summary


# =============================================================================
# Test Synthetic Data Generation
# =============================================================================


class TestSyntheticData:
    """Tests for synthetic data generators."""

    def test_generate_smile(self):
        df = generate_synthetic_smile(n_strikes=30, T=0.5, seed=42)

        assert len(df) == 30
        assert "log_moneyness" in df.columns
        assert "T" in df.columns
        assert "implied_volatility" in df.columns
        assert (df["T"] == 0.5).all()
        assert (df["implied_volatility"] > 0).all()

    def test_generate_smile_reproducibility(self):
        df1 = generate_synthetic_smile(seed=123)
        df2 = generate_synthetic_smile(seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_generate_surface(self):
        df = generate_synthetic_surface(
            n_strikes=20,
            maturities=[0.25, 0.5, 1.0],
            seed=42,
        )

        assert len(df) == 60  # 20 strikes × 3 maturities
        assert df["T"].nunique() == 3
        assert set(df["T"].unique()) == {0.25, 0.5, 1.0}

    def test_surface_smile_shape(self):
        """Verify the smile has expected shape (higher vol at wings)."""
        df = generate_synthetic_smile(
            n_strikes=50,
            atm_vol=0.2,
            skew=-0.3,
            smile=0.05,
            noise=0.0,  # No noise for deterministic test
            seed=42,
        )

        atm_mask = np.abs(df["log_moneyness"]) < 0.05
        wing_mask = np.abs(df["log_moneyness"]) > 0.2

        atm_vol = df.loc[atm_mask, "implied_volatility"].mean()
        wing_vol = df.loc[wing_mask, "implied_volatility"].mean()

        # Wings should have higher vol due to smile
        assert wing_vol > atm_vol


# =============================================================================
# Test Model Wrappers
# =============================================================================


class TestSVIWrapper:
    """Tests for SVI model wrapper."""

    def test_calibration(self):
        wrapper = SVIWrapper()

        # Generate simple smile data
        log_strikes = np.linspace(-0.3, 0.3, 20)
        market_vols = 0.2 + 0.05 * log_strikes**2

        wrapper.calibrate(log_strikes, market_vols, T=1.0)

        # Should be able to predict after calibration
        pred = wrapper.predict(log_strikes, T=1.0)
        assert len(pred) == 20
        assert np.all(pred > 0)

    def test_params(self):
        wrapper = SVIWrapper()
        log_strikes = np.linspace(-0.3, 0.3, 20)
        market_vols = 0.2 + 0.05 * log_strikes**2

        wrapper.calibrate(log_strikes, market_vols, T=1.0)
        params = wrapper.get_params()

        assert "a" in params
        assert "b" in params
        assert "rho" in params
        assert "m" in params
        assert "sigma" in params


class TestSABRWrapper:
    """Tests for SABR model wrapper."""

    def test_calibration(self):
        wrapper = SABRWrapper(beta=0.5)

        log_strikes = np.linspace(-0.2, 0.2, 15)
        market_vols = 0.2 - 0.1 * log_strikes + 0.03 * log_strikes**2

        wrapper.calibrate(log_strikes, market_vols, T=0.5, F=100.0)

        pred = wrapper.predict(log_strikes, T=0.5)
        assert len(pred) == 15
        assert np.all(pred > 0)

    def test_params(self):
        wrapper = SABRWrapper()
        log_strikes = np.linspace(-0.2, 0.2, 15)
        market_vols = np.full_like(log_strikes, 0.2)

        wrapper.calibrate(log_strikes, market_vols, T=1.0)
        params = wrapper.get_params()

        assert "alpha" in params
        assert "beta" in params
        assert "rho" in params
        assert "nu" in params


# =============================================================================
# Test Benchmark Runner
# =============================================================================


class TestVolSurfaceBenchmark:
    """Tests for the main benchmark runner."""

    def test_init_default_models(self):
        benchmark = VolSurfaceBenchmark()
        assert "svi" in benchmark.model_names
        assert "sabr" in benchmark.model_names
        assert "mlp" in benchmark.model_names

    def test_init_custom_models(self):
        benchmark = VolSurfaceBenchmark(models=["svi", "sabr"])
        assert benchmark.model_names == ["svi", "sabr"]

    def test_init_invalid_model(self):
        with pytest.raises(ValueError):
            VolSurfaceBenchmark(models=["svi", "invalid_model"])

    def test_run_quick(self):
        """Quick smoke test with minimal data."""
        data = generate_synthetic_smile(n_strikes=30, T=1.0, seed=42)

        benchmark = VolSurfaceBenchmark(
            models=["svi"],
            verbose=False,
        )
        results = benchmark.run(data, n_trials=1, test_size=0.2)

        assert len(results.models) == 1
        assert results.models[0].model_name == "SVI"
        assert results.models[0].error.rmse >= 0

    @pytest.mark.slow
    def test_run_full(self):
        """Full benchmark with multiple models (marked slow)."""
        data = generate_synthetic_surface(n_strikes=30, seed=42)

        benchmark = VolSurfaceBenchmark(
            models=["svi", "sabr"],
            verbose=False,
        )
        results = benchmark.run(data, n_trials=3, test_size=0.2)

        assert len(results.models) == 2
        df = results.to_dataframe()
        assert "SVI" in df.index
        assert "SABR" in df.index

    def test_results_dataframe(self):
        """Test that results convert to DataFrame correctly."""
        data = generate_synthetic_smile(n_strikes=30, T=1.0, seed=42)

        benchmark = VolSurfaceBenchmark(models=["svi"], verbose=False)
        results = benchmark.run(data, n_trials=1)

        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "RMSE" in df.columns
        assert "Calibration (ms)" in df.columns


# =============================================================================
# Test Integration
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self):
        """Test complete benchmark pipeline."""
        # Generate data
        data = generate_synthetic_surface(
            n_strikes=25,
            maturities=[0.25, 0.5],
            seed=42,
        )

        # Run benchmark
        benchmark = VolSurfaceBenchmark(
            models=["svi"],
            verbose=False,
        )
        results = benchmark.run(data, n_trials=2, test_size=0.3)

        # Verify results
        assert results.n_trials == 2
        assert results.dataset_info["n_samples"] == 50

        df = results.to_dataframe()
        assert df.loc["SVI", "RMSE"] < 1.0  # Reasonable error bound


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
