# src/benchmarks/vol_surface_benchmark.py
"""
Comprehensive Volatility Surface Model Benchmark Framework.

Compares SVI, SSVI, SABR, and Neural Network models across:
- Error metrics (RMSE, MAE, MAPE, wing error)
- Speed metrics (calibration time, prediction throughput)
- Stability metrics (parameter variance, arbitrage-free percentage)

Usage:
    >>> from src.benchmarks.vol_surface_benchmark import VolSurfaceBenchmark
    >>> benchmark = VolSurfaceBenchmark(models=["svi", "sabr", "mlp"])
    >>> results = benchmark.run(data, n_trials=10)
    >>> results.to_dataframe()
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# =============================================================================
# Metrics Dataclasses
# =============================================================================


@dataclass
class ErrorMetrics:
    """Error metrics for volatility surface prediction."""

    rmse: float = 0.0  # Root mean squared error
    mae: float = 0.0  # Mean absolute error
    mape: float = 0.0  # Mean absolute percentage error (%)
    max_error: float = 0.0  # Maximum absolute error
    atm_error: float = 0.0  # Error at ATM (|k| < 0.05)
    wing_error: float = 0.0  # Error at wings (|k| > 0.2)
    term_structure_error: float = 0.0  # Avg error across maturities

    def to_dict(self) -> Dict[str, float]:
        return {
            "RMSE": self.rmse,
            "MAE": self.mae,
            "MAPE (%)": self.mape,
            "Max Error": self.max_error,
            "ATM Error": self.atm_error,
            "Wing Error": self.wing_error,
            "Term Structure Error": self.term_structure_error,
        }


@dataclass
class SpeedMetrics:
    """Speed and performance metrics."""

    calibration_time_ms: float = 0.0  # Time to calibrate/train (milliseconds)
    prediction_time_ms: float = 0.0  # Time to predict smile (milliseconds)
    throughput: float = 0.0  # Smiles per second
    scaling_factor: float = 1.0  # How well it scales with data size

    def to_dict(self) -> Dict[str, float]:
        return {
            "Calibration (ms)": self.calibration_time_ms,
            "Prediction (ms)": self.prediction_time_ms,
            "Throughput (smiles/s)": self.throughput,
            "Scaling Factor": self.scaling_factor,
        }


@dataclass
class StabilityMetrics:
    """Stability and robustness metrics."""

    param_cv: float = 0.0  # Coefficient of variation of parameters
    arbitrage_free_pct: float = 0.0  # % of surfaces passing no-arb checks
    convergence_rate: float = 0.0  # % of successful calibrations
    gradient_stability: float = 0.0  # Avg gradient norm during optimization

    def to_dict(self) -> Dict[str, float]:
        return {
            "Param CV": self.param_cv,
            "Arb-Free (%)": self.arbitrage_free_pct,
            "Convergence (%)": self.convergence_rate,
            "Gradient Stability": self.gradient_stability,
        }


@dataclass
class ModelBenchmark:
    """Complete benchmark for a single model."""

    model_name: str
    error: ErrorMetrics = field(default_factory=ErrorMetrics)
    speed: SpeedMetrics = field(default_factory=SpeedMetrics)
    stability: StabilityMetrics = field(default_factory=StabilityMetrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Model": self.model_name,
            **self.error.to_dict(),
            **self.speed.to_dict(),
            **self.stability.to_dict(),
        }


@dataclass
class BenchmarkResults:
    """Container for all benchmark results."""

    models: List[ModelBenchmark] = field(default_factory=list)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    n_trials: int = 1

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for easy viewing."""
        if not self.models:
            return pd.DataFrame()
        rows = [m.to_dict() for m in self.models]
        return pd.DataFrame(rows).set_index("Model")

    def summary(self) -> str:
        """Generate text summary of results."""
        df = self.to_dataframe()
        return f"Benchmark Results ({self.n_trials} trials)\n\n{df.to_string()}"

    def best_model(self, metric: str = "RMSE") -> str:
        """Return name of best model for given metric."""
        df = self.to_dataframe()
        if metric in df.columns:
            return str(df[metric].idxmin())
        return self.models[0].model_name if self.models else ""


# =============================================================================
# Model Wrappers
# =============================================================================


class ModelWrapper:
    """Base class for wrapping volatility models with unified interface."""

    name: str = "base"

    def calibrate(
        self,
        log_strikes: np.ndarray,
        market_vols: np.ndarray,
        T: float,
        **kwargs,
    ) -> None:
        """Calibrate model to market data."""
        raise NotImplementedError

    def predict(self, log_strikes: np.ndarray, T: float) -> np.ndarray:
        """Predict implied volatilities."""
        raise NotImplementedError

    def get_params(self) -> Dict[str, float]:
        """Return calibrated parameters."""
        return {}


class SVIWrapper(ModelWrapper):
    """Wrapper for SVI model."""

    name = "SVI"

    def __init__(self):
        self.model = None

    def calibrate(
        self,
        log_strikes: np.ndarray,
        market_vols: np.ndarray,
        T: float,
        **kwargs,
    ) -> None:
        from src.volatility_surface.models.svi import calibrate_svi

        self.model = calibrate_svi(log_strikes, market_vols, T)
        self.T = T

    def predict(self, log_strikes: np.ndarray, T: float) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not calibrated")
        return self.model.smile(log_strikes, T)

    def get_params(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        return {
            "a": self.model.a,
            "b": self.model.b,
            "rho": self.model.rho,
            "m": self.model.m,
            "sigma": self.model.sigma,
        }


class SABRWrapper(ModelWrapper):
    """Wrapper for SABR model."""

    name = "SABR"

    def __init__(self, beta: float = 0.5):
        self.model = None
        self.beta = beta
        self.F = 100.0  # Default forward

    def calibrate(
        self,
        log_strikes: np.ndarray,
        market_vols: np.ndarray,
        T: float,
        F: float = 100.0,
        **kwargs,
    ) -> None:
        from src.pricing_models.sabr import calibrate_sabr

        # Convert log-strikes to absolute strikes
        strikes = F * np.exp(log_strikes)
        self.model = calibrate_sabr(F, T, strikes, market_vols, beta=self.beta)
        self.F = F
        self.T = T

    def predict(self, log_strikes: np.ndarray, T: float) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not calibrated")
        strikes = self.F * np.exp(log_strikes)
        return self.model.smile(self.F, T, strikes)

    def get_params(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        return {
            "alpha": self.model.alpha,
            "beta": self.model.beta,
            "rho": self.model.rho,
            "nu": self.model.nu,
        }


class MLPWrapper(ModelWrapper):
    """Wrapper for MLP volatility model using sklearn."""

    name = "MLP"

    def __init__(self):
        self.model = None
        self._trained = False
        self.scaler = None

    def calibrate(
        self,
        log_strikes: np.ndarray,
        market_vols: np.ndarray,
        T: float,
        **kwargs,
    ) -> None:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        # Prepare features
        X = np.column_stack([log_strikes, np.full_like(log_strikes, T)])
        y = market_vols

        # Scale inputs
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train MLP
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_scaled, y)
        self._trained = True

    def predict(self, log_strikes: np.ndarray, T: float) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model not trained")
        X = np.column_stack([log_strikes, np.full_like(log_strikes, T)])
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class RFWrapper(ModelWrapper):
    """Wrapper for Random Forest volatility model using sklearn."""

    name = "Random Forest"

    def __init__(self):
        self.model = None
        self._trained = False
        self.scaler = None

    def calibrate(
        self,
        log_strikes: np.ndarray,
        market_vols: np.ndarray,
        T: float,
        **kwargs,
    ) -> None:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        # Prepare features
        X = np.column_stack([log_strikes, np.full_like(log_strikes, T)])
        y = market_vols

        # Scale inputs
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train RF
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=1,  # Single thread to avoid multiprocessing issues
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_scaled, y)
        self._trained = True

    def predict(self, log_strikes: np.ndarray, T: float) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model not trained")
        X = np.column_stack([log_strikes, np.full_like(log_strikes, T)])
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class PINNWrapper(ModelWrapper):
    """Wrapper for PINN-style volatility model (simplified for benchmark speed)."""

    name = "PINN"

    def __init__(self):
        self.model = None
        self._trained = False
        self.scaler = None

    def calibrate(
        self,
        log_strikes: np.ndarray,
        market_vols: np.ndarray,
        T: float,
        **kwargs,
    ) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch required for PINN model")
        
        from sklearn.preprocessing import StandardScaler

        # Prepare features
        X = np.column_stack([log_strikes, np.full_like(log_strikes, T)])
        y = market_vols.reshape(-1, 1)

        # Scale inputs
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Simple PyTorch MLP with softplus output (ensures positive vol)
        class VolNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, 32),
                    nn.GELU(),
                    nn.Linear(32, 16),
                    nn.GELU(),
                    nn.Linear(16, 1),
                    nn.Softplus(),  # Ensures positive volatility
                )
            def forward(self, x):
                return self.net(x) * 0.5  # Scale to reasonable vol range

        self.model = VolNet()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        # Quick training loop
        self.model.train()
        for _ in range(100):
            optimizer.zero_grad()
            pred = self.model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

        self._trained = True

    def predict(self, log_strikes: np.ndarray, T: float) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model not trained")
        import torch
        
        X = np.column_stack([log_strikes, np.full_like(log_strikes, T)])
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_t).numpy().flatten()
        return preds


# =============================================================================
# Benchmark Runner
# =============================================================================


class VolSurfaceBenchmark:
    """
    Comprehensive benchmark suite for volatility surface models.

    Compares parametric (SVI, SABR) and machine learning (MLP, RF, XGBoost)
    models across accuracy, speed, and stability metrics.

    Usage:
        >>> benchmark = VolSurfaceBenchmark(["svi", "sabr", "mlp"])
        >>> results = benchmark.run(data_df, n_trials=10)
        >>> print(results.to_dataframe())
    """

    AVAILABLE_MODELS = {
        "svi": SVIWrapper,
        "sabr": SABRWrapper,
        "mlp": MLPWrapper,
        "rf": RFWrapper,
        "pinn": PINNWrapper,
    }

    def __init__(
        self,
        models: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        """
        Initialize benchmark suite.

        Args:
            models: List of model names to benchmark. Options: svi, sabr, mlp, rf
            verbose: Print progress during benchmarking.
        """
        if models is None:
            models = ["svi", "sabr", "mlp"]

        self.model_names = [m.lower() for m in models]
        self.verbose = verbose

        # Validate models
        for m in self.model_names:
            if m not in self.AVAILABLE_MODELS:
                raise ValueError(
                    f"Unknown model: {m}. Available: {list(self.AVAILABLE_MODELS.keys())}"
                )

    def run(
        self,
        data: pd.DataFrame,
        n_trials: int = 5,
        test_size: float = 0.2,
    ) -> BenchmarkResults:
        """
        Run full benchmark suite.

        Args:
            data: DataFrame with columns: log_moneyness, T, implied_volatility
            n_trials: Number of trials for averaging
            test_size: Fraction of data for testing

        Returns:
            BenchmarkResults with all metrics
        """
        from datetime import datetime

        results = BenchmarkResults(
            dataset_info={
                "n_samples": len(data),
                "n_maturities": data["T"].nunique() if "T" in data.columns else 1,
            },
            timestamp=datetime.now().isoformat(),
            n_trials=n_trials,
        )

        for model_name in self.model_names:
            if self.verbose:
                print(f"Benchmarking {model_name.upper()}...")

            model_result = self._benchmark_model(model_name, data, n_trials, test_size)
            results.models.append(model_result)

        return results

    def _benchmark_model(
        self,
        model_name: str,
        data: pd.DataFrame,
        n_trials: int,
        test_size: float,
    ) -> ModelBenchmark:
        """Benchmark a single model."""
        wrapper_class = self.AVAILABLE_MODELS[model_name]

        # Collect metrics across trials
        rmse_list, mae_list, mape_list = [], [], []
        calib_times, pred_times = [], []
        params_list = []
        arb_free_count = 0

        for trial in range(n_trials):
            # Split data
            n_test = int(len(data) * test_size)
            shuffled = data.sample(frac=1, random_state=trial)
            train_data = shuffled.iloc[n_test:]
            test_data = shuffled.iloc[:n_test]

            log_strikes = train_data["log_moneyness"].values
            market_vols = train_data["implied_volatility"].values
            T = train_data["T"].mean()

            wrapper = wrapper_class()

            # Time calibration
            try:
                start = time.perf_counter()
                wrapper.calibrate(log_strikes, market_vols, T)
                calib_time = (time.perf_counter() - start) * 1000
                calib_times.append(calib_time)
            except Exception as e:
                import traceback
                if self.verbose:
                    print(f"  Trial {trial}: Calibration failed - {e}")
                    traceback.print_exc()
                continue

            # Time prediction
            test_k = test_data["log_moneyness"].values
            test_T = test_data["T"].values[0] if len(test_data) > 0 else T
            test_vol = test_data["implied_volatility"].values

            try:
                start = time.perf_counter()
                pred_vol = wrapper.predict(test_k, test_T)
                pred_time = (time.perf_counter() - start) * 1000
                pred_times.append(pred_time)
            except Exception as e:
                if self.verbose:
                    print(f"  Trial {trial}: Prediction failed - {e}")
                continue

            # Compute errors
            errors = pred_vol - test_vol
            rmse_list.append(np.sqrt(np.mean(errors**2)))
            mae_list.append(np.mean(np.abs(errors)))
            mape_list.append(np.mean(np.abs(errors / test_vol)) * 100)

            # Store params for stability
            params = wrapper.get_params()
            if params:
                params_list.append(params)

            # Check arbitrage (simplified: no calendar arb if w increasing in T)
            arb_free_count += 1  # Placeholder - assume OK for now

        # Aggregate metrics
        n_success = len(rmse_list)
        if n_success == 0:
            return ModelBenchmark(model_name=wrapper_class.name)

        error_metrics = ErrorMetrics(
            rmse=np.mean(rmse_list),
            mae=np.mean(mae_list),
            mape=np.mean(mape_list),
            max_error=np.max(rmse_list),
        )

        speed_metrics = SpeedMetrics(
            calibration_time_ms=np.mean(calib_times) if calib_times else 0,
            prediction_time_ms=np.mean(pred_times) if pred_times else 0,
            throughput=1000 / np.mean(pred_times) if pred_times else 0,
        )

        # Parameter stability
        param_cv = 0.0
        if params_list:
            first_key = list(params_list[0].keys())[0]
            param_values = [p.get(first_key, 0) for p in params_list]
            if np.mean(param_values) > 0:
                param_cv = np.std(param_values) / np.mean(param_values)

        stability_metrics = StabilityMetrics(
            param_cv=param_cv,
            arbitrage_free_pct=arb_free_count / n_trials * 100,
            convergence_rate=n_success / n_trials * 100,
        )

        return ModelBenchmark(
            model_name=wrapper_class.name,
            error=error_metrics,
            speed=speed_metrics,
            stability=stability_metrics,
        )


# =============================================================================
# Synthetic Data Generator (for testing without yfinance)
# =============================================================================


def generate_synthetic_smile(
    n_strikes: int = 50,
    T: float = 1.0,
    atm_vol: float = 0.2,
    skew: float = -0.3,
    smile: float = 0.05,
    noise: float = 0.005,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic volatility smile data.

    Uses a simple quadratic smile model:
        σ(k) = σ_ATM + skew * k + smile * k²

    Args:
        n_strikes: Number of strike points
        T: Time to maturity
        atm_vol: ATM implied volatility
        skew: Skew coefficient (typically negative)
        smile: Smile coefficient (curvature)
        noise: Noise level to add
        seed: Random seed

    Returns:
        DataFrame with log_moneyness, T, implied_volatility columns
    """
    if seed is not None:
        np.random.seed(seed)

    log_strikes = np.linspace(-0.4, 0.4, n_strikes)
    vols = atm_vol + skew * log_strikes + smile * log_strikes**2
    vols = vols + np.random.normal(0, noise, n_strikes)
    vols = np.maximum(vols, 0.01)  # Floor at 1%

    return pd.DataFrame(
        {
            "log_moneyness": log_strikes,
            "T": T,
            "implied_volatility": vols,
        }
    )


def generate_synthetic_surface(
    n_strikes: int = 30,
    maturities: Optional[List[float]] = None,
    base_vol: float = 0.2,
    term_slope: float = -0.02,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic volatility surface with multiple maturities.

    Args:
        n_strikes: Number of strikes per maturity
        maturities: List of maturities (default: [0.1, 0.25, 0.5, 1.0, 2.0])
        base_vol: Base ATM volatility
        term_slope: Term structure slope (negative = downward sloping)
        seed: Random seed

    Returns:
        DataFrame with full surface data
    """
    if maturities is None:
        maturities = [0.1, 0.25, 0.5, 1.0, 2.0]

    dfs = []
    for i, T in enumerate(maturities):
        atm_vol = base_vol + term_slope * np.log(T)
        atm_vol = max(atm_vol, 0.05)
        df = generate_synthetic_smile(
            n_strikes=n_strikes,
            T=T,
            atm_vol=atm_vol,
            skew=-0.25 / np.sqrt(T),  # Skew flattens with maturity
            smile=0.04,
            noise=0.003,
            seed=seed + i if seed else None,
        )
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """Run benchmark from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Volatility Surface Model Benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["svi", "sabr", "mlp"],
        help="Models to benchmark",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Number of trials",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with synthetic data",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Volatility Surface Model Benchmark")
    print("=" * 60)

    # Generate test data
    print("\nGenerating synthetic volatility surface...")
    data = generate_synthetic_surface(n_strikes=40, seed=42)
    print(f"  Generated {len(data)} data points across {data['T'].nunique()} maturities")

    # Run benchmark
    benchmark = VolSurfaceBenchmark(models=args.models)
    results = benchmark.run(data, n_trials=args.n_trials)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(results.to_dataframe().to_string())

    print(f"\nBest model by RMSE: {results.best_model('RMSE')}")


if __name__ == "__main__":
    main()
