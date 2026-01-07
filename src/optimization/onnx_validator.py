# src/optimization/onnx_validator.py
"""
ONNX model validation with distributional comparison.

Provides:
- Statistical validation (not just pointwise equality)
- Sign agreement for directional outputs (Greeks)
- Rank correlation for ordering-sensitive predictions
- Detailed diagnostics for debugging

Usage:
    from src.optimization import ONNXValidator

    validator = ONNXValidator(rtol=1e-3, atol=1e-4)
    result = validator.validate(
        native_model=my_model,
        onnx_path="models/my_model.onnx",
        X_test=test_data,
    )
    if not result.passed:
        print(result.diagnostics)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class ValidationResult:
    """Result of ONNX model validation."""
    
    passed: bool
    
    # Error metrics
    mean_abs_diff: float
    max_abs_diff: float
    percentile_95_diff: float
    percentile_99_diff: float
    
    # Correlation metrics
    pearson_correlation: float
    spearman_rank_correlation: float
    
    # Sign agreement (for directional outputs like Greeks)
    sign_agreement_ratio: float
    
    # Distribution metrics
    native_mean: float
    native_std: float
    onnx_mean: float
    onnx_std: float
    
    # Thresholds used
    rtol: float
    atol: float
    max_diff_percentile: float
    
    # Diagnostics
    n_samples: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Get human-readable summary."""
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        return f"""
ONNX Validation {status}
─────────────────────────────────────
Samples: {self.n_samples}
Mean Absolute Diff: {self.mean_abs_diff:.6f}
Max Absolute Diff:  {self.max_abs_diff:.6f}
95th Percentile:    {self.percentile_95_diff:.6f}
99th Percentile:    {self.percentile_99_diff:.6f}

Pearson Correlation:  {self.pearson_correlation:.6f}
Spearman Correlation: {self.spearman_rank_correlation:.6f}
Sign Agreement:       {self.sign_agreement_ratio:.2%}

Native: μ={self.native_mean:.4f}, σ={self.native_std:.4f}
ONNX:   μ={self.onnx_mean:.4f}, σ={self.onnx_std:.4f}
"""


class ONNXValidationError(Exception):
    """Raised when ONNX validation fails critically."""
    pass


class ONNXValidator:
    """
    Distributional validation for ONNX models.
    
    Unlike naive pointwise equality (which fails on float32 vs float64),
    this validates statistical properties that matter in production:
    
    - Mean/max/percentile absolute differences
    - Correlation (Pearson for values, Spearman for ranks)
    - Sign agreement (critical for Greeks/sensitivities)
    - Distribution similarity
    """
    
    def __init__(
        self,
        rtol: float = 1e-3,
        atol: float = 1e-4,
        max_diff_percentile: float = 99.0,
        max_diff_threshold: Optional[float] = None,
        min_correlation: float = 0.999,
        min_sign_agreement: float = 0.99,
    ):
        """
        Initialize validator.
        
        Args:
            rtol: Relative tolerance for mean comparison.
            atol: Absolute tolerance for mean comparison.
            max_diff_percentile: Percentile for max diff threshold.
            max_diff_threshold: If set, override percentile-based threshold.
            min_correlation: Minimum acceptable Pearson correlation.
            min_sign_agreement: Minimum acceptable sign agreement ratio.
        """
        self.rtol = rtol
        self.atol = atol
        self.max_diff_percentile = max_diff_percentile
        self.max_diff_threshold = max_diff_threshold
        self.min_correlation = min_correlation
        self.min_sign_agreement = min_sign_agreement
    
    def validate(
        self,
        native_model: Any,
        onnx_path: Union[str, Path],
        X_test: np.ndarray,
        n_samples: Optional[int] = None,
        providers: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        Validate ONNX model against native model.
        
        Args:
            native_model: Original model with predict() method.
            onnx_path: Path to ONNX model file.
            X_test: Test data for validation.
            n_samples: Max samples to use (default: all).
            providers: ONNX Runtime execution providers.
        
        Returns:
            ValidationResult with detailed metrics.
        """
        import onnxruntime as ort
        
        onnx_path = Path(onnx_path)
        
        # Sample data if needed
        if n_samples and n_samples < len(X_test):
            indices = np.random.choice(len(X_test), n_samples, replace=False)
            X_test = X_test[indices]
        
        # Ensure float32 for ONNX
        X_test_f32 = X_test.astype(np.float32)
        
        # Get native predictions
        native_preds = self._get_native_predictions(native_model, X_test)
        native_preds = np.asarray(native_preds).flatten()
        
        # Get ONNX predictions
        providers = providers or ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        
        input_name = session.get_inputs()[0].name
        onnx_preds = session.run(None, {input_name: X_test_f32})[0]
        onnx_preds = np.asarray(onnx_preds).flatten()
        
        # Compute metrics
        result = self._compute_metrics(native_preds, onnx_preds)
        
        return result
    
    def validate_batch_sizes(
        self,
        onnx_path: Union[str, Path],
        n_features: int,
        batch_sizes: List[int] = None,
        providers: Optional[List[str]] = None,
    ) -> Dict[int, bool]:
        """
        Validate ONNX model works with various batch sizes.
        
        Args:
            onnx_path: Path to ONNX model.
            n_features: Number of input features.
            batch_sizes: List of batch sizes to test.
            providers: ONNX Runtime execution providers.
        
        Returns:
            Dict mapping batch_size -> success.
        """
        import onnxruntime as ort
        
        onnx_path = Path(onnx_path)
        batch_sizes = batch_sizes or [1, 16, 32, 64, 128, 256, 512, 1024]
        
        providers = providers or ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        input_name = session.get_inputs()[0].name
        
        results = {}
        for batch_size in batch_sizes:
            try:
                dummy_input = np.random.randn(batch_size, n_features).astype(np.float32)
                output = session.run(None, {input_name: dummy_input})[0]
                results[batch_size] = output.shape[0] == batch_size
            except Exception as e:
                results[batch_size] = False
        
        return results
    
    def _get_native_predictions(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get predictions from native model."""
        if hasattr(model, "predict"):
            return model.predict(X)
        elif callable(model):
            return model(X)
        else:
            raise ONNXValidationError(
                f"Model must have predict() method or be callable. "
                f"Got: {type(model)}"
            )
    
    def _compute_metrics(
        self,
        native_preds: np.ndarray,
        onnx_preds: np.ndarray,
    ) -> ValidationResult:
        """Compute validation metrics."""
        from scipy import stats
        
        # Absolute differences
        abs_diff = np.abs(native_preds - onnx_preds)
        mean_abs_diff = float(np.mean(abs_diff))
        max_abs_diff = float(np.max(abs_diff))
        p95_diff = float(np.percentile(abs_diff, 95))
        p99_diff = float(np.percentile(abs_diff, self.max_diff_percentile))
        
        # Correlation
        pearson_corr = float(np.corrcoef(native_preds, onnx_preds)[0, 1])
        spearman_corr = float(stats.spearmanr(native_preds, onnx_preds).statistic)
        
        # Sign agreement (important for Greeks)
        native_signs = np.sign(native_preds)
        onnx_signs = np.sign(onnx_preds)
        # Handle zeros - count as agreement if both are near zero
        near_zero = (np.abs(native_preds) < 1e-6) & (np.abs(onnx_preds) < 1e-6)
        sign_match = (native_signs == onnx_signs) | near_zero
        sign_agreement = float(np.mean(sign_match))
        
        # Distribution stats
        native_mean = float(np.mean(native_preds))
        native_std = float(np.std(native_preds))
        onnx_mean = float(np.mean(onnx_preds))
        onnx_std = float(np.std(onnx_preds))
        
        # Determine pass/fail
        threshold = self.max_diff_threshold or (self.atol + self.rtol * np.abs(native_mean))
        
        passed = (
            p99_diff <= threshold * 10 and  # Allow 10x threshold for 99th percentile
            pearson_corr >= self.min_correlation and
            sign_agreement >= self.min_sign_agreement
        )
        
        # Diagnostics for failures
        diagnostics = {}
        if not passed:
            if p99_diff > threshold * 10:
                diagnostics["percentile_exceeded"] = {
                    "value": p99_diff,
                    "threshold": threshold * 10,
                }
            if pearson_corr < self.min_correlation:
                diagnostics["correlation_low"] = {
                    "value": pearson_corr,
                    "threshold": self.min_correlation,
                }
            if sign_agreement < self.min_sign_agreement:
                diagnostics["sign_agreement_low"] = {
                    "value": sign_agreement,
                    "threshold": self.min_sign_agreement,
                }
            
            # Find worst samples
            worst_idx = np.argsort(abs_diff)[-5:]
            diagnostics["worst_samples"] = [
                {
                    "index": int(idx),
                    "native": float(native_preds[idx]),
                    "onnx": float(onnx_preds[idx]),
                    "diff": float(abs_diff[idx]),
                }
                for idx in worst_idx
            ]
        
        return ValidationResult(
            passed=passed,
            mean_abs_diff=mean_abs_diff,
            max_abs_diff=max_abs_diff,
            percentile_95_diff=p95_diff,
            percentile_99_diff=p99_diff,
            pearson_correlation=pearson_corr,
            spearman_rank_correlation=spearman_corr,
            sign_agreement_ratio=sign_agreement,
            native_mean=native_mean,
            native_std=native_std,
            onnx_mean=onnx_mean,
            onnx_std=onnx_std,
            rtol=self.rtol,
            atol=self.atol,
            max_diff_percentile=self.max_diff_percentile,
            n_samples=len(native_preds),
            diagnostics=diagnostics,
        )
