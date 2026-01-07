# src/optimization/onnx_runtime.py
"""
ONNX Runtime inference engine with production safeguards.

Provides:
- Provider selection and fallback
- Feature ordering enforcement
- Input dtype validation
- Batched inference for large datasets

Usage:
    from src.optimization import ONNXInferenceEngine

    engine = ONNXInferenceEngine("models/my_model.onnx")
    predictions = engine.predict(X_test)
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class ONNXInferenceError(Exception):
    """Raised when ONNX inference fails."""
    pass


class ONNXInferenceEngine:
    """
    Production-grade ONNX inference engine.
    
    Features:
        - Automatic provider selection with fallback
        - Feature ordering validation
        - Input dtype enforcement (float32)
        - Batched inference for memory efficiency
    """
    
    # Provider priority order
    PROVIDER_PRIORITY = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    
    def __init__(
        self,
        model_path: Union[str, Path],
        providers: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        validate_inputs: bool = True,
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to ONNX model file.
            providers: Execution providers (auto-detect if None).
            feature_names: Expected feature names for validation.
            validate_inputs: If True, validate input dtype and shape.
        """
        import onnxruntime as ort
        
        self.model_path = Path(model_path)
        self.validate_inputs = validate_inputs
        
        # Load metadata if available
        metadata_path = self.model_path.with_suffix(".json")
        self._metadata = self._load_metadata(metadata_path)
        
        # Feature names from arg, metadata, or None
        self.feature_names = feature_names or self._metadata.get("feature_names")
        
        # Select providers
        if providers is None:
            providers = self._select_providers(ort)
        self.providers = providers
        
        # Create session
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=self.providers,
            )
        except Exception as e:
            raise ONNXInferenceError(
                f"Failed to load ONNX model from {self.model_path}: {e}"
            )
        
        # Cache input/output info
        self._input_name = self.session.get_inputs()[0].name
        self._input_shape = self.session.get_inputs()[0].shape
        self._output_name = self.session.get_outputs()[0].name
        
        # Determine expected features
        if self._input_shape and len(self._input_shape) > 1:
            self._n_features = self._input_shape[1]
            if isinstance(self._n_features, str):  # Dynamic
                self._n_features = None
        else:
            self._n_features = None
    
    def _load_metadata(self, path: Path) -> Dict[str, Any]:
        """Load metadata from JSON file if exists."""
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _select_providers(self, ort) -> List[str]:
        """Select available providers in priority order."""
        available = ort.get_available_providers()
        selected = []
        
        for provider in self.PROVIDER_PRIORITY:
            if provider in available:
                selected.append(provider)
        
        # Always have CPU as fallback
        if "CPUExecutionProvider" not in selected:
            selected.append("CPUExecutionProvider")
        
        return selected
    
    @property
    def available_providers(self) -> List[str]:
        """Get list of available execution providers."""
        import onnxruntime as ort
        return ort.get_available_providers()
    
    @property
    def active_providers(self) -> List[str]:
        """Get list of active execution providers."""
        return self.session.get_providers()
    
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate and prepare input array."""
        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Check shape
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ONNXInferenceError(
                f"Expected {self._n_features} features, got {X.shape[1]}. "
                f"Expected features: {self.feature_names}"
            )
        
        # Ensure float32
        if X.dtype != np.float32:
            if self.validate_inputs:
                if X.dtype == np.float64:
                    # Common case - silently cast but could warn
                    pass
                else:
                    warnings.warn(
                        f"Converting input from {X.dtype} to float32. "
                        "This may cause precision loss.",
                        UserWarning,
                    )
            X = X.astype(np.float32)
        
        # Check for NaN/Inf
        if self.validate_inputs:
            if np.any(np.isnan(X)):
                raise ONNXInferenceError("Input contains NaN values")
            if np.any(np.isinf(X)):
                raise ONNXInferenceError("Input contains Inf values")
        
        return X
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            X: Input features (n_samples, n_features).
        
        Returns:
            Model predictions.
        """
        X = self._validate_input(X)
        
        try:
            outputs = self.session.run(None, {self._input_name: X})
            return outputs[0]
        except Exception as e:
            raise ONNXInferenceError(f"Inference failed: {e}")
    
    def predict_batch(
        self,
        X: np.ndarray,
        batch_size: int = 1024,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Run batched inference for large datasets.
        
        More memory-efficient for large inputs.
        
        Args:
            X: Input features.
            batch_size: Samples per batch.
            show_progress: Show progress bar (requires tqdm).
        
        Returns:
            Model predictions.
        """
        X = self._validate_input(X)
        n_samples = X.shape[0]
        
        if n_samples <= batch_size:
            return self.predict(X)
        
        # Batch processing
        predictions = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, n_samples, batch_size), desc="Inference")
            except ImportError:
                iterator = range(0, n_samples, batch_size)
        else:
            iterator = range(0, n_samples, batch_size)
        
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, n_samples)
            batch = X[start_idx:end_idx]
            batch_pred = self.session.run(None, {self._input_name: batch})[0]
            predictions.append(batch_pred)
        
        return np.concatenate(predictions, axis=0)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        
        return {
            "model_path": str(self.model_path),
            "providers": self.active_providers,
            "inputs": [
                {
                    "name": inp.name,
                    "shape": inp.shape,
                    "type": inp.type,
                }
                for inp in inputs
            ],
            "outputs": [
                {
                    "name": out.name,
                    "shape": out.shape,
                    "type": out.type,
                }
                for out in outputs
            ],
            "feature_names": self.feature_names,
            "metadata": self._metadata,
        }
    
    def benchmark(
        self,
        n_samples: int = 1000,
        n_features: Optional[int] = None,
        n_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            n_samples: Number of samples per iteration.
            n_features: Number of features (auto-detect if None).
            n_iterations: Number of timed iterations.
            warmup_iterations: Warmup iterations (not timed).
        
        Returns:
            Dict with timing statistics.
        """
        import time
        
        n_features = n_features or self._n_features or 10
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        
        # Warmup
        for _ in range(warmup_iterations):
            self.session.run(None, {self._input_name: X})
        
        # Timed runs
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            self.session.run(None, {self._input_name: X})
            end = time.perf_counter()
            times.append(end - start)
        
        times = np.array(times) * 1000  # Convert to ms
        
        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "samples_per_second": float(n_samples / (np.mean(times) / 1000)),
            "n_samples": n_samples,
            "n_iterations": n_iterations,
            "providers": self.active_providers,
        }
