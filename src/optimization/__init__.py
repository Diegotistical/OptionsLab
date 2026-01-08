# src/optimization/__init__.py

from src.optimization.model_wrappers import (
    create_mlp_optimizer,
    create_monte_carlo_ml_optimizer,
    optimize_and_export_onnx,
)
from src.optimization.objectives import (
    create_lgbm_objective,
    create_pytorch_objective,
    create_sklearn_objective,
)
from src.optimization.onnx_exporter import ONNXExporter, ONNXExportResult
from src.optimization.onnx_runtime import ONNXInferenceEngine
from src.optimization.onnx_validator import ONNXValidator, ValidationResult
from src.optimization.reproducibility import (
    ReproducibilityConfig,
    get_trial_seed,
    set_global_seed,
    set_thread_limits,
    set_torch_determinism,
)
from src.optimization.search_space import (
    InvalidSearchSpaceError,
    LightGBMSearchSpace,
    MLPSearchSpace,
    SearchSpace,
    XGBoostSearchSpace,
)
from src.optimization.study_manager import (
    OptunaStudyManager,
    StudyMetadata,
    StudyResult,
)

__all__ = [
    # Reproducibility
    "ReproducibilityConfig",
    "set_global_seed",
    "get_trial_seed",
    "set_thread_limits",
    "set_torch_determinism",
    # Search spaces
    "SearchSpace",
    "LightGBMSearchSpace",
    "MLPSearchSpace",
    "XGBoostSearchSpace",
    "InvalidSearchSpaceError",
    # Study management
    "OptunaStudyManager",
    "StudyResult",
    "StudyMetadata",
    # Objectives
    "create_lgbm_objective",
    "create_pytorch_objective",
    "create_sklearn_objective",
    # ONNX
    "ONNXExporter",
    "ONNXExportResult",
    "ONNXValidator",
    "ValidationResult",
    "ONNXInferenceEngine",
    # Model wrappers
    "create_monte_carlo_ml_optimizer",
    "create_mlp_optimizer",
    "optimize_and_export_onnx",
]
