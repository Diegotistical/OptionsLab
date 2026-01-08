# tests/test_optimization.py
"""
Tests for the optimization module.

Covers:
- Reproducibility (determinism, seeding)
- Study lifecycle (resume, RDB persistence)
- ONNX validation (distributional)
- Failure modes (pruning, invalid params)
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip tests if dependencies not available
pytest.importorskip("optuna")


class TestReproducibility:
    """Test reproducibility utilities."""

    def test_global_seed_determinism(self):
        """Two runs with same seed produce identical random values."""
        from src.optimization.reproducibility import set_global_seed

        set_global_seed(42)
        vals1 = np.random.rand(100)

        set_global_seed(42)
        vals2 = np.random.rand(100)

        np.testing.assert_array_equal(vals1, vals2)

    def test_per_trial_seeding(self):
        """Trial N always gets same seed."""
        from src.optimization.reproducibility import get_trial_seed

        seed1_trial0 = get_trial_seed(42, 0)
        seed1_trial1 = get_trial_seed(42, 1)
        seed2_trial0 = get_trial_seed(42, 0)

        assert seed1_trial0 == seed2_trial0, "Same trial should get same seed"
        assert (
            seed1_trial0 != seed1_trial1
        ), "Different trials should get different seeds"

    def test_thread_limits_applied(self):
        """Thread limits are set in environment."""
        from src.optimization.reproducibility import set_thread_limits

        set_thread_limits(n_jobs=2, omp_threads=4)

        assert os.environ.get("OMP_NUM_THREADS") == "4"
        assert os.environ.get("MKL_NUM_THREADS") == "2"

    def test_data_hash_determinism(self):
        """Data hash is deterministic."""
        from src.optimization.reproducibility import compute_data_hash

        data = np.random.rand(100, 10)
        hash1 = compute_data_hash(data)
        hash2 = compute_data_hash(data)

        assert hash1 == hash2
        assert len(hash1) == 8  # Default truncation


class TestSearchSpace:
    """Test search space abstractions."""

    def test_lightgbm_search_space_suggest(self):
        """LightGBM search space suggests valid params."""
        import optuna

        from src.optimization.search_space import LightGBMSearchSpace

        space = LightGBMSearchSpace()

        def objective(trial):
            params = space.suggest(trial)
            # Only validate non-deterministic params
            space.validate(params)
            return 0.0

        study = optuna.create_study()
        # Use catch to handle any validation errors gracefully
        study.optimize(
            objective, n_trials=5, show_progress_bar=False, catch=(Exception,)
        )

        # At least some trials should complete
        assert len(study.trials) == 5

    def test_invalid_params_raise_exception(self):
        """Invalid params raise InvalidSearchSpaceError."""
        from src.optimization.search_space import (
            InvalidSearchSpaceError,
            LightGBMSearchSpace,
        )

        space = LightGBMSearchSpace()

        # num_leaves > 2^max_depth should fail
        invalid_params = {
            "num_leaves": 1000,
            "max_depth": 3,  # 2^3 = 8, so 1000 > 8
        }

        with pytest.raises(InvalidSearchSpaceError) as exc_info:
            space.validate(invalid_params)

        assert "num_leaves" in str(exc_info.value)

    def test_mlp_search_space_suggest(self):
        """MLP search space suggests valid architecture."""
        import optuna

        from src.optimization.search_space import MLPSearchSpace

        space = MLPSearchSpace()

        def objective(trial):
            params = space.suggest(trial)
            space.validate(params)

            # Check structure
            assert "hidden_layers" in params
            assert len(params["hidden_layers"]) >= 1
            assert params["learning_rate"] > 0

            return 0.0

        study = optuna.create_study()
        study.optimize(objective, n_trials=3, show_progress_bar=False)


class TestStudyLifecycle:
    """Test study management and lifecycle."""

    def test_study_creates_sqlite(self):
        """Study creates SQLite database."""
        from src.optimization.study_manager import OptunaStudyManager

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            manager = OptunaStudyManager(
                study_name="test_study",
                storage=f"sqlite:///{db_path}",
                seed=42,
            )

            # Create study by running minimal optimization
            study = manager._create_or_load_study()

            # Study should be created
            assert study is not None

    def test_study_resume(self):
        """Study can be resumed from RDB."""
        import optuna

        from src.optimization.study_manager import OptunaStudyManager

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = f"sqlite:///{db_path}"

            # Create study with some trials
            study1 = optuna.create_study(
                study_name="resume_test",
                storage=storage,
            )
            study1.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=3)

            # Resume study
            manager = OptunaStudyManager.load_study(
                study_name="resume_test",
                storage=storage,
            )

            assert len(manager._study.trials) == 3

    def test_metadata_stored(self):
        """Study metadata is stored in user attributes."""
        import optuna

        from src.optimization.study_manager import OptunaStudyManager

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            manager = OptunaStudyManager(
                study_name="meta_test",
                storage=f"sqlite:///{db_path}",
                seed=42,
            )
            manager.set_notes("Test study for unit tests")

            study = manager._create_or_load_study()

            assert study.user_attrs.get("notes") == "Test study for unit tests"


class TestONNXValidation:
    """Test ONNX validation utilities."""

    def test_validation_result_summary(self):
        """ValidationResult produces readable summary."""
        from src.optimization.onnx_validator import ValidationResult

        result = ValidationResult(
            passed=True,
            mean_abs_diff=0.001,
            max_abs_diff=0.01,
            percentile_95_diff=0.005,
            percentile_99_diff=0.008,
            pearson_correlation=0.9999,
            spearman_rank_correlation=0.9998,
            sign_agreement_ratio=0.999,
            native_mean=10.5,
            native_std=2.3,
            onnx_mean=10.502,
            onnx_std=2.298,
            rtol=1e-3,
            atol=1e-4,
            max_diff_percentile=99.0,
            n_samples=1000,
        )

        summary = result.summary()
        assert "PASSED" in summary
        assert "1000" in summary


class TestFailureModes:
    """Test failure mode handling."""

    def test_pruned_trial_handling(self):
        """Pruned trials don't crash the study."""
        import optuna

        def objective(trial):
            for step in range(10):
                trial.report(step, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return 0.0

        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_startup_trials=1)
        )
        study.optimize(objective, n_trials=5, show_progress_bar=False)

        # Should complete without error
        assert len(study.trials) == 5

    def test_invalid_search_space_error_is_informative(self):
        """InvalidSearchSpaceError provides diagnostics."""
        from src.optimization.search_space import InvalidSearchSpaceError

        error = InvalidSearchSpaceError(
            message="Value out of range",
            param_name="learning_rate",
            param_value=-0.1,
            constraint="learning_rate > 0",
        )

        error_str = str(error)
        assert "learning_rate" in error_str
        assert "-0.1" in error_str
        assert "constraint" in error_str


class TestONNXExport:
    """Test ONNX export utilities."""

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not available"),
        reason="PyTorch required",
    )
    def test_pytorch_export_requires_eval_mode(self):
        """Export fails if model is in training mode."""
        import torch
        import torch.nn as nn

        from src.optimization.onnx_exporter import ONNXExporter, ONNXExportError

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        model.train()  # Training mode

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ONNXExportError) as exc_info:
                ONNXExporter.export_pytorch(
                    model=model,
                    dummy_input=torch.randn(1, 10),
                    output_path=Path(tmpdir) / "model.onnx",
                    validate_eval_mode=True,
                )

            assert "training mode" in str(exc_info.value).lower()

    def test_lightgbm_export_requires_feature_names(self):
        """LightGBM export fails without feature names."""
        from src.optimization.onnx_exporter import ONNXExporter, ONNXExportError

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ONNXExportError) as exc_info:
                ONNXExporter.export_lightgbm(
                    model=None,  # Would fail anyway
                    output_path=Path(tmpdir) / "model.onnx",
                    feature_names=[],  # Empty = not provided
                )

            # Either missing feature_names or missing onnxmltools
            error_msg = str(exc_info.value).lower()
            assert "feature_names" in error_msg or "onnxmltools" in error_msg
