# src/optimization/study_manager.py
"""
Optuna study lifecycle management with production-grade features.

Provides:
- RDB backend (SQLite/Postgres) for persistence
- Study resume and versioning
- Metadata tagging (git commit, data hash)
- Resource control per trial
- Parallel trial safety

Usage:
    from src.optimization import OptunaStudyManager, LightGBMSearchSpace

    manager = OptunaStudyManager(
        study_name="mc_ml_v1",
        storage="sqlite:///optuna_studies.db",
        seed=42,
    )
    result = manager.optimize(objective_fn, LightGBMSearchSpace(), n_trials=100)
"""

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.optimization.reproducibility import (
    ReproducibilityConfig,
    get_trial_seed,
    set_global_seed,
    set_thread_limits,
    compute_data_hash,
)
from src.optimization.search_space import SearchSpace


@dataclass
class StudyMetadata:
    """Metadata for study tracking and reproducibility."""
    
    study_name: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    git_commit: Optional[str] = None
    data_hash: Optional[str] = None
    feature_set_version: Optional[str] = None
    notes: Optional[str] = None
    
    # Auto-populated
    python_version: Optional[str] = None
    optuna_version: Optional[str] = None
    
    def __post_init__(self):
        import sys
        self.python_version = sys.version
        self.optuna_version = optuna.__version__
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_name": self.study_name,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "data_hash": self.data_hash,
            "feature_set_version": self.feature_set_version,
            "notes": self.notes,
            "python_version": self.python_version,
            "optuna_version": self.optuna_version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StudyMetadata":
        return cls(
            study_name=data["study_name"],
            created_at=data.get("created_at", ""),
            git_commit=data.get("git_commit"),
            data_hash=data.get("data_hash"),
            feature_set_version=data.get("feature_set_version"),
            notes=data.get("notes"),
        )


@dataclass
class StudyResult:
    """Result of an optimization study."""
    
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    n_complete: int
    n_pruned: int
    n_failed: int
    study_name: str
    duration_seconds: float
    metadata: StudyMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": self.n_trials,
            "n_complete": self.n_complete,
            "n_pruned": self.n_pruned,
            "n_failed": self.n_failed,
            "study_name": self.study_name,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata.to_dict(),
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save result to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class OptunaStudyManager:
    """
    Production-grade Optuna study management.
    
    Features:
        - RDB backend with resume support
        - Deterministic per-trial seeding
        - Thread/resource control
        - Metadata tagging
        - Pruning with warmup
    
    Example:
        manager = OptunaStudyManager("my_study", seed=42)
        result = manager.optimize(
            objective=my_objective,
            search_space=LightGBMSearchSpace(),
            n_trials=100,
        )
        result.save("results/my_study.json")
    """
    
    def __init__(
        self,
        study_name: str,
        storage: Optional[str] = None,
        seed: int = 42,
        n_jobs: int = 1,
        omp_threads: int = 4,
        torch_threads: int = 4,
        direction: str = "minimize",
        pruner_warmup_steps: int = 5,
        pruner_n_startup_trials: int = 10,
        load_if_exists: bool = True,
    ):
        """
        Initialize study manager.
        
        Args:
            study_name: Unique name for the study.
            storage: RDB URL (e.g., "sqlite:///studies.db"). None for in-memory.
            seed: Global seed for reproducibility.
            n_jobs: Number of parallel trials.
            omp_threads: OpenMP thread limit per trial.
            torch_threads: PyTorch thread limit per trial.
            direction: "minimize" or "maximize".
            pruner_warmup_steps: Min steps before pruning.
            pruner_n_startup_trials: Trials before pruning starts.
            load_if_exists: If True, resume existing study.
        """
        self.study_name = study_name
        self.storage = storage or "sqlite:///optuna_studies.db"
        self.seed = seed
        self.n_jobs = n_jobs
        self.omp_threads = omp_threads
        self.torch_threads = torch_threads
        self.direction = direction
        self.load_if_exists = load_if_exists
        
        # Create sampler with seed
        self.sampler = TPESampler(seed=seed)
        
        # Create pruner with warmup
        self.pruner = MedianPruner(
            n_startup_trials=pruner_n_startup_trials,
            n_warmup_steps=pruner_warmup_steps,
        )
        
        # Metadata
        self.metadata = StudyMetadata(
            study_name=study_name,
            git_commit=self._get_git_commit(),
        )
        
        # Study instance (created on optimize)
        self._study: Optional[optuna.Study] = None
    
    @staticmethod
    def _get_git_commit() -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass
        return None
    
    def set_data_hash(self, data: np.ndarray) -> None:
        """Set data hash for reproducibility tracking."""
        self.metadata.data_hash = compute_data_hash(data)
    
    def set_feature_version(self, version: str) -> None:
        """Set feature set version."""
        self.metadata.feature_set_version = version
    
    def set_notes(self, notes: str) -> None:
        """Set study notes."""
        self.metadata.notes = notes
    
    def _create_or_load_study(self) -> optuna.Study:
        """Create new study or load existing one."""
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=self.load_if_exists,
        )
        
        # Store metadata as user attributes
        for key, value in self.metadata.to_dict().items():
            if value is not None:
                study.set_user_attr(key, str(value))
        
        return study
    
    def _setup_trial_environment(self, trial: optuna.Trial) -> int:
        """Setup reproducible environment for a trial."""
        trial_seed = get_trial_seed(self.seed, trial.number)
        
        # Set global seed for this trial
        set_global_seed(trial_seed)
        
        # Set thread limits
        set_thread_limits(
            n_jobs=1,  # Single job per trial
            omp_threads=self.omp_threads,
            torch_threads=self.torch_threads,
        )
        
        return trial_seed
    
    def optimize(
        self,
        objective: Callable[[optuna.Trial, int], float],
        search_space: SearchSpace,
        n_trials: int,
        timeout: Optional[int] = None,
        show_progress_bar: bool = True,
        callbacks: Optional[List[Callable]] = None,
    ) -> StudyResult:
        """
        Run optimization study.
        
        Args:
            objective: Objective function taking (trial, seed) -> float.
            search_space: SearchSpace instance for hyperparameter suggestions.
            n_trials: Number of trials to run.
            timeout: Optional timeout in seconds.
            show_progress_bar: Show progress bar.
            callbacks: Optional list of Optuna callbacks.
        
        Returns:
            StudyResult with best params and metadata.
        """
        # Create or load study
        self._study = self._create_or_load_study()
        
        start_time = time.time()
        
        def wrapped_objective(trial: optuna.Trial) -> float:
            # Setup reproducible environment
            trial_seed = self._setup_trial_environment(trial)
            
            # Suggest params from search space
            params = search_space.suggest(trial)
            
            # Validate params
            search_space.validate(params)
            
            # Store params as user attributes
            trial.set_user_attr("trial_seed", trial_seed)
            
            # Call user objective
            return objective(trial, trial_seed)
        
        # Run optimization
        self._study.optimize(
            wrapped_objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=show_progress_bar,
            callbacks=callbacks or [],
            catch=(Exception,),  # Catch and log exceptions
        )
        
        duration = time.time() - start_time
        
        # Count trial states
        trials = self._study.trials
        n_complete = sum(1 for t in trials if t.state == optuna.trial.TrialState.COMPLETE)
        n_pruned = sum(1 for t in trials if t.state == optuna.trial.TrialState.PRUNED)
        n_failed = sum(1 for t in trials if t.state == optuna.trial.TrialState.FAIL)
        
        return StudyResult(
            best_params=self._study.best_params,
            best_value=self._study.best_value,
            n_trials=len(trials),
            n_complete=n_complete,
            n_pruned=n_pruned,
            n_failed=n_failed,
            study_name=self.study_name,
            duration_seconds=duration,
            metadata=self.metadata,
        )
    
    def get_study(self) -> Optional[optuna.Study]:
        """Get the underlying Optuna study."""
        return self._study
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters from completed study."""
        if self._study is None:
            raise RuntimeError("No study has been run. Call optimize() first.")
        return self._study.best_params
    
    def get_trials_dataframe(self):
        """Get trials as pandas DataFrame."""
        if self._study is None:
            raise RuntimeError("No study has been run. Call optimize() first.")
        return self._study.trials_dataframe()
    
    def export_results(
        self,
        output_dir: Union[str, Path],
        include_parquet: bool = True,
    ) -> None:
        """
        Export study results to files.
        
        Creates:
            - {study_name}_best_params.json
            - {study_name}_trials.parquet (if include_parquet)
            - {study_name}_metadata.json
        """
        if self._study is None:
            raise RuntimeError("No study has been run. Call optimize() first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Best params
        params_path = output_dir / f"{self.study_name}_best_params.json"
        with open(params_path, "w") as f:
            json.dump(self._study.best_params, f, indent=2)
        
        # Metadata
        meta_path = output_dir / f"{self.study_name}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2, default=str)
        
        # Trials DataFrame
        if include_parquet:
            try:
                df = self._study.trials_dataframe()
                parquet_path = output_dir / f"{self.study_name}_trials.parquet"
                df.to_parquet(parquet_path)
            except Exception:
                # Fallback to CSV if parquet fails
                csv_path = output_dir / f"{self.study_name}_trials.csv"
                df.to_csv(csv_path, index=False)
    
    @classmethod
    def load_study(
        cls,
        study_name: str,
        storage: str = "sqlite:///optuna_studies.db",
    ) -> "OptunaStudyManager":
        """Load an existing study."""
        manager = cls(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
        manager._study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )
        return manager
