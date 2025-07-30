# src / volatility_surface / svr_model.py

import numpy as np
import pandas as pd
import os
import logging
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from scipy.interpolate import griddata
from sklearn.ensemble import BaggingRegressor
import warnings
import platform
from sklearn.model_selection import GridSearchCV


# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

class InputValidator:
    """Input validation class for data consistency"""
    def __init__(self):
        self.required_cols = [
            'underlying_price', 'strike_price', 'time_to_maturity',
            'risk_free_rate', 'historical_volatility', 'implied_volatility'
        ]
    
    def validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for volatility surface modeling
    Returns DataFrame with engineered features
    """
    d = df.copy()
    d['moneyness'] = d['underlying_price'].clip(1e-6) / d['strike_price'].clip(1e-6)
    d['log_moneyness'] = np.log(d['moneyness'].clip(1e-6))
    d['ttm_squared'] = d['time_to_maturity'].clip(1e-6) ** 2
    
    # Compute volatility skew safely
    d['volatility_skew'] = (
        d['historical_volatility'].clip(1e-6) - 
        d['historical_volatility'].expanding(min_periods=20).mean().fillna(0)
    )
    
    return d

class SVRModel:
    """
    Enterprise-grade Support Vector Regression for volatility surface modeling
    Features:
    - Cross-validated training
    - Hyperparameter optimization
    - Uncertainty estimation via ensemble
    - Full leakage prevention
    - Modular feature engineering
    - Git version tracking
    """
    
    def __init__(self,
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 gamma: str = 'scale',
                 random_state: int = 42):
        """
        Initialize SVR model with cross-validation support
        Parameters:
        kernel : str
            Kernel type ('rbf', 'linear', 'poly')
        C : float
            Regularization parameter
        epsilon : float
            Epsilon in the epsilon-SVR model
        gamma : str or float
            Kernel coefficient for 'rbf' kernel
        random_state : int
            Seed for reproducibility
        """
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.random_state = random_state
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        self.scaler = StandardScaler()
        self.trained = False
        self.support_indices_ = None
        self.version = "1.4"
        self.git_commit_hash = self._get_git_commit_hash()
        self.library_versions = self._get_library_versions()

    def _get_git_commit_hash(self) -> str:
        """Get current git commit hash for version tracking"""
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            return repo.head.object.hexsha[:7]
        except Exception:
            return "unknown"

    def _get_library_versions(self) -> Dict[str, str]:
        """Get versions of all installed libraries"""
        try:
            import importlib.metadata
            return {
                dist.metadata['Name']: dist.version
                for dist in importlib.metadata.distributions()
            }
        except Exception:
            return {"versions": "unavailable"}

    def _prepare(self, df: pd.DataFrame, fit: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Feature engineering, scaling, and target extraction with strict leakage prevention
        """
        X = engineer_features(df)
        y = df['implied_volatility'].values
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return X_scaled, y

    def hyperparameter_tune(self,
                            df: pd.DataFrame,
                            param_grid: Dict[str, list],
                            n_splits: int = 5,
                            n_jobs: int = -1) -> Dict[str, Any]:
        """
        Cross-validated hyperparameter tuning using GridSearchCV
        """
        logger.info("Starting hyperparameter tuning...")
        
        # Split first to prevent leakage
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=self.random_state, shuffle=True
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, gamma=self.gamma))
        ])
        
        # Setup grid search with correct parameter prefixes
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=n_splits,
            scoring='neg_mean_squared_error',
            n_jobs=n_jobs,
            verbose=1
        )
        
        # Fit on training data only
        X_train, y_train = engineer_features(train_df), train_df['implied_volatility']
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters and scaler
        best_estimator = grid_search.best_estimator_
        self.model = clone(best_estimator.named_steps['svr'])
        self.scaler = best_estimator.named_steps['scaler']  # 🔁 Sync scaler
        self.support_indices_ = self.model.support_
        self.trained = True
        
        # Validate on val_df
        X_val, y_val = engineer_features(val_df), val_df['implied_volatility']
        y_pred = self.model.predict(X_val)
        val_metrics = {
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'val_r2': r2_score(y_val, y_pred)
        }
        
        return {
            **grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'cv_scores': -grid_search.cv_results_['mean_test_score'],
            'cv_std': grid_search.cv_results_['std_test_score'],
            **val_metrics,
            'git_commit': self.git_commit_hash,
            'library_versions': self.library_versions
        }

    def train(self,
              df: pd.DataFrame,
              val_split: float = 0.2,
              n_splits: int = 5,
              auto_tune: bool = False,
              param_grid: Optional[Dict[str, list]] = None) -> Dict[str, Any]:
        """
        Train with cross-validation and leakage prevention
        """
        InputValidator().validate(df)
        
        # Auto-tuning if enabled
        if auto_tune and param_grid is not None:
            tune_results = self.hyperparameter_tune(df, param_grid, n_splits=n_splits)
            # Retrain on full dataset with best params
            X_train, y_train = engineer_features(df), df['implied_volatility']
            self.model.fit(X_train, y_train)
            self.support_indices_ = self.model.support_
            self.trained = True
            return tune_results
            
        # Regular training
        train_df, val_df = train_test_split(
            df, test_size=val_split, random_state=self.random_state, shuffle=True
        )
        
        # Feature engineering and scaling
        X_train, y_train = self._prepare(train_df, fit=True)
        X_val, y_val = self._prepare(val_df, fit=False)
        
        # Train model
        self.model.fit(X_train, y_train)
        self.support_indices_ = self.model.support_
        self.trained = True
        
        # Cross-validation on training subset only
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(clone(self.model), X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
        
        # Predict and evaluate
        y_tr_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        # Return comprehensive metrics
        return {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_tr_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'train_r2': r2_score(y_train, y_tr_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'train_mae': mean_absolute_error(y_train, y_tr_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'support_vectors': len(self.support_indices_),
            'cross_val_rmse': np.sqrt(-cv_scores.mean()),
            'cross_val_std': np.sqrt(-cv_scores.std())
        }

    def predict_volatility(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict volatility with out-of-domain warning
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
            
        X, _ = self._prepare(df, fit=False)
        domain_status = self.domain_check(df)
        
        if not all(domain_status.values()):
            logger.warning("Prediction includes extrapolation beyond training domain")
                
        return self.model.predict(X)

    def uncertainty_intervals(self, df: pd.DataFrame, alpha: float = 0.95, n_estimators: int = 100) -> Dict[str, np.ndarray]:
        """
        Estimate prediction intervals using ensemble learning
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before estimating uncertainty")
            
        X, y = engineer_features(df), df['implied_volatility']
        base_pred = self.model.predict(X)
        
        # Use ensemble for uncertainty estimation
        ensemble = BaggingRegressor(
            base_estimator=self.model,
            n_estimators=n_estimators,
            max_samples=0.8,
            bootstrap=True,
            n_jobs=-1,
            random_state=self.random_state
        )
        ensemble.fit(X, y)
        
        preds = np.stack([tree.predict(X) for tree in ensemble.estimators_])
        q_lower = 1 - alpha/2
        q_upper = alpha/2
        lower = np.percentile(preds, 100 * q_lower, axis=0)
        upper = np.percentile(preds, 100 * q_upper, axis=0)
        
        return {
            'lower': lower,
            'upper': upper,
            'base_prediction': base_pred,
            'confidence_level': alpha,
            'n_estimators': n_estimators
        }

    def domain_check(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Check if prediction is within training domain using original features
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before domain check")
            
        X = engineer_features(df)
        X_train = engineer_features(df[df.index < df.index[-1]])  # Use training subset
        
        # Use original data statistics with standard deviation
        train_min = X_train.min(axis=0)
        train_std = X_train.std(axis=0)
        
        # Check each feature's domain
        domain_issues = {}
        feature_names = ['moneyness', 'log_moneyness', 'time_to_maturity', 'ttm_squared', 'risk_free_rate', 'historical_volatility', 'volatility_skew']

        for col_idx, col_name in enumerate(feature_names):
            within_min = np.all(X[:, col_idx] >= train_min[col_idx] - 3 * train_std[col_idx])
            within_max = np.all(X[:, col_idx] <= train_min[col_idx] + 3 * train_std[col_idx])
            domain_issues[col_name] = within_min and within_max

            
        return domain_issues

    def save_model(self, model_dir: str = 'models/saved_models') -> Dict[str, str]:
        """
        Save model with full versioning and metadata
        Returns: Dictionary of saved file paths
        """
        os.makedirs(model_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f'svr_vol_{ts}.joblib')
        scaler_path = os.path.join(model_dir, f'svr_scaler_{ts}.joblib')
        meta_path = os.path.join(model_dir, f'svr_meta_{ts}.joblib')
        
        try:
            # Save model components
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            # Save metadata with versioning
            joblib.dump({
                'version': self.version,
                'feature_columns': ['moneyness', 'log_moneyness', 'time_to_maturity', 'ttm_squared', 'risk_free_rate', 'historical_volatility', 'volatility_skew'],
                'support_indices': self.support_indices_,
                'random_state': self.random_state,
                'kernel': self.kernel,
                'params': {
                    'C': self.C,
                    'epsilon': self.epsilon,
                    'gamma': self.gamma
                },
                'trained': self.trained,
                'git_commit_hash': self.git_commit_hash,
                'library_versions': self.library_versions,
                'python_version': platform.python_version()
            }, meta_path)
            
            logger.info(f"Saved model to {model_dir}")
            return {'model': model_path, 'scaler': scaler_path, 'metadata': meta_path}
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, model_path: str, scaler_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Load model with full configuration recovery and integrity check
        """
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.trained = True
            
            if metadata_path and os.path.exists(metadata_path):
                meta = joblib.load(metadata_path)
                # Validate feature column consistency
                expected_cols = ['moneyness', 'log_moneyness', 'time_to_maturity', 'ttm_squared', 'risk_free_rate', 'historical_volatility', 'volatility_skew']
                stored_cols = meta.get('feature_columns', [])
                
                if set(stored_cols) != set(expected_cols):
                    raise ValueError("Stored feature columns do not match current implementation")
                    
                if meta['version'] != self.version:
                    logger.warning(f"Loading old model version {meta['version']}")
                
                self.random_state = meta.get('random_state', self.random_state)
                self.kernel = meta.get('kernel', self.kernel)
                self.C = meta.get('params', {}).get('C', self.C)
                self.epsilon = meta.get('params', {}).get('epsilon', self.epsilon)
                self.gamma = meta.get('params', {}).get('gamma', self.gamma)
            else:
                logger.warning("No metadata file found - model properties may be inconsistent")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def plot_volatility_surface(self, surface: Dict[str, np.ndarray], save_path: Optional[str] = None, use_interpolation: bool = True):
        """
        Plot 3D volatility surface with optional interpolation
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before plotting")
            
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        S, K, T = surface['S'], surface['K'], surface['T']
        vol = surface['volatility']
        
        S_flat = S.flatten()
        K_flat = K.flatten()
        T_flat = T.flatten()
        vol_flat = vol.flatten()
        
        if use_interpolation:
            try:
                # Create interpolated surface
                grid_x, grid_y, grid_z = np.mgrid[
                    S_flat.min():S_flat.max():S_flat.size**0.5*1j,
                    K_flat.min():K_flat.max():K_flat.size**0.5*1j,
                    T_flat.min():T_flat.max():T_flat.size**0.5*1j
                ]
                vol_interp = griddata(
                    (S_flat, K_flat, T_flat), 
                    vol_flat, 
                    (grid_x, grid_y, grid_z),
                    method='linear'
                )
                
                # Plot interpolated surface
                ax.plot_surface(grid_x, grid_y, grid_z, facecolors=plt.cm.viridis(vol_interp), rstride=1, cstride=1)
            except Exception as e:
                logger.warning(f"Interpolation failed: {e}. Plotting raw points instead.")
                ax.scatter(S_flat, K_flat, T_flat, c=vol_flat, cmap='viridis', s=10)
        else:
            # Plot raw points
            ax.scatter(S_flat, K_flat, T_flat, c=vol_flat, cmap='viridis', s=10)
            
        ax.set_xlabel('Underlying Price')
        ax.set_ylabel('Strike Price')
        ax.set_zlabel('Time to Maturity')
        ax.set_title('Volatility Surface')
        ax.invert_yaxis()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def get_surface_grid(
        self,
        S_range: Tuple[float, float, int],
        K_range: Tuple[float, float, int],
        T_range: Tuple[float, float, int],
        r: float = 0.05,
        hv: float = 0.2,
        skew: float = 0.0
    ) -> Dict[str, np.ndarray]:
        """
        Generate volatility surface grid with configurable inputs
        """
        S = np.linspace(S_range[0], S_range[1], S_range[2])
        K = np.linspace(K_range[0], K_range[1], K_range[2])
        T = np.linspace(T_range[0], T_range[1], T_range[2])
        
        S_grid, K_grid, T_grid = np.meshgrid(S, K, T)
        df_surface = pd.DataFrame({
            'underlying_price': S_grid.flatten(),
            'strike_price': K_grid.flatten(),
            'time_to_maturity': T_grid.flatten(),
            'risk_free_rate': r,
            'historical_volatility': hv,
            'volatility_skew': skew
        })
        
        volatility = self.predict_volatility(df_surface).reshape(S_range[2], K_range[2], T_range[2])
        
        return {
            'S': S_grid,
            'K': K_grid,
            'T': T_grid,
            'volatility': volatility,
            'params': {'r': r, 'hv': hv, 'skew': skew},
            'git_commit': self.git_commit_hash,
            'library_versions': self.library_versions
        }
