# src/volatility_surface/svr_model.py

import numpy as np
import pandas as pd
import os
import logging
import joblib
import matplotlib.pyplot as plt
import warnings
from scipy.interpolate import griddata
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from ..base import VolatilityModelBase, VolatilityModelConfig



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addFilter(lambda record: not record.getMessage().startswith('Using backend'))  # Suppress sklearn backend messages
warnings.filterwarnings("ignore", category=UserWarning)

class SVRModel(VolatilityModelBase):
    """
    Enterprise-grade Support Vector Regression for volatility surface modeling
    Features:
    - Cross-validated training
    - Hyperparameter optimization
    - Uncertainty estimation via bootstrapping
    - Data leakage prevention
    - Full model versioning
    - Modular feature engineering
    """
    
    def __init__(self, config: VolatilityModelConfig):
        super().__init__(config)
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize model based on config"""
        self.model = SVR(
            kernel=self.config.kernel, 
            C=self.config.C, 
            epsilon=self.config.epsilon, 
            gamma=self.config.gamma
        )
        self.support_indices_ = None

    def _create_model_clone(self):
        """Create a clone of the SVR model"""
        return clone(self.model)

    def hyperparameter_tune(self,
                            df: pd.DataFrame,
                            param_grid: Dict[str, list],
                            n_splits: int = 5,
                            n_jobs: int = -1) -> Dict[str, Any]:
        """
        Cross-validated hyperparameter tuning using GridSearchCV
        """
        logger.info("Starting hyperparameter tuning...")
        
        # Split data first to prevent leakage
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=self.config.random_state, shuffle=True
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel=self.config.kernel, C=self.config.C, epsilon=self.config.epsilon, gamma=self.config.gamma))
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
        X_train, y_train = self._engineer(train_df), train_df['implied_volatility']
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters and scaler
        best_estimator = grid_search.best_estimator_
        self.model = clone(best_estimator.named_steps['svr'])
        self.model.fit(X_train, y_train)
        self.scaler = best_estimator.named_steps['scaler']  # 🔁 Sync scaler
        self.support_indices_ = self.model.support_
        self.trained = True
        
        # Validate on val_df
        X_val, y_val = self._engineer(val_df), val_df['implied_volatility']
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
            **val_metrics
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
        # Input validation
        required_cols = ['underlying_price', 'strike_price', 'time_to_maturity',
                         'risk_free_rate', 'historical_volatility', 'implied_volatility']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
            
        # Auto-tuning if enabled
        if auto_tune and param_grid is not None:
            self.hyperparameter_tune(df, param_grid, n_splits=n_splits)
            
        # Split data first to prevent leakage
        train_df, val_df = train_test_split(
            df, test_size=val_split, random_state=self.config.random_state, shuffle=True
        )
        
        # Feature engineering and scaling
        X_train, y_train = self._engineer(train_df), train_df['implied_volatility']
        X_val, y_val = self._engineer(val_df), val_df['implied_volatility']
        
        # Train model
        self.model.fit(X_train, y_train)
        self.support_indices_ = self.model.support_
        self.trained = True
        
        # Cross-validation on training subset only
        metrics = super().cross_validate(train_df, n_splits=n_splits)
        
        # Predict and evaluate
        y_tr_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        # Add training metrics
        metrics.update({
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_tr_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'train_r2': r2_score(y_train, y_tr_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'train_mae': mean_absolute_error(y_train, y_tr_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'support_vectors': self.support_indices_.shape[0]
        })
        
        return metrics

    def predict_volatility(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict volatility with out-of-domain warning
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
            
        X, y = self._prepare(df, fit=False)
        domain_status = self.domain_check(df)
        
        if not all(domain_status.values()):
            logger.warning("Prediction includes extrapolation beyond training domain")
                
        return self.model.predict(X)

    def uncertainty_intervals(self, df: pd.DataFrame, alpha: float = 0.95, n_bootstrap: int = 100) -> Dict[str, np.ndarray]:
        """
        Estimate prediction intervals using bootstrap resampling
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before estimating uncertainty")
            
        X, y = self._prepare(df, fit=False)
        base_pred = self.model.predict(X)
        
        # Bootstrap resampling using original data
        preds = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(X), len(X), replace=True)
            bootstrap_model = clone(self.model)
            bootstrap_model.fit(X[indices], y[indices])
            preds.append(bootstrap_model.predict(X))
        
        preds_array = np.array(preds)
        q_lower = 1 - alpha/2
        q_upper = alpha/2
        lower = np.percentile(preds_array, 100 * q_lower, axis=0)
        upper = np.percentile(preds_array, 100 * q_upper, axis=0)
        
        return {
            'lower': lower,
            'upper': upper,
            'base_prediction': base_pred,
            'confidence_level': alpha
        }

    def domain_check(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Check if prediction is within training domain using original features
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before domain check")
            
        X_scaled, X_original, _ = self._prepare(df, fit=False, return_original=True)
        
        # Use original data statistics
        train_min = X_original.min(axis=0)
        train_max = X_original.max(axis=0)
        
        # Check each feature's domain
        domain_issues = {}
        for i, col in enumerate(self.feature_columns):
            within_min = np.all(X_original[:, i] >= train_min[i] - 3 * (train_max[i] - train_min[i]))
            within_max = np.all(X_original[:, i] <= train_max[i] + 3 * (train_max[i] - train_min[i]))
            domain_issues[col] = within_min and within_max
            
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
            
            # Save metadata
            joblib.dump({
                'version': self.version,
                'feature_columns': self.feature_columns,
                'support_indices': self.support_indices_,
                'config': self.config.dict(),
                'trained': self.trained
            }, meta_path)
            
            logger.info(f"Saved model to {model_path}")
            return {'model': model_path, 'scaler': scaler_path, 'metadata': meta_path}
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, model_path: str, scaler_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Load model with full configuration recovery
        Parameters:
        model_path : str
            Path to model file
        scaler_path : str
        Returns: None
        """
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.trained = True
            
            if metadata_path and os.path.exists(metadata_path):
                meta = joblib.load(metadata_path)
                if meta['version'] != self.version:
                    logger.warning(f"Loading old model version {meta['version']}")
                self.feature_columns = meta.get('feature_columns', self.feature_columns)
                self.support_indices_ = meta.get('support_indices', None)
                self.config = VolatilityModelConfig(**meta.get('config', {}))
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
            'params': {'r': r, 'hv': hv, 'skew': skew}
        }