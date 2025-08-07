# src/volatility_surface/mlp_model.py

import os
import joblib
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from ..base import VolatilityModelBase
from ..utils.feature_engineering import engineer_features
from ..utils.tensor_utils import ensuretensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FEATURE_COLUMNS = [
    'moneyness', 'log_moneyness', 'time_to_maturity',
    'ttm_squared', 'risk_free_rate',
    'historical_volatility', 'volatility_skew'
]


class MLPModel(VolatilityModelBase, nn.Module):
    def __init__(
        self,
        hidden_layers: List[int] = [64, 32],
        activation: str = 'GELU',
        activation_kwargs: Dict[str, Any] = None,
        dropout_rate: float = 0.2,
        smoothness_weight: float = 0.0,
        use_batchnorm: bool = True,
        use_dropout: bool = True,
        scaler_type: str = 'standard',
        learning_rate: float = 0.001,
        epochs: int = 200,
        random_seed: int = 42,
        early_stopping_patience: int = 15,
        num_workers: int = 0,
        **kwargs
    ):
        super().__init__(feature_columns=FEATURE_COLUMNS)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.activation_kwargs = activation_kwargs or {}
        self.activation = self._create_activation()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.smoothness_weight = smoothness_weight
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.num_workers = num_workers
        self.scaler_type = scaler_type

        layers = []
        prev_dim = len(self.feature_columns)
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(self.activation)
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.trained = False
        self._initialize_scaler()
        self.best_state = None
        self.best_loss = float('inf')
        self.best_optimizer_state = None
        self.optimizer = None
        self.scheduler = None
        self.train_metrics = {}
        self.val_metrics = {}
        self.arbitrage_violations = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _create_activation(self) -> nn.Module:
        activation_class = getattr(nn, self.activation_name, None)
        if activation_class is None:
            raise ValueError(f"Unsupported activation: {self.activation_name}")
        return activation_class(**self.activation_kwargs)

    def _initialize_scaler(self):
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        if self.scaler_type not in scalers:
            raise ValueError(f"Unsupported scaler type: {self.scaler_type}")
        self.scaler = scalers[self.scaler_type]

    def _prepare_data(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        targets: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            df['time_to_maturity'] = np.maximum(df['time_to_maturity'], 1e-5)
            df['underlying_price'] = np.maximum(df['underlying_price'], 1e-5)
            df['strike_price'] = np.maximum(df['strike_price'], 1e-5)
            features = engineer_features(df)
        elif isinstance(data, np.ndarray):
            data = data.copy()
            data[:, 2] = np.maximum(data[:, 2], 1e-5)
            data[:, 0] = np.maximum(data[:, 0], 1e-5)
            data[:, 1] = np.maximum(data[:, 1], 1e-5)
            df = pd.DataFrame(data, columns=[
                'underlying_price', 'strike_price', 'time_to_maturity',
                'risk_free_rate', 'historical_volatility'
            ])
            features = engineer_features(df)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        features_np = features.values
        scaled = self.scaler.fit_transform(features_np) if not self.trained else self.scaler.transform(features_np)
        features_tensor = ensuretensor(scaled, dtype=torch.float32, device=self.device)

        if targets is not None:
            targets_tensor = ensuretensor(targets, dtype=torch.float32, device=self.device)
            return features_tensor, targets_tensor

        return features_tensor

    def _compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        inputs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        targets = ensuretensor(targets, dtype=torch.float32, device=outputs.device)
        base_loss = nn.MSELoss()(outputs.squeeze(), targets)
        if self.smoothness_weight > 0 and inputs is not None and self.training:
            grads = torch.autograd.grad(outputs.sum(), inputs, create_graph=True)[0]
            smoothness = grads.pow(2).mean()
            return base_loss + self.smoothness_weight * smoothness
        return base_loss

    def fit(self, df: pd.DataFrame, val_split: float = 0.2, batch_size: int = 32) -> Dict[str, Any]:
        train_df, val_df = train_test_split(df, test_size=val_split, random_state=42)
        train_features, train_targets = self._prepare_data(train_df, train_df['implied_volatility'].values)
        val_features, val_targets = self._prepare_data(val_df, val_df['implied_volatility'].values)

        train_dataset = TensorDataset(train_features, train_targets)
        val_dataset = TensorDataset(val_features, val_targets)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        self.optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, factor=0.5)

        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(self.epochs):
            self.train()
            train_loss = 0.0
            for X, y in train_loader:
                self.optimizer.zero_grad()
                out = self.forward(X)
                loss = self._compute_loss(out, y, X if self.smoothness_weight > 0 else None)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item() * X.size(0)
            train_loss /= len(train_loader.dataset)

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X, y in val_loader:
                    out = self.forward(X)
                    loss = self._compute_loss(out, y)
                    val_loss += loss.item() * X.size(0)
            val_loss /= len(val_loader.dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_state = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

            self.scheduler.step(val_loss)

        self.load_state_dict(self.best_state)
        self.trained = True
        self.best_loss = best_val_loss
        return {"train_loss": train_loss, "val_loss": best_val_loss}

    def predict_volatility(self, df: pd.DataFrame, n_samples: int = 1) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        features_tensor = self._prepare_data(df)
        self.eval()

        if n_samples == 1:
            with torch.no_grad():
                preds = self.forward(features_tensor)
                preds = ensuretensor(preds, device='cpu')
            return preds.numpy().flatten(), None

        else:
            self.train()
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

            samples = []
            with torch.no_grad():
                for _ in range(n_samples):
                    out = self.forward(features_tensor)
                    out = ensuretensor(out, device='cpu')
                    samples.append(out.numpy().flatten())

            samples = np.array(samples)
            return samples.mean(axis=0), samples.std(axis=0)

    def save_model(self, model_dir: str = 'models/saved_models') -> Dict[str, str]:
        os.makedirs(model_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f'mlp_vol_{ts}.pt')
        scaler_path = os.path.join(model_dir, f'mlp_scaler_{ts}.joblib')
        torch.save({'model_state_dict': self.state_dict()}, model_path)
        joblib.dump(self.scaler, scaler_path)
        return {'model': model_path, 'scaler': scaler_path}

    @classmethod
    def load_model(cls, model_path: str, scaler_path: str) -> 'MLPModel':
        model_data = torch.load(model_path)
        scaler = joblib.load(scaler_path)
        model = cls()
        model.load_state_dict(model_data['model_state_dict'])
        model.scaler = scaler
        model.trained = True
        return model
