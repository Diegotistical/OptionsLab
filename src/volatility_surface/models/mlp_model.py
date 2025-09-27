# src / volatility_surface / models / mlp_model.py

import threading
import os
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from ..base import VolatilityModelBase
from ..utils.feature_engineering import engineer_features
from ..utils.tensor_utils import ensure_tensor


FEATURE_COLUMNS = [
    'moneyness', 'log_moneyness', 'time_to_maturity',
    'ttm_squared', 'risk_free_rate',
    'historical_volatility', 'volatility_skew'
]

class MLPModel(VolatilityModelBase, nn.Module):
    def __init__(self,
                 hidden_layers: list = [64, 32],
                 activation: str = 'GELU',
                 activation_kwargs: Optional[Dict] = None,
                 dropout_rate: float = 0.2,
                 smoothness_weight: float = 0.0,
                 use_batchnorm: bool = True,
                 use_dropout: bool = True,
                 scaler_type: str = 'standard',
                 learning_rate: float = 0.001,
                 epochs: int = 200,
                 early_stopping_patience: int = 15,
                 random_seed: int = 42,
                 **kwargs
                 ):
        VolatilityModelBase.__init__(self, feature_columns=FEATURE_COLUMNS, enable_benchmark=True)
        nn.Module.__init__(self)
        self._lock = threading.RLock()

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

        self._initialize_scaler()

        self.optimizer = None
        self.scheduler = None

    def _create_activation(self):
        act_cls = getattr(nn, self.activation_name, None)
        if act_cls is None:
            raise ValueError(f"Unsupported activation: {self.activation_name}")
        return act_cls(**self.activation_kwargs)

    def _initialize_scaler(self):
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        if self.scaler_type not in scalers:
            raise ValueError(f"Unsupported scaler type: {self.scaler_type}")
        self.scaler = scalers[self.scaler_type]

    def _prepare_data(self, df: pd.DataFrame, targets: Optional[np.ndarray] = None):
        df = df.copy()
        df['time_to_maturity'] = np.maximum(df['time_to_maturity'], 1e-5)
        features = engineer_features(df)
        features_np = features.values
        if not self.trained:
            scaled = self.scaler.fit_transform(features_np)
        else:
            scaled = self.scaler.transform(features_np)
        features_tensor = ensuretensor(scaled, dtype=torch.float32, device=self.device)
        if targets is not None:
            targets_tensor = ensuretensor(targets.astype(np.float32), device=self.device)
            return features_tensor, targets_tensor
        return features_tensor

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _compute_loss(self, outputs, targets, inputs=None):
        base_loss = nn.MSELoss()(outputs.squeeze(), targets)
        if self.smoothness_weight > 0 and inputs is not None and self.training:
            grads = torch.autograd.grad(outputs.sum(), inputs, create_graph=True)[0]
            smoothness = grads.pow(2).mean()
            return base_loss + self.smoothness_weight * smoothness
        return base_loss

    def _train_impl(self, df: pd.DataFrame, val_split: float = 0.2) -> Dict[str, float]:
        with self._lock:
            self._on_train_start(df)
            train_df, val_df = train_test_split(df, test_size=val_split, random_state=42)
            X_train, y_train = self._prepare_data(train_df, train_df['implied_volatility'].values)
            X_val, y_val = self._prepare_data(val_df, val_df['implied_volatility'].values)

            train_ds = torch.utils.data.TensorDataset(X_train, y_train)
            val_ds = torch.utils.data.TensorDataset(X_val, y_val)
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32)

            self.optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, factor=0.5)

            best_val_loss = float('inf')
            patience_counter = 0
            for epoch in range(self.epochs):
                self.train()
                train_loss = 0.0
                for Xb, yb in train_loader:
                    self.optimizer.zero_grad()
                    out = self.forward(Xb)
                    loss = self._compute_loss(out, yb, Xb if self.smoothness_weight > 0 else None)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    train_loss += loss.item() * Xb.size(0)
                train_loss /= len(train_loader.dataset)

                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for Xb, yb in val_loader:
                        out = self.forward(Xb)
                        loss = self._compute_loss(out, yb)
                        val_loss += loss.item() * Xb.size(0)
                val_loss /= len(val_loader.dataset)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_state = self.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    break

                self.scheduler.step(val_loss)

            self.load_state_dict(self.best_state)
            self.trained = True
            self._on_train_end({"train_loss": train_loss, "val_loss": best_val_loss})
            return {"train_loss": train_loss, "val_loss": best_val_loss}

    def predict_volatility(self, df: pd.DataFrame, mc_samples: int = 1) -> np.ndarray:
        with self._lock:
            self._on_predict_start(df)
            features = self._prepare_data(df)
            if mc_samples == 1:
                self.eval()
                with torch.no_grad():
                    preds = self.forward(features)
                preds_np = preds.cpu().numpy().flatten()
                self._on_predict_end(preds_np)
                return preds_np

            # MC Dropout inference
            self.train()  # enable dropout
            # Freeze BatchNorm stats
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

            preds_samples = []
            with torch.no_grad():
                for _ in range(mc_samples):
                    preds = self.forward(features)
                    preds_samples.append(preds.cpu().numpy().flatten())

            self.eval()  # back to eval mode

            preds_samples = np.array(preds_samples)
            mean_preds = preds_samples.mean(axis=0)
            self._on_predict_end(mean_preds)
            return mean_preds

    def _save_model_impl(self, model_path: str, scaler_path: str) -> None:
        with self._lock:
            torch.save({'model_state_dict': self.state_dict()}, model_path)
            joblib.dump(self.scaler, scaler_path)

    def _load_model_impl(self, model_path: str, scaler_path: str) -> None:
        with self._lock:
            data = torch.load(model_path, map_location=self.device)
            self.load_state_dict(data['model_state_dict'])
            self.scaler = joblib.load(scaler_path)
            self.trained = True

