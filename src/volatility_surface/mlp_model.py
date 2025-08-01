# src/volatility_surface/models/mlp_model.py

import os
import joblib
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from ..base import VolatilityModelBase
from ..utils.tensor_utils import ensuretensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def engineer_features(data: torch.Tensor) -> torch.Tensor:
    """Generates 7 features with 5 inputs: S, K, T, r, vol"""
    if not isinstance(data, torch.Tensor):
        data = ensuretensor(data, dtype=torch.float32, requires_grad=False)
    assert data.shape[1] == 5, "Esperaba 5 features: S, K, T, r, vol"

    S = data[:, 0].clamp(min=1e-6)
    K = data[:, 1].clamp(min=1e-6)
    T = data[:, 2].clamp(min=1e-6)
    r = data[:, 3]
    vol = data[:, 4].clamp(min=1e-6)

    moneyness = S / K
    log_moneyness = torch.log(moneyness.clamp(min=1e-6))
    ttm_squared = T ** 2
    vol_skew = vol - vol.mean()

    return torch.cat([
        moneyness.unsqueeze(1),
        log_moneyness.unsqueeze(1),
        T.unsqueeze(1),
        ttm_squared.unsqueeze(1),
        r.unsqueeze(1),
        vol.unsqueeze(1),
        vol_skew.unsqueeze(1)
    ], dim=1)


class MLPModel(VolatilityModelBase, nn.Module):
    def __init__(self,
                 input_dim: int = 7,
                 hidden_layers: list = [64, 32],
                 output_dim: int = 1,
                 activation: str = 'GELU',
                 dropout_rate: float = 0.2,
                 smoothness_weight: float = 0.0,
                 use_batchnorm: bool = True,
                 learning_rate: float = 0.001,
                 epochs: int = 50,
                 feature_columns: list = None,
                 **kwargs):

        if feature_columns is None:
            feature_columns = [
                'moneyness', 'log_moneyness', 'time_to_maturity',
                'ttm_squared', 'risk_free_rate',
                'historical_volatility', 'volatility_skew'
            ]
        VolatilityModelBase.__init__(self, feature_columns=feature_columns)
        nn.Module.__init__(self)

        # Hyperparameters
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.smoothness_weight = smoothness_weight
        self.activation = getattr(nn, activation)()
        self.use_batchnorm = use_batchnorm
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout_rate = dropout_rate

        # Architecture
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            else:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(self.activation)
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Estado de entrenamiento
        self.trained = False
        self.scaler = StandardScaler()
        self.best_state = None
        self.best_loss = float('inf')

    # Forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.smoothness_weight > 0 and self.training:
            x.requires_grad_(True)
        out = self.model(x)

        if self.smoothness_weight > 0 and self.training:
            grad_outputs = torch.autograd.grad(out, x, create_graph=True)[0]
            smoothness = sum(
                torch.autograd.grad(grad_outputs[:, i].sum(), x, create_graph=True)[0][:, i].pow(2).mean()
                for i in range(x.shape[1])
            )
            out = out + self.smoothness_weight * smoothness

        return out
        
    # Data Preparation
    def prepare_data(self, data: torch.Tensor, targets: Optional[torch.Tensor] = None):
        features = engineer_features(data)
        features_np = features.detach().cpu().numpy() if self.device.type == 'cuda' else features.numpy()

        if not self.trained:
            scaled = self.scaler.fit_transform(features_np)
        else:
            scaled = self.scaler.transform(features_np)

        features_tensor = torch.tensor(scaled, dtype=torch.float32, device=self.device)

        if targets is not None:
            targets_tensor = torch.tensor(targets, dtype=torch.float32, device=self.device)
            return features_tensor, targets_tensor
        return features_tensor

# Model training
    def train_model(self, data: torch.Tensor, targets: torch.Tensor,
                    epochs: int = 200, batch_size: int = 32, lr: float = 0.001,
                    val_split: float = 0.2, n_jobs: int = 4) -> Dict[str, Any]:

        features, targets = self.prepare_data(data, targets)
        dataset = TensorDataset(features, targets)
        train_size = int((1 - val_split) * len(dataset))
        train_data, val_data = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=n_jobs)
        val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=n_jobs)

        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        loss_fn = nn.MSELoss()

        train_losses, val_losses = [], []

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0

            for inputs, targs in train_loader:
                inputs, targs = inputs.to(self.device), targs.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs).squeeze()
                loss = loss_fn(outputs, targs)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * inputs.size(0)

            train_losses.append(total_loss / len(train_loader.dataset))

            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targs in val_loader:
                    inputs, targs = inputs.to(self.device), targs.to(self.device)
                    outputs = self(inputs).squeeze()
                    val_loss += loss_fn(outputs, targs).item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            # Best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_state = self.state_dict()

            # Early stopping
            if len(val_losses) > 15 and val_losses[-1] > val_losses[-5] * 1.05:
                self.load_state_dict(self.best_state)
                break

            scheduler.step(val_loss)

        self.trained = True
        return {'train_losses': train_losses, 'val_losses': val_losses, 'best_val_loss': self.best_loss}

    # Predictions
    
    def predict_volatility(self, data: torch.Tensor, n_samples: int = 1):
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")

        features = self.prepare_data(data)
        self.eval()
        with torch.no_grad():
            preds = self(features).cpu().numpy().flatten()
        return preds, None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        data_np = df[self.feature_columns].values
        data_tensor = torch.tensor(data_np, dtype=torch.float32)
        preds, _ = self.predict_volatility(data_tensor)
        return preds

    def train(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        data_np = df[self.feature_columns].values
        targets_np = df['implied_volatility'].values
        data_tensor = torch.tensor(data_np, dtype=torch.float32)
        targets_tensor = torch.tensor(targets_np, dtype=torch.float32)
        epochs = kwargs.get('epochs', self.epochs)
        lr = kwargs.get('learning_rate', self.learning_rate)
        batch_size = kwargs.get('batch_size', 32)
        return self.train_model(data_tensor, targets_tensor, epochs=epochs, lr=lr, batch_size=batch_size)

    # Save 
  
    def save_model(self, model_dir: str = 'models/saved_models') -> Dict[str, str]:
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = os.path.join(model_dir, f'mlp_vol_model_{timestamp}.pt')
        scaler_path = os.path.join(model_dir, f'mlp_scaler_{timestamp}.joblib')

        torch.save({
            'model_state_dict': self.best_state if self.best_state is not None else self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_layers': self.hidden_layers,
                'output_dim': self.output_dim,
                'activation': self.activation.__class__.__name__,
                'use_batchnorm': self.use_batchnorm,
                'smoothness_weight': self.smoothness_weight
            },
            'metadata': {
                'trained': self.trained,
                'best_loss': self.best_loss,
                'device': self.device.type
            }
        }, model_path)

        joblib.dump(self.scaler, scaler_path)
        return {'model': model_path, 'scaler': scaler_path}

    @classmethod
    def load_model(cls, model_path: str, scaler_path: str) -> 'MLPModel':
        model_data = torch.load(model_path)
        scaler = joblib.load(scaler_path)

        model = cls(
            input_dim=model_data['config']['input_dim'],
            hidden_layers=model_data['config']['hidden_layers'],
            output_dim=model_data['config']['output_dim'],
            activation=model_data['config']['activation'],
            use_batchnorm=model_data['config']['use_batchnorm'],
            smoothness_weight=model_data['config']['smoothness_weight']
        )

        model.load_state_dict(model_data['model_state_dict'])
        model.scaler = scaler
        model.trained = model_data['metadata']['trained']
        model.best_loss = model_data['metadata']['best_loss']
        return model

    def get_surface_grid(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['predicted_iv'] = self.predict(df)
        return df
