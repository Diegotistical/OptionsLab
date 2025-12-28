# src/volatility_surface/models/mlp_model.py

import threading
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.volatility_surface.base import VolatilityModelBase
from src.volatility_surface.utils.feature_engineering import engineer_features
from src.volatility_surface.utils.tensor_utils import ensure_tensor

FEATURE_COLUMNS = [
    "moneyness",
    "log_moneyness",
    "time_to_maturity",
    "ttm_squared",
    "risk_free_rate",
    "historical_volatility",
    "volatility_skew",
]


class MLPModel(VolatilityModelBase, nn.Module):
    def __init__(
        self,
        hidden_layers: list = [64, 32],
        activation: str = "GELU",
        activation_kwargs: Optional[Dict] = None,
        dropout_rate: float = 0.2,
        smoothness_weight: float = 0.0,
        use_batchnorm: bool = True,
        use_dropout: bool = True,
        scaler_type: str = "standard",
        learning_rate: float = 1e-3,
        epochs: int = 200,
        early_stopping_patience: int = 15,
        batch_size: int = 32,
        random_seed: int = 42,
        **kwargs,
    ):
        VolatilityModelBase.__init__(
            self, feature_columns=FEATURE_COLUMNS, enable_benchmark=True
        )
        nn.Module.__init__(self)
        self._lock = threading.RLock()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.scaler_type = scaler_type
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size

        self.model = self._build_model()
        self.to(self.device)
        self._initialize_scaler()

        self.optimizer = None
        self.scheduler = None
        self.best_state = None
        self.train_history = {"train_loss": [], "val_loss": []}

    def _create_activation(self):
        act_cls = getattr(nn, self.activation_name, None)
        if act_cls is None:
            raise ValueError(f"Unsupported activation: {self.activation_name}")
        return act_cls(**self.activation_kwargs)

    def _build_model(self):
        layers = []
        prev_dim = len(self.feature_columns)
        for h_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(self.activation)
            if self.use_dropout:
                layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def _initialize_scaler(self):
        scalers = {
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "minmax": MinMaxScaler(),
        }
        if self.scaler_type not in scalers:
            raise ValueError(f"Unsupported scaler type: {self.scaler_type}")
        self.scaler = scalers[self.scaler_type]

    def _prepare_data(self, df: pd.DataFrame, targets: Optional[np.ndarray] = None):
        df = df.copy()
        df["time_to_maturity"] = np.maximum(df["time_to_maturity"], 1e-5)
        features = engineer_features(df)
        features_np = features.values
        if not self.trained:
            scaled = self.scaler.fit_transform(features_np)
        else:
            scaled = self.scaler.transform(features_np)
        features_tensor = ensure_tensor(scaled, dtype=torch.float32, device=self.device)
        if targets is not None:
            targets_tensor = ensure_tensor(
                targets.astype(np.float32), device=self.device
            )
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
        if df is None:
            raise ValueError("DataFrame `df` must be provided to train the model.")
        with self._lock:
            self._on_train_start(df)
            train_df, val_df = train_test_split(
                df, test_size=val_split, random_state=42
            )
            X_train, y_train = self._prepare_data(
                train_df, train_df["implied_volatility"].values
            )
            X_val, y_val = self._prepare_data(
                val_df, val_df["implied_volatility"].values
            )

            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train, y_train),
                batch_size=self.batch_size,
                shuffle=True,
            )
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_val, y_val), batch_size=self.batch_size
            )

            self.optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, "min", patience=10, factor=0.5
            )

            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(self.epochs):
                self.train()
                train_loss = 0.0
                for Xb, yb in train_loader:
                    self.optimizer.zero_grad()
                    out = self.forward(Xb)
                    loss = self._compute_loss(
                        out, yb, Xb if self.smoothness_weight > 0 else None
                    )
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
                        val_loss += self._compute_loss(out, yb).item() * Xb.size(0)
                val_loss /= len(val_loader.dataset)

                self.train_history["train_loss"].append(train_loss)
                self.train_history["val_loss"].append(val_loss)

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

    def predict_volatility(
        self, df: pd.DataFrame, mc_samples: int = 1, compute_greeks: bool = False
    ):
        if df is None:
            raise ValueError("DataFrame `df` must be provided for prediction.")
        with self._lock:
            self._on_predict_start(df)
            features = self._prepare_data(df)

            if mc_samples == 1:
                self.eval()
                features.requires_grad_(compute_greeks)
                with torch.no_grad() if not compute_greeks else torch.enable_grad():
                    preds = self.forward(features)
                preds_np = preds.cpu().numpy().flatten()
                if compute_greeks:
                    grads = torch.autograd.grad(preds.sum(), features)[0].cpu().numpy()
                    return preds_np, grads
                self._on_predict_end(preds_np)
                return preds_np

            # MC Dropout inference
            self.train()
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

            preds_samples = []
            with torch.no_grad():
                for _ in range(mc_samples):
                    preds = self.forward(features)
                    preds_samples.append(preds.cpu().numpy().flatten())

            self.eval()
            mean_preds = np.mean(preds_samples, axis=0)
            self._on_predict_end(mean_preds)
            return mean_preds

    def plot_training_history(self, title="Training / Validation Loss"):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=self.train_history["train_loss"],
                mode="lines+markers",
                name="Train Loss",
            )
        )
        fig.add_trace(
            go.Scatter(
                y=self.train_history["val_loss"],
                mode="lines+markers",
                name="Validation Loss",
            )
        )
        fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title="MSE Loss")
        fig.show()

    def _save_model_impl(self, model_path: str, scaler_path: str) -> None:
        with self._lock:
            torch.save({"model_state_dict": self.state_dict()}, model_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.train_history, f"{model_path}_history.pkl")

    def _load_model_impl(self, model_path: str, scaler_path: str) -> None:
        with self._lock:
            data = torch.load(model_path, map_location=self.device)
            self.load_state_dict(data["model_state_dict"])
            self.scaler = joblib.load(scaler_path)
            try:
                self.train_history = joblib.load(f"{model_path}_history.pkl")
            except FileNotFoundError:
                self.train_history = {"train_loss": [], "val_loss": []}
            self.trained = True

    def _predict_impl(self, X: Any) -> Any:
        X_tensor = ensure_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y_pred = self.model(X_tensor)
        return y_pred.numpy()
