# src/volatility_surface/models/mlp_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from ..utils import engineer_features, validate_domain
from .base import VolatilityModelBase

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

class MLPModel(VolatilityModelBase):
    def __init__(self, lr=1e-3, epochs=100, batch_size=32):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()

    def train(self, df: pd.DataFrame, val_split: float = 0.2):
        X = engineer_features(df)
        y = df['implied_volatility'].values.reshape(-1, 1)

        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = MLP(X_tensor.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

        self.trained = True
        return {"final_loss": loss.item()}

    def predict_volatility(self, df: pd.DataFrame):
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        X = engineer_features(df)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy().flatten()
        return preds

    def save_model(self, model_dir='models/saved_models'):
        os.makedirs(model_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"mlp_vol_{ts}.pt")
        scaler_path = os.path.join(model_dir, f"mlp_scaler_{ts}.joblib")
        torch.save(self.model.state_dict(), model_path)
        joblib.dump(self.scaler, scaler_path)
        return {"model": model_path, "scaler": scaler_path}

    def load_model(self, model_path, scaler_path):
        self.scaler = joblib.load(scaler_path)
        dummy_input = torch.zeros((1, len(self.feature_columns)))
        self.model = MLP(dummy_input.shape[1])
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.trained = True
