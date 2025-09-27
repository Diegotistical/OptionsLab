import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from typing import List, Tuple

# ============================
# Tensor Utilities
# ============================

def ensure_tensor(x, dtype=torch.float32):
    """
    Convert numpy array, pandas DataFrame/Series, or list into a PyTorch tensor.
    """
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        return torch.tensor(x.values, dtype=dtype)
    if isinstance(x, np.ndarray) or isinstance(x, list):
        return torch.tensor(x, dtype=dtype)
    raise ValueError(f"Cannot convert type {type(x)} to tensor.")

def to_tensor(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert features and target from dataframe to tensors.
    """
    X = ensure_tensor(df[feature_cols])
    y = ensure_tensor(df[target_col]).view(-1, 1)
    return X, y

# ============================
# Simple MLP Model
# ============================

class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int = 1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ============================
# Training & Prediction
# ============================

def train_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int = 50, lr: float = 1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    return model

def predict_model(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
    return y_pred.numpy().flatten()
