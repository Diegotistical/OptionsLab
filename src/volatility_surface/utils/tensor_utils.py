# File: src/volatility_surface/utils/tensor_utils.py

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ============================
# Tensor Utilities
# ============================


def ensure_tensor(
    x, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert numpy array, pandas DataFrame/Series, list or scalar into a PyTorch tensor.
    - Keeps dtype control and moves to device if requested.
    - Accepts 1D/2D structures.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)

    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        arr = x.values
        return torch.tensor(arr, dtype=dtype, device=device)

    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=dtype, device=device)

    if isinstance(x, list):
        return torch.tensor(np.asarray(x), dtype=dtype, device=device)

    # handle scalars
    if isinstance(x, (int, float)):
        return torch.tensor([x], dtype=dtype, device=device)

    raise ValueError(f"Cannot convert type {type(x)} to tensor.")


def to_tensor(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Convert features and optional target from dataframe to tensors.
    Returns (X, y) where y is None if target_col is None.
    """
    X = ensure_tensor(df[feature_cols], dtype=dtype, device=device)
    if target_col is not None:
        y = ensure_tensor(df[target_col], dtype=dtype, device=device)
        # ensure column vector
        if y.ndim == 1:
            y = y.view(-1, 1)
        return X, y
    return X, None


# ============================
# Simple MLP Model
# ============================


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int = 1,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ============================
# Training & Prediction
# ============================


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: Optional[int] = None,
    verbose: bool = False,
) -> nn.Module:
    """
    Simple training loop using Adam + MSELoss. Accepts full-batch or mini-batch depending on batch_size.
    Returns the trained model (in-place).
    """
    device = (
        next(model.parameters()).device
        if any(p.requires_grad for p in model.parameters())
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n = X.shape[0]
    if batch_size is None or batch_size >= n:
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} loss={loss.item():.6f}")
    else:
        # minibatch
        for epoch in range(epochs):
            model.train()
            permutation = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = permutation[i : i + batch_size]
                xb = X[idx]
                yb = y[idx]
                optimizer.zero_grad()
                y_pred = model(xb)
                loss = criterion(y_pred, yb)
                loss.backward()
                optimizer.step()
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} last_batch_loss={loss.item():.6f}")

    return model


def predict_model(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """
    Run inference and return numpy 1D array.
    """
    model.eval()
    device = next(model.parameters()).device
    X = X.to(device)
    with torch.no_grad():
        y_pred = model(X)
    arr = y_pred.cpu().numpy()
    # flatten to 1D if single-output
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.flatten()
    return arr
