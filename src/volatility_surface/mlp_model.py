# src / volatility_surface / mlp_model.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import os
import joblib
from sklearn.preprocessing import StandardScaler
from ..pricing_models.utils import ensuretensor
from dataclasses import dataclass, field
import logging
import warnings
from typing import Optional, Tuple, Dict, Any


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# 🔧 Activation mapping
ACTIVATION_MAP = {
    'gelu': nn.GELU,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'tanh': nn.Tanh
}

@dataclass
class ModelConfig:
    """Configuration for volatility surface model"""
    input_dim: int = 7
    hidden_dims: list = field(default_factory=lambda: [64, 32])
    output_dim: int = 1
    activation: str = 'gelu'  # Save key instead of object
    dropout_rate: float = 0.2
    smoothness_weight: float = 0.0
    use_batchnorm: bool = True

class MLPModel(nn.Module):
    """
    Enterprise-grade volatility surface model with:
    - Modular feature engineering
    - Smoothness regularization
    - MC Dropout uncertainty estimation
    - TorchScript/ONNX compatibility
    """

    def __init__(self, config: ModelConfig):
        super(MLPModel, self).__init__()
        # Validate activation
        if config.activation not in ACTIVATION_MAP:
            raise ValueError(f"Unsupported activation: {config.activation}")
        self.activation = ACTIVATION_MAP[config.activation]()
        
        # Build network
        layers = []
        prev_dim = config.input_dim
        
        for h_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim) if config.use_batchnorm else nn.Dropout(config.dropout_rate),
                self.activation
            ])
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, config.output_dim))
        self.model = nn.Sequential(*layers)
        
        # Device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # Training state
        self.trained = False
        self.scaler = StandardScaler()
        self.best_state = None
        self.best_loss = float('inf')
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional smoothness regularization"""
        return self.model(x)

    def prepare_data(self, data: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        """
        Prepare data with proper device management
        """
        features = self._engineer_features(data)
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

    def _engineer_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Financial-aware feature engineering with defensive coding
        """
        if not isinstance(data, torch.Tensor):
            data = ensuretensor(data, dtype=torch.float32, requires_grad=False)

        assert data.shape[1] == 5, "Expected 5 input features: S, K, T, r, vol"
    
        S = data[:, 0].clamp(min=1e-6)
        K = data[:, 1].clamp(min=1e-6)
        T = data[:, 2].clamp(min=1e-6)
        r = data[:, 3]
        vol = data[:, 4].clamp(min=1e-6)
    
        moneyness = S / K
        log_moneyness = torch.log(torch.clamp(moneyness, min=1e-6))
        ttm_squared = T ** 2
    
        return torch.cat([
            moneyness.unsqueeze(1),
            log_moneyness.unsqueeze(1),
            T.unsqueeze(1),
            ttm_squared.unsqueeze(1),
            r.unsqueeze(1),
            vol.unsqueeze(1)
        ], dim=1)

    def train_model(self, 
                    data: torch.Tensor, 
                  targets: torch.Tensor,
                  epochs: int = 200,
                  batch_size: int = 32,
                  lr: float = 0.001,
                  val_split: float = 0.2,
                  n_jobs: int = 4) -> Dict[str, Any]:
        """
        Production-grade training with validation and checkpointing
        """
        features, targets = self.prepare_data(data, targets)
        dataset = TensorDataset(features, targets)
        train_size = int((1 - val_split) * len(dataset))
        train_data, val_data = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=n_jobs, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=n_jobs, pin_memory=True)
        
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True, factor=0.5)
        loss_fn = nn.MSELoss()
        
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs).squeeze()
                loss = loss_fn(outputs, targets)
                
                # 🔁 Smoothness regularization
                if self.config.smoothness_weight > 0:
                    grad_outputs = torch.autograd.grad(loss, inputs, create_graph=True)[0]
                    smoothness = sum(
                        torch.autograd.grad(grad_outputs[:, i].sum(), inputs, create_graph=True)[0][:, i].pow(2).mean() 
                        for i in range(inputs.shape[1])
                    )
                    loss += self.config.smoothness_weight * smoothness
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                
            train_losses.append(total_loss / len(train_loader.dataset))
            
            self.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self(inputs).squeeze()
                    batch_loss = loss_fn(outputs, targets)
                    val_loss += batch_loss.item() * inputs.size(0)
                    
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            
            # Model checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_state = self.state_dict()
                logger.info(f"New best model at epoch {epoch+1}")
                early_stop_counter = 0
                self.best_loss = val_loss
            else:
                early_stop_counter += 1
                
            # Early stopping
            if early_stop_counter >= 15:
                logger.info(f"Early stopping at epoch {epoch+1}")
                self.load_state_dict(self.best_state)
                break
                
            # Learning rate update
            scheduler.step(val_loss)
            
            # Logging
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
                
        self.trained = True
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }

    def predict_volatility(self, data: torch.Tensor, n_samples: int = 10) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict volatility with optional uncertainty estimation via MC Dropout
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
            
        features = self.prepare_data(data)
        
        # 🚨 Safety check for MC Dropout
        if n_samples > 1 and self.config.use_batchnorm:
            raise ValueError("MC Dropout incompatible with batchnorm. Set use_batchnorm=False.")
            
        if n_samples > 1:
            self.train()  # Enable dropout during prediction
            preds = []
            for _ in range(n_samples):
                preds.append(self(features).cpu().numpy().flatten())
            mean_pred = np.mean(preds, axis=0)
            std_pred = np.std(preds, axis=0)
            return mean_pred, std_pred
        else:
            self.eval()
            with torch.no_grad():
                return self(features).cpu().numpy().flatten(), None

    def save_model(self, model_dir: str = 'models/saved_models') -> Dict[str, str]:
        """
        Save model with full versioning and metadata
        """
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f'mlp_vol_model_{timestamp}.pt')
        scaler_path = os.path.join(model_dir, f'mlp_scaler_{timestamp}.joblib')
        
        # Save model
        torch.save({
            'model_state_dict': self.best_state if self.best_state is not None else self.state_dict(),
            'scaler': self.scaler,
            'config': {
                'input_dim': self.config.input_dim,
                'hidden_dims': self.config.hidden_dims,
                'output_dim': self.config.output_dim,
                'activation': self.config.activation,  # Save key from activation_map
                'use_batchnorm': self.config.use_batchnorm,
                'smoothness_weight': self.config.smoothness_weight
            },
            'metadata': {
                'trained': self.trained,
                'best_loss': self.best_loss,
                'device': self.device.type
            }
        }, model_path)
        
        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Saved model to {model_path}")
        return {'model': model_path, 'scaler': scaler_path}

    @classmethod
    def load_model(cls, model_path: str, scaler_path: str) -> 'MLPModel':
        """
        Load model with full configuration recovery
        """
        model_data = torch.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Reconstruct model from config
        config = model_data['config']
        model = cls(ModelConfig(**config))
        
        model.load_state_dict(model_data['model_state_dict'])
        model.scaler = scaler
        model.trained = model_data['metadata']['trained']
        model.best_loss = model_data['metadata']['best_loss']
        
        return model

    def export_onnx(self, output_path: str):
        """Export to ONNX with production-ready settings"""
        example_input = torch.rand(1, self.config.input_dim).to(self.device)
        torch.onnx.export(
            self,
            example_input,
            output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input_features'],
            output_names=['volatility_output'],
            dynamic_axes={
                'input_features': {0: 'batch_size'},
                'volatility_output': {0: 'batch_size'}
            }
        )
        logger.info(f"Exported ONNX model to {output_path}")

    def get_surface_grid(self, 
                        S_range: Tuple[float, float, int],
                        K_range: Tuple[float, float, int],
                        T_range: Tuple[float, float, int],
                        r_value: float = 0.05,
                        vol_value: float = 0.2) -> Dict[str, np.ndarray]:
        """
        Generate volatility surface grid with configurable inputs
        """
        S = np.linspace(S_range[0], S_range[1], S_range[2])
        K = np.linspace(K_range[0], K_range[1], K_range[2])
        T = np.linspace(T_range[0], T_range[1], T_range[2])
        
        S_grid, K_grid, T_grid = np.meshgrid(S, K, T)
        df_surface = torch.tensor(np.column_stack([
            S_grid.flatten(), K_grid.flatten(), T_grid.flatten(),
            np.full_like(S_grid.flatten(), r_value),
            np.full_like(S_grid.flatten(), vol_value)
        ]))
        
        volatility, _ = self.predict_volatility(df_surface)
        volatility = volatility.reshape(S_range[2], K_range[2], T_range[2])
        
        return {
            'S': S_grid,
            'K': K_grid,
            'T': T_grid,
            'volatility': volatility
        }
