# src/pricing_models/monte_carlo_ml.py

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import numpy as np
from monte_carlo import MonteCarloPricer

class MonteCarloML(MonteCarloPricer):
    """
    ML surrogate for Monte Carlo outputs (price + Greeks).
    
    - Trains regression model(s) to predict:
      - price
      - delta
      - gamma
    - Input features: S, K, T, r, sigma, q
    """

    def __init__(self, num_simulations: int = 10000, num_steps: int = 100,
                 seed: int = None, ml_model=None):
        super().__init__(num_simulations, num_steps, seed)
        # Multi-output regressor for price, delta, gamma
        base_model = ml_model or GradientBoostingRegressor(
            n_estimators=200, max_depth=5, random_state=seed
        )
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MultiOutputRegressor(base_model))
        ])
        self.trained = False

    def generate_training_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Compute Monte Carlo price, delta, gamma for each row in DataFrame.
        
        Args:
            df: DataFrame with columns ["S","K","T","r","sigma","q"]
        
        Returns:
            X: Features (same as input)
            y: np.ndarray of shape (n_samples, 3) [price, delta, gamma]
        """
        X = df.copy()
        prices, deltas, gammas = [], [], []
        for _, row in X.iterrows():
            S, K, T, r, sigma, q = row[['S','K','T','r','sigma','q']]
            price = self.price(S, K, T, r, sigma, 'call', q)
            delta = self.delta(S, K, T, r, sigma, 'call', q)
            gamma = self.gamma(S, K, T, r, sigma, 'call', q)
            prices.append(price)
            deltas.append(delta)
            gammas.append(gamma)
        y = np.vstack([prices, deltas, gammas]).T
        return X, y

    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        """
        Train ML surrogate on Monte Carlo generated targets.
        
        If y is None, compute price, delta, gamma using Monte Carlo.
        """
        if y is None:
            X, y = self.generate_training_data(X)
        self.model.fit(X.values, y)
        self.trained = True

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict price, delta, gamma using trained ML surrogate.
        
        Returns:
            DataFrame with columns ['price','delta','gamma']
        """
        if not self.trained:
            raise RuntimeError("ML surrogate not trained")
        preds = self.model.predict(X.values)
        return pd.DataFrame(preds, columns=['price','delta','gamma'])
