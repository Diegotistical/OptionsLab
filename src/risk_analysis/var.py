# src / risk_analysis / var.py

"""
Value at Risk (VaR) Calculation Module

Implements:
- Historical VaR
- Monte Carlo VaR (with BinomialTree integration)
- Parametric VaR (normal/log-normal)
- Conditional VaR (CVaR)
- Stress test scenarios
"""

from typing import Union, List, Dict, Optional, Tuple, Callable
import numpy as np
import pandas as pd
import warnings
from scipy.stats import norm
from src.exceptions import RiskAnalysisError, InputValidationError
from src.pricing_models.binomial_tree import BinomialTree


class VaRAnalyzer:
    """
    Features:
    - Historical simulation
    - Monte Carlo simulation (option-aware)
    - Parametric normal/log-normal VaR
    - Conditional VaR (CVaR)
    - Portfolio-level stress testing
    """

    def __init__(self, confidence_level: float = 0.95, time_horizon_days: int = 1):
        if not 0 < confidence_level < 1:
            raise InputValidationError("Confidence level must be between 0 and 1")
        if time_horizon_days <= 0:
            raise InputValidationError("Time horizon must be positive")

        self.confidence_level = confidence_level
        self.time_horizon_days = time_horizon_days
        self.z_score = norm.ppf(1 - (1 - confidence_level)) # critical z-value for VaR
        self.scaling_factor = self.time_horizon_days / 365  # convert annual to time horizon scale

    def _validate_inputs(self, portfolio_value: float):
        if portfolio_value <= 0:
            raise InputValidationError("Portfolio value must be positive")
        if not isinstance(portfolio_value, (int, float)):
            raise InputValidationError("Portfolio value must be numeric")

    def _calculate_cvar(self, sorted_losses: np.ndarray) -> float:
        """Helper for CVaR calculation"""
        index = int(np.floor((1 - self.confidence_level) * len(sorted_losses)))
        return np.mean(sorted_losses[:index])
        """
        * Given an array of sorted losses (ascending), compute the Conditional VaR (CVaR).

        * CVaR is the average loss in the worst-case tail beyond the VaR percentile.

        * More conservative risk metric than VaR alone.
        """

    def historical_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Calculate VaR using historical simulation
        
        Parameters:
        -----------
        returns : array-like
            Historical returns (daily or adjusted for time horizon)
        portfolio_value : float
            Total portfolio value in base currency
            
        Returns:
        --------
        dict with VaR and CVaR metrics
        """
        self._validate_inputs(portfolio_value)
        
        # Convert to numpy array if needed
        returns_array = np.array(returns)
        
        # Validate input data
        if len(returns_array) < 252:
            warnings.warn("Insufficient historical data for reliable VaR calculation")
        
        # Sort returns and calculate VaR
        sorted_returns = np.sort(returns_array)
        index = int(np.floor((1 - self.confidence_level) * len(sorted_returns)))
        
        var = -sorted_returns[index] * portfolio_value
        cvar = -self._calculate_cvar(sorted_returns) * portfolio_value
        
        return {
            'var': var,
            'cvar': cvar,
            'confidence_level': self.confidence_level,
            'time_horizon_days': self.time_horizon_days
        }

    def monte_carlo_var(
        self,
        initial_price: float,
        mu: float,
        sigma: float,
        portfolio_value: float,
        num_simulations: int = 10000,
        use_log_returns: bool = True,
        binomial_pricer: Optional[BinomialTree] = None
    ) -> Dict[str, float]:
        """
        Calculate VaR using Monte Carlo simulation
        
        Parameters:
        -----------
        initial_price : float
            Current price of the asset
        mu : float
            Expected return (annualized)
        sigma : float
            Volatility (annualized)
        portfolio_value : float
            Total portfolio value
        num_simulations : int
            Number of simulation paths
        use_log_returns : bool
            Use log returns (log-normal) instead of simple returns
        binomial_pricer : BinomialTree
            Option pricer for path-dependent instruments
            
        Returns:
        --------
        dict with VaR and CVaR metrics
        """
        self._validate_inputs(portfolio_value)
        
        if sigma <= 0:
            raise InputValidationError("Volatility must be positive")
        if num_simulations < 1000:
            warnings.warn("Fewer than 1000 simulations may reduce accuracy")

        # Generate simulated returns
        if use_log_returns:
            simulated_returns = np.random.normal(
                loc=mu * self.scaling_factor,
                scale=sigma * np.sqrt(self.scaling_factor),
                size=num_simulations
            )
            
            # Calculate final prices and losses
            final_prices = initial_price * np.exp(simulated_returns)
            losses = initial_price - final_prices
        else:
            simulated_returns = np.random.normal(
                loc=mu * self.scaling_factor,
                scale=sigma * np.sqrt(self.scaling_factor),
                size=num_simulations
            )
            losses = -simulated_returns * portfolio_value

        # Calculate VaR and CVaR
        sorted_losses = np.sort(losses)
        index = int(np.floor((1 - self.confidence_level) * num_simulations))
        var = sorted_losses[index]
        cvar = self._calculate_cvar(sorted_losses)

        return {
            'var': var,
            'cvar': cvar,
            'confidence_level': self.confidence_level,
            'time_horizon_days': self.time_horizon_days
        }

    def parametric_var(
        self,
        initial_price: float,
        mu: float,
        sigma: float,
        portfolio_value: float,
        use_log_returns: bool = True
    ) -> Dict[str, float]:
        """
        Calculate parametric VaR assuming normal/log-normal distribution
        
        Parameters:
        initial_price : float
            Current price of the asset
        mu : float
            Expected return (annualized)
        sigma : float
            Volatility (annualized)
        portfolio_value : float
            Total portfolio value
        use_log_returns : bool
            Use log returns (log-normal) instead of simple returns
            
        Returns:
        dict with VaR and CVaR metrics
        """
        self._validate_inputs(portfolio_value)
        
        if sigma <= 0:
            raise InputValidationError("Volatility must be positive")

        # Calculate VaR and CVaR
        if use_log_returns:
            # Log-normal distribution (Geometric Brownian Motion)
            mu_t = mu * self.scaling_factor
            sigma_t = sigma * np.sqrt(self.scaling_factor)
            
            # Log price distribution
            log_price = np.log(initial_price)
            log_var = log_price + mu_t - self.z_score * sigma_t
            var = np.exp(log_var) - initial_price
            cvar = self._calculate_log_cvar(log_price, mu_t, sigma_t)
        else:
            # Normal distribution
            mu_t = mu * self.scaling_factor
            sigma_t = sigma * np.sqrt(self.scaling_factor)
            var = -(mu_t - self.z_score * sigma_t) * initial_price
            cvar = -(mu_t - (norm.pdf(self.z_score) / (1 - self.confidence_level)) * sigma_t) * initial_price

        return {
            'var': var * portfolio_value / initial_price,
            'cvar': cvar * portfolio_value / initial_price,
            'confidence_level': self.confidence_level,
            'time_horizon_days': self.time_horizon_days
        }

    def _calculate_log_cvar(self, log_price: float, mu_t: float, sigma_t: float) -> float:
        """CVaR calculation for log-normal distribution"""
        z = norm.ppf(1 - self.confidence_level)
        exp_mu = np.exp(mu_t - 0.5 * sigma_t**2)
        exp_sigma = np.exp(-sigma_t**2/2)
        cvar = -exp_mu * exp_sigma * norm.pdf(z) / (1 - self.confidence_level)
        return cvar

    def portfolio_var(
        self,
        weights: List[float],
        expected_returns: List[float],
        cov_matrix: np.ndarray,
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Calculate VaR for a multi-asset portfolio
        
        Parameters:
        weights : list of float
            Portfolio weights (sum to 1)
        expected_returns : list of float
            Annualized expected returns
        cov_matrix : np.ndarray
            Covariance matrix of returns
        portfolio_value : float
            Total portfolio value
            
        Returns:
        dict with VaR and CVaR metrics
        """
        self._validate_inputs(portfolio_value)
        
        if abs(sum(weights) - 1.0) > 1e-8:
            raise InputValidationError("Portfolio weights must sum to 1")
        if len(weights) != len(expected_returns):
            raise InputValidationError("Weights and returns arrays must match")
        if cov_matrix.shape[0] != len(weights):
            raise InputValidationError("Covariance matrix dimensions must match number of assets")

        # Portfolio parameters
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Annualized to daily/horizon conversion
        sigma_t = portfolio_volatility * np.sqrt(self.scaling_factor)
        mu_t = portfolio_return * self.scaling_factor

        # Calculate VaR and CVaR
        var = -(mu_t - self.z_score * sigma_t) * portfolio_value
        cvar = -(mu_t - (norm.pdf(self.z_score) / (1 - self.confidence_level)) * sigma_t) * portfolio_value

        return {
            'var': var,
            'cvar': cvar,
            'confidence_level': self.confidence_level,
            'time_horizon_days': self.time_horizon_days
        }

    def stress_test_var(
        self,
        initial_price: float,
        mu: float,
        sigma: float,
        portfolio_value: float,
        stress_factors: List[float] = [1.5, 2.0, 2.5]
    ) -> pd.DataFrame:
        """
        Calculate stress test VaR with extreme scenarios
        
        Parameters:
        initial_price : float
            Current price of the asset
        mu : float
            Expected return (annualized)
        sigma : float
            Volatility (annualized)
        portfolio_value : float
            Total portfolio value
        stress_factors : list of float
            Number of standard deviations for stress scenarios
            
        Returns:
        DataFrame with stress test VaR metrics
        """
        self._validate_inputs(portfolio_value)
        
        if sigma <= 0:
            raise InputValidationError("Volatility must be positive")
        if any(sf <= 0 for sf in stress_factors):
            raise InputValidationError("Stress factors must be positive")

        results = []
        for factor in stress_factors:
            # Stress scenario return
            stress_return = mu * self.scaling_factor - factor * sigma * np.sqrt(self.scaling_factor)
            stress_loss = -stress_return * initial_price
            
            results.append({
                'stress_factor': factor,
                'var': stress_loss * portfolio_value,
                'confidence_level': self.confidence_level,
                'time_horizon_days': self.time_horizon_days
            })

        return pd.DataFrame(results).set_index('stress_factor')

    def delta_normal_var(
        self,
        delta: float,
        gamma: float,
        sigma: float,
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Delta-normal VaR approximation for options portfolios
        
        Parameters:
        delta : float
            Option delta (price sensitivity)
        gamma : float
            Option gamma (convexity)
        sigma : float
            Underlying volatility (annualized)
        portfolio_value : float
            Total portfolio value
            
        Returns:
        dict with VaR and CVaR metrics
        """
        self._validate_inputs(portfolio_value)
        
        if sigma <= 0:
            raise InputValidationError("Volatility must be positive")

        # Delta-normal approximation
        sigma_t = sigma * np.sqrt(self.scaling_factor)
        var = -delta * sigma_t * self.z_score * portfolio_value
        
        # Delta-gamma correction (optional)
        gamma_correction = 0.5 * gamma * sigma_t**2 * portfolio_value
        var += gamma_correction
        
        return {
            'var': var,
            'confidence_level': self.confidence_level,
            'time_horizon_days': self.time_horizon_days
        }

    def option_var(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        q: float = 0.0,
        num_simulations: int = 10000,
        num_steps: int = 500
    ) -> Dict[str, float]:
        """
        VaR calculation for options using Monte Carlo
        
        Parameters:
        S : float
            Spot price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
        option_type : str
            Option type ('call' or 'put')
        q : float
            Dividend yield
        num_simulations : int
            Number of simulations
        num_steps : int
            Steps for binomial pricing
            
        Returns:
        dict with VaR and CVaR metrics
        """
        if T <= 0:
            raise InputValidationError("Time to maturity must be positive")
        if option_type not in ['call', 'put']:
            raise InputValidationError("Option type must be 'call' or 'put'")

        # Initialize binomial pricer
        pricer = BinomialTree(num_steps=num_steps)
        simulated_prices = []

        # Simulate underlying price paths
        np.random.seed(42)
        simulated_returns = np.random.normal(
            loc=r * self.scaling_factor,
            scale=sigma * np.sqrt(self.scaling_factor),
            size=num_simulations
        )
        
        # Price options under simulated prices
        for ret in simulated_returns:
            final_price = S * np.exp(ret)
            price = pricer.price(final_price, K, T, r, sigma, option_type, q=q)
            simulated_prices.append(price)

        # Calculate losses and VaR
        sorted_losses = np.sort(simulated_prices)
        index = int(np.floor((1 - self.confidence_level) * num_simulations))
        
        var = sorted_losses[index]
        cvar = self._calculate_cvar(simulated_prices)

        return {
            'var': var,
            'cvar': cvar,
            'confidence_level': self.confidence_level,
            'time_horizon_days': self.time_horizon_days
        }

    def batch_stress_test(
        self,
        initial_price: float,
        mu_range: List[float],
        sigma_range: List[float],
        portfolio_value: float
    ) -> pd.DataFrame:
        """
        Batch stress test with parameter grids
        
        Parameters:
        initial_price : float
            Current price of the asset
        mu_range : list of float
            Expected return scenarios
        sigma_range : list of float
            Volatility scenarios
        portfolio_value : float
            
        Returns:
        DataFrame with VaR scenarios
        """
        self._validate_inputs(portfolio_value)
        
        results = []
        for mu in mu_range:
            for sigma in sigma_range:
                # Calculate stress VaR
                stress_return = mu * self.scaling_factor - 2.0 * sigma * np.sqrt(self.scaling_factor)
                stress_loss = -stress_return * initial_price
                
                results.append({
                    'mu': mu,
                    'sigma': sigma,
                    'var': stress_loss * portfolio_value,
                    'confidence_level': self.confidence_level,
                    'time_horizon_days': self.time_horizon_days
                })

        return pd.DataFrame(results).pivot(index='mu', columns='sigma', values='var')
