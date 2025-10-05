# 📈 Options Lab

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/Diegotistical/OptionsLab)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/Diegotistical/OptionsLab)](https://github.com/Diegotistical/quant-options-toolkit/commits/main)
[![Streamlit](https://img.shields.io/badge/Open%20in%20Streamlit-App-red?logo=streamlit&logoColor=white)](https://optionslab.streamlit.app/)

# 🧠 OptionsLab

**OptionsLab** is a modular quantitative research framework focused on **volatility surface modeling**, **Options Pricing using different models**, and **risk analysis** for derivatives pricing and trading strategy development.  
It provides a unified interface for building, fitting, and evaluating option pricing models — from classical methods (Binomial Trees, Black-Scholes) to machine learning–based volatility surface models (MLP, SVR, Random Forest & XGBoost).

---

## 🚀 Key Features

- **Volatility Surface Modeling**
  - Parametric and non-parametric surface fitting.
  - ML-based volatility modeling using PyTorch and Scikit-learn.
  - Support for arbitrage-free enforcement and surface regularization.
  
- **Monte Carlo Engine**
  - Classic stochastic simulation with configurable paths and payoffs.
  - ML-augmented Monte Carlo models for accelerated pricing and scenario analysis.
  - Unified API across stochastic and ML-based simulations.

- **Pricing Models**
  - Black-Scholes, Binomial Trees, and Monte Carlo (standard and ML-driven).
  - Support for greeks computation and sensitivity analysis.

- **Risk Analysis Tools**
  - Value at Risk (VaR), Expected Shortfall (ES), and Stress Testing.
  - Scenario-based risk metrics for portfolios and individual instruments.

- **Streamlit App**
  - Interactive dashboards for pricing, visualization, and volatility surface exploration.
  - Modular page system (Monte Carlo, Binomial Tree, Vol Surface, Risk Analysis).

---

## 🧩 Project Highlights

- Fully modular **`src/`** structure with clear separation between models, utils, and core logic.
- Robust **exception handling** and **validation layers** across all modules.
- Comprehensive **unit testing suite** (`pytest`) for model reliability.
- Compatible with **CI/CD workflows** via GitHub Actions (coming soon)
- Ready-to-deploy **Streamlit app** for interactive exploration and demonstrations.


---

## 🏗️ Project Structure

    |   LICENSE
    |   README.md
    |   requirements.txt
    |   runtime.txt
    |   setup.py
    |   structure.txt
    |   
    +---.devcontainer
    |       devcontainer.json
    |       
    +---.github
    |   \---workflows
    |           ci.yml
    |           
    +---data
    |   +---processed
    |   |       .gitkeep
    |   |       
    |   \---raw
    |           .gitkeep
    |           
    +---docs
    |       .gitkeep
    |       
    +---models
    |   +---my_models
    |   +---saved_models
    |   |       .gitkeep
    |   |       
    |   \---training_logs
    |           .gitkeep
    |           
    +---notebooks
    |       .gitkeep
    |       backtesting.ipynb
    |       binomial_tree.ipynb
    |       exploratory_data_analysis.ipynb
    |       learnings.ipynb
    |       volatility_model_tuning.ipynb
    |       
    +---src
    |   |   __init__.py
    |   |   
    |   +---common
    |   |       config.py
    |   |       helpers.py
    |   |       logging_config.py
    |   |       validation.py
    |   |       __init__.py
    |   |       
    |   +---exceptions
    |   |       data_exceptions.py
    |   |       greek_exceptions.py
    |   |       model_exceptions.py
    |   |       montecarlo_exceptions.py
    |   |       pricing_exceptions.py
    |   |       risk_exceptions.py
    |   |       __init__.py
    |   |       
    |   +---greeks
    |   |       .gitkeep
    |   |       greeks.py
    |   |       
    |   +---pricing_models
    |   |       .gitkeep
    |   |       binomial_tree.py
    |   |       black_scholes.py
    |   |       monte_carlo.py
    |   |       monte_carlo_ml.py
    |   |       monte_carlo_unified.py
    |   |       __init__.py
    |   |       
    |   +---risk_analysis
    |   |       .gitkeep
    |   |       expected_shortfall.py
    |   |       sensitivity_analysis.py
    |   |       stress_testing.py
    |   |       var.py
    |   |       __init__.py
    |   |       
    |   +---utils
    |   |   |   .gitkeep
    |   |   |   utils.py
    |   |   |   __init__.py
    |   |   |   
    |   |   \---decorators
    |   |           caching.py
    |   |           timing.py
    |   |           __init__.py
    |   |           
    |   \---volatility_surface
    |       |   .gitkeep
    |       |   base.py
    |       |   surface_generator.py
    |       |   __init__.py
    |       |   
    |       +---common
    |       |       validation.py
    |       |       __init__.py
    |       |       
    |       +---models
    |       |       mlp_model.py
    |       |       random_forest.py
    |       |       svr_model.py
    |       |       xgboost_model.py
    |       |       __init__.py
    |       |       
    |       \---utils
    |               arbitrage.py
    |               arbitrage_enforcement.py
    |               arbitrage_utils.py
    |               data_preprocessing.py
    |               feature_engineering.py
    |               grid_search.py
    |               tensor_utils.py
    |               __init__.py
    |               
    +---streamlit_app
    |   |   app.py
    |   |   st_utils.py
    |   |   __init__.py
    |   |   
    |   \---pages
    |           .gitkeep
    |           1_MonteCarlo_Basic.py
    |           2_MonteCarlo_ML.py
    |           3_MonteCarlo_Unified.py
    |           4_Binomial_Tree.py
    |           benchmarks.py
    |           risk_analysis.py
    |           volatility_surface.py
    |           __init__.py
    |           
    \---tests
            .gitkeep
            test_benchmarks.py
            test_benchmarks2.py
            test_black_scholes.py
            test_models.py
            test_monte_carlo.py
            test_risk_analysis.py
            test_var.py
            test_vol_surface.py
---

## 🚀 Getting Started

### Requirements

- Python ≥ 3.10 (Recommended 3.13)
- Recommended: virtual environment (venv, conda, etc.)

### Installation

Clone the repository:

    git clone https://github.com/Diegotistical/OptionsLab.git
    cd OptionsLab
    
*(Note: Ensure you fill in `requirements.txt` as you add dependencies.)*

Install requirements:

    pip install -r requirements.txt

---

## 🛠️ Usage Examples

# Monte Carlo Option Pricer Tutorial

This tutorial demonstrates how to use the `MonteCarloPricer` class to price European options and compute Greeks. Optional Numba acceleration is supported for faster simulations.

---

## 1. Import the Pricer

```bash
from src.pricing_models.monte_carlo import MonteCarloPricer
```

---

## 2. Initialize a Pricer

```bash
pricer = MonteCarloPricer(
    num_simulations=100_000,
    num_steps=100,
    seed=42,
    use_numba=False
)
```

---

## 3. Price a European Option

```bash
S, K, T, r, sigma, q = 100, 110, 1, 0.01, 0.2, 0.0

call_price = pricer.price(S, K, T, r, sigma, option_type="call", q=q)
put_price  = pricer.price(S, K, T, r, sigma, option_type="put", q=q)

print(f"Call Price: {call_price:.4f}")
print(f"Put Price: {put_price:.4f}")
```

---

## 4. Compute Greeks

```bash
delta = pricer.delta(S, K, T, r, sigma, option_type="call")
gamma = pricer.gamma(S, K, T, r, sigma, option_type="call")
vega  = pricer.vega(S, K, T, r, sigma, option_type="call")
theta = pricer.theta(S, K, T, r, sigma, option_type="call")
rho   = pricer.rho(S, K, T, r, sigma, option_type="call")

print(f"Delta: {delta:.4f}, Gamma: {gamma:.4f}, Vega: {vega:.4f}, Theta: {theta:.4f}, Rho: {rho:.4f}")
```

---

## 5. Enable Numba Acceleration (Optional)

```bash
pricer_numba = MonteCarloPricer(
    num_simulations=100_000,
    num_steps=100,
    use_numba=True
)

fast_price = pricer_numba.price(S, K, T, r, sigma, option_type="call")
print(f"Call Price with Numba: {fast_price:.4f}")
```

---

## 🤝 Contributing

Contributions are welcome!

- Fork the repository
- Create a feature branch: `git checkout -b feature/MyFeature`
- Commit your changes: `git commit -m "Add MyFeature"`
- Push to your branch: `git push origin feature/MyFeature`
- Open a pull request

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

Project maintained by [Diegotistical](https://github.com/Diegotistical).

---

*Note: This project is in active development. Expect breaking changes as new features are added.*








