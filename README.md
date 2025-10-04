# 📈 Options Lab

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/Diegotistical/OptionsLab)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/Diegotistical/OptionsLab)](https://github.com/Diegotistical/quant-options-toolkit/commits/main)
[![Streamlit](https://img.shields.io/badge/Open%20in%20Streamlit-App-red?logo=streamlit&logoColor=white)](https://optionslab.streamlit.app/)


**Options Lab** is a Python library for quantitative finance practitioners, researchers, and students. It provides models, simulations, and risk analysis tools for pricing and analyzing options and derivatives portfolios.

The toolkit is designed to be modular and extensible, enabling you to integrate custom models, stochastic processes, or analytics pipelines.

---

## Features

✅ **Option Pricing Models**
- Black-Scholes (European options, closed-form)
- Binomial Tree (flexible steps, planned American support)
- Easily extendable for custom models

✅ **Monte Carlo Simulation**
- Simulate option pricing under Geometric Brownian Motion (GBM)
- Several different MC options with Numba-JIT integrated
- Customizable drift, volatility, time steps, paths

✅ **Greeks Calculation**
- Delta, Gamma, Vega, Theta, Rho & Second-order Greeks
- Analytical and finite-difference methods

✅ **Portfolio Risk Analysis**
- Aggregate Greeks across portfolios
- Analyze exposures and risk profiles

✅ **Extensible Architecture**
- Plug in custom pricing models or risk engines

✅ **Visualization**
- Payoff diagrams
- Volatility surfaces
- Greeks heatmaps

---

## 🏗️ Project Structure

    |   LICENSE
    |   README.md
    |   requirements.txt
    |   runtime.txt
    |   setup.py
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

- Python ≥ 3.10+
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

### Black-Scholes Pricing

Compute call and put prices using Black-Scholes:

    from src.pricing_models.black_scholes import call_price, put_price

    price_call = call_price(
        S=100,        # Underlying price
        K=110,        # Strike price
        T=0.5,        # Time to maturity (years)
        r=0.01,       # Risk-free rate
        sigma=0.2     # Volatility
    )

    price_put = put_price(
        S=100,
        K=110,
        T=0.5,
        r=0.01,
        sigma=0.2
    )

    print(f"Call Price: {price_call:.4f}")
    print(f"Put Price: {price_put:.4f}")

---

### MonteCarloPricer Tutorial

MonteCarloPricer is a Monte Carlo option pricer with Greeks computation and optional Numba acceleration. It allows you to price European options and compute Delta, Gamma, Vega, Theta, and Rho.

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

### Greeks Calculation

Compute option Greeks:

    from src.greeks.calculate_greeks import calculate_greeks

    greeks = calculate_greeks(
        S=100,
        K=110,
        T=0.5,
        r=0.01,
        sigma=0.2,
        option_type="call"
    )

    print(greeks)

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










