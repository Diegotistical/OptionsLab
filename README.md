# 📈 Quant Options Toolkit

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/Diegotistical/quant-options-toolkit)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/Diegotistical/quant-options-toolkit)](https://github.com/Diegotistical/quant-options-toolkit/commits/main)

**Quant Options Toolkit** is a Python library for quantitative finance practitioners, researchers, and students. It provides models, simulations, and risk analysis tools for pricing and analyzing options and derivatives portfolios.

The toolkit is designed to be modular and extensible, enabling you to integrate custom models, stochastic processes, or analytics pipelines.

---

## Features

✅ **Option Pricing Models**
- Black-Scholes (European options, closed-form)
- Binomial Tree (flexible steps, planned American support)
- Easily extendable for custom models

✅ **Monte Carlo Simulation**
- Simulate option pricing under Geometric Brownian Motion (GBM)
- Customizable drift, volatility, time steps, paths

✅ **Greeks Calculation**
- Delta, Gamma, Vega, Theta, Rho
- Analytical and finite-difference methods

✅ **Portfolio Risk Analysis**
- Aggregate Greeks across portfolios
- Analyze exposures and risk profiles

✅ **Extensible Architecture**
- Plug in custom pricing models or risk engines

✅ **Visualization (Planned)**
- Payoff diagrams
- Volatility surfaces
- Greeks heatmaps

---

## 🏗️ Project Structure

    quant-options-toolkit/
    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    └── src
        ├── greeks/
        │   ├── __init__.py
        │   └── calculate_greeks.py
        ├── pricing_models/
        │   ├── __init__.py
        │   ├── black_scholes.py
        │   ├── binomial_tree.py
        │   └── utils.py
        ├── risk_analysis/
        │   ├── __init__.py
        │   └── portfolio_risk.py
        └── simulation/
            ├── __init__.py
            └── monte_carlo.py

---

## 🚀 Getting Started

### Requirements

- Python ≥ 3.8
- Recommended: virtual environment (venv, conda, etc.)

### Installation

Clone the repository:

    git clone https://github.com/Diegotistical/quant-options-toolkit.git
    cd quant-options-toolkit

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

### Monte Carlo Simulation

Estimate an option price via Monte Carlo simulation:

    from src.simulation.monte_carlo import simulate_option_price

    price, stddev = simulate_option_price(
        S0=100,
        K=110,
        T=1,
        r=0.01,
        sigma=0.2,
        n_simulations=100_000,
        option_type="call"
    )

    print(f"Estimated Call Price: {price:.4f} ± {stddev:.4f}")

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

## 🛣️ Roadmap

- [x] Black-Scholes pricing (European)
- [x] Monte Carlo engine for European options
- [x] Greeks calculation (analytical & numerical)
- [ ] Binomial Tree American options support
- [ ] Visualization: payoff diagrams, volatility surfaces
- [ ] Market data integration (e.g. Yahoo Finance, Alpha Vantage)
- [ ] Advanced risk metrics (VaR, CVaR)

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
