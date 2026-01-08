# References

A comprehensive list of academic papers, books, and resources used in the development of OptionsLab.

---

## Option Pricing Theory

### Black-Scholes Model

- **Black, F., & Scholes, M.** (1973). _The Pricing of Options and Corporate Liabilities_. Journal of Political Economy, 81(3), 637-654.
  - Foundation for European option pricing
  - Used in: `src/pricing_models/black_scholes.py`

### Binomial Tree Model

- **Cox, J. C., Ross, S. A., & Rubinstein, M.** (1979). _Option Pricing: A Simplified Approach_. Journal of Financial Economics, 7(3), 229-263.
  - Discrete-time option pricing model
  - Used in: `src/pricing_models/binomial_tree.py`

---

## Monte Carlo Methods

### Standard Monte Carlo

- **Boyle, P. P.** (1977). _Options: A Monte Carlo Approach_. Journal of Financial Economics, 4(3), 323-338.
  - First application of MC to option pricing
  - Used in: `src/pricing_models/monte_carlo.py`

### Variance Reduction

- **Glasserman, P.** (2003). _Monte Carlo Methods in Financial Engineering_. Springer.
  - Antithetic variates, control variates
  - Reference for advanced MC techniques

---

## American Options

### Longstaff-Schwartz LSM

- **Longstaff, F. A., & Schwartz, E. S.** (2001). _Valuing American Options by Simulation: A Simple Least-Squares Approach_. Review of Financial Studies, 14(1), 113-147.
  - Regression-based early exercise boundary
  - Used in: `src/pricing_models/exotic_options.py` → `AmericanOption`

---

## Exotic Options

### Asian Options

- **Kemna, A. G. Z., & Vorst, A. C. F.** (1990). _A Pricing Method for Options Based on Average Asset Values_. Journal of Banking & Finance, 14(1), 113-129.
  - Geometric Asian closed-form solution
  - Used in: `src/pricing_models/exotic_options.py` → `AsianOption`

### Barrier Options

- **Merton, R. C.** (1973). _Theory of Rational Option Pricing_. Bell Journal of Economics and Management Science, 4(1), 141-183.
  - Continuous barrier pricing theory
  - Used in: `src/pricing_models/exotic_options.py` → `BarrierOption`

---

## Implied Volatility

### Newton-Raphson Method

- **Manaster, S., & Koehler, G.** (1982). _The Calculation of Implied Variances from the Black-Scholes Model_. Journal of Finance, 37(1), 227-230.
  - Newton-Raphson IV solver
  - Used in: `src/pricing_models/iv_solver.py`

### Volatility Smile/Surface

- **Dupire, B.** (1994). _Pricing with a Smile_. Risk Magazine, 7(1), 18-20.
  - Local volatility theory
  - Reference for IV surface modeling

---

## Machine Learning in Finance

### Gradient Boosting

- **Ke, G., et al.** (2017). _LightGBM: A Highly Efficient Gradient Boosting Decision Tree_. NeurIPS.
  - LightGBM algorithm
  - Used in: `src/pricing_models/monte_carlo_ml.py`

### Hyperparameter Optimization

- **Akiba, T., et al.** (2019). _Optuna: A Next-generation Hyperparameter Optimization Framework_. KDD.
  - Bayesian optimization with TPE
  - Used in: `src/optimization/study_manager.py`

---

## Risk Management

### Value at Risk

- **Jorion, P.** (2006). _Value at Risk: The New Benchmark for Managing Financial Risk_. McGraw-Hill.
  - VaR methodology
  - Used in: `src/risk_analysis/var.py`

### Expected Shortfall

- **Artzner, P., et al.** (1999). _Coherent Measures of Risk_. Mathematical Finance, 9(3), 203-228.
  - CVaR/ES as coherent risk measure
  - Used in: `src/risk_analysis/expected_shortfall.py`

---

## Software & Libraries

| Library      | Version | Purpose                     | Reference                                                                |
| ------------ | ------- | --------------------------- | ------------------------------------------------------------------------ |
| NumPy        | ≥1.24   | Numerical computing         | [numpy.org](https://numpy.org)                                           |
| SciPy        | ≥1.11   | Scientific computing        | [scipy.org](https://scipy.org)                                           |
| PyTorch      | ≥2.0    | Deep learning               | [pytorch.org](https://pytorch.org)                                       |
| LightGBM     | ≥4.1    | Gradient boosting           | [lightgbm.readthedocs.io](https://lightgbm.readthedocs.io)               |
| Optuna       | ≥3.5    | Hyperparameter optimization | [optuna.org](https://optuna.org)                                         |
| ONNX Runtime | ≥1.17   | Model inference             | [onnxruntime.ai](https://onnxruntime.ai)                                 |
| yfinance     | ≥0.2    | Market data                 | [github.com/ranaroussi/yfinance](https://github.com/ranaroussi/yfinance) |
| Streamlit    | ≥1.32   | Web UI                      | [streamlit.io](https://streamlit.io)                                     |

---

## Additional Resources

### Textbooks

- Hull, J. C. (2021). _Options, Futures, and Other Derivatives_ (11th ed.). Pearson.
- Wilmott, P. (2006). _Paul Wilmott on Quantitative Finance_ (2nd ed.). Wiley.

### Online Resources

- [QuantLib](https://www.quantlib.org/) - Open-source quantitative finance library
- [Wilmott Forums](https://wilmott.com/) - Quantitative finance community

---

_Last updated: 2026-01-08_
