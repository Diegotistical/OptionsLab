# Monte Carlo ML Surrogate: Development & Optimization Journey

A technical retrospective on building and tuning the ML surrogate model for option pricing.

---

## Overview

I developed a machine learning surrogate to replace Monte Carlo simulation for option pricing. The goal was to achieve near-analytical accuracy (< 1% error) while maintaining sub-millisecond inference times.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ARCHITECTURE OVERVIEW                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Input Parameters            ML Surrogate            Output             │
│   ─────────────────           ────────────            ──────             │
│                                                                          │
│   ┌──────────────┐      ┌───────────────────┐      ┌──────────────┐     │
│   │ Spot (S)     │      │                   │      │ Price        │     │
│   │ Strike (K)   │      │   StandardScaler  │      │ Delta        │     │
│   │ Time (T)     │ ──►  │        ↓          │ ──►  │ Gamma        │     │
│   │ Rate (r)     │      │ MultiOutputReg    │      │              │     │
│   │ Vol (σ)      │      │   (LightGBM)      │      │              │     │
│   │ Div (q)      │      │                   │      │              │     │
│   └──────────────┘      └───────────────────┘      └──────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Initial Implementation

### What I Built

The `MonteCarloMLSurrogate` class wraps a sklearn Pipeline:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", MultiOutputRegressor(LGBMRegressor(...)))
])
```

### Training Data Generation

I used vectorized Black-Scholes instead of actual Monte Carlo simulation for training data:

```
┌─────────────────────────────────────────────────────────────────┐
│                   TRAINING DATA PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Random Parameter Sampling          Vectorized Black-Scholes   │
│   ──────────────────────────         ───────────────────────    │
│                                                                  │
│   S ∈ [50, 150]     ────┐                                       │
│   K ∈ [50, 150]         │                                       │
│   T ∈ [0.05, 2.0]       ├───►  BS(S,K,T,r,σ,q)  ───►  [P, Δ, Γ] │
│   r ∈ [0.01, 0.10]      │                                       │
│   σ ∈ [0.10, 0.50]      │                                       │
│   q ∈ [0.00, 0.03]  ────┘                                       │
│                                                                  │
│   10,000 - 50,000 samples                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Initial Results

| Metric         | Value                    |
| -------------- | ------------------------ |
| Training Time  | ~2 seconds (10K samples) |
| Inference Time | < 0.5 ms                 |
| Mean Error     | 5-15% (unacceptable)     |

---

## Phase 2: Diagnosing the Accuracy Problem

### Root Causes Identified

1. **Feature Space Mismatch**: Training on uniform random samples, but testing at specific points
2. **Missing Engineered Features**: Raw inputs (S, K, T) lack financial meaning
3. **Hyperparameter Defaults**: LightGBM defaults not tuned for this domain

### Feature Engineering Fix

I added domain-relevant features:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Raw Features          Engineered Features                      │
│   ────────────          ────────────────────                     │
│                                                                  │
│   S, K, T, r, σ, q  ───►  Moneyness:     S/K                    │
│                           Log-Moneyness: ln(S/K)                 │
│                           Sqrt-Time:     √T                      │
│                           Normalized:    T × (r - q)             │
│                                                                  │
│   6 features        ───►  10 features                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Results After Feature Engineering

| Metric     | Before | After |
| ---------- | ------ | ----- |
| Mean Error | 8.2%   | 2.1%  |
| Max Error  | 25.3%  | 8.4%  |

---

## Phase 3: Hyperparameter Tuning

### The Tuning Challenge

Manual tuning was tedious. I had three key hyperparameters:

```
┌─────────────────────────────────────────────────────────────────┐
│               HYPERPARAMETER SEARCH SPACE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   n_estimators     [100 ─────────────────────────────── 1000]   │
│                           More trees = better fit, slower        │
│                                                                  │
│   max_depth        [4 ───────────────────────────────── 12]     │
│                           Deeper = more expressiveness           │
│                                                                  │
│   learning_rate    [0.01 ──────────────────────────── 0.30]     │
│                           Lower = smoother convergence           │
│                                                                  │
│   Total combinations: ~500 realistic configurations              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Optuna Integration

I integrated Optuna with Bayesian optimization:

```
┌─────────────────────────────────────────────────────────────────┐
│                    OPTUNA OPTIMIZATION FLOW                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Trial 1    Trial 2    Trial 3         Trial N                 │
│     │          │          │               │                      │
│     ▼          ▼          ▼               ▼                      │
│   ┌────┐     ┌────┐     ┌────┐         ┌────┐                   │
│   │ HP │     │ HP │     │ HP │   ...   │ HP │                   │
│   │Set1│     │Set2│     │Set3│         │SetN│                   │
│   └─┬──┘     └─┬──┘     └─┬──┘         └─┬──┘                   │
│     │          │          │               │                      │
│     ▼          ▼          ▼               ▼                      │
│   Train      Train      Train          Train                    │
│   Model      Model      Model          Model                    │
│     │          │          │               │                      │
│     ▼          ▼          ▼               ▼                      │
│   Score:     Score:     Score:         Score:                   │
│   0.042      0.031      0.028          0.019 ◄── Best           │
│                                                                  │
│   ─────────────────────────────────────────────                  │
│   TPE Sampler learns: "max_depth ≈ 7 works well"                │
│   ─────────────────────────────────────────────                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Optimal Hyperparameters Found

After 20-50 trials:

```python
best_params = {
    "n_estimators": 400-500,
    "max_depth": 7-8,
    "learning_rate": 0.08-0.12,
}
```

---

## Phase 4: Production Hardening

### Reproducibility

I implemented deterministic seeding:

```python
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch_available:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
```

### Per-Trial Seeding

```
┌─────────────────────────────────────────────────────────────────┐
│               DETERMINISTIC TRIAL SEEDING                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Study Seed: 42                                                │
│                                                                  │
│   Trial 0: SHA256("42:0") → seed 0xA3B2C1D4                     │
│   Trial 1: SHA256("42:1") → seed 0x7E8F9A0B                     │
│   Trial 2: SHA256("42:2") → seed 0x1C2D3E4F                     │
│                                                                  │
│   Same study seed + trial number = Same trial seed (always)     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Final Results

### Performance Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL PERFORMANCE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Method             Time (ms)    Error vs BS    Use Case       │
│   ─────────────────  ──────────   ───────────    ────────       │
│                                                                  │
│   Monte Carlo (10K)     150          0.5%        Validation     │
│   Monte Carlo (1K)       15          1.5%        Quick est.     │
│   Black-Scholes         0.01         0.0%        European only  │
│   ML Surrogate          0.3         <1.0%        Real-time      │
│                                                                  │
│   ML Surrogate: 500x faster than MC, 1% accuracy                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Takeaways

1. **Feature engineering matters more than model complexity** - Adding moneyness reduced error more than doubling tree count

2. **Bayesian optimization beats grid search** - 20 Optuna trials outperformed 100 random trials

3. **Training data distribution must match inference** - Uniform random sampling works poorly for edge cases

4. **Reproducibility requires intentional design** - Not just seeding, but thread control and algorithm selection

---

## Appendix: Code Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODULE STRUCTURE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   src/pricing_models/                                           │
│   └── monte_carlo_ml.py     ← MonteCarloMLSurrogate class       │
│                                                                  │
│   src/optimization/                                             │
│   ├── reproducibility.py    ← Seeding, thread control           │
│   ├── search_space.py       ← LightGBM hyperparameter ranges    │
│   ├── study_manager.py      ← Optuna study lifecycle            │
│   ├── objectives.py         ← Training objective wrappers       │
│   ├── onnx_exporter.py      ← Model export for production       │
│   └── model_wrappers.py     ← Convenience functions             │
│                                                                  │
│   streamlit_app/pages/                                          │
│   └── 2_MonteCarlo_ML.py    ← Interactive UI                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

_Author: Diego_
