# Optuna & ONNX Integration: Lessons Learned

A practical guide based on integrating production-grade hyperparameter optimization and model export into OptionsLab.

---

## 1. LightGBM + ONNX = Pain

### The Problem

LightGBM wrapped in sklearn's `Pipeline` or `MultiOutputRegressor` doesn't export cleanly to ONNX.

### What Failed

```python
# This doesn't work out of the box
ONNXExporter.export_sklearn(pipeline_with_lgbm, ...)
# Error: Unable to find alias for 'LGBMRegressor'
```

### The Fix

Register LightGBM's converter with skl2onnx **before** conversion:

```python
from skl2onnx import update_registered_converter
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm

update_registered_converter(
    LGBMRegressor,
    "LightGbmLGBMRegressor",
    calculate_linear_regressor_output_shapes,
    convert_lightgbm,
)
```

Also need to pin the ML domain opset:

```python
target_opset={"": 15, "ai.onnx.ml": 3}
```

### Lesson

**Don't assume sklearn-compatible = ONNX-compatible.** Always test export before building the UI around it.

---

## 2. Streamlit Caches Python Modules

### The Problem

After editing `onnx_exporter.py`, the Streamlit app kept using the old code.

### The Fix

Restart Streamlit server after modifying any imported Python module:

```bash
# Ctrl+C to stop
python -m streamlit run streamlit_app/app.py
```

### Lesson

Streamlit doesn't hot-reload imported modules the way you'd expect. This bit us multiple times during debugging.

---

## 3. Optimization ≠ Training

### The Problem

Running Optuna optimization finds great hyperparameters, but the main "Train & Predict" button used hardcoded defaults.

### The Gap

```
[Optuna] → Best params: n_estimators=420, max_depth=7
[Train Button] → Uses: n_estimators=300, max_depth=8  ← WRONG
```

### The Fix

Store optimized params in `st.session_state` and auto-fill UI controls:

```python
has_optimized = "optuna_result" in st.session_state
if has_optimized:
    default_trees = st.session_state["optuna_result"].best_params.get("n_estimators", 300)
```

### Lesson

**Don't just find optimal params—surface them to the user.** Otherwise optimization is useless.

---

## 4. Reproducibility Is Hard

### What We Did

- Global seeding (NumPy, PyTorch, random, LightGBM)
- Per-trial deterministic seeds via SHA256 hashing
- Thread limit controls (OMP, MKL, PyTorch)
- GPU determinism flags

### What Still Breaks

- LightGBM's `early_stopping_callback` has non-determinism
- Different NumPy versions can produce different random sequences
- Windows vs Linux floating-point differences

### Lesson

**100% reproducibility is a myth.** Aim for "close enough to debug" not "bit-perfect."

---

## 5. Validation > Pointwise Equality

### The Problem

ONNX uses float32, native model uses float64. Pointwise `==` fails:

```python
assert np.allclose(native_pred, onnx_pred)  # Fails at 1e-6 precision
```

### The Fix

Distributional validation:

```python
validator = ONNXValidator(rtol=1e-3, atol=1e-4)
result = validator.validate(native_model, onnx_path, X_test)
# Checks: correlation, percentile diffs, sign agreement
```

### Lesson

**For ML models, statistical equivalence matters more than numerical equality.**

---

## 6. Dependencies Are Fragile

### Version Conflicts Encountered

| Package       | Issue                           |
| ------------- | ------------------------------- |
| `onnxmltools` | Required for LightGBM converter |
| `skl2onnx`    | Needed for Pipeline export      |
| `protobuf`    | ONNX requires specific versions |
| `onnxruntime` | Must match ONNX opset version   |

### The Fix

Pin versions in requirements.txt:

```
optuna>=3.5.0,<4.0.0
onnx>=1.15.0,<1.17.0
onnxruntime>=1.17.0,<1.19.0
onnxmltools>=1.12.0,<1.13.0
```

### Lesson

**Pin everything.** ONNX ecosystem is particularly sensitive to version mismatches.

---

## 7. Separation of Concerns Saved Us

### Architecture Decision

Models stay "dumb" (just `fit`, `predict`). Optimization and export are external wrappers.

```
src/optimization/
├── study_manager.py      ← Optuna lifecycle
├── objectives.py         ← Wraps model training
├── onnx_exporter.py      ← Export logic
└── model_wrappers.py     ← Convenience functions
```

### Why It Worked

- Easy to test each component independently
- Could swap optimizers without touching models
- ONNX export bugs didn't break core pricing

### Lesson

**Keep models clean. Tooling wraps, not invades.**

---

## Summary: What Would I Do Differently

1. **Test ONNX export early** - before building any UI
2. **Use joblib for quick wins** - ONNX is for production, not prototyping
3. **Always show optimized params in UI** - don't make users copy-paste
4. **Document version requirements** - ONNX stack is brittle
5. **Add a "use optimized" button** - explicit is better than implicit

---
