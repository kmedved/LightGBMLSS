# Troubleshooting

## Installation Issues

### `pyro-ppl` vs `Pyro4`

LightGBMLSS requires `pyro-ppl` (Uber's probabilistic programming library), **not** `Pyro4` (a remote object framework). If you see import errors related to Pyro transforms:

```bash
pip uninstall Pyro4        # remove wrong package if installed
pip install pyro-ppl       # install correct package
```

### NumPy / pandas version conflicts

LightGBMLSS pins compatible ranges for core dependencies. If you encounter conflicts with other packages in your environment, consider using a dedicated virtual environment:

```bash
python -m venv lightgbmlss-env
source lightgbmlss-env/bin/activate
pip install lightgbmlss
```

### SHAP installation on Python 3.13+

The `shap` dependency is restricted to Python < 3.14. On newer Python versions, install without extras:

```bash
pip install lightgbmlss  # core only, no shap
```

## Training Issues

### Model does not converge or converges slowly

**Try gradient stabilization:**

```python
from lightgbmlss.distributions.Gaussian import Gaussian

dist = Gaussian(stabilization="MAD")  # or "L2"
```

Stabilization normalizes gradients and Hessians to be comparable in magnitude across distributional parameters. Options:
- `"None"` (default): No stabilization
- `"MAD"`: Median Absolute Deviation normalization
- `"L2"`: L2-norm normalization

**Try standardizing the response:**

If your response variable has a very different scale from the gradient magnitudes, divide by a constant (e.g., `y / 100`) before training.

### NaN loss values

NaN losses typically occur when:
1. Distribution parameters go out of bounds (e.g., negative variance)
2. The response is outside the distribution's support (e.g., negative values with Gamma)

**Solutions:**
- Verify your response variable matches the distribution's support
- Use `initialize=True` to set better starting values
- Lower the learning rate
- Try a different distribution via `dist_select()`

### Starting value optimization fails

If `calculate_start_values` does not converge, the model falls back to default values (0.5 for each parameter). This is usually fine for training but may slow convergence. To debug:

```python
dist = Gaussian()
model = LightGBMLSS(dist)
# Manually check start values
loss, start_values = dist.calculate_start_values(y_train)
print(f"Start values: {start_values}, Loss: {loss}")
```

## Prediction Issues

### Slow predictions

If `predict()` is slow, check your `pred_type`:
- `"parameters"` is fast (no sampling required)
- `"samples"` and `"quantiles"` require drawing from the distribution

For quantile predictions, reducing `n_samples` speeds up prediction at the cost of quantile precision:

```python
model.predict(X, pred_type="quantiles", n_samples=500, quantiles=[0.1, 0.5, 0.9])
```

### Discrete distributions return floats

Discrete distributions (Poisson, NegativeBinomial, etc.) automatically cast predictions to integers. If you need continuous relaxations, use the underlying parameter predictions:

```python
params = model.predict(X, pred_type="parameters")
```

## Distribution Selection

Use `dist_select()` to automatically compare candidate distributions:

```python
from lightgbmlss.distributions import Gaussian, Gamma, StudentT, LogNormal

candidate_distributions = [Gaussian, Gamma, StudentT, LogNormal]

dist = Gaussian()
model = LightGBMLSS(dist)
model.dist_select(
    target=y_train,
    candidate_distributions=candidate_distributions,
    max_iter=50
)
```

This ranks distributions by NLL on the training data to help choose the best parametric family.

## SHAP Plots

### SHAP errors with certain distributions

SHAP requires predictions to be numeric arrays. If you encounter errors:

```python
# Use pred_type="parameters" for SHAP
model.plot(X, parameter="loc", plot_type="beeswarm")
```

### Import errors for SHAP

SHAP is an optional dependency. Install it separately:

```bash
pip install "lightgbmlss[all_extras]"
```
