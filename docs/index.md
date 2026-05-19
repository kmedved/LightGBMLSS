<img align="right" width="156.5223" height="181.3" src="LightGBMLSS.png">

# LightGBMLSS - An extension of LightGBM to probabilistic modelling

LightGBMLSS is a comprehensive framework that models and predicts the full conditional distribution of a univariate target as a function of covariates. Choosing from a wide range of continuous, discrete, and mixed discrete-continuous distributions, it enhances LightGBM with probabilistic forecasts from which prediction intervals and quantiles of interest can be derived.

## Features

- Estimation of all distributional parameters
- [Normalizing Flows](dgbm.md#normalizing-flows) for complex and multi-modal distributions
- [Mixture Densities](dgbm.md#mixture-distributions) for diverse data characteristics
- Zero-Adjusted and Zero-Inflated Distributions for excess zeros
- Automatic gradient/Hessian derivation via [PyTorch autograd](https://pytorch.org/docs/stable/autograd.html)
- Hyperparameter search via [Optuna](https://optuna.org/)
- Feature importance via [SHAP](https://github.com/dsgibbons/shap)
- Full compatibility with LightGBM features

## Installation

```bash
pip install lightgbmlss
```

Or for the development version:

```bash
pip install git+https://github.com/StatMixedML/LightGBMLSS.git
```

To install with optional dependencies (SHAP, Optuna, plotting):

```bash
pip install "lightgbmlss[all_extras]"
```

## Quickstart

```python
from lightgbmlss.distributions.Gaussian import Gaussian
from lightgbmlss.model import LightGBMLSS
import lightgbm as lgb

dist = Gaussian(stabilization="None", loss_fn="nll")
model = LightGBMLSS(dist)

dtrain = lgb.Dataset(X_train, label=y_train)
params = {"learning_rate": 0.05, "max_depth": 3}
model.train(params, dtrain, num_boost_round=100)

pred_params = model.predict(X_test, pred_type="parameters")
pred_quantiles = model.predict(X_test, pred_type="quantiles", quantiles=[0.1, 0.5, 0.9])
```

## Documentation

- [Distributional Modelling](dgbm.md) - Theory behind GAMLSS and distributional gradient boosting
- [Available Distributions](distributions.md) - Complete reference of supported distributions
- [Architecture](architecture.md) - System design, component diagrams, training/prediction flow
- [Development Guide](development.md) - Local setup, running tests, adding new distributions
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [API Reference](api.md) - Auto-generated from docstrings

## Notes

### Stabilization
Since LightGBMLSS updates parameter estimates by optimizing gradients and Hessians, it is important that these are comparable in magnitude for all distributional parameters. Use `stabilization="MAD"` or `stabilization="L2"` if estimation is unstable. For improved convergence, standardizing the response variable (e.g., `y / 100`) can also help.

### Runtime
LightGBMLSS uses a one-vs-all estimation strategy where a separate tree is grown for each distributional parameter. Training requires `[iterations] * [parameters]` trees, so runtime scales with the number of distributional parameters.

## Reference

- [Distributional Gradient Boosting Machines](https://arxiv.org/abs/2204.00778) (Marz and Kneib, 2022)
- [XGBoostLSS: An extension of XGBoost to probabilistic forecasting](https://arxiv.org/abs/1907.03178) (Marz, 2019)
