# Development Guide

## Prerequisites

- Python 3.10+
- pip or conda

## Local Setup

```bash
# Clone the repository
git clone https://github.com/StatMixedML/LightGBMLSS.git
cd LightGBMLSS

# Install in editable mode with all extras
pip install -e ".[all_extras,dev]"

# Or install core only
pip install -e .
```

### Dependency Groups

| Group | Install | Contents |
|-------|---------|----------|
| Core | `pip install .` | lightgbm, torch, pyro-ppl, numpy, pandas, scipy, scikit-learn |
| Extras | `pip install ".[all_extras]"` | + matplotlib, optuna, seaborn, shap, ipython |
| Docs | `pip install ".[docs]"` | + mkdocs, mkdocstrings, mkdocs-jupyter |
| Notebooks | `pip install ".[notebooks]"` | + jupyter |
| Dev | `pip install ".[dev]"` | + pytest, pytest-cov, flake8 |

## Running Tests

```bash
# Full test suite (1577 tests, ~50s)
pytest tests/ -q

# Specific test module
pytest tests/test_distribution_utils/ -q

# Single test file
pytest tests/test_distribution_utils/test_crps_score.py -v

# With coverage
pytest tests/ --cov=lightgbmlss --cov-report=term-missing
```

### Test Organization

```
tests/
├── test_distribution_utils/    # DistributionClass methods
│   ├── test_crps_score.py
│   ├── test_stabilize_derivative.py
│   ├── test_compute_gradients_and_hessians.py
│   ├── test_draw_samples.py
│   ├── test_predict_dist.py
│   └── ...
├── test_flow_utils/            # NormalizingFlowClass methods
│   ├── test_crps_score.py
│   ├── test_create_spline_flow.py
│   └── test_replace_parameters.py
├── test_mixture_distribution_utils/  # MixtureDensityClass methods
├── test_distributions/         # End-to-end distribution tests
│   ├── test_univariate_cont_distns.py
│   ├── test_univariate_discrete_distns.py
│   ├── test_spline_flow.py
│   ├── test_mixture.py
│   └── test_expectile.py
├── test_model/                 # LightGBMLSS model integration tests
├── test_utils/                 # Response function tests
└── utils.py                    # Shared test fixtures
```

## Adding a New Distribution

1. **Create the distribution file** in `lightgbmlss/distributions/`. Follow an existing distribution as a template (e.g., `Gaussian.py` for continuous, `Poisson.py` for discrete):

```python
from .distribution_utils import DistributionClass
from ..utils import *

class NewDist(DistributionClass):
    def __init__(self, stabilization="None", loss_fn="nll"):
        param_dict = {
            "loc": identity_fn,      # response function for each parameter
            "scale": exp_fn,
        }
        distribution_arg_names = list(param_dict.keys())
        super().__init__(
            distribution=torch.distributions.Normal,  # PyTorch distribution class
            univariate=True,
            discrete=False,
            n_dist_param=len(param_dict),
            stabilization=stabilization,
            param_dict=param_dict,
            distribution_arg_names=distribution_arg_names,
            loss_fn=loss_fn,
        )
```

2. **Register in `__init__.py`**: Add the import in `lightgbmlss/distributions/__init__.py`.

3. **Add tests**: Add the distribution to the parametrized test lists in `tests/test_distributions/test_univariate_cont_distns.py` or `test_univariate_discrete_distns.py`.

4. **Update the distributions table** in `docs/distributions.md`.

## Building Documentation

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Serve locally with live reload
mkdocs serve

# Build static site
mkdocs build
```

The site is deployed via GitHub Actions to GitHub Pages on push to `master`.

## Code Style

- Follow existing conventions (no separate formatter configured)
- Type hints are used throughout
- Docstrings use NumPy-style formatting
- Keep imports organized: stdlib, third-party, local

## CI/CD

- **Unit tests**: GitHub Actions on push/PR (`.github/workflows/unit-tests.yml`)
- **Documentation**: Auto-deployed via mkdocs on push to master (`.github/workflows/mkdocs.yaml`)
- **Coverage**: Reported via Codecov
