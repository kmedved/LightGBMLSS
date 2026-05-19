# Architecture

LightGBMLSS extends LightGBM from point predictions to full distributional modelling. Instead of estimating only the conditional mean, it estimates all parameters of a chosen distribution as functions of covariates.

## High-Level Design

```mermaid
flowchart TD
    A[User Code] --> B[LightGBMLSS Model]
    B --> C[LightGBM Booster]
    B --> D[Distribution Layer]
    D --> E[DistributionClass]
    D --> F[NormalizingFlowClass]
    D --> G[MixtureDensityClass]
    E --> H[PyTorch Distributions]
    F --> I[Pyro Transforms]
    G --> H
    C -->|raw scores| J[Response Functions]
    J -->|transformed params| D
    D -->|gradients, hessians| C
```

## Core Components

### `LightGBMLSS` (model.py)

The main user-facing class. Wraps a LightGBM Booster and delegates distributional logic to one of the three base classes. Key methods:

| Method | Purpose |
|--------|---------|
| `train()` | Train the model with custom objective and metric functions |
| `predict()` | Generate predictions: parameters, samples, quantiles, or expectiles |
| `hyper_opt()` | Bayesian hyperparameter search via Optuna |
| `cv()` | Cross-validation with distributional objective |
| `plot()` | SHAP-based feature importance for each distributional parameter |
| `save_model()` / `load_model()` | Persistence via pickle |

### Distribution Base Classes

LightGBMLSS uses three base classes, each providing `objective_fn`, `metric_fn`, `predict_dist`, and gradient/Hessian computation:

```mermaid
classDiagram
    class DistributionClass {
        +objective_fn()
        +metric_fn()
        +predict_dist()
        +compute_gradients_and_hessians()
        +calculate_start_values()
        +crps_score()
        +stabilize_derivative()
        +draw_samples()
    }
    class NormalizingFlowClass {
        +create_spline_flow()
        +replace_parameters()
        +objective_fn()
        +predict_dist()
        +crps_score()
    }
    class MixtureDensityClass {
        +get_component_distributions()
        +create_mixture_distribution()
        +objective_fn()
        +predict_dist()
    }
    DistributionClass <|-- Gaussian
    DistributionClass <|-- Gamma
    DistributionClass <|-- StudentT
    DistributionClass <|-- Beta
    NormalizingFlowClass <|-- SplineFlow
    MixtureDensityClass <|-- Mixture
```

- **DistributionClass** (`distribution_utils.py`): Standard parametric distributions (Gaussian, Gamma, StudentT, Beta, Weibull, etc.). Maps LightGBM raw scores through response functions to distribution parameters, then uses PyTorch distributions for likelihood and sampling.

- **NormalizingFlowClass** (`flow_utils.py`): Spline-based normalizing flows via Pyro. Learns flexible invertible transformations of a base distribution, enabling modelling of complex and multi-modal densities.

- **MixtureDensityClass** (`mixture_distribution_utils.py`): Mixture density models. Combines M component distributions with learned mixing weights via Gumbel-Softmax.

### Response Functions (utils.py)

Map unbounded raw LightGBM scores to valid parameter ranges:

| Function | Maps to | Used for |
|----------|---------|----------|
| `identity_fn` | (-inf, inf) | Location parameters (mean) |
| `exp_fn` | (0, inf) | Scale parameters (variance) |
| `softplus_fn` | (0, inf) | Smooth positive parameters |
| `sigmoid_fn` | (0, 1) | Probability parameters |
| `softmax_fn` | Simplex | Mixture weights |

## Training Flow

```mermaid
sequenceDiagram
    participant User
    participant Model as LightGBMLSS
    participant LGB as LightGBM Booster
    participant Dist as Distribution

    User->>Model: train(params, dtrain)
    Model->>Dist: calculate_start_values(target)
    Dist-->>Model: init_scores
    Model->>LGB: train(fobj=objective_fn, feval=metric_fn)

    loop Each Boosting Iteration
        LGB->>Dist: objective_fn(predictions, data)
        Dist->>Dist: get_params_loss() [transform + build distribution]
        Dist->>Dist: torch.autograd → gradients, hessians
        Dist->>Dist: stabilize_derivative() [MAD or L2]
        Dist-->>LGB: (grad, hess) as numpy arrays
        LGB->>LGB: Build tree for each parameter
        LGB->>Dist: metric_fn(predictions, data)
        Dist-->>LGB: loss value
    end

    LGB-->>Model: trained booster
```

## Prediction Flow

```mermaid
flowchart LR
    A[Input Data] --> B[Booster.predict → raw scores]
    B --> C[Add init_scores]
    C --> D[Apply response functions]
    D --> E{pred_type?}
    E -->|parameters| F[Return parameter DataFrame]
    E -->|samples| G[Draw from distribution]
    E -->|quantiles| H[Sample → compute quantiles]
    E -->|expectiles| F
```

## Loss Functions

LightGBMLSS supports two loss functions:

- **NLL** (Negative Log-Likelihood): Standard maximum likelihood. Gradients and Hessians computed via `torch.autograd`.
- **CRPS** (Continuous Ranked Probability Score): Sample-based scoring rule. Uses the energy form: `E|X-y| - 0.5*E|X-X'|`. When CRPS is used, Hessians are set to 1 (CRPS is not twice differentiable).

## Directory Structure

```
lightgbmlss/
├── model.py                    # LightGBMLSS class (train, predict, hyper_opt)
├── utils.py                    # Response functions (exp, softplus, sigmoid, etc.)
├── logger.py                   # Custom LightGBM logger (suppresses warnings)
├── datasets/
│   └── data_loader.py          # Simulated dataset loaders
└── distributions/
    ├── distribution_utils.py   # Base class for parametric distributions
    ├── flow_utils.py           # Base class for normalizing flows
    ├── mixture_distribution_utils.py  # Base class for mixture densities
    ├── zero_inflated.py        # Zero-inflated distribution helpers
    ├── Gaussian.py             # Individual distribution definitions
    ├── Gamma.py                # (one file per distribution)
    ├── StudentT.py
    ├── Beta.py
    ├── SplineFlow.py
    ├── Mixture.py
    └── ...                     # 20+ distribution files total
```
