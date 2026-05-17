# NAML Aerodynamic Surrogate Models

A unified Python framework for training, optimizing, and evaluating surrogate models for aerodynamic coefficient prediction. Four canonical models:

1. **mlp** — Multi-Layer Perceptron (single hidden block, activation-driven init)
2. **gp** — Gaussian Process (Kriging) with learnable exponential kernel
3. **mlpgp_ns** — MLP + non-stationary (Gibbs) GP kernel; MLP predicts local σ(x), ℓ(x)
4. **mlpgp_dkl** — MLP + Deep Kernel Learning; MLP maps inputs to a small feature space, then a stationary kernel acts on the features

## Features

- **Unified CLI**: Single entry point (`main.py`) with two workflow modes — `single_run` and `convergence` (HPO is hosted inside convergence via `convergence_config.yaml: mode: optimization`).
- **Modular architecture**: Clean separation across data, models, training, inference, and visualization.
- **YAML configuration**: Hierarchical composable configs; `training_common_config.yaml` holds defaults shared across trainers.
- **Learnable kernels**: ARD exponential kernel with constrained power exponent.
- **Hyperparameter optimization**: Optuna integration (optional dependency) invoked through convergence sub-mode.
- **Uncertainty quantification**: GP / MLPGP variants emit posterior std at inference time.
- **Two-level error boundary**: failures in a single model don't crash the batch.
- **Reproducibility**: Config snapshots, deterministic training, global seed management.

## Installation

### Prerequisites
- Python 3.10+
- CUDA 12.1+ (optional, for GPU)

### Setup

```bash
cd naml-surrogate-models
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# Optional: install optuna for hyperparameter optimization
pip install optuna
```

## Quick Start

### Single training run

```bash
# Configure dataset and models in config/main_config.yaml (`models_to_train: [...]`).
python main.py --mode single_run

# Override the model list from the CLI (comma-separated, "all" allowed):
python main.py --mode single_run --models mlp,gp,mlpgp_ns
```

### Convergence analysis (repeated split/seed sweep)

```bash
# Set convergence_config.yaml: mode: analysis
python main.py --mode convergence
```

### Hyperparameter optimization (HPO is a convergence sub-mode)

```bash
# Set convergence_config.yaml: mode: optimization, then run convergence.
python main.py --mode convergence
```

### Migration from the previous release

- Top-level `--mode optimization` has been retired. Set
  `convergence_config.yaml: mode: optimization` and run `--mode convergence` instead.
- `model_to_train: "mlp"` → `models_to_train: ["mlp"]`. `"all"` still works as a one-element alias for
  `["mlp", "gp", "mlpgp_ns", "mlpgp_dkl"]`.
- The single `mlpgp` model name is gone — pick `mlpgp_ns` (Gibbs kernel) or `mlpgp_dkl` (DKL) explicitly.
  Old checkpoints under `outputs/<dataset>/single_run/mlpgp/...` are not auto-migrated.
- SVGP variants and the legacy Hyperband knobs in `optimize_*_config.yaml` have been removed.

## Project Structure

```
├── README.md
├── requirements.txt
├── main.py                          # Unified mode dispatcher
│
├── src/
│   ├── convergence_analysis.py      # Convergence analysis + HPO orchestration
│   ├── entrypoints/
│   │   └── convergence.py           # Package-level convergence entrypoint
│   ├── pipeline.py                  # Shared train_model() and model/trainer building
│   ├── optimization_support.py      # Objective functions and HPO utilities
│   ├── data/
│   │   └── dataset.py              # Data loading, preprocessing, splitting
│   ├── models/
│   │   ├── mlp.py                  # Single-block MLP regressor
│   │   ├── gp.py                   # Exact GP with exponential kernel
│   │   ├── mlpgp.py                # MLPGPModel + MLPGPNonStationaryModel + MLPGPDeepKernelModel
│   │   └── gp_utils.py             # Kernel factories and shared helpers
│   ├── training/
│   │   ├── trainer.py              # MLPTrainer, GPTrainer, MLPGPTrainer
│   │   └── metrics.py              # MSE, MAE, R², RMSE, MAPE
│   ├── inference/
│   │   └── predictor.py            # Unified prediction interface
│   ├── visualization/
│   │   ├── runner.py               # Per-mode dispatcher
│   │   ├── plots/                  # Leaf plot functions (diagnostics, metrics, surface, …)
│   │   ├── suites/                 # Per-mode orchestration (single_run, optimization, convergence)
│   │   ├── datasets/               # Dataset-specific physics suites (HYDDB)
│   │   ├── config.py / labels.py / utils.py
│   └── utils/
│       ├── config.py               # YAML I/O, device resolution
│       ├── paths.py                # Path resolution and output structure
│       ├── entrypoints.py          # Shared config bootstrap
│       └── __init__.py             # setup_logger, set_global_seed, etc.
│
├── config/                          # All configs are YAML-based
│   ├── main_config.yaml            # Top-level: mode, models_to_train (list), seed, precision
│   ├── dataset/
│   │   └── dataset_config.yaml     # All datasets with metadata
│   ├── model/
│   │   ├── mlp_config.yaml
│   │   ├── gp_config.yaml
│   │   ├── mlpgp_ns_config.yaml
│   │   └── mlpgp_dkl_config.yaml
│   ├── training/
│   │   ├── training_common_config.yaml   # Shared scheduler, early-stopping, log cadence
│   │   ├── training_mlp_config.yaml
│   │   ├── training_gp_config.yaml
│   │   ├── training_mlpgp_ns_config.yaml
│   │   └── training_mlpgp_dkl_config.yaml
│   ├── optimization/
│   │   ├── optimize_mlp_config.yaml
│   │   ├── optimize_gp_config.yaml
│   │   ├── optimize_mlpgp_ns_config.yaml
│   │   └── optimize_mlpgp_dkl_config.yaml
│   ├── convergence/
│   │   └── convergence_config.yaml       # `mode: analysis | optimization` selects sub-mode
│   └── visualization/
│       └── visualization_config.yaml
│
├── data/                            # CSV datasets
│   ├── HYDDB_v2_output.csv
│   ├── AEDB_v1_output.csv
│   ├── airfoil_self_noise.csv
│   └── ...
│
└── outputs/                         # Run outputs (timestamped, gitignored)
    └── <dataset>/
        ├── single_run/
        ├── testing/
        └── convergence_analysis/
```

## Configuration Guide

### Main Orchestrator (`config/main_config.yaml`)

```yaml
mode: "single_run"          # "single_run" or "convergence" (HPO lives inside convergence)
dataset: "HYDDB_v2"         # Dataset key from dataset_config.yaml
models_to_train: ["mlp"]    # Any subset of [mlp, gp, mlpgp_ns, mlpgp_dkl]; ["all"] is an alias
run_mode: "training"        # "training", "test", or "training+test"
global_seed: 21
precision: "float32"        # "float32" or "float64"
```

### Model Configuration (e.g., `config/model/gp_config.yaml`)

Example for exponential GP kernel:

```yaml
trend_degree: 1
kernel:
  type: "exponential"
  theta: 1.0                # Learnable parameter (init value)
  power_exponent: 2.0       # Learnable parameter (init value)
  tau2: 1.0                 # Outputscale (variance)
  likelihood_noise: 1e-4    # Likelihood noise
```

### Training Configuration (`config/training/training_common_config.yaml`)

```yaml
optimizer: "adam"
learning_rate: 1e-3
weight_decay: 0.0
scheduler_type: "steplr"
scheduler_step_size: 50
scheduler_gamma: 0.5
cholesky_jitter_float32: 1e-6
cholesky_jitter_float64: 1e-8
n_epochs: 100               # MLP
batch_size: 128             # MLP
device: "auto"              # "cuda", "cpu", or "auto"
```

## Usage Examples

### Train MLP with custom config

```bash
python main.py --mode single_run --models mlp --output-dir outputs
```

### Train all canonical models in a single batch

```bash
python main.py --mode single_run --models all
```

### Optimize GP hyperparameters

Set `convergence_config.yaml: mode: optimization` and adjust `optimization_overrides.study.n_trials[gp]`,
then run:

```bash
python main.py --mode convergence
```

### Run convergence analysis

```bash
python main.py --mode convergence --config-path config/convergence/convergence_config.yaml
```

### Python API

```python
from src.utils import load_run_configs
from src.data import load_data_from_config
from src.pipeline import train_model

# Load configs
config_dir = Path("config")
configs = load_run_configs(config_dir, "gp", include_optimize=False)

# Load and preprocess data
data = load_data_from_config(config_dir.parent, configs["dataset_cfg"], 42, "float32")

# Train model
results = train_model(
    model_name="gp",
    data_bundle=data,
    dataset_cfg=configs["dataset_cfg"],
    model_cfg=configs["model_cfg"],
    training_cfg=configs["training_cfg"],
    output_resolver=output_resolver,
)
```

## Metrics Computed

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error  
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error

## Kernel Parametrization

### Exponential Kernel

- **theta**: Learnable distance-scaling parameter (ARD)
- **power_exponent**: Learnable exponent (constrained 1.5–2.0)
- Both initialized from config and optimized during training

### RBF & Matérn Kernels

Wrapped in ScaleKernel with learnable outputscale (tau2).

## Reproducibility

- Global seed controls NumPy, PyTorch, and Python random
- Deterministic training via `torch.backends.cudnn.deterministic=True`
- All configs saved to output directory for run reconstruction

## References

This section documents all papers, methods, and frameworks used throughout the project, organized by functionality.

### Dataset Handling

1. **Santner, T. J., Williams, B. J., & Notz, W. I. (2003).** *The Design and Analysis of Computer Experiments.* Springer.
   - Foundational reference for surrogate modeling and design of experiments.
   - Used for dataset design principles and computer experiment methodology.

2. **Kennedy, M. C., & O'Hagan, A. (2000).** Predicting the output from a complex computer code when fast approximations are available. *Biometrika*, 87(1), 1–13.
   - Kennedy-O'Hagan emulator framework for computer code approximation.
   - Used as a methodological basis for surrogate model training strategies.

3. **Data Preprocessing & Normalization**
   - Standard z-score normalization: $\tilde{x} = \frac{x - \mu}{\sigma}$
   - Train/validation/test splitting: 70/15/15 default ratios (configurable).
   - CSV loading with Pandas; PyTorch DataLoader for batching.

### Model Definition

#### Gaussian Process (GP)

4. **Rasmussen, C. E., & Williams, C. K. I. (2006).** *Gaussian Processes for Machine Learning.* MIT Press.
   - Comprehensive reference for GP theory, kernels, and inference.
   - Used for exact Gaussian Process regression (Kriging) with learnable kernels.

5. **Learnable Exponential Kernel**
   - Custom exponential kernel with Automatic Relevance Determination (ARD) scaling.
   - Parametrization:
     - `theta`: Distance-scaling parameter (learnable).
     - `power_exponent`: Kernel exponent ∈ [1.5, 2.0] (constrained, learnable).
     - `tau2`: Outputscale (variance).
   - Enables feature-specific lengthscale learning.

6. **RBF & Matérn Kernels** (optional alternatives)
   - Radial Basis Function (RBF) kernel: $k(x, x') = \exp\left(-\frac{\|x-x'\|^2}{2\ell^2}\right)$
   - Matérn kernel (ν-parameterized): interpolates between exponential (ν→0) and squared-exponential (ν→∞).
   - Wrapped in `ScaleKernel` for learnable outputscale.

#### Multi-Layer Perceptron (MLP)

7. **MLPRegressor (Block-Based Architecture)**
   - Three independent blocks (configurable layers/neurons per block).
   - Flexible architecture: each block can have different activation functions (ReLU, Tanh, etc.).
   - Dropout regularization for generalization.
   - Batch normalization optional per block.

#### Hybrid MLPGP Models

8. **Wilson, A. G., Hu, Z., Salakhudinov, R., & Xing, E. P. (2016).** Deep Kernel Learning. In *Proceedings of the 19th International Conference on Artificial Intelligence and Statistics (AISTATS)*.
   - Deep Kernel Learning (DKL) combines neural networks with GP kernels.
   - Implemented as `DKLFeatureExtractor` that learns a feature map φ(x) → ℝ^d.
   - Kernel in feature space: $k_{\text{DKL}}(x, x') = k_{\text{base}}(\phi(x), \phi(x'))$
   - Reduces model complexity while maintaining expressiveness.

9. **Paciorek, C. J., & Schervish, M. J. (2006).** Nonstationary Covariance Functions for Gaussian Process Regression. In *Advances in Neural Information Processing Systems (NIPS)*.
   - Gibbs non-stationary kernel for input-dependent variance and lengthscale.
   - Implemented as `NonstationaryExponentialKernel` with harmonic mean lengthscales.
   - Formula: $K_{\text{Gibbs}}(x, x') = \sigma(x)\sigma(x') \cdot k_{\text{stat}}\left(\frac{x}{\ell(x)}, \frac{x'}{\ell(x')}\right)$
   - Learned via MLP feature extraction (`NonstationaryFeatureExtractor`).

10. **Hybrid Architecture Variants**
    - **Non-Stationary MLPGP**: Uses exponential kernel with position-dependent variance σ(x) and lengthscale ℓ(x).
    - **DKL MLPGP**: Uses standard or learnable kernels with deep feature extraction; supports exponential, RBF, and Matérn.

### Training Logic

11. **Prechelt, L. (1998).** Early Stopping - But When? In *Neural Networks: Tricks of the Trade.*
    - Validation-based early stopping criterion with generalization gap monitoring.
    - Implemented as: stop if best_val_loss - val_loss < rel_tol × |best_val_loss| + abs_tol for `val_patience` consecutive steps.
    - Prevents overfitting without manual validation-set tuning.

12. **Convergence Criterion (Least-Squares Style)**
    - Train-loss convergence: $|L_{t-1} - L_t| \leq \text{abs\_tol} + \text{rel\_tol} \times |L_{t-1}|$
    - Adapted from `scipy.optimize.least_squares` stopping rule.
    - Used for detecting plateau in training loss; fires after `train_patience` consecutive steps.
    - Training stops when **either** validation patience **or** train convergence fires.

13. **Exact Marginal Log-Likelihood (EMLL) for GP**
    - GP training maximizes marginal likelihood: $\log p(y | X, \theta) = -\frac{1}{2}(y^T K^{-1} y + \log |K| + n \log 2\pi)$
    - Uses GPyTorch's `ExactMarginalLogLikelihood` objective.
    - Full-batch optimization with no minibatching (all data in one step).
    - Supports exact Cholesky-based solves with numerical jitter for stability.

14. **Cholesky Jitter for Numerical Stability**
    - GPyTorch's `psd_safe_cholesky` adds small jitter when Cholesky decomposition fails.
    - Configurable jitter: separate values for float32 and float64 precisions.
    - Default: 1e-6 (float32), 1e-8 (float64).

15. **Optimizers**
    - Supported: AdamW, Adam, SGD, LBFGS.
    - Warmup epochs: delay early stopping criteria for N initial steps (allows learning rate to stabilize).
    - Learning rate scheduling: optional StepLR scheduler (e.g., decay by γ every K steps).

### Metric Comparison

16. **Standard Regression Metrics**
    - **Mean Squared Error (MSE)**: $\text{MSE} = \frac{1}{n}\sum (y_i - \hat{y}_i)^2$
    - **Mean Absolute Error (MAE)**: $\text{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i|$
    - **Root Mean Squared Error (RMSE)**: $\text{RMSE} = \sqrt{\text{MSE}}$
    - **R² Score**: $R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ (fraction of variance explained).
    - **Mean Absolute Percentage Error (MAPE)**: $\text{MAPE} = \frac{1}{n}\sum\left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$

17. **Uncertainty Quantification (GP/MLPGP)**
    - Posterior variance from GP predictive distribution: $\text{Var}[\hat{y}(x)] = \text{diag}(K_* K^{-1} K_*^T)$
    - Enables calibration plots and uncertainty band visualization.
    - Used to diagnose model reliability across input space.

### Optimization (Optuna)

18. **Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019).** Optuna: A Next-generation Hyperparameter Optimization Framework. In *Proceedings of the 25th ACM SIGKDD Conference on Knowledge Discovery & Data Mining*.
    - Bayesian hyperparameter optimization framework with pruning support.
    - Samplers: TPE (Tree-structured Parzen Estimator, default), Grid, Random, Bayesian.
    - Pruners: Median, Hyperband, SuccessiveHalving for early termination of unpromising trials.
    - Configuration: study direction (minimize/maximize), n_trials, timeout, metric name.

19. **Hyperparameter Search Spaces**
    - **MLP**: Block layer counts, hidden dimensions, activations, dropout rates, learning rates.
    - **GP**: Kernel hyperparameters (theta, power_exponent, tau2), likelihood noise.
    - **MLPGP (DKL)**: Feature extractor hidden dims, output dims, learning rates.
    - Each trial trains a model from scratch and logs val_loss to the study.

20. **Sampling & Pruning Strategy**
    - Default: TPE sampler + Median pruner (stops trials performing below median at checkpoints).
    - Enables efficient search over high-dimensional hyperparameter spaces.
    - Objective function: typically validation loss (cross-entropy or regression loss).

### Core Libraries & Frameworks

21. **PyTorch**: https://pytorch.org/
    - Deep learning framework for neural network training.
    - Automatic differentiation (autograd) for gradient computation.
    - CUDA support for GPU acceleration.

22. **GPyTorch**: https://gpytorch.ai/
    - Gaussian Process library built on PyTorch.
    - Supports exact and approximate GP inference.
    - Flexible kernel and mean function definitions.
    - Scalable training via MLL objectives.

23. **Optuna**: https://optuna.org/ (optional dependency)
    - Hyperparameter optimization framework.
    - Supports multiple samplers and pruners for efficient search.

24. **Pandas & NumPy**
    - CSV I/O and data manipulation (Pandas).
    - Numerical computation and linear algebra (NumPy).

## Development Notes

- Code follows PEP 8 style conventions
- Docstrings use Google-style format
- Inline comments explain *why*, not *what*
- Configuration is centralized in YAML for repeatability
- Models are fully deterministic with proper seed management

## License

Proprietary - NAML Project