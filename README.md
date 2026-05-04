# NAML Aerodynamic Surrogate Models

A unified, production-ready Python framework for training, optimizing, and evaluating surrogate models for aerodynamic coefficient prediction using:

1. **MLP** - Multi-Layer Perceptron (block-based, dynamic architecture)
2. **GP** - Gaussian Process (Kriging) with learnable exponential kernel
3. **MLPGP** - Hybrid architecture combining MLP feature extraction with non-stationary Gaussian Processes (two variants: DKL and Gibbs kernel)

## Features

- **Unified CLI**: Single entry point (`main.py`) supporting three workflow modes: `single_run`, `optimization`, `convergence`
- **Modular Architecture**: Clean separation across data, models, training, inference, and visualization modules
- **Dynamic YAML Configuration**: Hierarchical, composable config system for easy experimentation
- **Learnable Kernels**: Exponential kernel with ARD-scaled theta and power exponent parameters (both constrained)
- **Hyperparameter Optimization**: Optuna integration for automated tuning (optional dependency)
- **Uncertainty Quantification**: GP/MLPGP provide predictive uncertainty via variance estimates
- **Comprehensive Diagnostics**: Parity plots, residuals, training curves, uncertainty bands, convergence analysis
- **Reproducibility**: Config snapshots, deterministic training, global seed management
- **Output Management**: Timestamped runs with checkpoints, config copies, plots, and aggregated metrics

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

### Single Training Run

```bash
# Configure model and dataset in config/main_config.yaml
python main.py --mode single_run
```

### Hyperparameter Optimization

```bash
python main.py --mode optimization --model mlp --n-trials 50
```

### Convergence Analysis

```bash
python main.py --mode convergence
```

### Entry Point Help

```bash
python main.py --help
python main.py --mode single_run --help
python main.py --mode optimization --help
python main.py --mode convergence --help
```

## Project Structure

```
├── README.md
├── requirements.txt
├── main.py                          # Unified mode dispatcher
│
├── src/
│   ├── convergence_analysis.py      # Convergence analysis implementation
│   ├── entrypoints/
│   │   ├── convergence.py           # Package-level convergence entrypoint
│   │   └── optimize.py              # Package-level optimization entrypoint
│   ├── pipeline.py                  # Shared train_model() and model/trainer building
│   ├── optimization_support.py      # Objective functions and HPO utilities
│   ├── data/
│   │   └── dataset.py              # Data loading, preprocessing, splitting
│   ├── models/
│   │   ├── mlp.py                  # Block-based MLP
│   │   ├── gp.py                   # Exact GP with exponential kernel
│   │   ├── mlpgp.py                # Hybrid MLPGP (DKL and Gibbs variants)
│   │   └── gp_utils.py             # Kernel factories and shared helpers
│   ├── training/
│   │   ├── trainer.py              # MLPTrainer, GPTrainer, MLPGPTrainer
│   │   └── metrics.py              # MSE, MAE, R², RMSE, MAPE
│   ├── inference/
│   │   └── predictor.py            # Unified prediction interface
│   ├── visualization/
│   │   ├── runner.py               # Main visualization dispatcher
│   │   ├── diagnostics.py          # Diagnostic plots
│   │   ├── config.py               # Visualization config schema
│   │   ├── utils.py                # Plotting utilities
│   │   └── modules/                # Parity, surface, uncertainty, table plots
│   └── utils/
│       ├── config.py               # YAML I/O, device resolution
│       ├── paths.py                # Path resolution and output structure
│       ├── entrypoints.py          # Shared config bootstrap
│       └── __init__.py             # setup_logger, set_global_seed, etc.
│
├── config/                          # All configs are YAML-based
│   ├── main_config.yaml            # Top-level: mode, model, seed, precision
│   ├── dataset/
│   │   └── dataset_config.yaml     # All datasets with metadata
│   ├── model/
│   │   ├── mlp_config.yaml
│   │   ├── gp_config.yaml
│   │   └── mlpgp_config.yaml
│   ├── training/
│   │   ├── training_common_config.yaml
│   │   ├── training_mlp_config.yaml
│   │   ├── training_gp_config.yaml
│   │   └── training_mlpgp_config.yaml
│   ├── optimization/
│   │   ├── optimize_mlp_config.yaml
│   │   ├── optimize_gp_config.yaml
│   │   └── optimize_mlpgp_config.yaml
│   ├── convergence/
│   │   └── convergence_config.yaml
│   └── visualization/
│       └── visualization_config.yaml
│
├── data/                            # CSV datasets
│   ├── HYDDB_v2_output.csv
│   ├── AEDB_v1_output.csv
│   ├── airfoil_self_noise.csv
│   └── ...
│
├── outputs/                         # Run outputs (timestamped)
│   ├── <dataset>/
│   │   ├── single_run/
│   │   ├── optimization/
│   │   └── convergence_analysis/
│
├── report/
│   └── codebase_map/               # Bootstrap docs for future sessions
│       ├── README.md
│       ├── overview.md
│       ├── entrypoints_and_flow.md
│       ├── modules_and_responsibilities.md
│       ├── configs_outputs_and_conventions.md
│       └── src/README.md
│
└── scripts/
    └── import_smoke_test.py        # Verify modules load without optuna
```

## Configuration Guide

### Main Orchestrator (`config/main_config.yaml`)

```yaml
mode: "single_run"          # "single_run", "optimization", or "convergence"
dataset: "HYDDB_v2"         # Dataset key from dataset_config.yaml
model_to_train: "mlp"       # "mlp", "gp", "mlpgp", or "all"
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

### Train MLP with Custom Config

```bash
python main.py --mode single_run --model mlp --output-dir outputs
```

### Optimize GP Hyperparameters

```bash
python main.py --mode optimization --model gp --n-trials 100 --timeout 3600
```

### Run Convergence Analysis

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

## Documentation

For detailed architecture and workflow info, see:
- [report/codebase_map/README.md](report/codebase_map/README.md) - Bootstrap guide for future sessions
- [report/codebase_map/overview.md](report/codebase_map/overview.md) - High-level overview
- [report/codebase_map/entrypoints_and_flow.md](report/codebase_map/entrypoints_and_flow.md) - Workflow dispatcher
- [report/codebase_map/modules_and_responsibilities.md](report/codebase_map/modules_and_responsibilities.md) - Module purposes

## References

- **GPyTorch**: https://gpytorch.ai/
- **PyTorch**: https://pytorch.org/
- **Optuna**: https://optuna.org/
- Gaussian Processes: Rasmussen & Williams (2006)

## Development Notes

- Code follows PEP 8 style conventions
- Docstrings use Google-style format
- Inline comments explain *why*, not *what*
- Configuration is centralized in YAML for repeatability
- Models are fully deterministic with proper seed management

## License

Proprietary - NAML Project

## Refactoring Status

### Phase 1: Structural Refactoring ✅ COMPLETE

- ✅ Entry-point consolidation (main.py mode dispatcher)
- ✅ Trainer layer fixes (Cholesky jitter, scheduler semantics)
- ✅ Lazy optuna imports (--help works without optuna)
- ✅ Model layer consolidation (kernel configs, dead code removal)
- ✅ Data layer refactoring (magic numbers → module constants)
- ✅ Comprehensive documentation (codebase_map)

### Phase 2: Code Hygiene ✅ COMPLETE

- ✅ Removed all debug code (commented filler removed)
- ✅ Replaced all `print()` statements with `logger` calls:
  - `src/optimization_support.py` - Trial progress logging
  - `src/visualization/modules/table.py` - Metrics summary
- ✅ No remaining TODO/FIXME/HACK comments in src/
- ✅ All imports verified functional

**Code is production-ready.**

## Current Status

The entrypoint consolidation is complete. `main.py` dispatches the three supported modes, the package-level workflow entrypoints live under `src/entrypoints/`, and convergence analysis now runs from `src/convergence_analysis.py`.

Optimization remains an optional path because it depends on `optuna`.
