# NAML Aerodynamic Surrogate Models

A unified, production-ready Python framework for training and comparing three different surrogate models for aerodynamic coefficient prediction:

1. **MLP** - Multi-Layer Perceptron
2. **GP (Kriging)** - Gaussian Process / Kriging  
3. **Hybrid MLP+Kriging** - Combined feature extraction + non-stationary Kriging (under development)

## Features

- **Modular Architecture**: Clean separation of concerns across data, models, training, inference, and plotting modules.
- **Dynamic Configuration**: YAML-based, hierarchical configuration system for easy experimentation.
- **MLP Model**: Block-based architecture with flexible layer definitions, batch normalization, dropout, and multiple activation functions.
- **GP/Kriging Model**: GPyTorch-based exact GP with learnable hyperparameters (lengthscale, power exponent).
- **Uncertainty Quantification**: GP provides predictive uncertainty (±2σ confidence bands).
- **Comprehensive Diagnostics**: Parity plots, residual analysis, training curves, uncertainty visualization.
- **Reproducibility**: Config snapshots, random seed management, deterministic training.
- **Output Management**: Timestamped runs with checkpoints, config copies, plots, and summary metrics.

## Installation

### Prerequisites
- Python 3.10 or higher
- CUDA 12.1+ (optional, for GPU acceleration)

### Setup

1. Clone and navigate to project:
   ```bash
   cd naml-surrogate-models
   ```

2. Create Python environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Quick Start

### Train MLP Model

```bash
# Configure to use MLP
# config/main_config.yaml: model_to_train: "mlp"

python main.py
```

### Train Gaussian Process Model

```bash
# Configure to use GP
# config/main_config.yaml: model_to_train: "gp"

python main.py
```

### Run with Custom Config

```bash
python main.py --main-config config/main_config.yaml --output-dir outputs
```

## Project Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                          # Main entry point
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py              # Data loading, preprocessing, splitting
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py                  # MLP regression model
│   │   ├── gp.py                   # GPyTorch-based Kriging model
│   │   └── mlpgp.py                # Hybrid MLP+Kriging (placeholder)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # MLPTrainer, GPTrainer classes
│   │   └── metrics.py              # Evaluation metrics
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py            # Unified prediction interface
│   ├── plot/
│   │   ├── __init__.py
│   │   └── diagnostics.py          # Plotting and visualization
│   └── utils/
│       ├── __init__.py
│       └── config.py               # YAML I/O, logging, device management
├── config/
│   ├── main_config.yaml
│   ├── dataset/
│   │   └── dataset_config.yaml
│   ├── model/
│   │   ├── mlp_config.yaml
│   │   ├── gp_config.yaml
│   │   └── mlpgp_config.yaml
│   ├── training/
│   │   ├── training_mlp_config.yaml
│   │   ├── training_gp_config.yaml
│   │   └── training_mlpgp_config.yaml
│   └── visualization/
│       └── visualization_config.yaml
└── outputs/
    ├── mlp/
    ├── gp/
    └── mlpgp/
```

## Configuration

### Main Orchestrator (`config/main_config.yaml`)

Controls which model to train and execution mode:
- `global_seed`: Random seed for reproducibility
- `model_to_train`: "mlp", "gp", "mlpgp", or "all"
- `run_mode`: "training", "test", or "training+test"

### MLP Architecture (`config/model/mlp_config.yaml`)

Block-based layer definition:
```yaml
architecture:
  - ([128, 128, 64], "relu")
  - ([32], "relu")
dropout: 0.0
batch_norm: false
output_activation: "identity"
```

### GP Hyperparameters (`config/model/gp_config.yaml`)

Kernel configuration and optimization:
```yaml
kernel_type: "matern"
initial_lengthscale: 1.0
likelihood_noise: 1e-4
```

### Training Settings

Edit training configs to control optimization:
- `optimizer`: "adam", "sgd", "adamw" for MLP, GP, and MLPGP
- `learning_rate`: Learning rate
- `n_epochs` (MLP) / `n_training_steps` (GP)
- `train_residual_check_interval`, `train_residual_tol`
- `validation_check_interval`, `validation_residual_tol`

## Outputs

Each training run creates `outputs/<model>/ run_<YYYYMMDD_HHMMSS>/` with:
- **checkpoint.pt** - Trained model weights
- **config/** - Config files used (for reproducibility)
- **plot/** - Diagnostic plots (parity, residuals, training history, uncertainty)

At model level: `outputs/<model>/summary.csv` aggregates metrics across all runs.

## Python API Usage

```python
from src.data import load_and_preprocess_data
from src.models import MLPRegressor
from src.training import MLPTrainer
from src.inference import Predictor

# Load data
data_bundle = load_and_preprocess_data(
    csv_path="data/HYDDB_v2_output.csv",
    input_columns=["col1", "col2", ...],
    target_columns=["target"],
)

# Create and train model
model = MLPRegressor(data_bundle.n_inputs, data_bundle.n_outputs, model_cfg)
trainer = MLPTrainer(model, train_loader, val_loader, training_cfg)
trainer.train()

# Predict
predictor = Predictor(model, target_scaler_mean, target_scaler_std)
predictions = predictor.predict(X_test)
```

## Metrics

Supported evaluation metrics:
- MSE, MAE, RMSE, R², MAPE

## References

- **GPyTorch**: https://gpytorch.ai/
- **PyTorch**: https://pytorch.org/
- **Gaussian Processes**: Rasmussen & Williams (2006)

## License

Proprietary - NAML Project

## Dependencies

The path setup in [mainCoefficientModel.m](mainCoefficientModel.m) currently references external folders:

- ../common
- ../../dissileMatcom

Make sure your local environment includes these paths, or adapt the setupPaths section to your workspace layout.

## Suggested workflow for experiments

- Start with multiFidelity mode and fixed split ratio.
- Tune optimization bounds and polynomial trend degree via config files.
- Run convergence sweeps by increasing CFD training budget.
- Compare against CFD-only Kriging baseline with same test set.
- Collect timing and error metrics in a reproducible table.

## License

No license file is currently provided in this repository.
