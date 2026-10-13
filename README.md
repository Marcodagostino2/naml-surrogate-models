# Neural-Gaussian Surrogates for Rocket Aerodynamics

Course project for Numerical Analysis for Machine Learning (Politecnico di Milano): training and evaluation of surrogate models for aerodynamic coefficient prediction on the HYDDB dataset (variable-airbrake rocket geometry, generated via an automated CFD pipeline). Four model variants:

1. **mlp** — Multi-Layer Perceptron with a single hidden block.
2. **gp** — Gaussian Process (Universal Kriging) with an anisotropic power-exponential kernel.
3. **mlpgp_ns** — MLP + non-stationary GP. The MLP predicts local σ(x) and ℓ(x) for a Gibbs kernel.
4. **mlpgp_dkl** — MLP + Deep Kernel Learning. The MLP maps inputs to a small feature space; a stationary kernel acts on the features.

The full methodology and validation campaign are documented in the project report ([report.pdf](report.pdf)).

## Results

Test-set metrics on HYDDB (single run, default configs, 70/15/15 split, normalized targets):

| model     |     R² |   RMSE |    MAE |  MAPE % | max error |
|-----------|-------:|-------:|-------:|--------:|----------:|
| mlp       | 0.9974 | 0.0123 | 0.0095 |  1.653  |    0.0436 |
| gp        | 0.9979 | 0.0112 | 0.0071 |  1.136  |    0.0456 |
| mlpgp_ns  | 0.9979 | 0.0110 | 0.0076 |  1.278  |    0.0347 |
| mlpgp_dkl | 0.9985 | 0.0093 | 0.0065 |  1.077  |    0.0247 |

## Installation

Requirements: Python 3.10+. CUDA is optional (GPU is auto-detected).

```bash
python -m venv venv
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
# Optional, only needed for hyperparameter optimization:
pip install optuna
```

## Quick start

The default configuration in `config/main_config.yaml` runs all four models on HYDDB in `single_run` mode.

```bash
# Single training run (uses main_config.yaml as-is)
python main.py --mode single_run

# Restrict the model list from the CLI (comma-separated, 'all' is accepted)
python main.py --mode single_run --models mlp,gp

# Convergence mode: repeated split/seed sweep
# (set convergence_config.yaml: mode: analysis)
python main.py --mode convergence

# Hyperparameter optimization
# (set convergence_config.yaml: mode: optimization)
python main.py --mode convergence
```

CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--mode` | from `main_config.yaml` | `single_run` or `convergence` |
| `--config-dir` | `config` | Path to the config directory |
| `--output-dir` | `outputs` | Base output directory |
| `--models` | from `main_config.yaml` | Comma-separated list or `all` |
| `--run-id` | timestamp | Override the output folder name |
| `--resume-run-id` | latest | For `single_run` in test mode: which checkpoint to load |
| `--config-path` | `config/convergence/convergence_config.yaml` | Path to the convergence config |

## Project layout

```
├── main.py                      # CLI dispatcher (single_run / convergence)
├── requirements.txt
├── report.pdf                   # project report
├── config/
│   ├── main_config.yaml         # mode, dataset, models_to_train, seed, precision
│   ├── dataset/dataset_config.yaml
│   ├── model/
│   │   ├── mlp_config.yaml
│   │   ├── gp_config.yaml
│   │   ├── mlpgp_ns_config.yaml
│   │   └── mlpgp_dkl_config.yaml
│   ├── training/
│   │   ├── training_common_config.yaml
│   │   └── training_<model>_config.yaml
│   ├── optimization/optimize_<model>_config.yaml
│   ├── convergence/convergence_config.yaml
│   └── visualization/visualization_config.yaml
├── src/
│   ├── pipeline.py              # train_model() and model/trainer wiring
│   ├── convergence_analysis.py  # convergence + HPO driver
│   ├── optimization_support.py  # Optuna objective and helpers
│   ├── entrypoints/             # package-level entry points
│   ├── data/dataset.py          # CSV loading, z-score normalization, splits
│   ├── models/                  # mlp.py, gp.py, mlpgp.py, gp_utils.py, init.py
│   ├── training/                # trainer.py, metrics.py
│   ├── inference/predictor.py
│   ├── visualization/           # plots/, suites/, datasets/
│   └── utils/                   # config I/O, path resolution, logging, seeding
├── data/                        # CSV datasets (only HYDDB.csv and the four UCI benchmarks are tracked)
└── outputs/                     # Run outputs (gitignored)
    └── <dataset>/<run_type>/<model>/runs/<run_id>/
```

## Configuration

### `config/main_config.yaml`

```yaml
mode: "single_run"          # "single_run" or "convergence"
global_seed: 21
precision: "float32"        # "float32" or "float64"
dataset: "HYDDB"            # key in config/dataset/dataset_config.yaml
models_to_train: ["all"]    # subset of [mlp, gp, mlpgp_ns, mlpgp_dkl] or ["all"]
run_mode: "training"        # "training", "test", or "training+test"
verbose: true
```

### `config/model/gp_config.yaml`

```yaml
kernel:
  type: "exponential"               # "exponential" | "rbf" | "matern"
  theta_init: 1.0e-2
  power_exponent_init: 1.8          # constrained to [1.5, 2.0]
  tau2_init: 1.0e-1
  likelihood_noise_init: 1.0e-2
  # bounds for the learnable parameters
  theta_floor: 1.0e-5
  theta_ceiling: 1.0e3
  tau2_floor: 1.0e-5
  tau2_ceiling: 1.0e3
trend_degree: 1                     # polynomial trend degree
```

### `config/model/mlp_config.yaml`

```yaml
architecture:
  layers: [128]                     # one hidden block of width 128
  activation: "tanh"                # relu | leaky_relu | selu | tanh | mish
dropout: 0.0
batch_norm: false
output_activation: "identity"
```

### `config/training/training_common_config.yaml` (defaults shared by all trainers)

```yaml
scheduler_type: cosine              # "steplr" | "cosine" | "none"
scheduler_T_max: null               # null → use n_epochs
scheduler_eta_min: 1.0e-6
scheduler_warmup_steps: 200
progress_log_interval: 30
rel_tol: 1.0e-4
abs_tol: 1.0e-6
train_patience: 200
val_patience: 400
warmup_epochs: 400
train_metrics_log_interval: 10
validation_check_interval: 10
```

Per-model overrides live in `training_mlp_config.yaml`, `training_gp_config.yaml`, etc.

## Data splits and normalization

`src/data/dataset.py` performs:

- 70/15/15 train/validation/test split (overridable in `dataset_config.yaml`).
- Per-feature z-score normalization computed on the training split and applied to all splits.
- DataLoader wrapping for MLP training; GP/MLPGP use full-batch tensors.

## Metrics

Computed on train/val/test in `src/training/metrics.py`:

- MSE, MAE, RMSE
- R² (coefficient of determination)
- MAPE (skipped when targets contain zeros)

For GP-family models, the posterior standard deviation is also returned at inference time and used to produce ±2σ uncertainty bands in the slice plots and a reliability/coverage curve on the test set.

## References

- C. E. Rasmussen and C. K. I. Williams. *Gaussian Processes for Machine Learning*. MIT Press, 2006.
- A. I. J. Forrester, A. Sóbester, and A. J. Keane. *Engineering Design via Surrogate Modelling: A Practical Guide*. Wiley, 2008. (power-exponential kernel)
- A. G. Wilson, Z. Hu, R. Salakhutdinov, E. P. Xing. *Deep Kernel Learning*. AISTATS, 2016.
- C. J. Paciorek and M. J. Schervish. *Nonstationary Covariance Functions for Gaussian Process Regression*. NeurIPS, 2004.
- L. Prechelt. *Early Stopping — But When?* In *Neural Networks: Tricks of the Trade*, 1998.
- M. D. McKay, R. J. Beckman, W. J. Conover. *A comparison of three methods for selecting values of input variables in the analysis of output from a computer code*. Technometrics, 1979. (Latin Hypercube Sampling)

Software:

- PyTorch (https://pytorch.org/)
- GPyTorch (https://gpytorch.ai/)
- Optuna (https://optuna.org/) — optional, only for the HPO sub-mode.

## Dataset attribution

`data/HYDDB.csv` is an aerodynamic coefficient database of the Hydra sounding
rocket (variable-airbrake geometry), generated with an automated CFD pipeline
and released with the consent of [Skyward Experimental Rocketry](https://skywarder.eu).
The remaining CSV files are standard public benchmark datasets (UCI).

## Authors

Project for the *Numerical Analysis for Machine Learning* course,
Politecnico di Milano (a.y. 2025/2026):

- Riccardo Bellotti
- Marco D'Agostino

Supervised by Prof. Edie Miglio.

## License

MIT — see [LICENSE](LICENSE).
