"""User-friendly split convergence analysis for MLP, GP, and MLPGP."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data import DataBundle, load_and_preprocess_data
from src.inference import Predictor
from src.models import GPRegressionModel, MLPGPModel, MLPRegressor
from src.training import GPTrainer, MLPGPTrainer, MLPTrainer, compute_all_metrics
from src.utils import OutputPathResolver, RunType, load_run_configs, load_yaml, resolve_device, set_global_seed, setup_torch_dtype

PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_analysis_paths(dataset_name: str, model_name: str) -> OutputPathResolver:
    """Resolve convergence-analysis output paths for one dataset/model pair."""
    return OutputPathResolver(
        dataset_name=dataset_name,
        run_type=RunType.CONVERGENCE_ANALYSIS,
        model_name=model_name,
        output_base_dir=PROJECT_ROOT / "outputs",
    )


def _normalize_split_value(value: float) -> float:
    """Normalize split values from percentage to fraction when required."""
    return value / 100.0 if value > 1.0 else value


def _build_split_grid(raw_splits: List[float]) -> List[float]:
    """Build a validated, sorted train-split grid."""
    if not raw_splits:
        raise ValueError("split_percentages cannot be empty.")

    split_grid = []
    for raw in raw_splits:
        train_ratio = _normalize_split_value(float(raw))
        if train_ratio <= 0.0 or train_ratio >= 1.0:
            raise ValueError(f"Invalid train split {raw}. Must be in (0,1) or (0,100).")
        split_grid.append(round(train_ratio, 6))

    return sorted(set(split_grid))


def _load_configs(config_path: Path) -> Dict[str, Any]:
    """Load analysis, dataset, model, and training configs from current APIs."""
    analysis_cfg = load_yaml(config_path)
    config_dir = PROJECT_ROOT / "config"

    loaded_mlp = load_run_configs(config_dir, "mlp", include_optimize=False)
    loaded_gp = load_run_configs(config_dir, "gp", include_optimize=False)
    loaded_mlpgp = load_run_configs(config_dir, "mlpgp", include_optimize=False)

    return {
        "analysis": analysis_cfg,
        "main": loaded_mlp["main_cfg"],
        "dataset": loaded_mlp["dataset_cfg"],
        "dataset_name": loaded_mlp["dataset_name"],
        "model_map": {
            "mlp": loaded_mlp["model_cfg"],
            "gp": loaded_gp["model_cfg"],
            "mlpgp": loaded_mlpgp["model_cfg"],
        },
        "training_map": {
            "mlp": loaded_mlp["training_cfg"],
            "gp": loaded_gp["training_cfg"],
            "mlpgp": loaded_mlpgp["training_cfg"],
        },
    }


def _prepare_data_bundle(dataset_cfg: Dict[str, Any], train_ratio: float, seed: int) -> DataBundle:
    """Load and preprocess the dataset for one train split."""
    val_ratio = (1.0 - train_ratio) / 2.0
    test_ratio = val_ratio

    return load_and_preprocess_data(
        csv_path=PROJECT_ROOT / dataset_cfg["csv_path"],
        input_columns=dataset_cfg["input_columns"],
        target_columns=dataset_cfg["target_columns"],
        dropna=bool(dataset_cfg.get("dropna", True)),
        normalize_inputs=bool(dataset_cfg.get("normalize_inputs", True)),
        normalize_targets=bool(dataset_cfg.get("normalize_targets", True)),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        random_seed=int(seed),
    )


def _evaluate_with_predictor(predictor: Predictor, data_bundle: DataBundle) -> Dict[str, float]:
    """Evaluate test metrics using a trained predictor."""
    y_test_pred = predictor.predict(data_bundle.X_test)
    y_test_true = data_bundle.y_test * data_bundle.target_scaler_std + data_bundle.target_scaler_mean
    metrics = compute_all_metrics(y_test_true, y_test_pred)
    return {key: float(value) for key, value in metrics.items()}


def _run_mlp(
    data_bundle: DataBundle,
    dataset_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one MLP training/evaluation pass for a split."""
    device = resolve_device(training_cfg.get("device", "auto"))
    model = MLPRegressor(
        input_dim=data_bundle.n_inputs,
        output_dim=data_bundle.n_outputs,
        model_cfg=model_cfg,
    )
    train_loader = data_bundle.get_train_loader(batch_size=int(dataset_cfg.get("batch_size", 128)))
    val_loader = data_bundle.get_val_loader(batch_size=int(dataset_cfg.get("batch_size", 128)))
    trainer = MLPTrainer(model=model, train_loader=train_loader, val_loader=val_loader, training_cfg=training_cfg, device=device)
    history = trainer.train()
    predictor = Predictor(model=model, target_scaler_mean=data_bundle.target_scaler_mean, target_scaler_std=data_bundle.target_scaler_std, device=device)
    metrics = _evaluate_with_predictor(predictor, data_bundle)
    return {
        "metrics": metrics,
        "elapsed_time": float(history["elapsed_time"]),
        "history_values": [float(v) for v in history["train_losses"]],
        "history_label": "train_loss",
    }


def _run_gp(data_bundle: DataBundle, model_cfg: Dict[str, Any], training_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run one GP training/evaluation pass for a split."""
    device = resolve_device(training_cfg.get("device", "auto"))
    X_train_t = torch.from_numpy(data_bundle.X_train).float()
    y_train_t = torch.from_numpy(data_bundle.y_train).float()
    if y_train_t.ndim == 2 and y_train_t.shape[1] == 1:
        y_train_t = y_train_t.squeeze(-1)

    model = GPRegressionModel(X_train=X_train_t, y_train=y_train_t, model_cfg=model_cfg, likelihood=None)
    trainer = GPTrainer(
        model=model,
        X_train=X_train_t,
        y_train=y_train_t,
        X_val=torch.from_numpy(data_bundle.X_val),
        y_val=torch.from_numpy(data_bundle.y_val),
        training_cfg=training_cfg,
        device=device,
    )
    history = trainer.train()
    predictor = Predictor(model=model, target_scaler_mean=data_bundle.target_scaler_mean, target_scaler_std=data_bundle.target_scaler_std, device=device)
    metrics = _evaluate_with_predictor(predictor, data_bundle)
    return {
        "metrics": metrics,
        "elapsed_time": float(history["elapsed_time"]),
        "history_values": [float(v) for v in history["mll_history"]],
        "history_label": "mll",
    }


def _run_mlpgp(data_bundle: DataBundle, model_cfg: Dict[str, Any], training_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run one MLPGP training/evaluation pass for a split."""
    device = resolve_device(training_cfg.get("device", "auto"))
    X_train_t = torch.from_numpy(data_bundle.X_train).float()
    y_train_t = torch.from_numpy(data_bundle.y_train).float()
    if y_train_t.ndim == 2 and y_train_t.shape[1] == 1:
        y_train_t = y_train_t.squeeze(-1)

    model = MLPGPModel(X_train=X_train_t, y_train=y_train_t, model_cfg=model_cfg, likelihood=None)
    trainer = MLPGPTrainer(
        model=model,
        X_train=X_train_t,
        y_train=y_train_t,
        X_val=torch.from_numpy(data_bundle.X_val),
        y_val=torch.from_numpy(data_bundle.y_val),
        training_cfg=training_cfg,
        device=device,
    )
    history = trainer.train()
    predictor = Predictor(model=model, target_scaler_mean=data_bundle.target_scaler_mean, target_scaler_std=data_bundle.target_scaler_std, device=device)
    metrics = _evaluate_with_predictor(predictor, data_bundle)
    return {
        "metrics": metrics,
        "elapsed_time": float(history["elapsed_time"]),
        "history_values": [float(v) for v in history["mll_history"]],
        "history_label": "mll",
    }


def _plot_metric_curve(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Plot one metric against train-split values for all models."""
    plt.figure(figsize=(9, 6))
    for model_name in ["mlp", "gp", "mlpgp"]:
        subset = df[df["model"] == model_name].sort_values("train_split")
        plt.plot(
            subset["train_split"],
            subset[metric],
            marker="o",
            linewidth=2,
            label=model_name.upper(),
        )
    plt.xlabel("Train split")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs train split")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_training_histories(history_records: List[Dict[str, Any]], output_path: Path) -> None:
    """Plot training-history curves grouped by model and split."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False)
    model_order = ["mlp", "gp", "mlpgp"]

    for ax, model_name in zip(axes, model_order):
        model_records = [record for record in history_records if record["model"] == model_name]
        for record in sorted(model_records, key=lambda item: item["train_split"]):
            values = np.asarray(record["history_values"], dtype=float)
            if values.size == 0:
                continue
            x = np.arange(1, values.size + 1)
            ax.plot(x, values, linewidth=1.8, label=f"split={record['train_split']:.2f}")

        ylabel = model_records[0]["history_label"] if model_records else "history"
        ax.set_title(model_name.upper())
        ax.set_xlabel("epoch/step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if model_records:
            ax.legend(fontsize=8)

    fig.suptitle("Training histories across splits")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_analysis(config_path: Path) -> None:
    """Run convergence analysis over all configured train splits."""
    config_bundle = _load_configs(config_path)
    analysis_cfg = config_bundle["analysis"]
    main_cfg = config_bundle["main"]
    dataset_cfg = config_bundle["dataset"]
    dataset_name = config_bundle["dataset_name"]
    model_map = config_bundle["model_map"]
    training_map = config_bundle["training_map"]

    split_grid = _build_split_grid(list(analysis_cfg.get("split_percentages", [0.7])))
    seed = int(analysis_cfg.get("global_seed", 31))
    setup_torch_dtype(str(main_cfg.get("precision", "float32")))

    model_sequence = ["mlp", "gp", "mlpgp"]
    model_paths = {name: _resolve_analysis_paths(dataset_name, name) for name in model_sequence}
    for resolver in model_paths.values():
        resolver.root.mkdir(parents=True, exist_ok=True)
        resolver.plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []

    for split in split_grid:
        set_global_seed(seed)
        data_bundle = _prepare_data_bundle(dataset_cfg=dataset_cfg, train_ratio=split, seed=seed)

        for model_name in model_sequence:
            set_global_seed(seed)
            model_cfg = deepcopy(model_map[model_name])
            training_cfg = deepcopy(training_map[model_name])

            if model_name == "mlp":
                result = _run_mlp(data_bundle, dataset_cfg, model_cfg, training_cfg)
            elif model_name == "gp":
                result = _run_gp(data_bundle, model_cfg, training_cfg)
            else:
                result = _run_mlpgp(data_bundle, model_cfg, training_cfg)

            metrics_rows.append(
                {
                    "model": model_name,
                    "train_split": split,
                    "val_split": (1.0 - split) / 2.0,
                    "test_split": (1.0 - split) / 2.0,
                    "n_train": int(data_bundle.X_train.shape[0]),
                    "n_val": int(data_bundle.X_val.shape[0]),
                    "n_test": int(data_bundle.X_test.shape[0]),
                    "r2": result["metrics"]["r2"],
                    "rmse": result["metrics"]["rmse"],
                    "mae": result["metrics"]["mae"],
                    "mape": result["metrics"]["mape"],
                    "elapsed_time": result["elapsed_time"],
                }
            )

            histories.append(
                {
                    "model": model_name,
                    "train_split": split,
                    "history_label": result["history_label"],
                    "history_values": result["history_values"],
                }
            )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["model", "train_split"]).reset_index(drop=True)
    for resolver in model_paths.values():
        metrics_df.to_csv(resolver.summary_csv, index=False)

    # Log convergence analysis summary
    from src.utils import setup_logger
    logger = setup_logger(__name__)
    logger.info("=" * 70)
    logger.info("CONVERGENCE ANALYSIS RESULTS")
    logger.info("=" * 70)
    for model_name in model_sequence:
        model_data = metrics_df[metrics_df["model"] == model_name].sort_values("train_split")
        logger.info(f"\n{model_name.upper()} Results:")
        logger.info("-" * 70)
        for _, row in model_data.iterrows():
            logger.info(
                f"  Split={row['train_split']:.2f} (n_train={int(row['n_train'])}): "
                f"R²={row['r2']:.6f}, RMSE={row['rmse']:.6f}, MAE={row['mae']:.6f}, MAPE={row['mape']:.2f}%"
            )
    logger.info("=" * 70)

    for resolver in model_paths.values():
        _plot_metric_curve(metrics_df, metric="r2", output_path=resolver.plots_dir / "r2_vs_split.png")
        _plot_metric_curve(metrics_df, metric="rmse", output_path=resolver.plots_dir / "rmse_vs_split.png")
        _plot_metric_curve(metrics_df, metric="mae", output_path=resolver.plots_dir / "mae_vs_split.png")
        _plot_metric_curve(metrics_df, metric="mape", output_path=resolver.plots_dir / "mape_vs_split.png")
        _plot_training_histories(histories, output_path=resolver.plots_dir / "training_history.png")

    print("Done. Outputs written under outputs/<dataset>/convergence_analysis/<model>.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for convergence analysis."""
    parser = argparse.ArgumentParser(description="Split convergence analysis for MLP, GP, MLPGP.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/convergence/convergence_config.yaml",
        help="Path to analysis config YAML.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    run_analysis(config_path)
