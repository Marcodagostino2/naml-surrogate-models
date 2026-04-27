from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data import build_data_bundle
from src.models import build_regressor
from src.training import evaluate_model, train_model
from src.training.trainer import _resolve_device
from src.utils import load_yaml, save_json, set_global_seed


@dataclass
class ConvergenceRun:
    train_ratio: float
    val_ratio: float
    test_ratio: float
    best_epoch: int
    best_val_loss: float
    train_loss_final: float
    train_loss_min: float
    val_loss_final: float
    val_loss_min: float
    train_time_seconds: float
    val_rmse: float
    val_mae: float
    val_r2: float
    test_rmse: float
    test_mae: float
    test_r2: float


def _apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _format_ratio(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _build_split_config(base_cfg: Dict[str, Any], train_ratio: float) -> Dict[str, Any]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    configured_val_ratio = float(base_cfg.get("val_ratio", 0.3))
    configured_test_ratio = float(base_cfg.get("test_ratio", 0.3))
    remaining_ratio = 1.0 - train_ratio

    if configured_val_ratio + configured_test_ratio <= 0.0:
        val_share = 0.5
    else:
        val_share = configured_val_ratio / (configured_val_ratio + configured_test_ratio)

    split_cfg = deepcopy(base_cfg)
    split_cfg["train_ratio"] = float(train_ratio)
    split_cfg["val_ratio"] = float(remaining_ratio * val_share)
    split_cfg["test_ratio"] = float(remaining_ratio * (1.0 - val_share))
    return split_cfg


def _resolve_train_ratios(train_ratios: Sequence[float] | None) -> List[float]:
    if train_ratios is None or len(train_ratios) == 0:
        return [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
    unique_ratios = []
    seen = set()
    for ratio in train_ratios:
        ratio_value = float(ratio)
        if ratio_value not in seen:
            unique_ratios.append(ratio_value)
            seen.add(ratio_value)
    return sorted(unique_ratios)


def _run_single_experiment(
    project_root: Path,
    dataset_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    visualization_cfg: Dict[str, Any],
    train_ratio: float,
    run_output_dir: Path,
) -> tuple[ConvergenceRun, Dict[str, List[float]]]:
    run_dataset_cfg = _build_split_config(dataset_cfg, train_ratio)
    run_training_cfg = deepcopy(training_cfg)
    run_training_cfg["generate_plots"] = False

    data_bundle = build_data_bundle(run_dataset_cfg, project_root)
    model = build_regressor(
        input_dim=data_bundle.input_dim,
        output_dim=data_bundle.output_dim,
        model_cfg=model_cfg,
    )

    start_time = datetime.now()
    training_result, best_state = train_model(model, data_bundle, run_training_cfg)
    elapsed_seconds = (datetime.now() - start_time).total_seconds()

    model.load_state_dict(best_state)
    device = _resolve_device(str(run_training_cfg.get("device", "auto")))
    model.to(device)

    val_metrics = evaluate_model(
        model,
        data_bundle.val_loader,
        device,
        target_scaler=data_bundle.target_scaler,
    )
    test_metrics = evaluate_model(
        model,
        data_bundle.test_loader,
        device,
        target_scaler=data_bundle.target_scaler,
    )

    train_losses = training_result.history["train_loss"]
    val_losses = training_result.history["val_loss"]
    run_summary = ConvergenceRun(
        train_ratio=float(train_ratio),
        val_ratio=float(run_dataset_cfg["val_ratio"]),
        test_ratio=float(run_dataset_cfg["test_ratio"]),
        best_epoch=int(training_result.best_epoch),
        best_val_loss=float(training_result.best_val_loss),
        train_loss_final=float(train_losses[-1]),
        train_loss_min=float(min(train_losses)),
        val_loss_final=float(val_losses[-1]),
        val_loss_min=float(min(val_losses)),
        train_time_seconds=float(elapsed_seconds),
        val_rmse=float(val_metrics["rmse"]),
        val_mae=float(val_metrics["mae"]),
        val_r2=float(val_metrics["r2"]),
        test_rmse=float(test_metrics["rmse"]),
        test_mae=float(test_metrics["mae"]),
        test_r2=float(test_metrics["r2"]),
    )

    run_output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        run_output_dir / "summary.json",
        {
            **asdict(run_summary),
            "history": training_result.history,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
    )
    return run_summary, training_result.history


def _plot_metric_curves(results: List[ConvergenceRun], output_dir: Path) -> Path:
    train_ratios = np.array([result.train_ratio for result in results], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    palette = {
        "validation": "tab:blue",
        "test": "tab:orange",
        "train": "tab:green",
    }

    def _curve(ax, y_values: Sequence[float], label: str, color: str, marker: str = "o") -> None:
        ax.plot(train_ratios, y_values, marker=marker, linewidth=2.2, color=color, label=label)
        ax.scatter(train_ratios, y_values, s=50, color=color, edgecolor="white", linewidth=0.7, zorder=3)

    _curve(axes[0, 0], [result.val_rmse for result in results], "Validation RMSE", palette["validation"])
    _curve(axes[0, 0], [result.test_rmse for result in results], "Test RMSE", palette["test"])
    axes[0, 0].set_title("RMSE vs train ratio")
    axes[0, 0].set_ylabel("RMSE")
    axes[0, 0].legend(loc="best")

    _curve(axes[0, 1], [result.val_mae for result in results], "Validation MAE", palette["validation"])
    _curve(axes[0, 1], [result.test_mae for result in results], "Test MAE", palette["test"])
    axes[0, 1].set_title("MAE vs train ratio")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].legend(loc="best")

    _curve(axes[1, 0], [result.val_r2 for result in results], "Validation R2", palette["validation"])
    _curve(axes[1, 0], [result.test_r2 for result in results], "Test R2", palette["test"])
    axes[1, 0].set_title("R2 vs train ratio")
    axes[1, 0].set_xlabel("Train ratio")
    axes[1, 0].set_ylabel("R2")
    axes[1, 0].legend(loc="best")

    _curve(axes[1, 1], [result.best_val_loss for result in results], "Best val loss", palette["validation"])
    _curve(axes[1, 1], [result.train_loss_final for result in results], "Final train loss", palette["train"])
    axes[1, 1].set_title("Loss at convergence")
    axes[1, 1].set_xlabel("Train ratio")
    axes[1, 1].set_ylabel("MSE loss")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend(loc="best")

    for ax in axes.ravel():
        ax.set_xticks(train_ratios)
        ax.set_xticklabels([_format_ratio(value) for value in train_ratios])
        ax.set_xlim(train_ratios.min() - 0.02, train_ratios.max() + 0.02)
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("Model convergence as training set increases", fontsize=15, y=1.02)
    fig.tight_layout()
    output_path = output_dir / "convergence_metrics.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_loss_curves(
    results: List[ConvergenceRun],
    histories: List[Dict[str, List[float]]],
    output_dir: Path,
) -> Path:
    num_runs = len(histories)
    num_cols = 2 if num_runs > 1 else 1
    num_rows = int(np.ceil(num_runs / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6.8 * num_cols, 3.8 * num_rows), squeeze=False)

    for idx, (result, history) in enumerate(zip(results, histories, strict=True)):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col]

        epochs = np.arange(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], color="tab:green", linewidth=2.0, label="Train loss")
        ax.plot(epochs, history["val_loss"], color="tab:blue", linewidth=2.0, label="Validation loss")
        ax.axvline(result.best_epoch, color="tab:red", linestyle="--", linewidth=1.4, alpha=0.75, label="Best epoch")
        ax.set_title(f"train_ratio = {_format_ratio(result.train_ratio)}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.text(
            0.02,
            0.98,
            f"best val = {result.best_val_loss:.4g}\nR2 test = {result.test_r2:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
        ax.legend(loc="best")

    for idx in range(num_runs, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].axis("off")

    fig.suptitle("Training convergence curves by train ratio", fontsize=15, y=1.01)
    fig.tight_layout()
    output_path = output_dir / "training_curves.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a train-ratio convergence sweep for the PyTorch aerodynamic surrogate model."
    )
    parser.add_argument("--convergence-config", default="config/convergence_config.yaml")
    parser.add_argument("--dataset-config", default="config/dataset_config.yaml")
    parser.add_argument("--model-config", default="config/model_config.yaml")
    parser.add_argument("--training-config", default="config/training_config.yaml")
    parser.add_argument("--visualization-config", default="config/visualization_config.yaml")
    parser.add_argument(
        "--train-ratios",
        type=float,
        nargs="+",
        default=None,
        help="List of train ratios to evaluate, e.g. 0.15 0.25 0.35 0.45 0.55.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/convergence_analysis",
        help="Base directory where plots and summaries are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed used for all runs.",
    )
    return parser.parse_args()


def main() -> None:
    _apply_publication_style()
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    convergence_cfg_path = project_root / args.convergence_config
    convergence_cfg = load_yaml(convergence_cfg_path) if convergence_cfg_path.exists() else {}

    dataset_cfg_path = project_root / str(convergence_cfg.get("dataset_config", args.dataset_config))
    model_cfg_path = project_root / str(convergence_cfg.get("model_config", args.model_config))
    training_cfg_path = project_root / str(convergence_cfg.get("training_config", args.training_config))
    visualization_cfg_path = project_root / str(
        convergence_cfg.get("visualization_config", args.visualization_config)
    )

    dataset_cfg = load_yaml(dataset_cfg_path)
    model_cfg = load_yaml(model_cfg_path)
    training_cfg = load_yaml(training_cfg_path)
    visualization_cfg = load_yaml(visualization_cfg_path) if visualization_cfg_path.exists() else {}

    train_ratios = _resolve_train_ratios(
        args.train_ratios if args.train_ratios is not None else convergence_cfg.get("train_ratios")
    )

    convergence_seed = convergence_cfg.get("seed")
    seed = int(
        args.seed
        if args.seed is not None
        else (convergence_seed if convergence_seed is not None else training_cfg.get("random_seed", dataset_cfg.get("random_seed", 42)))
    )
    set_global_seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir_root = str(convergence_cfg.get("output_dir", args.output_dir))
    output_root = project_root / output_dir_root / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[ConvergenceRun] = []
    histories: List[Dict[str, List[float]]] = []

    for train_ratio in train_ratios:
        run_name = f"train_{_format_ratio(train_ratio).replace('.', 'p')}"
        run_dir = output_root / run_name
        print(f"Running convergence experiment for train_ratio={train_ratio:.3f}...")
        run_summary, history = _run_single_experiment(
            project_root=project_root,
            dataset_cfg=dataset_cfg,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            visualization_cfg=visualization_cfg,
            train_ratio=train_ratio,
            run_output_dir=run_dir,
        )
        results.append(run_summary)
        histories.append(history)

    summary_frame = pd.DataFrame([asdict(result) for result in results]).sort_values("train_ratio")
    summary_csv_path = output_root / "convergence_summary.csv"
    summary_frame.to_csv(summary_csv_path, index=False)

    summary_json_path = output_root / "convergence_summary.json"
    save_json(
        summary_json_path,
        {
            "seed": seed,
            "train_ratios": train_ratios,
            "results": [asdict(result) for result in results],
            "convergence_config": str(convergence_cfg_path.relative_to(project_root))
            if convergence_cfg_path.exists()
            else None,
            "dataset_config": str(dataset_cfg_path.relative_to(project_root)),
            "model_config": str(model_cfg_path.relative_to(project_root)),
            "training_config": str(training_cfg_path.relative_to(project_root)),
            "visualization_config": str(visualization_cfg_path.relative_to(project_root))
            if visualization_cfg_path.exists()
            else None,
        },
    )

    metrics_plot_path = _plot_metric_curves(results, output_root)
    losses_plot_path = _plot_loss_curves(results, histories, output_root)

    print("Convergence sweep completed.")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Metrics plot: {metrics_plot_path}")
    print(f"Loss curves plot: {losses_plot_path}")


if __name__ == "__main__":
    main()