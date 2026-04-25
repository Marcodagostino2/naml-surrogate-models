from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.training.metrics import regression_metrics


def _pretty_label(name: str, label_map: Optional[Dict[str, str]] = None) -> str:
    if label_map and name in label_map:
        return str(label_map[name])
    return str(name).replace("_", " ")


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)


def _inverse_targets(values: np.ndarray, target_scaler) -> np.ndarray:
    if target_scaler is None:
        return values
    return target_scaler.inverse_transform(values)


def _resolve_visualization_cfg(
    visualization_cfg: Optional[Dict[str, Any]],
    legacy_num_points: int,
) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "labels": {},
        "diagnostics": {
            "parity": {
                "point_size": 18,
                "point_alpha": 0.7,
                "annotate_metrics": True,
                "metrics_decimals": 5,
            },
            "residuals": {
                "bins": 30,
            },
            "percent_error": {
                "enabled": True,
                "epsilon": 1e-8,
                "percentiles": [50, 80, 90, 95],
                "error_thresholds_percent": [1, 2, 5, 10],
            },
        },
        "sweeps": {
            "enabled": True,
            "num_points_1d": int(legacy_num_points),
            "num_points_2d": 80,
            "num_points_3d": 32,
            "freeze_strategy": "median",
            "freeze_values": {},
            "include_training_points": True,
            "max_training_points_overlay": 1500,
            "generate_1d": True,
            "generate_2d_contour": True,
            "generate_3d_surface": True,
            "targets": "all",
        },
    }

    if visualization_cfg is None:
        return defaults

    merged: Dict[str, Any] = {
        "labels": {**defaults["labels"]},
        "diagnostics": {**defaults["diagnostics"]},
        "sweeps": {**defaults["sweeps"]},
    }
    if isinstance(visualization_cfg.get("labels", {}), dict):
        merged["labels"] = {**merged["labels"], **visualization_cfg.get("labels", {})}

    for top_key in ("diagnostics", "sweeps"):
        section = visualization_cfg.get(top_key, {})
        if isinstance(section, dict):
            for key, value in section.items():
                if isinstance(value, dict) and isinstance(merged[top_key].get(key), dict):
                    merged[top_key][key] = {**merged[top_key][key], **value}
                else:
                    merged[top_key][key] = value

    return merged


def _subsample_indices(num_samples: int, max_samples: int) -> np.ndarray:
    if num_samples <= max_samples:
        return np.arange(num_samples)
    rng = np.random.default_rng(42)
    return np.sort(rng.choice(num_samples, size=max_samples, replace=False))


def _build_reference_point(
    input_columns,
    feature_bounds,
    train_features_physical: Optional[np.ndarray],
    default_reference: Optional[np.ndarray],
    sweep_cfg: Dict[str, Any],
) -> np.ndarray:
    num_features = len(input_columns)
    if default_reference is not None:
        reference = np.asarray(default_reference, dtype=np.float32).copy()
    else:
        reference = np.zeros(num_features, dtype=np.float32)

    freeze_strategy = str(sweep_cfg.get("freeze_strategy", "median")).lower()
    for idx, feature_name in enumerate(input_columns):
        lo, hi = feature_bounds[feature_name]
        if freeze_strategy == "midrange":
            reference[idx] = 0.5 * (lo + hi)
        elif freeze_strategy == "mean" and train_features_physical is not None:
            reference[idx] = float(np.mean(train_features_physical[:, idx]))
        elif freeze_strategy == "median" and train_features_physical is not None:
            reference[idx] = float(np.median(train_features_physical[:, idx]))
        elif default_reference is None:
            reference[idx] = 0.5 * (lo + hi)

    freeze_values = sweep_cfg.get("freeze_values", {})
    if isinstance(freeze_values, dict):
        for feature_name, value in freeze_values.items():
            if feature_name in input_columns:
                reference[input_columns.index(feature_name)] = float(value)

    return reference


def _freeze_caption(
    input_columns,
    reference_point: np.ndarray,
    varying_indices: set[int],
    label_map: Optional[Dict[str, str]] = None,
) -> str:
    frozen_parts = []
    for idx, feature_name in enumerate(input_columns):
        if idx in varying_indices:
            continue
        frozen_parts.append(f"{_pretty_label(feature_name, label_map)}={reference_point[idx]:.4g}")
    return "Frozen: " + ", ".join(frozen_parts) if frozen_parts else "No frozen dimensions"


def _resolve_target_indices(target_names, targets_cfg) -> list[int]:
    if targets_cfg is None or str(targets_cfg).lower() == "all":
        return list(range(len(target_names)))

    if isinstance(targets_cfg, list):
        resolved: list[int] = []
        for item in targets_cfg:
            if isinstance(item, int):
                if 0 <= item < len(target_names):
                    resolved.append(item)
            else:
                item_name = str(item)
                if item_name in target_names:
                    resolved.append(target_names.index(item_name))
        unique = []
        seen = set()
        for idx in resolved:
            if idx not in seen:
                unique.append(idx)
                seen.add(idx)
        return unique

    return list(range(len(target_names)))


def _percent_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float) -> np.ndarray:
    denominator = np.maximum(np.abs(y_true), epsilon)
    return 100.0 * np.abs((y_pred - y_true) / denominator)


@torch.no_grad()
def _collect_predictions(model, loader, device: torch.device):
    model.eval()
    y_true_batches = []
    y_pred_batches = []
    x_batches = []
    for features, targets in loader:
        features = features.to(device)
        predictions = model(features)
        x_batches.append(features.cpu().numpy())
        y_true_batches.append(targets.cpu().numpy())
        y_pred_batches.append(predictions.cpu().numpy())

    return (
        np.concatenate(x_batches, axis=0),
        np.concatenate(y_true_batches, axis=0),
        np.concatenate(y_pred_batches, axis=0),
    )


def plot_diagnostics(
    model,
    loader,
    device: torch.device,
    target_scaler,
    target_names,
    output_dir: Path,
    visualization_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _resolve_visualization_cfg(visualization_cfg, legacy_num_points=200)
    label_map = cfg.get("labels", {})
    parity_cfg = cfg["diagnostics"]["parity"]
    residual_cfg = cfg["diagnostics"]["residuals"]
    percent_error_cfg = cfg["diagnostics"]["percent_error"]

    _, y_true, y_pred = _collect_predictions(model, loader, device)

    y_true_physical = _inverse_targets(y_true, target_scaler)
    y_pred_physical = _inverse_targets(y_pred, target_scaler)
    residuals = y_true_physical - y_pred_physical

    produced_files: Dict[str, Path] = {}

    num_targets = y_true_physical.shape[1]
    fig, axes = plt.subplots(1, num_targets, figsize=(6 * num_targets, 5), squeeze=False)
    for idx in range(num_targets):
        ax = axes[0, idx]
        ax.scatter(
            y_true_physical[:, idx],
            y_pred_physical[:, idx],
            s=float(parity_cfg.get("point_size", 18)),
            alpha=float(parity_cfg.get("point_alpha", 0.7)),
        )
        lower = min(y_true_physical[:, idx].min(), y_pred_physical[:, idx].min())
        upper = max(y_true_physical[:, idx].max(), y_pred_physical[:, idx].max())
        ax.plot([lower, upper], [lower, upper], "k--", linewidth=1.2)
        pretty_name = _pretty_label(target_names[idx], label_map)
        ax.set_xlabel(f"True {pretty_name}")
        ax.set_ylabel(f"Predicted {pretty_name}")
        ax.set_title(f"Parity plot - {pretty_name}")
        ax.grid(True, alpha=0.3)

        if bool(parity_cfg.get("annotate_metrics", True)):
            decimals = int(parity_cfg.get("metrics_decimals", 5))
            metrics = regression_metrics(y_true_physical[:, idx], y_pred_physical[:, idx])
            metrics_text = (
                f"RMSE = {metrics['rmse']:.{decimals}g}\n"
                f"MAE = {metrics['mae']:.{decimals}g}\n"
                f"R2 = {metrics['r2']:.{decimals}g}"
            )
            ax.text(
                0.03,
                0.97,
                metrics_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.6"},
            )
    fig.tight_layout()
    parity_path = output_dir / "parity_plot.png"
    fig.savefig(parity_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    produced_files["parity"] = parity_path

    fig, axes = plt.subplots(1, num_targets, figsize=(6 * num_targets, 5), squeeze=False)
    for idx in range(num_targets):
        ax = axes[0, idx]
        ax.hist(
            residuals[:, idx].ravel(),
            bins=int(residual_cfg.get("bins", 30)),
            alpha=0.8,
            color="tab:blue",
        )
        ax.set_xlabel(f"Residual {_pretty_label(target_names[idx], label_map)}")
        ax.set_ylabel("Count")
        ax.set_title(f"Residuals - {_pretty_label(target_names[idx], label_map)}")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    residual_path = output_dir / "residuals_histogram.png"
    fig.savefig(residual_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    produced_files["residuals"] = residual_path

    if bool(percent_error_cfg.get("enabled", True)):
        epsilon = float(percent_error_cfg.get("epsilon", 1e-8))
        percentiles = [float(v) for v in percent_error_cfg.get("percentiles", [50, 80, 90, 95])]
        thresholds = [
            float(v) for v in percent_error_cfg.get("error_thresholds_percent", [1, 2, 5, 10])
        ]

        fig, axes = plt.subplots(1, num_targets, figsize=(6 * num_targets, 5), squeeze=False)
        for idx in range(num_targets):
            ax = axes[0, idx]
            target_ape = _percent_error(y_true_physical[:, idx], y_pred_physical[:, idx], epsilon)
            sorted_error = np.sort(target_ape)
            cdf = 100.0 * (np.arange(sorted_error.shape[0]) + 1) / sorted_error.shape[0]

            ax.plot(sorted_error, cdf, color="tab:green", linewidth=2.0)
            for percentile in percentiles:
                error_at_percentile = float(np.percentile(target_ape, percentile))
                ax.axvline(error_at_percentile, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.6)
                ax.text(
                    error_at_percentile,
                    min(99.0, percentile + 2.0),
                    f"p{int(percentile)}={error_at_percentile:.3g}%",
                    rotation=90,
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="tab:red",
                )

            within_threshold = {
                threshold: 100.0 * float(np.mean(target_ape <= threshold)) for threshold in thresholds
            }
            threshold_text = "\n".join(
                f"<= {thr:g}%: {coverage:.1f}%" for thr, coverage in within_threshold.items()
            )
            ax.text(
                0.98,
                0.03,
                threshold_text,
                transform=ax.transAxes,
                va="bottom",
                ha="right",
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.6"},
            )

            pretty_name = _pretty_label(target_names[idx], label_map)
            ax.set_title(f"Percent Error CDF - {pretty_name}")
            ax.set_xlabel("Absolute percentage error [%]")
            ax.set_ylabel("Samples within error [%]")
            ax.set_ylim(0.0, 100.0)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        percent_error_path = output_dir / "percent_error_cdf.png"
        fig.savefig(percent_error_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        produced_files["percent_error_cdf"] = percent_error_path

    return produced_files


@torch.no_grad()
def plot_physical_sweeps(
    model,
    device: torch.device,
    feature_scaler,
    target_scaler,
    input_columns,
    target_names,
    feature_bounds,
    reference_point: Optional[np.ndarray],
    output_dir: Path,
    train_features_physical: Optional[np.ndarray] = None,
    train_targets_physical: Optional[np.ndarray] = None,
    visualization_cfg: Optional[Dict[str, Any]] = None,
    num_points: int = 200,
) -> tuple[Dict[str, Path], Dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _resolve_visualization_cfg(visualization_cfg, legacy_num_points=num_points)
    label_map = cfg.get("labels", {})
    sweep_cfg = cfg["sweeps"]

    produced_files: Dict[str, Path] = {}
    if not bool(sweep_cfg.get("enabled", True)):
        return produced_files, {}

    target_indices = _resolve_target_indices(target_names, sweep_cfg.get("targets", "all"))
    if len(target_indices) == 0:
        target_indices = list(range(len(target_names)))

    reference_point_resolved = _build_reference_point(
        input_columns=input_columns,
        feature_bounds=feature_bounds,
        train_features_physical=train_features_physical,
        default_reference=reference_point,
        sweep_cfg=sweep_cfg,
    )

    include_training_points = bool(sweep_cfg.get("include_training_points", True))
    max_training_points = int(sweep_cfg.get("max_training_points_overlay", 1500))
    train_indices = np.array([], dtype=np.int64)
    if include_training_points and train_features_physical is not None:
        train_indices = _subsample_indices(train_features_physical.shape[0], max_training_points)

    if bool(sweep_cfg.get("generate_1d", True)):
        for target_index in target_indices:
            target_name = target_names[target_index]
            target_slug = _safe_name(target_name)
            for feature_index, feature_name in enumerate(input_columns):
                lower, upper = feature_bounds[feature_name]
                n_1d = int(sweep_cfg.get("num_points_1d", num_points))
                physical_grid = np.linspace(lower, upper, n_1d, dtype=np.float32)
                physical_inputs = np.tile(reference_point_resolved.reshape(1, -1), (n_1d, 1)).astype(np.float32)
                physical_inputs[:, feature_index] = physical_grid

                if feature_scaler is not None:
                    model_inputs = feature_scaler.transform(physical_inputs)
                else:
                    model_inputs = physical_inputs

                inputs_tensor = torch.tensor(model_inputs, dtype=torch.float32, device=device)
                predictions = model(inputs_tensor).cpu().numpy()
                predictions_physical = _inverse_targets(predictions, target_scaler)[:, target_index]

                fig, ax = plt.subplots(figsize=(7, 5))
                ax.plot(
                    physical_grid,
                    predictions_physical,
                    linewidth=2.0,
                    color="tab:blue",
                    label="Model sweep",
                )

                if (
                    include_training_points
                    and train_features_physical is not None
                    and train_targets_physical is not None
                    and train_indices.size > 0
                ):
                    ax.scatter(
                        train_features_physical[train_indices, feature_index],
                        train_targets_physical[train_indices, target_index],
                        s=10,
                        alpha=0.35,
                        color="tab:orange",
                        label="Training points",
                    )

                ax.set_xlabel(_pretty_label(feature_name, label_map))
                ax.set_ylabel(_pretty_label(target_name, label_map))
                ax.set_title(
                    f"1D sweep: {_pretty_label(target_name, label_map)} vs {_pretty_label(feature_name, label_map)}"
                )
                ax.text(
                    0.02,
                    0.02,
                    _freeze_caption(
                        input_columns,
                        reference_point_resolved,
                        varying_indices={feature_index},
                        label_map=label_map,
                    ),
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    fontsize=8,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.6"},
                )
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best")
                fig.tight_layout()

                feature_slug = _safe_name(feature_name)
                sweep_path = output_dir / f"physical_sweep_{target_slug}_{feature_slug}.png"
                fig.savefig(sweep_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                produced_files[f"{target_name}__{feature_name}"] = sweep_path

    if bool(sweep_cfg.get("generate_2d_contour", True)) and len(input_columns) >= 2:
        n_2d = int(sweep_cfg.get("num_points_2d", 80))
        for target_index in target_indices:
            target_name = target_names[target_index]
            target_slug = _safe_name(target_name)
            for idx_a, idx_b in combinations(range(len(input_columns)), 2):
                feature_a = input_columns[idx_a]
                feature_b = input_columns[idx_b]

                grid_a = np.linspace(feature_bounds[feature_a][0], feature_bounds[feature_a][1], n_2d)
                grid_b = np.linspace(feature_bounds[feature_b][0], feature_bounds[feature_b][1], n_2d)
                mesh_a, mesh_b = np.meshgrid(grid_a, grid_b)
                flat_size = mesh_a.size

                physical_inputs = np.tile(reference_point_resolved.reshape(1, -1), (flat_size, 1)).astype(np.float32)
                physical_inputs[:, idx_a] = mesh_a.ravel()
                physical_inputs[:, idx_b] = mesh_b.ravel()

                if feature_scaler is not None:
                    model_inputs = feature_scaler.transform(physical_inputs)
                else:
                    model_inputs = physical_inputs

                inputs_tensor = torch.tensor(model_inputs, dtype=torch.float32, device=device)
                predictions = model(inputs_tensor).cpu().numpy()
                prediction_surface = _inverse_targets(predictions, target_scaler)[:, target_index].reshape(
                    mesh_a.shape
                )

                fig, ax = plt.subplots(figsize=(8, 6))
                contour = ax.contourf(mesh_a, mesh_b, prediction_surface, levels=25, cmap="viridis")
                colorbar = fig.colorbar(contour, ax=ax)
                colorbar.set_label(_pretty_label(target_name, label_map))

                if include_training_points and train_features_physical is not None and train_indices.size > 0:
                    ax.scatter(
                        train_features_physical[train_indices, idx_a],
                        train_features_physical[train_indices, idx_b],
                        s=10,
                        color="white",
                        alpha=0.45,
                        edgecolors="black",
                        linewidths=0.2,
                    )

                ax.set_xlabel(_pretty_label(feature_a, label_map))
                ax.set_ylabel(_pretty_label(feature_b, label_map))
                ax.set_title(
                    f"2D contour: {_pretty_label(target_name, label_map)}({_pretty_label(feature_a, label_map)}, {_pretty_label(feature_b, label_map)})"
                )
                ax.text(
                    0.02,
                    0.02,
                    _freeze_caption(
                        input_columns,
                        reference_point_resolved,
                        varying_indices={idx_a, idx_b},
                        label_map=label_map,
                    ),
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    fontsize=8,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.6"},
                )
                fig.tight_layout()

                feature_a_slug = _safe_name(feature_a)
                feature_b_slug = _safe_name(feature_b)
                contour_path = output_dir / f"physical_contour_{target_slug}_{feature_a_slug}_{feature_b_slug}.png"
                fig.savefig(contour_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                produced_files[f"contour_{target_name}__{feature_a}__{feature_b}"] = contour_path

    if bool(sweep_cfg.get("generate_3d_surface", True)) and len(input_columns) >= 2:
        n_3d = int(sweep_cfg.get("num_points_3d", 32))
        for target_index in target_indices:
            target_name = target_names[target_index]
            target_slug = _safe_name(target_name)
            for idx_a, idx_b in combinations(range(len(input_columns)), 2):
                feature_a = input_columns[idx_a]
                feature_b = input_columns[idx_b]

                grid_a = np.linspace(feature_bounds[feature_a][0], feature_bounds[feature_a][1], n_3d)
                grid_b = np.linspace(feature_bounds[feature_b][0], feature_bounds[feature_b][1], n_3d)
                mesh_a, mesh_b = np.meshgrid(grid_a, grid_b)
                flat_size = mesh_a.size

                physical_inputs = np.tile(reference_point_resolved.reshape(1, -1), (flat_size, 1)).astype(np.float32)
                physical_inputs[:, idx_a] = mesh_a.ravel()
                physical_inputs[:, idx_b] = mesh_b.ravel()

                if feature_scaler is not None:
                    model_inputs = feature_scaler.transform(physical_inputs)
                else:
                    model_inputs = physical_inputs

                inputs_tensor = torch.tensor(model_inputs, dtype=torch.float32, device=device)
                predictions = model(inputs_tensor).cpu().numpy()
                prediction_surface = _inverse_targets(predictions, target_scaler)[:, target_index].reshape(
                    mesh_a.shape
                )

                fig = plt.figure(figsize=(9, 7))
                ax = fig.add_subplot(111, projection="3d")
                surface = ax.plot_surface(mesh_a, mesh_b, prediction_surface, cmap="viridis", alpha=0.9)
                fig.colorbar(
                    surface,
                    ax=ax,
                    shrink=0.7,
                    aspect=12,
                    label=_pretty_label(target_name, label_map),
                )

                if (
                    include_training_points
                    and train_features_physical is not None
                    and train_targets_physical is not None
                    and train_indices.size > 0
                ):
                    ax.scatter(
                        train_features_physical[train_indices, idx_a],
                        train_features_physical[train_indices, idx_b],
                        train_targets_physical[train_indices, target_index],
                        s=8,
                        color="black",
                        alpha=0.35,
                    )

                ax.set_xlabel(_pretty_label(feature_a, label_map))
                ax.set_ylabel(_pretty_label(feature_b, label_map))
                ax.set_zlabel(_pretty_label(target_name, label_map))
                ax.set_title(
                    f"3D surface: {_pretty_label(target_name, label_map)}({_pretty_label(feature_a, label_map)}, {_pretty_label(feature_b, label_map)})"
                )
                ax.text2D(
                    0.02,
                    0.02,
                    _freeze_caption(
                        input_columns,
                        reference_point_resolved,
                        varying_indices={idx_a, idx_b},
                        label_map=label_map,
                    ),
                    transform=ax.transAxes,
                    fontsize=8,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.6"},
                )
                fig.tight_layout()

                feature_a_slug = _safe_name(feature_a)
                feature_b_slug = _safe_name(feature_b)
                surface_path = output_dir / f"physical_surface_{target_slug}_{feature_a_slug}_{feature_b_slug}.png"
                fig.savefig(surface_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                produced_files[f"surface_{target_name}__{feature_a}__{feature_b}"] = surface_path

    sweep_reference = {
        feature_name: float(reference_point_resolved[idx])
        for idx, feature_name in enumerate(input_columns)
    }
    return produced_files, sweep_reference
