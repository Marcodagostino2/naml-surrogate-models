from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def _pretty_label(name: str) -> str:
    labels = {
        "sim_machNumberValue": "Mach [-]",
        "sim_aoaValue": "AoA [deg]",
        "geom_abk_openingPercentage": "ABK [%]",
        "cfd_cfx": "Cfx [-]",
    }
    return labels.get(name, name)


def _inverse_targets(values: np.ndarray, target_scaler) -> np.ndarray:
    if target_scaler is None:
        return values
    return target_scaler.inverse_transform(values)


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
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, y_true, y_pred = _collect_predictions(model, loader, device)

    y_true_physical = _inverse_targets(y_true, target_scaler)
    y_pred_physical = _inverse_targets(y_pred, target_scaler)
    residuals = y_true_physical - y_pred_physical

    produced_files: Dict[str, Path] = {}

    num_targets = y_true_physical.shape[1]
    fig, axes = plt.subplots(1, num_targets, figsize=(6 * num_targets, 5), squeeze=False)
    for idx in range(num_targets):
        ax = axes[0, idx]
        ax.scatter(y_true_physical[:, idx], y_pred_physical[:, idx], s=18, alpha=0.7)
        lower = min(y_true_physical[:, idx].min(), y_pred_physical[:, idx].min())
        upper = max(y_true_physical[:, idx].max(), y_pred_physical[:, idx].max())
        ax.plot([lower, upper], [lower, upper], "k--", linewidth=1.2)
        pretty_name = _pretty_label(target_names[idx])
        ax.set_xlabel(f"True {pretty_name}")
        ax.set_ylabel(f"Predicted {pretty_name}")
        ax.set_title(f"Parity plot - {pretty_name}")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    parity_path = output_dir / "parity_plot.png"
    fig.savefig(parity_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    produced_files["parity"] = parity_path

    fig, axes = plt.subplots(1, num_targets, figsize=(6 * num_targets, 5), squeeze=False)
    for idx in range(num_targets):
        ax = axes[0, idx]
        ax.hist(residuals[:, idx].ravel(), bins=30, alpha=0.8, color="tab:blue")
        ax.set_xlabel(f"Residual {_pretty_label(target_names[idx])}")
        ax.set_ylabel("Count")
        ax.set_title(f"Residuals - {_pretty_label(target_names[idx])}")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    residual_path = output_dir / "residuals_histogram.png"
    fig.savefig(residual_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    produced_files["residuals"] = residual_path

    return produced_files


@torch.no_grad()
def plot_physical_sweeps(
    model,
    device: torch.device,
    feature_scaler,
    target_scaler,
    input_columns,
    target_name: str,
    feature_bounds,
    reference_point: np.ndarray,
    output_dir: Path,
    num_points: int = 200,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    produced_files: Dict[str, Path] = {}
    target_index = 0

    for feature_index, feature_name in enumerate(input_columns):
        lower, upper = feature_bounds[feature_name]
        physical_grid = np.linspace(lower, upper, num_points, dtype=np.float32)
        physical_inputs = np.tile(reference_point.reshape(1, -1), (num_points, 1)).astype(np.float32)
        physical_inputs[:, feature_index] = physical_grid

        if feature_scaler is not None:
            model_inputs = feature_scaler.transform(physical_inputs)
        else:
            model_inputs = physical_inputs

        inputs_tensor = torch.tensor(model_inputs, dtype=torch.float32, device=device)
        predictions = model(inputs_tensor).cpu().numpy()
        predictions_physical = _inverse_targets(predictions, target_scaler)[:, target_index]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(physical_grid, predictions_physical, linewidth=2.0, color="tab:blue")
        ax.set_xlabel(_pretty_label(feature_name))
        ax.set_ylabel(_pretty_label(target_name))
        ax.set_title(f"{_pretty_label(target_name)} vs {_pretty_label(feature_name)}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        sweep_path = output_dir / f"physical_sweep_{feature_name}.png"
        fig.savefig(sweep_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        produced_files[feature_name] = sweep_path

    return produced_files
