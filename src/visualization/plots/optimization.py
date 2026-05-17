"""Leaf plot for the optimization workflow: a 2x3 summary dashboard.

Reads completed Optuna trials, ignores those with non-finite objective, and
emits one figure with: convergence, per-trial validation metrics, objective
histogram, incremental improvement, top-10 trials table, and best-trial
train/val loss curves.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..config import VisualizationConfig
from ..labels import TEX_MAE, TEX_MAPE, TEX_R2, TEX_RMSE
from ..utils import finalize_figure

logger = logging.getLogger(__name__)


def _completed_trials(study: Any) -> list[Any]:
    import optuna

    return [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and np.isfinite(float(t.value))
    ]


def run_summary_dashboard(study: Any, config: VisualizationConfig) -> None:
    completed = _completed_trials(study)
    if not completed:
        logger.warning("Skipping summary dashboard: no completed trials with finite objective.")
        return

    trial_numbers = np.asarray([t.number for t in completed], dtype=int)
    objective_values = np.asarray([float(t.value) for t in completed], dtype=float)
    best_so_far = np.minimum.accumulate(objective_values)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax_convergence, ax_metrics, ax_hist = axes[0]
    ax_improvement, ax_top10, ax_losses = axes[1]

    ax_convergence.plot(trial_numbers, best_so_far, color="tab:blue", linewidth=2)
    ax_convergence.scatter(trial_numbers, objective_values, color="tab:blue", s=20, alpha=0.45)
    ax_convergence.set_title("Convergence (best so far)")
    ax_convergence.set_xlabel("Trial")
    ax_convergence.set_ylabel("Objective")
    ax_convergence.grid(alpha=0.3)

    metric_specs = [
        ("val_rmse", "tab:red", TEX_RMSE),
        ("val_mae", "tab:orange", TEX_MAE),
        ("val_mape", "tab:green", TEX_MAPE),
    ]
    metric_r2 = np.asarray(
        [float(t.user_attrs.get("val_r2", np.nan)) for t in completed], dtype=float
    )

    has_primary = False
    for key, color, label in metric_specs:
        vals = np.asarray(
            [float(t.user_attrs.get(key, np.nan)) for t in completed], dtype=float
        )
        finite_mask = np.isfinite(vals)
        if finite_mask.any():
            has_primary = True
            ax_metrics.plot(
                trial_numbers[finite_mask], vals[finite_mask],
                marker="o", linewidth=1.5, color=color, label=label,
            )

    ax_metrics_r2 = ax_metrics.twinx()
    r2_mask = np.isfinite(metric_r2)
    has_r2 = r2_mask.any()
    if has_r2:
        ax_metrics_r2.plot(
            trial_numbers[r2_mask], metric_r2[r2_mask],
            marker="x", linewidth=1.4, color="tab:purple", label=TEX_R2,
        )
        ax_metrics_r2.set_ylabel(TEX_R2)

    ax_metrics.set_title("Val metrics over trials")
    ax_metrics.set_xlabel("Trial")
    ax_metrics.set_ylabel("Error metrics")
    ax_metrics.grid(alpha=0.3)
    if has_primary or has_r2:
        handles_1, labels_1 = ax_metrics.get_legend_handles_labels()
        handles_2, labels_2 = ax_metrics_r2.get_legend_handles_labels()
        ax_metrics.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best", fontsize=8)
    else:
        ax_metrics.text(0.5, 0.5, "No val_* user attrs", ha="center", va="center",
                        transform=ax_metrics.transAxes)

    ax_hist.hist(objective_values, bins=min(20, len(objective_values)),
                 color="tab:cyan", edgecolor="black")
    ax_hist.axvline(float(np.min(objective_values)), color="tab:red", linestyle="--", linewidth=2)
    ax_hist.set_title("Distribution of values")
    ax_hist.set_xlabel("Objective")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(alpha=0.3)

    improvement = np.concatenate([[np.nan], np.diff(best_so_far)])
    ax_improvement.plot(trial_numbers, improvement, color="tab:green", marker="o", linewidth=1.5)
    ax_improvement.axhline(0.0, color="black", linewidth=1, linestyle=":")
    ax_improvement.set_title("Incremental improvement")
    ax_improvement.set_xlabel("Trial")
    ax_improvement.set_ylabel("Delta best")
    ax_improvement.grid(alpha=0.3)

    top_trials = sorted(completed, key=lambda t: float(t.value))[:10]
    table_rows = [
        [
            str(t.number),
            f"{float(t.value):.5f}",
            f"{float(t.user_attrs.get('val_rmse', np.nan)):.4f}",
            f"{float(t.user_attrs.get('val_r2',   np.nan)):.4f}",
        ]
        for t in top_trials
    ]
    ax_top10.axis("off")
    ax_top10.set_title("Top-10 trials")
    table = ax_top10.table(
        cellText=table_rows,
        colLabels=["Trial", "Objective", f"Val {TEX_RMSE}", f"Val {TEX_R2}"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    best_trial = study.best_trial if study.best_trial is not None else None
    train_losses = best_trial.user_attrs.get("train_losses") if best_trial is not None else None
    val_losses = best_trial.user_attrs.get("val_losses") if best_trial is not None else None

    if isinstance(train_losses, list) and len(train_losses) > 0:
        train_arr = np.asarray(train_losses, dtype=float)
        train_finite = np.isfinite(train_arr)
        epochs = np.arange(1, len(train_arr) + 1)
        ax_losses.plot(epochs[train_finite], train_arr[train_finite],
                       color="tab:blue", linewidth=1.8, label="Train")
        if isinstance(val_losses, list) and len(val_losses) > 0:
            val_arr = np.asarray(val_losses, dtype=float)
            val_finite = np.isfinite(val_arr)
            ax_losses.plot(
                np.arange(1, len(val_arr) + 1)[val_finite], val_arr[val_finite],
                color="tab:orange", linewidth=1.8, label="Val",
            )
        ax_losses.set_title("Train vs Val loss (best trial)")
        ax_losses.set_xlabel("Epoch")
        ax_losses.set_ylabel("Loss")
        ax_losses.grid(alpha=0.3)
        ax_losses.legend(loc="best", fontsize=8)
    else:
        ax_losses.text(0.5, 0.5, "No train/val loss history available",
                       ha="center", va="center", transform=ax_losses.transAxes)
        ax_losses.set_title("Train vs Val loss (best trial)")
        ax_losses.set_xticks([])
        ax_losses.set_yticks([])

    fig.suptitle("Optimization Summary Dashboard", fontsize=14)
    fig.tight_layout()
    finalize_figure(fig, "summary_dashboard.png", config)
