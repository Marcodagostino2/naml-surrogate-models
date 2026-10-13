"""Plot suite for the ``convergence`` workflow.

``convergence_records`` is a tuple ``(metrics_df, histories, model_sequence)``
assembled by ``src.convergence_analysis``. The suite mirrors the per-model
folder layout under ``config.output_dir`` and a ``complete/`` directory with
multi-model overlays when more than one model is present.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from ..config import VisualizationConfig
from ..plots.convergence import (
    plot_metric_curve,
    plot_metric_curve_all_models,
    plot_training_histories,
)

logger = logging.getLogger(__name__)

ConvergenceRecords = Tuple[pd.DataFrame, List[Dict[str, Any]], Sequence[str]]


def run(convergence_records: ConvergenceRecords, config: VisualizationConfig) -> None:
    if not config.convergence.enabled:
        return

    metrics_df, histories, model_sequence = convergence_records
    analysis_root = Path(config.output_dir)
    analysis_root.mkdir(parents=True, exist_ok=True)

    for model_name in model_sequence:
        subset = metrics_df[metrics_df["model"] == model_name]
        if subset.empty:
            continue
        model_root = analysis_root / model_name
        model_root.mkdir(parents=True, exist_ok=True)
        for metric in ("r2", "rmse", "mae", "mape"):
            plot_metric_curve(
                subset, metric,
                model_root / f"{metric}_vs_split.png",
                f"{model_name.upper()}: {metric.upper()} vs train split",
            )

    if len(model_sequence) > 1:
        complete_root = analysis_root / "complete"
        complete_root.mkdir(parents=True, exist_ok=True)
        for metric, label in (
            ("r2", "R²"),
            ("rmse", "RMSE"),
            ("mae", "MAE"),
            ("mape", "MAPE"),
        ):
            plot_metric_curve_all_models(
                metrics_df, metric,
                complete_root / f"{metric}_vs_split.png",
                f"All Models: {label} vs train split",
                model_sequence,
            )
        plot_training_histories(
            histories,
            complete_root / "training_history.png",
            "All Models: training histories across splits",
        )
