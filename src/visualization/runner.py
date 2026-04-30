"""Visualization orchestrator for model diagnostics and summaries."""

from __future__ import annotations

import logging
import os
from typing import Any

from .config import VisualizationConfig
from .utils import is_gp_model

logger = logging.getLogger(__name__)
ALL_MODELS = frozenset({"mlp", "gp", "mlpgp"})


def _require_keys(data: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"Visualization data is missing required keys: {missing}")


def _validate_data(data: dict[str, Any]) -> None:
    _require_keys(
        data,
        [
            "X_train",
            "y_train",
            "X_val",
            "y_val",
            "X_test",
            "y_test",
            "feature_names",
            "output_name",
            "input_scaler_mean",
            "input_scaler_std",
            "target_scaler_mean",
            "target_scaler_std",
        ],
    )


def run_visualization(
    models: dict[str, Any],
    data: dict[str, Any],
    config: VisualizationConfig,
    mode: str,
) -> None:
    """Dispatch enabled visualization modules for the provided models."""

    config.validate()
    _validate_data(data)
    os.makedirs(config.output_dir, exist_ok=True)

    from .modules.parity import run_parity
    from .modules.surface import run_surface
    from .modules.table import run_table
    from .modules.uncertainty_gp import run_uncertainty_gp

    for model_name, model in models.items():
        if config.parity.enabled:
            try:
                run_parity(model, model_name, data, config, mode)
            except Exception:
                logger.exception("Parity visualization failed for model '%s'", model_name)

        if config.surface.enabled:
            try:
                run_surface(model, model_name, data, config, mode)
            except Exception:
                logger.exception("Surface visualization failed for model '%s'", model_name)

        if config.uncertainty_gp.enabled and is_gp_model(model):
            try:
                run_uncertainty_gp(model, model_name, data, config, mode)
            except Exception:
                logger.exception("Uncertainty visualization failed for model '%s'", model_name)

    if config.table.enabled and set(models.keys()) == ALL_MODELS:
        try:
            run_table(models, data, config, mode)
        except Exception:
            logger.exception("Summary table visualization failed")
