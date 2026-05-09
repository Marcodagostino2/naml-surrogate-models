"""Plot suite for the ``single_run`` workflow.

Calls the leaf plot functions for every trained model: metrics (parity /
residuals / MAPE), uncertainty (GP-only), generic surface explorations, and
the comparison table when all three model families are present. Diagnostics
(training history, val/test uncertainty bands) are emitted directly from
``pipeline.train_model`` because they need the per-model history dict.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..config import VisualizationConfig
from ..datasets import DATASET_SUITES, get_suite
from ..plots.metrics import run_parity
from ..plots.surface_generic import run_surface
from ..plots.table import run_table
from ..plots.uncertainty import run_uncertainty_gp
from ..utils import is_gp_model

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
            "X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
            "feature_names", "output_name",
            "input_scaler_mean", "input_scaler_std",
            "target_scaler_mean", "target_scaler_std",
        ],
    )


def run(
    models: dict[str, Any],
    data: dict[str, Any],
    config: VisualizationConfig,
    dataset_name: str = "",
) -> None:
    """Generate the single_run plot suite for ``models``.

    ``dataset_name`` is reserved for dataset-specific suites (stage 7 wires
    HYDDB plots in here); passing the empty string disables dataset routing.
    """

    _validate_data(data)
    os.makedirs(config.output_dir, exist_ok=True)

    suite_name = DATASET_SUITES.get(dataset_name)

    for model_name, model in models.items():
        if config.parity.enabled:
            try:
                run_parity(model, model_name, data, config)
            except Exception:
                logger.exception("Parity visualization failed for model '%s'", model_name)

        if config.surface.enabled:
            try:
                run_surface(model, model_name, data, config)
            except Exception:
                logger.exception("Surface visualization failed for model '%s'", model_name)

        if config.uncertainty_gp.enabled and is_gp_model(model):
            try:
                run_uncertainty_gp(model, model_name, data, config)
            except Exception:
                logger.exception("Uncertainty visualization failed for model '%s'", model_name)

        if suite_name is not None:
            try:
                get_suite(suite_name).run(model, model_name, data, config)
            except Exception:
                logger.exception("%s suite visualization failed for model '%s'", suite_name, model_name)

    if config.table.enabled and set(models.keys()) == ALL_MODELS:
        try:
            run_table(models, data, config)
        except Exception:
            logger.exception("Summary table visualization failed")
