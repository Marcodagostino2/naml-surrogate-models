"""Plot suite for the ``optimization`` workflow.

Builds the Optuna summary dashboard (and any future optimization-only views).
The runner already validates that ``study`` is non-None before calling here.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..config import VisualizationConfig
from ..plots.optimization import run_summary_dashboard

logger = logging.getLogger(__name__)


def run(study: Any, config: VisualizationConfig) -> None:
    if not config.optimization.enabled:
        return
    os.makedirs(config.output_dir, exist_ok=True)
    try:
        run_summary_dashboard(study, config)
    except Exception:
        logger.exception("Optimization summary dashboard failed")
