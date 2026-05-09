"""Per-workflow-mode visualization dispatcher.

One entry point — ``run_visualization`` — routes to the matching suite under
``src/visualization/suites/`` based on ``workflow_mode``. Each suite owns its
own outputs and decides which leaf plots to call.

The optimization and convergence suites are placeholders for now; stages 5/6
of the refactor move existing plotting code from ``optimization_support.py``
and ``convergence_analysis.py`` into them.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import VisualizationConfig

logger = logging.getLogger(__name__)


def run_visualization(
    workflow_mode: str,
    *,
    config: VisualizationConfig,
    dataset_name: str = "",
    models: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    study: Any = None,
    convergence_records: Any = None,
) -> None:
    """Dispatch to the per-mode plot suite.

    ``workflow_mode`` is ``"single_run"``, ``"optimization"``, or ``"convergence"``.
    Mode-specific arguments are validated by the matching suite.
    """

    config.validate()
    mode = workflow_mode.lower()

    if mode == "single_run":
        if models is None or data is None:
            raise ValueError("single_run visualization requires both 'models' and 'data'")
        from .suites.single_run import run as run_single_run
        run_single_run(models, data, config, dataset_name)
        return

    if mode == "optimization":
        if study is None:
            raise ValueError("optimization visualization requires 'study'")
        from .suites.optimization import run as run_optimization
        run_optimization(study, config)
        return

    if mode == "convergence":
        if convergence_records is None:
            raise ValueError("convergence visualization requires 'convergence_records'")
        from .suites.convergence import run as run_convergence
        run_convergence(convergence_records, config)
        return

    raise ValueError(
        f"Unknown workflow_mode: '{workflow_mode}'. Expected 'single_run', 'optimization', or 'convergence'."
    )
