"""Entrypoint helpers for convergence analysis mode.

This module provides `run_analysis` which mirrors the top-level
`convergence_analysis.py` functionality but lives inside the `src`
package so `main.py` can delegate to it and the top-level script can
be removed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.utils import load_yaml, setup_logger
from src.convergence_analysis import run_analysis as _run_analysis_impl

logger = setup_logger(__name__)


def run_analysis(config_path: Path) -> None:
    """Run convergence analysis using the internal implementation.

    This is a thin wrapper that keeps the public API stable while
    moving the entrypoint into the package namespace.
    """
    # Reuse the existing implementation under src.convergence_analysis
    _run_analysis_impl(config_path)
