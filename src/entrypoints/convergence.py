"""Convergence-analysis entrypoint."""

from __future__ import annotations

from pathlib import Path

from src.convergence_analysis import run_analysis as _run_analysis_impl
from src.utils import setup_logger

logger = setup_logger(__name__)


def run_analysis(config_path: Path) -> None:
    _run_analysis_impl(config_path)
