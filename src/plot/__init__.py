"""
Initialization module for src.plot subpackage.
"""

from src.plot.diagnostics import (
    generate_diagnostics,
    plot_parity,
    plot_residuals,
    plot_training_history,
    plot_uncertainty,
)

__all__ = [
    "plot_parity",
    "plot_residuals",
    "plot_training_history",
    "plot_uncertainty",
    "generate_diagnostics",
]
