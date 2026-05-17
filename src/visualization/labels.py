"""Shared TeX-formatted axis and legend labels.

mathtext-friendly strings (no system LaTeX required). Dataset-specific
labels (e.g. HYDDB physics quantities) live next to their suite, not here.
"""

from __future__ import annotations

TEX_R2 = r"$R^2$"
TEX_RMSE = "RMSE"
TEX_MAE = "MAE"
TEX_MAPE = "MAPE"

TEX_PRED = r"$\hat{y}$"
TEX_TRUE = r"$y$"
TEX_RESIDUAL = r"$y - \hat{y}$"
TEX_SIGMA = r"$\sigma$"


def with_unit(name: str, unit: str | None) -> str:
    """Format ``name [unit]`` only when ``unit`` is non-empty."""
    if unit is None or not str(unit).strip():
        return str(name)
    return f"{name} [{unit}]"
