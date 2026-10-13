"""HYDDB physics-driven plot suite.

Five figures exercising the trained surrogate at fixed operating conditions:

1. Cfx vs Mach, parameterized by airbrake opening (alpha = 0).
2. Cfx vs alpha, parameterized by Mach (abk = 0).
3. Phi sensitivity (single curve; sanity check that phi is a near-null axis).
4. 2D heatmaps of mean Cfx and predicted sigma over a Mach x alpha grid.
5. 3D surface of mean Cfx + 2D sigma heatmap (toggle: ``config.hyddb.enable_3d``;
   skipped for non-GP models since sigma is meaningless there).

Locked values for the inputs that are neither axis nor sweep parameter come
from ``config.hyddb.locked_values``. A missing required entry raises
``ValueError`` listing every column the call needed — no silent fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from ..utils import (
    FIG_STYLE,
    PALETTE,
    apply_style,
    finalize_figure,
    is_gp_model,
    predict_mean_std,
)

if TYPE_CHECKING:
    from ..config import VisualizationConfig

# HYDDB CSV columns (denormalized space).
COL_MACH = "sim_machNumberValue"
COL_ALPHA = "sim_aoaValue"
COL_PHI = "sim_phiValue"
COL_ALTITUDE = "sim_altitudeValue"
COL_ABK = "geom_abk_openingPercentage"

# TeX labels for the HYDDB axes / parameters.
TEX_M = r"$M$"
TEX_ALPHA = r"$\alpha\ [\mathrm{deg}]$"
TEX_PHI = r"$\phi\ [\mathrm{deg}]$"
TEX_ABK = r"$\delta_{abk}\ [\%]$"
TEX_CFX = r"$C_{Fx}$"

_FEATURE_ORDER = (COL_MACH, COL_ALPHA, COL_PHI, COL_ALTITUDE, COL_ABK)


def _column_index(data: dict[str, Any], column: str) -> int:
    """Return the index of ``column`` in ``data["feature_names"]`` (input columns)."""
    names = list(data.get("feature_names", []))
    if column not in names:
        raise KeyError(f"HYDDB plot expects column '{column}' in feature_names, got {names}")
    return names.index(column)


def _required_locked(config: "VisualizationConfig", required: list[str]) -> dict[str, float]:
    """Look up ``required`` columns in config.hyddb.locked_values; raise on any missing."""
    locked = config.hyddb.locked_values
    missing = [name for name in required if name not in locked]
    if missing:
        raise ValueError(
            "Missing locked_values for HYDDB plot — add the following entries under "
            f"visualization.hyddb.locked_values: {missing}"
        )
    return {name: float(locked[name]) for name in required}


def _build_inputs(
    data: dict[str, Any],
    base_locked: dict[str, float],
    sweeps: dict[str, np.ndarray],
) -> np.ndarray:
    """Assemble an (n_points, n_features) array with columns in ``feature_names`` order.

    ``sweeps`` overrides ``base_locked`` per-column; sweep arrays must share length.
    """
    names = list(data["feature_names"])
    sweep_lengths = {len(arr) for arr in sweeps.values()}
    if len(sweep_lengths) > 1:
        raise ValueError(f"sweep arrays must share length, got {sweep_lengths}")
    n_points = next(iter(sweep_lengths)) if sweep_lengths else 1

    rows = np.empty((n_points, len(names)), dtype=float)
    for j, name in enumerate(names):
        if name in sweeps:
            rows[:, j] = sweeps[name]
        elif name in base_locked:
            rows[:, j] = base_locked[name]
        else:
            raise ValueError(f"HYDDB plot: column '{name}' is neither swept nor locked")
    return rows


def _predict(model: Any, features: np.ndarray, data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Predict (mean, std) on denormalized ``features``."""
    return predict_mean_std(
        model,
        features,
        data["input_scaler_mean"],
        data["input_scaler_std"],
        data["target_scaler_mean"],
        data["target_scaler_std"],
    )


def _curve_palette(n: int) -> np.ndarray:
    """Discretized viridis (colorblind-safe) for parametric curves."""
    return plt.get_cmap("viridis")(np.linspace(0.05, 0.95, max(n, 1)))


def _set_xlim_to_range(ax: plt.Axes, span: tuple) -> None:
    ax.set_xlim(float(span[0]), float(span[1]))


def _plot_cfx_vs_mach_by_abk(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig") -> None:
    locked = _required_locked(config, [COL_PHI, COL_ALTITUDE])
    locked[COL_ALPHA] = 0.0  # spec: alpha = 0
    abk_values = list(config.hyddb.abk_curves)
    mach_grid = np.linspace(*config.hyddb.mach_range, config.hyddb.n_sweep_points)
    show_band = is_gp_model(model)

    fig, ax = plt.subplots(figsize=(float(FIG_STYLE["figure_width"]), float(FIG_STYLE["figure_height"])))
    colors = _curve_palette(len(abk_values))

    for color, abk in zip(colors, abk_values):
        features = _build_inputs(
            data,
            base_locked={**locked, COL_ABK: float(abk)},
            sweeps={COL_MACH: mach_grid},
        )
        mean, std = _predict(model, features, data)
        label = rf"$\delta_{{abk}} = {abk:g}\%$"
        ax.plot(mach_grid, mean, color=color, linewidth=float(FIG_STYLE["line_width"]), label=label)
        if show_band:
            ax.fill_between(mach_grid, mean - 2 * std, mean + 2 * std, color=color, alpha=float(FIG_STYLE["band_alpha"]))

    apply_style(ax, rf"$C_{{Fx}}$ vs $M$ — {model_name} ($\alpha = 0$)", TEX_M, TEX_CFX)
    _set_xlim_to_range(ax, config.hyddb.mach_range)
    ax.legend(loc="best", ncol=2, fontsize=int(FIG_STYLE["annotation_size"]))
    finalize_figure(fig, f"cfx_vs_mach_by_abk_{model_name}.png", config, subdir="hyddb")


def _plot_cfx_vs_alpha_by_mach(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig") -> None:
    locked = _required_locked(config, [COL_PHI, COL_ALTITUDE])
    locked[COL_ABK] = 0.0  # spec: abk = 0
    mach_values = list(config.hyddb.mach_curves)
    alpha_grid = np.linspace(*config.hyddb.alpha_range, config.hyddb.n_sweep_points)
    show_band = is_gp_model(model)

    fig, ax = plt.subplots(figsize=(float(FIG_STYLE["figure_width"]), float(FIG_STYLE["figure_height"])))
    colors = _curve_palette(len(mach_values))

    for color, mach in zip(colors, mach_values):
        features = _build_inputs(
            data,
            base_locked={**locked, COL_MACH: float(mach)},
            sweeps={COL_ALPHA: alpha_grid},
        )
        mean, std = _predict(model, features, data)
        label = rf"$M = {mach:g}$"
        ax.plot(alpha_grid, mean, color=color, linewidth=float(FIG_STYLE["line_width"]), label=label)
        if show_band:
            ax.fill_between(alpha_grid, mean - 2 * std, mean + 2 * std, color=color, alpha=float(FIG_STYLE["band_alpha"]))

    apply_style(ax, rf"$C_{{Fx}}$ vs $\alpha$ — {model_name} ($\delta_{{abk}} = 0$)", TEX_ALPHA, TEX_CFX)
    _set_xlim_to_range(ax, config.hyddb.alpha_range)
    ax.legend(loc="best", fontsize=int(FIG_STYLE["annotation_size"]))
    finalize_figure(fig, f"cfx_vs_alpha_by_mach_{model_name}.png", config, subdir="hyddb")


def _plot_phi_sensitivity(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig") -> None:
    locked = _required_locked(config, [COL_MACH, COL_ALPHA, COL_ABK, COL_ALTITUDE])
    phi_grid = np.linspace(*config.hyddb.phi_range, config.hyddb.n_sweep_points)
    show_band = is_gp_model(model)

    features = _build_inputs(data, base_locked=locked, sweeps={COL_PHI: phi_grid})
    mean, std = _predict(model, features, data)

    fig, ax = plt.subplots(figsize=(float(FIG_STYLE["figure_width"]), float(FIG_STYLE["figure_height"])))
    ax.plot(phi_grid, mean, color=PALETTE["test"], linewidth=float(FIG_STYLE["line_width"]), label=r"prediction")
    if show_band:
        ax.fill_between(phi_grid, mean - 2 * std, mean + 2 * std, color=PALETTE["test"], alpha=float(FIG_STYLE["band_alpha"]), label=r"$\pm 2\sigma$")

    mean_value = float(np.mean(mean))
    ax.axhline(mean_value, linestyle="--", linewidth=float(FIG_STYLE["reference_line_width"]), color=PALETTE["ref"], label=rf"$\bar{{C}}_{{Fx}} = {mean_value:.4f}$")
    delta = float(np.max(mean) - np.min(mean))
    ax.text(
        0.02, 0.98,
        rf"$\Delta C_{{Fx}} = {delta:.4f}$",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=int(FIG_STYLE["annotation_size"]),
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    apply_style(ax, rf"Phi sensitivity — {model_name}", TEX_PHI, TEX_CFX)
    _set_xlim_to_range(ax, config.hyddb.phi_range)
    ax.legend(loc="best")
    finalize_figure(fig, f"phi_sensitivity_{model_name}.png", config, subdir="hyddb")


def _build_2d_grid(data: dict[str, Any], config: "VisualizationConfig", base_locked: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (mach_axis, alpha_axis, mean_grid, std_grid) on a Mach x alpha mesh."""
    mach_axis = np.linspace(*config.hyddb.mach_range, config.hyddb.n_grid_points)
    alpha_axis = np.linspace(*config.hyddb.alpha_range, config.hyddb.n_grid_points)
    mach_grid, alpha_grid = np.meshgrid(mach_axis, alpha_axis)
    flat = mach_grid.size
    features = _build_inputs(
        data,
        base_locked=base_locked,
        sweeps={COL_MACH: mach_grid.ravel(), COL_ALPHA: alpha_grid.ravel()},
    )
    mean, std = _predict(_build_2d_grid._model, features, data)
    return mach_axis, alpha_axis, mean.reshape(mach_grid.shape), std.reshape(mach_grid.shape)


def _plot_cfx_2d_pred_unc(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig") -> None:
    base_locked = _required_locked(config, [COL_PHI, COL_ALTITUDE, COL_ABK])
    _build_2d_grid._model = model  # transient handle; avoids duplicating the gridding code
    mach_axis, alpha_axis, mean_grid, std_grid = _build_2d_grid(data, config, base_locked)
    show_unc = is_gp_model(model)

    fig, axes = plt.subplots(1, 2, figsize=(2 * float(FIG_STYLE["figure_width"]) * 0.55, float(FIG_STYLE["figure_height"])))
    pred_levels = int(FIG_STYLE["contour_levels"])
    contour_pred = axes[0].contourf(mach_axis, alpha_axis, mean_grid, levels=pred_levels, cmap="viridis")
    axes[0].contour(mach_axis, alpha_axis, mean_grid, levels=int(FIG_STYLE["contour_line_levels"]), colors="black", linewidths=0.6)
    fig.colorbar(contour_pred, ax=axes[0], label=TEX_CFX)
    apply_style(axes[0], rf"Predicted {TEX_CFX}", TEX_M, TEX_ALPHA)

    if show_unc:
        contour_std = axes[1].contourf(mach_axis, alpha_axis, std_grid, levels=pred_levels, cmap="magma")
        fig.colorbar(contour_std, ax=axes[1], label=r"$\sigma(C_{Fx})$")
        apply_style(axes[1], r"Predicted $\sigma$", TEX_M, TEX_ALPHA)
    else:
        axes[1].text(0.5, 0.5, "Uncertainty unavailable\nfor this model", transform=axes[1].transAxes, ha="center", va="center", fontsize=int(FIG_STYLE["title_size"]))
        axes[1].set_xticks([])
        axes[1].set_yticks([])

    fig.suptitle(rf"$C_{{Fx}}$ over $(M, \alpha)$ — {model_name}", fontsize=int(FIG_STYLE["title_size"]))
    fig.tight_layout()
    finalize_figure(fig, f"cfx_2d_pred_unc_{model_name}.png", config, subdir="hyddb")


def _plot_cfx_3d_pred_sigma(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig") -> None:
    if not is_gp_model(model):
        return  # 3D + sigma panel is GP-only by design; MLP gets only the 2D pred from plot 4
    base_locked = _required_locked(config, [COL_PHI, COL_ALTITUDE, COL_ABK])
    _build_2d_grid._model = model
    mach_axis, alpha_axis, mean_grid, std_grid = _build_2d_grid(data, config, base_locked)
    mach_mesh, alpha_mesh = np.meshgrid(mach_axis, alpha_axis)

    fig = plt.figure(figsize=(2 * float(FIG_STYLE["figure_width"]) * 0.55, float(FIG_STYLE["figure_height"])))
    ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax_3d.plot_surface(mach_mesh, alpha_mesh, mean_grid, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    fig.colorbar(surf, ax=ax_3d, shrink=0.65, pad=0.1, label=TEX_CFX)
    ax_3d.set_xlabel(TEX_M)
    ax_3d.set_ylabel(TEX_ALPHA)
    ax_3d.set_zlabel(TEX_CFX)
    ax_3d.set_title(rf"Predicted {TEX_CFX} — {model_name}")

    ax_2d = fig.add_subplot(1, 2, 2)
    contour_std = ax_2d.contourf(mach_axis, alpha_axis, std_grid, levels=int(FIG_STYLE["contour_levels"]), cmap="magma")
    fig.colorbar(contour_std, ax=ax_2d, label=r"$\sigma(C_{Fx})$")
    apply_style(ax_2d, r"Predicted $\sigma$", TEX_M, TEX_ALPHA)

    fig.tight_layout()
    finalize_figure(fig, f"cfx_3d_pred_sigma_{model_name}.png", config, subdir="hyddb")


def run(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig") -> None:
    """Run the HYDDB suite for a single model. Output lands under config.output_dir/hyddb/."""

    if not config.hyddb.enabled:
        return
    _plot_cfx_vs_mach_by_abk(model, model_name, data, config)
    _plot_cfx_vs_alpha_by_mach(model, model_name, data, config)
    _plot_phi_sensitivity(model, model_name, data, config)
    _plot_cfx_2d_pred_unc(model, model_name, data, config)
    if config.hyddb.enable_3d:
        _plot_cfx_3d_pred_sigma(model, model_name, data, config)
