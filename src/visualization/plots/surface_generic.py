"""Feature-space surface visualizations (1D slice + 2D heatmap) for one model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..utils import (
    FIG_STYLE,
    PALETTE,
    apply_style,
    axis_label,
    finalize_figure,
    get_display_names,
    get_input_units,
    get_output_display,
    predict_mean,
    sanitize_filename_component,
)

if TYPE_CHECKING:
    from ..config import VisualizationConfig


def _predict_grid(model: Any, grid: np.ndarray, data: dict[str, Any]) -> np.ndarray:
    return predict_mean(
        model,
        grid,
        data["input_scaler_mean"],
        data["input_scaler_std"],
        data["target_scaler_mean"],
        data["target_scaler_std"],
    )


def _locked_defaults(data: dict[str, Any], config: "VisualizationConfig") -> np.ndarray:
    """Resolve per-dim locked values (config overrides → median of train data)."""

    x_train = np.asarray(data["X_train"])
    defaults = np.median(x_train, axis=0)
    names = get_display_names(data)
    overrides = config.single_run.locked_values
    for index, name in enumerate(names):
        if name in overrides:
            defaults[index] = float(overrides[name])
    return defaults


def _locked_annotation(
    data: dict[str, Any],
    locked: np.ndarray,
    plotted: Sequence[int],
) -> str:
    names = get_display_names(data)
    units = get_input_units(data)
    plotted_set = set(plotted)
    lines: list[str] = []
    for index, name in enumerate(names):
        if index in plotted_set:
            continue
        unit_suffix = f" {units[index]}" if units[index] else ""
        lines.append(f"{name} = {locked[index]:.4g}{unit_suffix}")
    return "Locked: " + ", ".join(lines) if lines else ""


def _feature_variance(model: Any, data: dict[str, Any], feature_index: int, config: "VisualizationConfig", locked: np.ndarray) -> float:
    x_train = np.asarray(data["X_train"])
    feature_grid = np.linspace(
        float(x_train[:, feature_index].min()),
        float(x_train[:, feature_index].max()),
        config.surface.n_grid_points,
    )
    candidates = np.repeat(locked[None, :], feature_grid.size, axis=0)
    candidates[:, feature_index] = feature_grid
    predictions = _predict_grid(model, candidates, data)
    return float(np.var(predictions))


def _resolve_index(name: str, names: Sequence[str]) -> int:
    if name not in names:
        raise ValueError(f"Input dimension '{name}' not in feature list {list(names)}")
    return names.index(name)


def _resolve_slice_indices(
    model: Any,
    data: dict[str, Any],
    config: "VisualizationConfig",
    locked: np.ndarray,
) -> list[int]:
    names = get_display_names(data)
    if config.single_run.slice_dims:
        return [_resolve_index(name, names) for name in config.single_run.slice_dims]
    n_features = np.asarray(data["X_train"]).shape[1]
    scores = [(_feature_variance(model, data, idx, config, locked), idx) for idx in range(n_features)]
    ranked = sorted(scores, reverse=True)
    if len(ranked) < config.surface.n_top_features:
        raise ValueError("Not enough input dimensions for the configured slice count")
    return [idx for _, idx in ranked[: max(2, config.surface.n_top_features)]]


def _resolve_heatmap_pair(
    data: dict[str, Any],
    config: "VisualizationConfig",
    fallback: Sequence[int],
) -> tuple[int, int]:
    names = get_display_names(data)
    if config.single_run.heatmap_dim_pair:
        a, b = config.single_run.heatmap_dim_pair
        return _resolve_index(a, names), _resolve_index(b, names)
    if len(fallback) < 2:
        raise ValueError("Heatmap requires at least two slice dimensions")
    return fallback[0], fallback[1]


def _annotate_locked(ax: plt.Axes, text: str) -> None:
    if not text:
        return
    ax.text(
        0.02,
        -0.18,
        text,
        transform=ax.transAxes,
        fontsize=int(FIG_STYLE["annotation_size"]),
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": PALETTE["band"], "alpha": 0.7},
    )


def _plot_1d_slice(
    model: Any,
    model_name: str,
    data: dict[str, Any],
    config: "VisualizationConfig",
    feature_index: int,
    locked: np.ndarray,
) -> None:
    x_train = np.asarray(data["X_train"])
    names = get_display_names(data)
    units = get_input_units(data)
    output_name, output_unit = get_output_display(data)
    feature_values = np.linspace(
        float(x_train[:, feature_index].min()),
        float(x_train[:, feature_index].max()),
        config.surface.n_grid_points,
    )
    sweep = np.repeat(locked[None, :], feature_values.size, axis=0)
    sweep[:, feature_index] = feature_values
    predictions = _predict_grid(model, sweep, data)
    fig, ax = plt.subplots(figsize=(float(FIG_STYLE["figure_width"]), float(FIG_STYLE["figure_height"])))
    ax.plot(feature_values, predictions, color=PALETTE["test"], linewidth=float(FIG_STYLE["line_width"]))
    apply_style(
        ax,
        f"1D Slice — {model_name}",
        axis_label(names[feature_index], units[feature_index]),
        axis_label(output_name, output_unit),
    )
    _annotate_locked(ax, _locked_annotation(data, locked, [feature_index]))
    feature_slug = sanitize_filename_component(names[feature_index])
    finalize_figure(fig, f"surface_1d_{model_name}_{feature_slug}.png", config, subdir="surface")


def _plot_2d_heatmap(
    model: Any,
    model_name: str,
    data: dict[str, Any],
    config: "VisualizationConfig",
    feature_a: int,
    feature_b: int,
    locked: np.ndarray,
) -> None:
    x_train = np.asarray(data["X_train"])
    names = get_display_names(data)
    units = get_input_units(data)
    output_name, output_unit = get_output_display(data)
    feature_a_values = np.linspace(
        float(x_train[:, feature_a].min()),
        float(x_train[:, feature_a].max()),
        config.surface.n_grid_points,
    )
    feature_b_values = np.linspace(
        float(x_train[:, feature_b].min()),
        float(x_train[:, feature_b].max()),
        config.surface.n_grid_points,
    )
    grid_a, grid_b = np.meshgrid(feature_a_values, feature_b_values)
    grid = np.repeat(locked[None, :], grid_a.size, axis=0)
    grid[:, feature_a] = grid_a.ravel()
    grid[:, feature_b] = grid_b.ravel()
    predictions = _predict_grid(model, grid, data).reshape(grid_a.shape)
    fig, ax = plt.subplots(figsize=(float(FIG_STYLE["figure_width"]), float(FIG_STYLE["figure_height"])))
    contour = ax.contourf(grid_a, grid_b, predictions, levels=int(FIG_STYLE["contour_levels"]), cmap="viridis")
    lines = ax.contour(grid_a, grid_b, predictions, levels=int(FIG_STYLE["contour_line_levels"]), colors="black", linewidths=0.6)
    ax.clabel(lines, inline=True, fontsize=int(FIG_STYLE["tick_size"]))
    fig.colorbar(contour, ax=ax, label=axis_label(output_name, output_unit))
    apply_style(
        ax,
        f"2D Surface — {model_name}",
        axis_label(names[feature_a], units[feature_a]),
        axis_label(names[feature_b], units[feature_b]),
    )
    _annotate_locked(ax, _locked_annotation(data, locked, [feature_a, feature_b]))
    feature_a_slug = sanitize_filename_component(names[feature_a])
    feature_b_slug = sanitize_filename_component(names[feature_b])
    finalize_figure(fig, f"surface_2d_{model_name}_{feature_a_slug}_{feature_b_slug}.png", config, subdir="surface")


def run_surface(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig") -> None:
    """Generate feature-space surface visualizations for one model."""

    locked = _locked_defaults(data, config)
    slice_indices = _resolve_slice_indices(model, data, config, locked)
    for index in slice_indices:
        _plot_1d_slice(model, model_name, data, config, index, locked)
    feature_a, feature_b = _resolve_heatmap_pair(data, config, slice_indices)
    _plot_2d_heatmap(model, model_name, data, config, feature_a, feature_b, locked)
