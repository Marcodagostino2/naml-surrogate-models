"""Feature-space and PCA surface visualizations for one model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

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
    mode: str,
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
    ax.axvline(
        float(locked[feature_index]),
        linestyle="--",
        linewidth=float(FIG_STYLE["reference_line_width"]),
        color=PALETTE["ref"],
        label="lock",
    )
    apply_style(
        ax,
        f"1D Slice — {model_name}",
        axis_label(names[feature_index], units[feature_index]),
        axis_label(output_name, output_unit),
    )
    _annotate_locked(ax, _locked_annotation(data, locked, [feature_index]))
    ax.legend(loc="best")
    feature_slug = sanitize_filename_component(names[feature_index])
    finalize_figure(fig, f"surface_1d_{model_name}_{feature_slug}.png", config, mode)


def _plot_2d_heatmap(
    model: Any,
    model_name: str,
    data: dict[str, Any],
    config: "VisualizationConfig",
    mode: str,
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
    finalize_figure(fig, f"surface_2d_{model_name}_{feature_a_slug}_{feature_b_slug}.png", config, mode)


def _fit_pca(data: dict[str, Any], config: "VisualizationConfig") -> tuple[PCA, np.ndarray]:
    x_train = np.asarray(data["X_train"])
    n_components = min(config.surface.pca_n_components, x_train.shape[1])
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(x_train)
    return pca, x_pca


def _format_loading(name: str, weight: float) -> str:
    sign = "+" if weight >= 0 else "-"
    return f"{sign}{abs(weight):.2f}·{name}"


def _pc_loading_text(pca: PCA, data: dict[str, Any], component_index: int, top_k: int) -> str:
    names = get_display_names(data)
    components = pca.components_[component_index]
    order = np.argsort(np.abs(components))[::-1][:top_k]
    parts = [_format_loading(names[idx], float(components[idx])) for idx in order]
    explained = float(pca.explained_variance_ratio_[component_index]) * 100.0
    return f"PC{component_index + 1} ({explained:.1f}% var) ≈ " + " ".join(parts)


def _sample_pca_coordinates(pca: PCA, x_pca: np.ndarray, rng: np.random.Generator, sweep_values: np.ndarray) -> np.ndarray:
    coordinates = np.repeat(np.median(x_pca, axis=0)[None, :], sweep_values.size, axis=0)
    coordinates[:, 0] = sweep_values
    if coordinates.shape[1] > 1:
        coordinates[:, 1] = np.median(x_pca[:, 1])
    for component in range(2, coordinates.shape[1]):
        low, high = np.percentile(x_pca[:, component], [5, 95])
        coordinates[:, component] = rng.uniform(low, high, size=sweep_values.size)
    return pca.inverse_transform(coordinates)


def _plot_pca_1d(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig", mode: str, pca: PCA, x_pca: np.ndarray) -> None:
    output_name, output_unit = get_output_display(data)
    rng = np.random.default_rng(config.random_seed)
    sweep_values = np.linspace(float(x_pca[:, 0].min()), float(x_pca[:, 0].max()), config.surface.n_grid_points)
    samples = []
    for _ in range(config.surface.pca_n_other_samples):
        x_phys = _sample_pca_coordinates(pca, x_pca, rng, sweep_values)
        samples.append(_predict_grid(model, x_phys, data))
    predictions = np.vstack(samples) if samples else _predict_grid(model, _sample_pca_coordinates(pca, x_pca, rng, sweep_values), data)[None, :]
    figure, ax = plt.subplots(figsize=(float(FIG_STYLE["figure_width"]), float(FIG_STYLE["figure_height"])))
    mean_prediction = predictions.mean(axis=0)
    ax.plot(sweep_values, mean_prediction, color=PALETTE["test"], linewidth=float(FIG_STYLE["line_width"]), label="mean")
    ax.fill_between(sweep_values, predictions.min(axis=0), predictions.max(axis=0), color=PALETTE["band"], alpha=float(FIG_STYLE["band_alpha"]), label="band (PC3+ jitter)")
    loading_text = _pc_loading_text(pca, data, 0, config.surface.pca_loadings_top_k)
    ax.text(
        0.02,
        0.98,
        loading_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=int(FIG_STYLE["annotation_size"]),
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
    )
    apply_style(ax, f"PCA 1D Slice — {model_name}", "PC1", axis_label(output_name, output_unit))
    ax.legend()
    finalize_figure(figure, f"surface_pca_1d_{model_name}.png", config, mode)


def _plot_pca_2d(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig", mode: str, pca: PCA, x_pca: np.ndarray) -> None:
    output_name, output_unit = get_output_display(data)
    sweep_a = np.linspace(float(x_pca[:, 0].min()), float(x_pca[:, 0].max()), config.surface.n_grid_points)
    sweep_b = np.linspace(float(x_pca[:, 1].min()), float(x_pca[:, 1].max()), config.surface.n_grid_points)
    grid_a, grid_b = np.meshgrid(sweep_a, sweep_b)
    coordinates = np.repeat(np.median(x_pca, axis=0)[None, :], grid_a.size, axis=0)
    coordinates[:, 0] = grid_a.ravel()
    coordinates[:, 1] = grid_b.ravel()
    predictions = _predict_grid(model, pca.inverse_transform(coordinates), data).reshape(grid_a.shape)
    fig, ax = plt.subplots(figsize=(float(FIG_STYLE["figure_width"]), float(FIG_STYLE["figure_height"])))
    contour = ax.contourf(grid_a, grid_b, predictions, levels=int(FIG_STYLE["contour_levels"]), cmap="viridis")
    lines = ax.contour(grid_a, grid_b, predictions, levels=int(FIG_STYLE["contour_line_levels"]), colors="black", linewidths=0.6)
    ax.clabel(lines, inline=True, fontsize=int(FIG_STYLE["tick_size"]))
    fig.colorbar(contour, ax=ax, label=axis_label(output_name, output_unit))
    top_k = config.surface.pca_loadings_top_k
    annotation = _pc_loading_text(pca, data, 0, top_k) + "\n" + _pc_loading_text(pca, data, 1, top_k)
    apply_style(ax, f"PCA 2D Surface — {model_name}", "PC1", "PC2")
    ax.text(
        0.02,
        -0.22,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=int(FIG_STYLE["annotation_size"]),
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": PALETTE["band"], "alpha": 0.7},
    )
    finalize_figure(fig, f"surface_pca_2d_{model_name}.png", config, mode)


def _plot_pca_3d(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig", mode: str, pca: PCA, x_pca: np.ndarray) -> None:
    if x_pca.shape[1] < 3:
        raise ValueError("PCA 3D surfaces require at least three principal components")

    output_name, output_unit = get_output_display(data)
    sweep_a = np.linspace(float(x_pca[:, 0].min()), float(x_pca[:, 0].max()), config.surface.n_grid_points)
    sweep_b = np.linspace(float(x_pca[:, 1].min()), float(x_pca[:, 1].max()), config.surface.n_grid_points)
    grid_a, grid_b = np.meshgrid(sweep_a, sweep_b)
    base_coordinates = np.repeat(np.median(x_pca, axis=0)[None, :], grid_a.size, axis=0)
    base_coordinates[:, 0] = grid_a.ravel()
    base_coordinates[:, 1] = grid_b.ravel()
    rng = np.random.default_rng(config.random_seed)
    surfaces = []
    for _ in range(config.surface.pca_n_other_samples):
        sample_coordinates = base_coordinates.copy()
        for component in range(2, sample_coordinates.shape[1]):
            low, high = np.percentile(x_pca[:, component], [5, 95])
            sample_coordinates[:, component] = rng.uniform(low, high, size=sample_coordinates.shape[0])
        surfaces.append(_predict_grid(model, pca.inverse_transform(sample_coordinates), data).reshape(grid_a.shape))
    surface_stack = np.stack(surfaces, axis=0) if surfaces else np.zeros_like(grid_a)[None, ...]
    z_mean = surface_stack.mean(axis=0)
    z_min = surface_stack.min(axis=0)
    z_max = surface_stack.max(axis=0)
    fig = plt.figure(figsize=(float(FIG_STYLE["surface_width"]), float(FIG_STYLE["surface_height"])))
    ax = fig.add_subplot(projection="3d")
    main_surface = ax.plot_surface(grid_a, grid_b, z_mean, cmap="plasma", alpha=0.9, linewidth=0, antialiased=True)
    ax.plot_surface(grid_a, grid_b, z_min, color=PALETTE["band"], alpha=0.15, linewidth=0)
    ax.plot_surface(grid_a, grid_b, z_max, color=PALETTE["band"], alpha=0.15, linewidth=0)
    fig.colorbar(main_surface, ax=ax, shrink=0.65, pad=0.1, label=axis_label(output_name, output_unit))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel(axis_label(output_name, output_unit))
    ax.set_title(f"PCA 3D Surface — {model_name}", fontsize=int(FIG_STYLE["title_size"]), fontweight="bold")
    top_k = config.surface.pca_loadings_top_k
    annotation = _pc_loading_text(pca, data, 0, top_k) + "\n" + _pc_loading_text(pca, data, 1, top_k)
    fig.text(0.02, 0.02, annotation, fontsize=int(FIG_STYLE["annotation_size"]), bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": PALETTE["band"], "alpha": 0.7})
    finalize_figure(fig, f"surface_pca_3d_{model_name}.png", config, mode)


def run_surface(model: Any, model_name: str, data: dict[str, Any], config: "VisualizationConfig", mode: str) -> None:
    """Generate feature-space and PCA surface visualizations for one model."""

    locked = _locked_defaults(data, config)
    slice_indices = _resolve_slice_indices(model, data, config, locked)
    for index in slice_indices:
        _plot_1d_slice(model, model_name, data, config, mode, index, locked)
    feature_a, feature_b = _resolve_heatmap_pair(data, config, slice_indices)
    _plot_2d_heatmap(model, model_name, data, config, mode, feature_a, feature_b, locked)
    pca, x_pca = _fit_pca(data, config)
    _plot_pca_1d(model, model_name, data, config, mode, pca, x_pca)
    _plot_pca_2d(model, model_name, data, config, mode, pca, x_pca)
    _plot_pca_3d(model, model_name, data, config, mode, pca, x_pca)
