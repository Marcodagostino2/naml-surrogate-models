"""Configuration objects for the visualization package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass(slots=True)
class ParityConfig:
    """Configuration for parity, residual, and MAPE-distribution plots."""

    enabled: bool = True
    show_residuals: bool = True
    show_mape: bool = True


@dataclass(slots=True)
class SurfaceConfig:
    """Configuration for feature-space and PCA surface plots."""

    enabled: bool = True
    n_top_features: int = 2
    n_grid_points: int = 50
    pca_n_components: int = 3
    pca_n_other_samples: int = 50
    pca_loadings_top_k: int = 3


@dataclass(slots=True)
class UncertaintyGPConfig:
    """Configuration for Gaussian-process uncertainty plots."""

    enabled: bool = True


@dataclass(slots=True)
class TableConfig:
    """Configuration for the metrics summary table."""

    enabled: bool = True
    split: str = "test"


@dataclass(slots=True)
class SingleRunConfig:
    """Configuration for single-run plot dimension selection.

    ``slice_dims`` lists input names to use for 1D slice plots; an empty list
    means auto-select by predictive variance. ``heatmap_dim_pair`` is a pair of
    names for the 2D heatmap; an empty list means auto. ``locked_values`` maps
    input name -> value to use for non-plotted dimensions; missing entries fall
    back to the median of the training data.
    """

    slice_dims: List[str] = field(default_factory=list)
    heatmap_dim_pair: List[str] = field(default_factory=list)
    locked_values: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class OptimizationConfig:
    """Placeholder for optimization-mode visualization options."""

    enabled: bool = True


@dataclass(slots=True)
class ConvergenceConfig:
    """Placeholder for convergence-mode visualization options."""

    enabled: bool = True


@dataclass(slots=True)
class VisualizationConfig:
    """Top-level visualization configuration."""

    save_plots: bool = True
    output_dir: str = "plots/"
    random_seed: int = 42
    parity: ParityConfig = field(default_factory=ParityConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    uncertainty_gp: UncertaintyGPConfig = field(default_factory=UncertaintyGPConfig)
    table: TableConfig = field(default_factory=TableConfig)
    single_run: SingleRunConfig = field(default_factory=SingleRunConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None = None,
        *,
        output_dir: str | None = None,
    ) -> "VisualizationConfig":
        """Build a config from a mapping, including legacy flat YAML shapes."""

        raw = dict(mapping or {})
        plots = raw.get("plots", {}) if isinstance(raw.get("plots", {}), Mapping) else {}
        parity_raw = raw.get("parity", {}) if isinstance(raw.get("parity", {}), Mapping) else {}
        surface_raw = raw.get("surface", {}) if isinstance(raw.get("surface", {}), Mapping) else {}
        gp_raw = raw.get("uncertainty_gp", {}) if isinstance(raw.get("uncertainty_gp", {}), Mapping) else {}
        table_raw = raw.get("table", {}) if isinstance(raw.get("table", {}), Mapping) else {}
        single_raw = raw.get("single_run", {}) if isinstance(raw.get("single_run", {}), Mapping) else {}
        opt_raw = raw.get("optimization", {}) if isinstance(raw.get("optimization", {}), Mapping) else {}
        conv_raw = raw.get("convergence", {}) if isinstance(raw.get("convergence", {}), Mapping) else {}

        parity_enabled = bool(parity_raw.get("enabled", plots.get("parity_plot", True)))
        residuals_enabled = bool(parity_raw.get("show_residuals", plots.get("residual_plot", True)))
        mape_enabled = bool(parity_raw.get("show_mape", True))
        surface_enabled = bool(surface_raw.get("enabled", plots.get("kernel_surface", True)))
        uncertainty_enabled = bool(gp_raw.get("enabled", plots.get("prediction_uncertainty", True)))
        table_enabled = bool(table_raw.get("enabled", raw.get("table_enabled", True)))

        locked_raw = single_raw.get("locked_values", {}) if isinstance(single_raw.get("locked_values", {}), Mapping) else {}
        locked_values = {str(k): float(v) for k, v in locked_raw.items()}

        return cls(
            save_plots=bool(raw.get("save_plots", True)),
            output_dir=str(output_dir or raw.get("output_dir", "plots/")),
            random_seed=int(raw.get("random_seed", 42)),
            parity=ParityConfig(
                enabled=parity_enabled,
                show_residuals=residuals_enabled,
                show_mape=mape_enabled,
            ),
            surface=SurfaceConfig(
                enabled=surface_enabled,
                n_top_features=int(surface_raw.get("n_top_features", 2)),
                n_grid_points=int(surface_raw.get("n_grid_points", 50)),
                pca_n_components=int(surface_raw.get("pca_n_components", 3)),
                pca_n_other_samples=int(surface_raw.get("pca_n_other_samples", 50)),
                pca_loadings_top_k=int(surface_raw.get("pca_loadings_top_k", 3)),
            ),
            uncertainty_gp=UncertaintyGPConfig(enabled=uncertainty_enabled),
            table=TableConfig(
                enabled=table_enabled,
                split=str(table_raw.get("split", raw.get("table_split", "test"))),
            ),
            single_run=SingleRunConfig(
                slice_dims=[str(s) for s in single_raw.get("slice_dims", []) or []],
                heatmap_dim_pair=[str(s) for s in single_raw.get("heatmap_dim_pair", []) or []],
                locked_values=locked_values,
            ),
            optimization=OptimizationConfig(enabled=bool(opt_raw.get("enabled", True))),
            convergence=ConvergenceConfig(enabled=bool(conv_raw.get("enabled", True))),
        )

    def validate(self) -> None:
        """Raise ValueError when any configuration field is invalid."""

        if not isinstance(self.save_plots, bool):
            raise ValueError("save_plots must be a boolean")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a non-empty string")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if self.surface.n_top_features < 2:
            raise ValueError("surface.n_top_features must be >= 2")
        if self.surface.n_grid_points < 10:
            raise ValueError("surface.n_grid_points must be >= 10")
        if self.surface.pca_n_components < 2:
            raise ValueError("surface.pca_n_components must be >= 2")
        if self.surface.pca_n_other_samples < 1:
            raise ValueError("surface.pca_n_other_samples must be >= 1")
        if self.surface.pca_loadings_top_k < 1:
            raise ValueError("surface.pca_loadings_top_k must be >= 1")
        if self.table.split != "test":
            raise ValueError("table.split must remain 'test'")
        if not isinstance(self.parity.enabled, bool):
            raise ValueError("parity.enabled must be a boolean")
        if not isinstance(self.parity.show_residuals, bool):
            raise ValueError("parity.show_residuals must be a boolean")
        if not isinstance(self.parity.show_mape, bool):
            raise ValueError("parity.show_mape must be a boolean")
        if not isinstance(self.surface.enabled, bool):
            raise ValueError("surface.enabled must be a boolean")
        if not isinstance(self.uncertainty_gp.enabled, bool):
            raise ValueError("uncertainty_gp.enabled must be a boolean")
        if not isinstance(self.table.enabled, bool):
            raise ValueError("table.enabled must be a boolean")
        if self.single_run.heatmap_dim_pair and len(self.single_run.heatmap_dim_pair) != 2:
            raise ValueError("single_run.heatmap_dim_pair must contain exactly two names when set")
