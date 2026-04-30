"""Configuration objects for the visualization package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class ParityConfig:
    """Configuration for parity and residual plots."""

    enabled: bool = True
    show_residuals: bool = True


@dataclass(slots=True)
class SurfaceConfig:
    """Configuration for feature-space and PCA surface plots."""

    enabled: bool = True
    n_top_features: int = 2
    n_grid_points: int = 50
    pca_n_components: int = 3
    pca_n_other_samples: int = 50


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
class VisualizationConfig:
    """Top-level visualization configuration."""

    save_plots: bool = True
    output_dir: str = "plots/"
    random_seed: int = 42
    parity: ParityConfig = field(default_factory=ParityConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    uncertainty_gp: UncertaintyGPConfig = field(default_factory=UncertaintyGPConfig)
    table: TableConfig = field(default_factory=TableConfig)

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

        parity_enabled = bool(parity_raw.get("enabled", plots.get("parity_plot", True)))
        residuals_enabled = bool(parity_raw.get("show_residuals", plots.get("residual_plot", True)))
        surface_enabled = bool(surface_raw.get("enabled", plots.get("kernel_surface", True)))
        uncertainty_enabled = bool(gp_raw.get("enabled", plots.get("prediction_uncertainty", True)))
        table_enabled = bool(table_raw.get("enabled", raw.get("table_enabled", True)))

        return cls(
            save_plots=bool(raw.get("save_plots", True)),
            output_dir=str(output_dir or raw.get("output_dir", "plots/")),
            random_seed=int(raw.get("random_seed", 42)),
            parity=ParityConfig(enabled=parity_enabled, show_residuals=residuals_enabled),
            surface=SurfaceConfig(
                enabled=surface_enabled,
                n_top_features=int(surface_raw.get("n_top_features", 2)),
                n_grid_points=int(surface_raw.get("n_grid_points", 50)),
                pca_n_components=int(surface_raw.get("pca_n_components", 3)),
                pca_n_other_samples=int(surface_raw.get("pca_n_other_samples", 50)),
            ),
            uncertainty_gp=UncertaintyGPConfig(enabled=uncertainty_enabled),
            table=TableConfig(
                enabled=table_enabled,
                split=str(table_raw.get("split", raw.get("table_split", "test"))),
            ),
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
        if self.table.split != "test":
            raise ValueError("table.split must remain 'test'")
        if not isinstance(self.parity.enabled, bool):
            raise ValueError("parity.enabled must be a boolean")
        if not isinstance(self.parity.show_residuals, bool):
            raise ValueError("parity.show_residuals must be a boolean")
        if not isinstance(self.surface.enabled, bool):
            raise ValueError("surface.enabled must be a boolean")
        if not isinstance(self.uncertainty_gp.enabled, bool):
            raise ValueError("uncertainty_gp.enabled must be a boolean")
        if not isinstance(self.table.enabled, bool):
            raise ValueError("table.enabled must be a boolean")
