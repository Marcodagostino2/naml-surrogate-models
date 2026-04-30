"""Centralized output path resolution and naming constants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal


OUTPUTS_DIRNAME = "outputs"
CONFIG_DIRNAME = "config"
PLOTS_DIRNAME = "plots"


class RunType(str, Enum):
    """Supported output run-type folders."""

    SINGLE_RUN = "single_run"
    OPTIMIZATION = "optimization"
    CONVERGENCE_ANALYSIS = "convergence_analysis"


class ModelName(str, Enum):
    """Supported model-name folders."""

    GP = "gp"
    MLP = "mlp"
    MLPGP = "mlpgp"


RunTypeLike = Literal[
    RunType.SINGLE_RUN,
    RunType.OPTIMIZATION,
    RunType.CONVERGENCE_ANALYSIS,
    "single_run",
    "optimization",
    "convergence_analysis",
]
ModelNameLike = Literal[ModelName.GP, ModelName.MLP, ModelName.MLPGP, "gp", "mlp", "mlpgp"]


@dataclass(frozen=True)
class OutputPathResolver:
    """Resolve all output paths for a dataset, run type, and model.

    All output path arithmetic is centralized in this class to keep callers
    explicit and free from duplicated folder-building logic.
    """

    dataset_name: str
    run_type: RunTypeLike
    model_name: ModelNameLike
    output_base_dir: Path = Path(OUTPUTS_DIRNAME)

    def __post_init__(self) -> None:
        if not str(self.dataset_name).strip():
            raise ValueError("dataset_name must be a non-empty string")
        run_type = self._run_type_value
        model_name = self._model_name_value
        if run_type not in {item.value for item in RunType}:
            raise ValueError(f"Unsupported run_type: {self.run_type}")
        if model_name not in {item.value for item in ModelName}:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

    @property
    def _run_type_value(self) -> str:
        value = self.run_type.value if isinstance(self.run_type, RunType) else str(self.run_type)
        return value.lower()

    @property
    def _model_name_value(self) -> str:
        value = self.model_name.value if isinstance(self.model_name, ModelName) else str(self.model_name)
        return value.lower()

    @property
    def root(self) -> Path:
        """Model-level root output directory."""
        return self.output_base_dir / self.dataset_name / self._run_type_value / self._model_name_value

    @property
    def config_dir(self) -> Path:
        """Directory where run/optimization configs are stored."""
        return self.root / CONFIG_DIRNAME

    @property
    def plots_dir(self) -> Path:
        """Directory where plots are stored."""
        return self.root / PLOTS_DIRNAME

    @property
    def checkpoint(self) -> Path:
        """Path of the model checkpoint file."""
        return self.root / "checkpoint.pt"

    @property
    def summary_csv(self) -> Path:
        """Path of the summary CSV file."""
        return self.root / "summary.csv"

    @property
    def best_params(self) -> Path:
        """Path of best hyperparameters JSON file."""
        return self.root / "best_params.json"

    @property
    def all_trials_csv(self) -> Path:
        """Path of the all-trials CSV file."""
        return self.root / "all_trials.csv"
