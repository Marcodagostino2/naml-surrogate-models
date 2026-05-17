"""Output path resolver: dataset / run-type / model-name folder layout."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Optional


OUTPUTS_DIRNAME = "outputs"
CONFIG_DIRNAME = "config"
PLOTS_DIRNAME = "plots"
RUNS_DIRNAME = "runs"


class RunType(str, Enum):
    SINGLE_RUN = "single_run"
    CONVERGENCE_ANALYSIS = "convergence_analysis"
    TESTING = "testing"


class ModelName(str, Enum):
    GP = "gp"
    MLP = "mlp"
    MLPGP_NS = "mlpgp_ns"
    MLPGP_DKL = "mlpgp_dkl"


RunTypeLike = Literal[
    RunType.SINGLE_RUN,
    RunType.CONVERGENCE_ANALYSIS,
    RunType.TESTING,
    "single_run",
    "convergence_analysis",
    "testing",
]
ModelNameLike = Literal[
    ModelName.GP,
    ModelName.MLP,
    ModelName.MLPGP_NS,
    ModelName.MLPGP_DKL,
    "gp",
    "mlp",
    "mlpgp_ns",
    "mlpgp_dkl",
]


@dataclass(frozen=True)
class OutputPathResolver:
    """Resolve all output paths for a (dataset, run_type, model) tuple."""

    dataset_name: str
    run_type: RunTypeLike
    model_name: ModelNameLike
    output_base_dir: Path = Path(OUTPUTS_DIRNAME)
    run_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not str(self.dataset_name).strip():
            raise ValueError("dataset_name must be a non-empty string")
        run_type = self._run_type_value
        model_name = self._model_name_value
        if run_type not in {item.value for item in RunType}:
            raise ValueError(f"Unsupported run_type: {self.run_type}")
        if model_name not in {item.value for item in ModelName}:
            raise ValueError(f"Unsupported model_name: {self.model_name}")
        if self.run_id is not None and not str(self.run_id).strip():
            raise ValueError("run_id must be a non-empty string when provided")

    @property
    def _run_type_value(self) -> str:
        value = self.run_type.value if isinstance(self.run_type, RunType) else str(self.run_type)
        return value.lower()

    @property
    def _model_name_value(self) -> str:
        value = self.model_name.value if isinstance(self.model_name, ModelName) else str(self.model_name)
        return value.lower()

    @property
    def model_root(self) -> Path:
        return self.output_base_dir / self.dataset_name / self._run_type_value / self._model_name_value

    @property
    def runs_root(self) -> Path:
        return self.model_root / RUNS_DIRNAME

    @property
    def root(self) -> Path:
        if self.run_id is None:
            return self.model_root
        return self.runs_root / str(self.run_id)

    @property
    def config_dir(self) -> Path:
        return self.root / CONFIG_DIRNAME

    @property
    def plots_dir(self) -> Path:
        return self.root / PLOTS_DIRNAME

    @property
    def checkpoint(self) -> Path:
        return self.root / "checkpoint.pt"

    @property
    def summary_csv(self) -> Path:
        return self.root / "summary.csv"

    @property
    def best_params(self) -> Path:
        return self.root / "best_params.json"

    @property
    def all_trials_csv(self) -> Path:
        return self.root / "all_trials.csv"


def make_run_id(prefix: Optional[str] = None) -> str:
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix is None:
        return base
    clean = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(prefix).strip())
    return f"{clean}_{base}" if clean else base
