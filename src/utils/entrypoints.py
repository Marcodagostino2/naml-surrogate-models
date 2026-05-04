"""Shared bootstrapping helpers for top-level entry points.

The scripts under the repository root all need the same first step: load the
configured run set, then initialize global randomness and PyTorch dtype from
``main_config.yaml``. Keeping that logic in one place avoids drift between
``main.py`` and ``optimize.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .config import load_run_configs, set_global_seed, setup_torch_dtype


def load_and_initialize_run_configs(
    config_dir: Path,
    model_name: str,
    include_optimize: bool = False,
) -> Dict[str, Any]:
    """Load run configs and initialize global seed/dtype from main config."""

    loaded_configs = load_run_configs(config_dir, model_name, include_optimize=include_optimize)
    main_cfg = loaded_configs["main_cfg"]

    set_global_seed(int(main_cfg.get("global_seed", 42)))
    setup_torch_dtype(str(main_cfg.get("precision", "float32")))

    return loaded_configs
