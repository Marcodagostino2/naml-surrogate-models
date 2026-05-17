"""YAML/JSON/pickle I/O, run-config loading, RNG seeding, device + dtype setup, logging."""

from __future__ import annotations

import json
import logging
import pickle
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_run_configs(
    config_dir: Path,
    model_name: str,
    include_optimize: bool = False,
) -> Dict[str, Any]:
    """Load main, dataset, model, training, and visualization configs for a run.

    The training config is built by overlaying the model-specific file on top
    of ``training_common_config.yaml`` (when present), so shared defaults like
    the LR scheduler live in one place.
    """
    main_cfg = load_yaml(config_dir / "main_config.yaml")
    all_datasets = load_yaml(config_dir / "dataset" / "dataset_config.yaml")
    common_training_cfg_path = config_dir / "training" / "training_common_config.yaml"
    common_training_cfg = load_yaml(common_training_cfg_path) if common_training_cfg_path.exists() else {}
    dataset_name = str(main_cfg.get("dataset", "airfoil_self_noise"))
    datasets = all_datasets.get("datasets", {})

    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(datasets.keys())}")

    model_training_cfg = load_yaml(config_dir / "training" / f"training_{model_name}_config.yaml")
    training_cfg = {**common_training_cfg, **model_training_cfg}

    result: Dict[str, Any] = {
        "main_cfg": main_cfg,
        "dataset_cfg": datasets[dataset_name],
        "dataset_name": dataset_name,
        "model_cfg": load_yaml(config_dir / "model" / f"{model_name}_config.yaml"),
        "training_cfg": training_cfg,
        "viz_cfg": load_yaml(config_dir / "visualization" / "visualization_config.yaml"),
    }
    if include_optimize:
        result["opt_cfg"] = load_yaml(config_dir / "optimization" / f"optimize_{model_name}_config.yaml")
    return result


def load_yaml(config_path: Path | str) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config if config is not None else {}


def save_yaml(config: Dict[str, Any], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA); enable cudnn determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_str: str | None = None) -> torch.device:
    """Accepts ``"auto"``, ``"cuda"``, ``"cpu"``, or ``None`` (treated as ``"auto"``)."""
    spec = (device_str or "auto").lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            logging.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if spec == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {device_str}")


def setup_torch_dtype(precision: str | None = None) -> torch.dtype:
    """Set PyTorch's default dtype. ``precision`` is ``"float32"`` (default) or ``"float64"``."""
    spec = (precision or "float32").lower()
    if spec == "float32":
        torch.set_default_dtype(torch.float32)
        return torch.float32
    if spec == "float64":
        torch.set_default_dtype(torch.float64)
        return torch.float64
    raise ValueError(f"Unknown precision: {precision}. Supported: 'float32', 'float64'.")


def setup_logger(name: str, log_file: Path | str | None = None, level=logging.INFO) -> logging.Logger:
    """Console (always) + file (optional) handlers; module-named loggers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def save_json(data: Dict[str, Any], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(file_path: Path | str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)


def save_pickle(data: Any, output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(file_path: Path | str) -> Any:
    with open(file_path, "rb") as f:
        return pickle.load(f)
