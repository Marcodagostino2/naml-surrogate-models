"""
Utility functions for YAML config loading, random seed management, and logging.

This module provides helper functions for:
- Loading YAML configuration files
- Setting global random seeds (NumPy, PyTorch, Python)
- Device resolution (CPU/GPU)
- Logging setup
"""

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
    """
    Load the configuration set for a model run.

    Parameters
    ----------
    config_dir : Path
        Base configuration directory.
    model_name : str
        Model identifier (``"mlp"``, ``"gp"``, or ``"mlpgp"``).
    include_optimize : bool, optional
        Whether to include the optimization config. Default: ``False``.

    Returns
    -------
    Dict[str, Any]
        Loaded configs. Keys: ``main_cfg``, ``dataset_cfg``, ``dataset_name``,
        ``model_cfg``, ``training_cfg``, ``viz_cfg`` and optionally ``opt_cfg``.
    """
    main_cfg = load_yaml(config_dir / "main_config.yaml")
    all_datasets = load_yaml(config_dir / "dataset" / "dataset_config.yaml")
    common_training_cfg_path = config_dir / "training" / "common_training_config.yaml"
    common_training_cfg = load_yaml(common_training_cfg_path) if common_training_cfg_path.exists() else {}
    dataset_name = str(main_cfg.get("dataset", "airfoil_self_noise"))
    datasets = all_datasets.get("datasets", {})

    if dataset_name not in datasets:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available: {list(datasets.keys())}"
        )

    model_training_cfg = load_yaml(config_dir / "training" / f"training_{model_name}_config.yaml")

    # Merge shared defaults with model-specific overrides so scheduler settings
    # live in a single source of truth.
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
        optimize_cfg_path = config_dir / "optimization" / f"optimize_{model_name}_config.yaml"
        result["opt_cfg"] = load_yaml(optimize_cfg_path)

    return result


def load_yaml(config_path: Path | str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : Path or str
        Absolute or relative path to the YAML file.

    Returns
    -------
    Dict[str, Any]
        Parsed YAML configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    yaml.YAMLError
        If the YAML is malformed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config if config is not None else {}


def save_yaml(config: Dict[str, Any], output_path: Path | str) -> None:
    """
    Save a dictionary to a YAML file.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to save.
    output_path : Path or str
        Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across NumPy, PyTorch, and Python.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_str: str | None = None) -> torch.device:
    """
    Resolve device specification to a PyTorch device.

    Parameters
    ----------
    device_str : str or None, optional
        Device specification: "cuda", "cpu", "auto", or None.
        - "auto": uses GPU if available, otherwise CPU
        - "cuda": explicitly use GPU
        - "cpu": explicitly use CPU
        - None: defaults to "auto"

    Returns
    -------
    torch.device
        Resolved PyTorch device.
    """
    if device_str is None or device_str.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str.lower() == "cuda":
        if not torch.cuda.is_available():
            logging.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    elif device_str.lower() == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Unknown device: {device_str}")


def setup_torch_dtype(precision: str | None = None) -> torch.dtype:
    """
    Set PyTorch's default dtype globally based on precision specification.

    Parameters
    ----------
    precision : str or None, optional
        Precision specification: "float32", "float64", or None.
        - "float32": sets torch.float32 (default)
        - "float64": sets torch.float64 (double precision)
        - None: defaults to "float32"

    Returns
    -------
    torch.dtype
        The dtype that was set as default.

    Raises
    ------
    ValueError
        If precision is not recognized.
    """
    if precision is None or precision.lower() == "float32":
        torch.set_default_dtype(torch.float32)
        return torch.float32
    elif precision.lower() == "float64":
        torch.set_default_dtype(torch.float64)
        return torch.float64
    else:
        raise ValueError(
            f"Unknown precision: {precision}. "
            f"Supported values: 'float32', 'float64', or None."
        )


def setup_logger(name: str, log_file: Path | str | None = None, level=logging.INFO) -> logging.Logger:
    """
    Configure a logger with console and optional file output.

    Parameters
    ----------
    name : str
        Logger name.
    log_file : Path, str, or None, optional
        Path to log file. If None, only console output is used.
    level
        Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_json(data: Dict[str, Any], output_path: Path | str) -> None:
    """
    Save a dictionary to a JSON file.

    Parameters
    ----------
    data : Dict[str, Any]
        Data to save.
    output_path : Path or str
        Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(file_path: Path | str) -> Dict[str, Any]:
    """
    Load a dictionary from a JSON file.

    Parameters
    ----------
    file_path : Path or str
        Path to JSON file.

    Returns
    -------
    Dict[str, Any]
        Loaded data.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def save_pickle(data: Any, output_path: Path | str) -> None:
    """
    Serialize and save data to a pickle file.

    Parameters
    ----------
    data : Any
        Data to save.
    output_path : Path or str
        Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(file_path: Path | str) -> Any:
    """
    Load data from a pickle file.

    Parameters
    ----------
    file_path : Path or str
        Path to pickle file.

    Returns
    -------
    Any
        Loaded data.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)
