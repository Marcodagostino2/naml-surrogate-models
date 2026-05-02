"""
Initialization module for src.data subpackage.
"""

from pathlib import Path
from typing import Any, Dict

from src.data.dataset import DataBundle, load_and_preprocess_data


def load_data_from_config(
	project_root: Path,
	dataset_cfg: Dict[str, Any],
	global_seed: int,
	precision: str = "float32",
) -> DataBundle:
	"""
	Load and preprocess a dataset from a dataset configuration dictionary.

	Parameters
	----------
	project_root : Path
		Repository root directory.
	dataset_cfg : Dict[str, Any]
		Selected dataset configuration.
	global_seed : int
		Random seed used for the train/val/test split.
	precision : str, optional
		Default PyTorch dtype name. Default: ``"float32"``.

	Returns
	-------
	DataBundle
		Loaded and preprocessed dataset bundle.
	"""
	return load_and_preprocess_data(
		csv_path=project_root / dataset_cfg["csv_path"],
		input_columns=dataset_cfg["input_columns"],
		target_columns=dataset_cfg["target_columns"],
		dropna=bool(dataset_cfg.get("dropna", True)),
		normalize_inputs=bool(dataset_cfg.get("normalize_inputs", True)),
		normalize_targets=bool(dataset_cfg.get("normalize_targets", True)),
		train_ratio=float(dataset_cfg.get("train_ratio", 0.70)),
		val_ratio=float(dataset_cfg.get("val_ratio", 0.15)),
		test_ratio=float(dataset_cfg.get("test_ratio", 0.15)),
		random_seed=global_seed,
		default_type=precision,
		input_display_names=dataset_cfg.get("input_display_names"),
		input_units=dataset_cfg.get("input_units"),
		output_display_name=dataset_cfg.get("output_display_name"),
		output_unit=dataset_cfg.get("output_unit"),
	)


__all__ = ["DataBundle", "load_and_preprocess_data", "load_data_from_config"]
