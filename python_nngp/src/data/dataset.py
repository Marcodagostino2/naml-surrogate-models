from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class TabularRegressionDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    input_dim: int
    output_dim: int
    input_columns: List[str]
    target_columns: List[str]
    feature_scaler: Optional[StandardScaler]
    target_scaler: Optional[StandardScaler]
    feature_bounds: Dict[str, tuple[float, float]]
    target_bounds: Dict[str, tuple[float, float]]
    train_features_raw: np.ndarray
    train_targets_raw: np.ndarray
    val_features_raw: np.ndarray
    val_targets_raw: np.ndarray
    test_features_raw: np.ndarray
    test_targets_raw: np.ndarray


def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {total:.6f}")


def _extract_arrays(
    frame: pd.DataFrame,
    input_columns: List[str],
    target_columns: List[str],
) -> tuple[np.ndarray, np.ndarray]:
    missing_inputs = [col for col in input_columns if col not in frame.columns]
    missing_targets = [col for col in target_columns if col not in frame.columns]
    if missing_inputs or missing_targets:
        missing = missing_inputs + missing_targets
        raise KeyError(f"Missing columns in dataset: {missing}")

    x = frame.loc[:, input_columns].to_numpy(dtype=np.float32)
    y = frame.loc[:, target_columns].to_numpy(dtype=np.float32)
    return x, y


def build_data_bundle(dataset_cfg: Dict, project_root: Path) -> DataBundle:
    csv_path = project_root / dataset_cfg["csv_path"]
    input_columns = list(dataset_cfg["input_columns"])
    target_columns = list(dataset_cfg["target_columns"])

    _validate_split_ratios(
        dataset_cfg["train_ratio"],
        dataset_cfg["val_ratio"],
        dataset_cfg["test_ratio"],
    )

    frame = pd.read_csv(csv_path)
    if dataset_cfg.get("dropna", True):
        frame = frame.dropna(subset=input_columns + target_columns)

    x_all, y_all = _extract_arrays(frame, input_columns, target_columns)

    seed = int(dataset_cfg.get("random_seed", 42))
    val_ratio = float(dataset_cfg["val_ratio"])
    test_ratio = float(dataset_cfg["test_ratio"])

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x_all,
        y_all,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        shuffle=True,
    )

    test_size_in_tmp = test_ratio / (val_ratio + test_ratio)
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp,
        y_tmp,
        test_size=test_size_in_tmp,
        random_state=seed,
        shuffle=True,
    )

    # Preserve physical-domain splits before optional normalization.
    x_train_raw = x_train.copy()
    y_train_raw = y_train.copy()
    x_val_raw = x_val.copy()
    y_val_raw = y_val.copy()
    x_test_raw = x_test.copy()
    y_test_raw = y_test.copy()

    feature_scaler: Optional[StandardScaler] = None
    target_scaler: Optional[StandardScaler] = None

    feature_bounds = {
        column_name: (float(x_train[:, idx].min()), float(x_train[:, idx].max()))
        for idx, column_name in enumerate(input_columns)
    }
    target_bounds = {
        column_name: (float(y_train[:, idx].min()), float(y_train[:, idx].max()))
        for idx, column_name in enumerate(target_columns)
    }

    if dataset_cfg.get("normalize_inputs", True):
        feature_scaler = StandardScaler()
        x_train = feature_scaler.fit_transform(x_train)
        x_val = feature_scaler.transform(x_val)
        x_test = feature_scaler.transform(x_test)

    if dataset_cfg.get("normalize_targets", True):
        target_scaler = StandardScaler()
        y_train = target_scaler.fit_transform(y_train)
        y_val = target_scaler.transform(y_val)
        y_test = target_scaler.transform(y_test)

    batch_size = int(dataset_cfg.get("batch_size", 64))
    num_workers = int(dataset_cfg.get("num_workers", 0))
    pin_memory = bool(dataset_cfg.get("pin_memory", False))

    train_loader = DataLoader(
        TabularRegressionDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TabularRegressionDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        TabularRegressionDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_dim=x_train.shape[1],
        output_dim=y_train.shape[1],
        input_columns=input_columns,
        target_columns=target_columns,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_bounds=feature_bounds,
        target_bounds=target_bounds,
        train_features_raw=x_train_raw,
        train_targets_raw=y_train_raw,
        val_features_raw=x_val_raw,
        val_targets_raw=y_val_raw,
        test_features_raw=x_test_raw,
        test_targets_raw=y_test_raw,
    )
