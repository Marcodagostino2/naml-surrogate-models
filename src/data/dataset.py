"""
Dataset module for loading, preprocessing, and creating PyTorch DataLoaders.

Provides:
- Data loading from CSV files
- Normalization (input/output features)
- Train/val/test splitting
- PyTorch DataLoaders for batch training
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils import setup_logger

logger = setup_logger(__name__)

# ============================================================================
# Module Constants: Data Preprocessing Configuration
# ============================================================================

# Default train/val/test split ratios. Must sum to 1.0.
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15

# Epsilon for numerical stability during normalization.
# Used to prevent division by zero when std is very small.
NORMALIZATION_EPSILON = 1e-8

# Tolerance for validating that split ratios sum to 1.0.
RATIO_VALIDATION_TOLERANCE = 1e-6


class DataBundle:
    """
    Container for train/val/test datasets and their associated statistics.

    Attributes
    ----------
    X_train : np.ndarray
        Training input features (n_train, n_features).
    y_train : np.ndarray
        Training targets (n_train, n_targets).
    X_val : np.ndarray
        Validation input features (n_val, n_features).
    y_val : np.ndarray
        Validation targets (n_val, n_targets).
    X_test : np.ndarray
        Test input features (n_test, n_features).
    y_test : np.ndarray
        Test targets (n_test, n_targets).
    input_scaler_mean : np.ndarray
        Mean of training inputs for normalization (n_features,).
    input_scaler_std : np.ndarray
        Std of training inputs for normalization (n_features,).
    target_scaler_mean : np.ndarray
        Mean of training targets for normalization (n_targets,).
    target_scaler_std : np.ndarray
        Std of training targets for normalization (n_targets,).
    feature_names : list[str]
        Names of input features.
    target_names : list[str]
        Names of target outputs.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        input_scaler_mean: np.ndarray,
        input_scaler_std: np.ndarray,
        target_scaler_mean: np.ndarray,
        target_scaler_std: np.ndarray,
        feature_names: list[str],
        target_names: list[str],
        input_display_names: list[str] | None = None,
        input_units: list[str] | None = None,
        output_display_name: str | None = None,
        output_unit: str | None = None,
    ) -> None:
        """
        Initialize DataBundle.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training data.
        X_val, y_val : np.ndarray
            Validation data.
        X_test, y_test : np.ndarray
            Test data.
        input_scaler_mean, input_scaler_std : np.ndarray
            Input normalization parameters.
        target_scaler_mean, target_scaler_std : np.ndarray
            Target normalization parameters.
        feature_names : list[str]
            Names of input features.
        target_names : list[str]
            Names of target outputs.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.input_scaler_mean = input_scaler_mean
        self.input_scaler_std = input_scaler_std
        self.target_scaler_mean = target_scaler_mean
        self.target_scaler_std = target_scaler_std
        self.feature_names = feature_names
        self.target_names = target_names
        self.input_display_names = list(input_display_names) if input_display_names else list(feature_names)
        self.input_units = list(input_units) if input_units else [""] * len(feature_names)
        self.output_display_name = str(output_display_name) if output_display_name else (target_names[0] if target_names else "")
        self.output_unit = str(output_unit) if output_unit else ""

    def get_train_loader(self, batch_size: int = 32, num_workers: int = 0, pin_memory: bool = False) -> DataLoader:
        """Create a DataLoader for training data."""
        X_train_t = torch.from_numpy(self.X_train).to(torch.get_default_dtype())
        y_train_t = torch.from_numpy(self.y_train).to(torch.get_default_dtype())
        dataset = TensorDataset(X_train_t, y_train_t)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    def get_val_loader(self, batch_size: int = 32, num_workers: int = 0, pin_memory: bool = False) -> DataLoader:
        """Create a DataLoader for validation data."""
        X_val_t = torch.from_numpy(self.X_val).to(torch.get_default_dtype())
        y_val_t = torch.from_numpy(self.y_val).to(torch.get_default_dtype())
        dataset = TensorDataset(X_val_t, y_val_t)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    def get_test_loader(self, batch_size: int = 32, num_workers: int = 0, pin_memory: bool = False) -> DataLoader:
        """Create a DataLoader for test data."""
        X_test_t = torch.from_numpy(self.X_test).to(torch.get_default_dtype())
        y_test_t = torch.from_numpy(self.y_test).to(torch.get_default_dtype())
        dataset = TensorDataset(X_test_t, y_test_t)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    @property
    def n_inputs(self) -> int:
        """Number of input features."""
        return self.X_train.shape[1]

    @property
    def n_outputs(self) -> int:
        """Number of output targets."""
        return self.y_train.shape[1]


def load_and_preprocess_data(
    csv_path: Path | str,
    input_columns: list[str],
    target_columns: list[str],
    dropna: bool = True,
    normalize_inputs: bool = True,
    normalize_targets: bool = True,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_seed: int = 42,
    default_type: str = "float32",
    input_display_names: list[str] | None = None,
    input_units: list[str] | None = None,
    output_display_name: str | None = None,
    output_unit: str | None = None,
) -> DataBundle:
    """
    Load data from CSV, preprocess, normalize, and split into train/val/test.

    Parameters
    ----------
    csv_path : Path or str
        Path to CSV file.
    input_columns : list[str]
        List of input feature column names.
    target_columns : list[str]
        List of target column names.
    dropna : bool, optional
        Whether to drop rows with NaN values. Default: True.
    normalize_inputs : bool, optional
        Whether to normalize inputs (zero-mean, unit-std). Default: True.
    normalize_targets : bool, optional
        Whether to normalize targets. Default: True.
    train_ratio : float, optional
        Fraction of data for training. Default: 0.70.
    val_ratio : float, optional
        Fraction of data for validation. Default: 0.15.
    test_ratio : float, optional
        Fraction of data for testing. Default: 0.15.
    random_seed : int, optional
        Random seed for reproducibility. Default: 42.

    Returns
    -------
    DataBundle
        Container with preprocessed datasets and normalization statistics.

    Raises
    ------
    AssertionError
        If train_ratio + val_ratio + test_ratio != 1.0.
    ValueError
        If specified columns are missing from CSV.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < RATIO_VALIDATION_TOLERANCE, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    csv_path = Path(csv_path)
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Validation
    missing_inputs = set(input_columns) - set(df.columns)
    missing_targets = set(target_columns) - set(df.columns)
    if missing_inputs or missing_targets:
        raise ValueError(f"Missing columns: inputs={missing_inputs}, targets={missing_targets}")

    # Extract features and targets
    X = df[input_columns].to_numpy(default_type)
    y = df[target_columns].to_numpy(default_type)

    logger.info(f"Loaded data shape: X={X.shape}, y={y.shape}")

    # Handle missing values
    if dropna:
        before = X.shape[0]
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
        X = X[valid_mask]
        y = y[valid_mask]
        if X.shape[0] != before:
            logger.info(f"Dropped {before - X.shape[0]} rows containing NaN")

    if normalize_inputs:
        input_mean = X.mean(axis=0, keepdims=False)
        input_std = X.std(axis=0, keepdims=False)
        input_std = np.where(input_std < NORMALIZATION_EPSILON, 1.0, input_std)
        X = (X - input_mean) / input_std
    else:
        input_mean = np.zeros(X.shape[1])
        input_std = np.ones(X.shape[1])

    if normalize_targets:
        target_mean = y.mean(axis=0, keepdims=False)
        target_std = y.std(axis=0, keepdims=False)
        target_std = np.where(target_std < NORMALIZATION_EPSILON, 1.0, target_std)
        y = (y - target_mean) / target_std
    else:
        target_mean = np.zeros(y.shape[1])
        target_std = np.ones(y.shape[1])

    # Split data
    n_samples = X.shape[0]
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    indices = np.arange(n_samples)
    np.random.RandomState(random_seed).shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    return DataBundle(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        input_scaler_mean=input_mean,
        input_scaler_std=input_std,
        target_scaler_mean=target_mean,
        target_scaler_std=target_std,
        feature_names=input_columns,
        target_names=target_columns,
        input_display_names=input_display_names,
        input_units=input_units,
        output_display_name=output_display_name,
        output_unit=output_unit,
    )
