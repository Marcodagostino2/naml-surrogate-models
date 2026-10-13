"""
Initialization module for src subpackage.
"""

__version__ = "1.0.0"
__author__ = "Riccardo Bellotti, Marco D'Agostino"

from src.data import DataBundle, load_and_preprocess_data
from src.inference import Predictor
from src.models import GPRegressionModel, MLPRegressor, MLPGPModel
from src.training import GPTrainer, MLPTrainer, MLPGPTrainer, compute_all_metrics
from src.utils import (
    load_json,
    load_pickle,
    load_yaml,
    resolve_device,
    save_json,
    save_pickle,
    save_yaml,
    set_global_seed,
    setup_logger,
)

__all__ = [
    "DataBundle",
    "load_and_preprocess_data",
    "MLPRegressor",
    "GPRegressionModel",
    "MLPGPModel",
    "MLPTrainer",
    "GPTrainer",
    "MLPGPTrainer",
    "compute_all_metrics",
    "Predictor",
    "load_yaml",
    "save_yaml",
    "load_json",
    "save_json",
    "load_pickle",
    "save_pickle",
    "set_global_seed",
    "resolve_device",
    "setup_logger",
]
