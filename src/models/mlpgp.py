"""
Hybrid MLP+Kriging model combining feature extraction with non-stationary GP.

The MLP acts as a feature extractor that outputs local hyperparameters
(lengthscale, nugget) which are fed into a custom non-stationary kernel.

This module is under active development. Core components will be integrated
in the final version.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from src.utils import setup_logger

logger = setup_logger(__name__)


class MLPGPHybrid(nn.Module):
    """
    Hybrid MLP+Kriging model (placeholder for future development).

    This class will combine:
    - MLPRegressor as feature extractor
    - Non-stationary GP kernel parameterized by MLP outputs
    - Joint optimization via Exact Marginal Log-Likelihood

    Parameters
    ----------
    input_dim : int
        Number of input features.
    output_dim : int
        Number of output targets.
    model_cfg : Dict
        Hybrid model configuration.
    """

    def __init__(self, input_dim: int, output_dim: int, model_cfg: Dict) -> None:
        """Initialize MLPGPHybrid (placeholder)."""
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_cfg = model_cfg

        logger.warning("MLPGPHybrid is a placeholder. Full implementation coming soon.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder forward pass."""
        raise NotImplementedError("MLPGPHybrid forward pass not yet implemented.")
