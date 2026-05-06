"""
Multi-Layer Perceptron (MLP) model for aerodynamic coefficient prediction.

Features:
- Dynamic architecture definition via YAML configuration
- Block-based layer specification with per-block activations
- Optional batch normalization and dropout
- Flexible output activation
"""

from __future__ import annotations

import ast
from typing import Dict, List, Tuple

import torch
from torch import nn

from src.utils import setup_logger

logger = setup_logger(__name__)


def _resolve_activation(activation_name: str) -> nn.Module:
    """
    Resolve activation function name to PyTorch module.

    Parameters
    ----------
    activation_name : str
        Activation name (case-insensitive). Supported: relu, gelu, tanh,
        elu, leaky_relu, selu, sigmoid, softplus, mish, silu, identity.

    Returns
    -------
    nn.Module
        PyTorch activation module.

    Raises
    ------
    ValueError
        If activation name is not recognized.
    """
    activations: Dict[str, nn.Module] = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(),
        "leaky_relu": nn.LeakyReLU(),
        "selu": nn.SELU(),
        "sigmoid": nn.Sigmoid(),
        "softplus": nn.Softplus(),
        "mish": nn.Mish(),
        "silu": nn.SiLU(),
        "identity": nn.Identity(),
    }
    key = activation_name.lower()
    if key not in activations:
        raise ValueError(
            f"Unsupported activation '{activation_name}'. "
            f"Available: {sorted(activations.keys())}"
        )
    return activations[key]


class MLPRegressor(nn.Module):
    """
    Multi-Layer Perceptron for regression.

    The architecture is defined in config as a list of blocks. Each block
    is a tuple: ([layer_sizes], "activation_name").

    Example config architecture:
        - ([128, 128], "relu")
        - ([64], "tanh")

    This creates:
        input -> Linear(in, 128) -> ReLU -> Linear(128, 128) -> ReLU ->
        Linear(128, 64) -> Tanh -> Linear(64, output) -> output_activation

    Parameters
    ----------
    input_dim : int
        Number of input features.
    output_dim : int
        Number of output targets.
    model_cfg : Dict
        Configuration dictionary containing:
        - architecture: List[Tuple[List[int], str]]
            List of (layer_sizes, activation_name) blocks.
        - dropout: float, optional
            Dropout probability (default: 0.0).
        - batch_norm: bool, optional
            Whether to use batch normalization (default: False).
        - output_activation: str, optional
            Activation for output layer (default: "identity").
    """

    def __init__(self, input_dim: int, output_dim: int, model_cfg: Dict) -> None:
        """
        Initialize MLPRegressor.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        output_dim : int
            Number of output targets.
        model_cfg : Dict
            Model configuration.
        """
        super().__init__()

        raw_architecture = model_cfg.get(
            "architecture",
            [{"layers": [128, 128], "activation": "relu"}],
        )
        dropout_prob: float = float(model_cfg.get("dropout", 0.0))
        use_batch_norm: bool = bool(model_cfg.get("batch_norm", False))
        output_activation_name: str = str(model_cfg.get("output_activation", "identity"))

        architecture: List[Tuple[List[int], str]] = self._parse_architecture(raw_architecture)

        layers: List[nn.Module] = []
        in_features = input_dim

        # Build hidden layers from architecture blocks
        for block_layers, block_activation in architecture:
            activation_module = _resolve_activation(block_activation)

            for out_features in block_layers:
                layers.append(nn.Linear(in_features, out_features))

                layers.append(activation_module)

                if dropout_prob > 0.0:
                    layers.append(nn.Dropout(p=dropout_prob))

                in_features = out_features

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
            
        # Output layer
        layers.append(nn.Linear(in_features, output_dim))

        # Output activation
        output_activation = _resolve_activation(output_activation_name)
        layers.append(output_activation)
        
        self.network = nn.Sequential(*layers)

        logger.info(
            f"MLPRegressor initialized: input_dim={input_dim}, output_dim={output_dim}, "
            f"architecture_blocks={len(architecture)}"
        )

    @staticmethod
    def _parse_architecture(raw_architecture: List) -> List[Tuple[List[int], str]]:
        """
        Parse architecture blocks from YAML-friendly formats.

        Supported block formats:
        - {"layers": [128, 64], "activation": "relu"}
        - ([128, 64], "relu") as a string legacy representation
        - [[128, 64], "relu"]
        """
        parsed: List[Tuple[List[int], str]] = []

        for block in raw_architecture:
            if isinstance(block, dict):
                layers = [int(v) for v in block["layers"]]
                activation = str(block["activation"])
            elif isinstance(block, str):
                block_eval = ast.literal_eval(block)
                layers = [int(v) for v in block_eval[0]]
                activation = str(block_eval[1])
            elif isinstance(block, (list, tuple)) and len(block) == 2:
                layers = [int(v) for v in block[0]]
                activation = str(block[1])
            else:
                raise ValueError(f"Unsupported architecture block format: {block}")

            if not layers:
                raise ValueError("Each architecture block must contain at least one layer size")
            parsed.append((layers, activation))

        if not parsed:
            raise ValueError("Architecture must contain at least one block")

        return parsed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        return self.network(x)
