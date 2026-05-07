"""MLP regressor with block-based architecture from YAML."""

from __future__ import annotations

import ast
from typing import Dict, List, Tuple

import torch
from torch import nn

from src.utils import setup_logger

logger = setup_logger(__name__)


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "mish": nn.Mish,
    "silu": nn.SiLU,
    "identity": nn.Identity,
}


def _resolve_activation(name: str) -> nn.Module:
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation '{name}'. Available: {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[key]()


class MLPRegressor(nn.Module):
    """MLP defined as a list of (layer_sizes, activation) blocks in ``model_cfg["architecture"]``."""

    def __init__(self, input_dim: int, output_dim: int, model_cfg: Dict) -> None:
        super().__init__()

        raw_architecture = model_cfg.get(
            "architecture",
            [{"layers": [128, 128], "activation": "relu"}],
        )
        dropout_prob = float(model_cfg.get("dropout", 0.0))
        use_batch_norm = bool(model_cfg.get("batch_norm", False))
        output_activation_name = str(model_cfg.get("output_activation", "identity"))

        architecture = self._parse_architecture(raw_architecture)

        layers: List[nn.Module] = []
        in_features = input_dim
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

        layers.append(nn.Linear(in_features, output_dim))
        layers.append(_resolve_activation(output_activation_name))

        self.network = nn.Sequential(*layers)

        logger.info(
            f"MLPRegressor initialized: input_dim={input_dim}, output_dim={output_dim}, "
            f"architecture_blocks={len(architecture)}"
        )

    @staticmethod
    def _parse_architecture(raw_architecture: List) -> List[Tuple[List[int], str]]:
        """Accept dict, string-tuple, or list-pair block formats."""
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
        return self.network(x)
