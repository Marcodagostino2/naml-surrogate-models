"""MLP regressor with a single hidden block (one activation, variable per-layer width)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from src.models.init import init_block_linear, init_output_linear
from src.utils import setup_logger

logger = setup_logger(__name__)


# nn-module factories — superset of block activations (which are also valid as
# output activations) plus output-only options like sigmoid/identity.
_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
    "tanh": nn.Tanh,
    "mish": nn.Mish,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
}


def _resolve_activation(name: str) -> nn.Module:
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation '{name}'. Available: {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[key]()


def _parse_architecture(raw: Dict) -> Tuple[List[int], str]:
    """``raw`` must be a dict ``{layers: [int, ...], activation: str}``."""
    if not isinstance(raw, dict) or "layers" not in raw or "activation" not in raw:
        raise ValueError(
            "architecture must be a dict with 'layers' (list of int) and 'activation' (str)."
        )
    layer_sizes = [int(v) for v in raw["layers"]]
    if not layer_sizes:
        raise ValueError("architecture.layers must contain at least one layer size.")
    return layer_sizes, str(raw["activation"])


class MLPRegressor(nn.Module):
    """MLP defined by a single block: one activation shared across the hidden
    layers, each layer's width given by ``model_cfg["architecture"]["layers"][i]``.
    The output projection's init is agnostic to the block activation.
    """

    def __init__(self, input_dim: int, output_dim: int, model_cfg: Dict) -> None:
        super().__init__()

        layer_sizes, activation_name = _parse_architecture(
            model_cfg.get("architecture", {"layers": [128, 128], "activation": "relu"})
        )
        dropout_prob = float(model_cfg.get("dropout", 0.0))
        use_batch_norm = bool(model_cfg.get("batch_norm", False))
        output_activation_name = str(model_cfg.get("output_activation", "identity"))

        layers: List[nn.Module] = []
        in_features = input_dim
        for out_features in layer_sizes:
            linear = nn.Linear(in_features, out_features)
            init_block_linear(linear, activation_name)
            layers.append(linear)
            layers.append(_resolve_activation(activation_name))
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(p=dropout_prob))
            in_features = out_features

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(in_features))

        output_linear = nn.Linear(in_features, output_dim)
        init_output_linear(output_linear)
        layers.append(output_linear)
        layers.append(_resolve_activation(output_activation_name))

        self.network = nn.Sequential(*layers)

        logger.info(
            f"MLPRegressor initialized: input_dim={input_dim}, output_dim={output_dim}, "
            f"depth={len(layer_sizes)}, activation={activation_name}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)