"""Activation-driven weight init for ``nn.Linear`` modules in MLPs.

Each block activation is paired with a fixed PyTorch initializer. The output
projection uses an activation-agnostic init. To add an activation, register
it in ``_BLOCK_INIT``; nothing else changes.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

# Block-activation -> initializer (operates on layer.weight in place).
_BLOCK_INIT: Dict[str, Callable[[nn.Linear], None]] = {
    "relu":       lambda L: nn.init.kaiming_normal_(L.weight, nonlinearity="relu"),
    "leaky_relu": lambda L: nn.init.kaiming_normal_(L.weight, nonlinearity="leaky_relu"),
    # SELU's recommended init is LeCun normal, equivalent to Kaiming-normal with
    # a linear nonlinearity (gain=1) under fan-in mode.
    "selu":       lambda L: nn.init.kaiming_normal_(L.weight, nonlinearity="linear"),
    "tanh":       lambda L: nn.init.xavier_normal_(L.weight, gain=nn.init.calculate_gain("tanh")),
    # Mish: ReLU-family Kaiming is the practical default.
    "mish":       lambda L: nn.init.kaiming_normal_(L.weight, nonlinearity="relu"),
}


def init_block_linear(layer: nn.Linear, activation: str) -> None:
    """Init a hidden-layer ``nn.Linear`` according to its block activation."""
    name = activation.lower()
    if name not in _BLOCK_INIT:
        raise ValueError(
            f"No init registered for activation '{activation}'. "
            f"Available: {sorted(_BLOCK_INIT)}"
        )
    _BLOCK_INIT[name](layer)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_output_linear(layer: nn.Linear) -> None:
    """Init the output projection — activation-agnostic (Xavier-normal, gain=1)."""
    nn.init.xavier_normal_(layer.weight, gain=1.0)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)