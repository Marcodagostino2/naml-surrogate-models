from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn


def _resolve_activation(name: str) -> nn.Module:
    activations = {
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
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Unsupported activation '{name}'. Available: {sorted(activations)}")
    return activations[key]()


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, model_cfg: Dict) -> None:
        super().__init__()
        hidden_layers: List[int] = list(model_cfg.get("hidden_layers", [128, 128]))
        dropout = float(model_cfg.get("dropout", 0.0))
        activation_name = str(model_cfg.get("activation", "relu"))

        layers = []
        in_features = input_dim
        for hidden_features in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_features))
            layers.append(_resolve_activation(activation_name))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_features = hidden_features

        layers.append(nn.Linear(in_features, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
