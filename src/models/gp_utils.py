from __future__ import annotations

from typing import Any, Dict, Sequence

import gpytorch
import torch
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import ScaleKernel

from src.utils import setup_logger

logger = setup_logger(__name__)


class ExponentialKernel(gpytorch.kernels.Kernel):
    """Anisotropic generalized exponential covariance with learnable theta, and power exponent."""

    is_stationary = True
    has_lengthscale = False

    def __init__(
        self,
        input_dim: int,
        initial_theta: float | Sequence[float],
        initial_power_exponent: float | Sequence[float],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)

        initial_theta_tensor = self._coerce_vector(initial_theta, "initial_theta")
        initial_power_tensor = self._coerce_vector(initial_power_exponent, "initial_power_exponent")

        self.register_parameter(
            "raw_theta",
            torch.nn.Parameter(initial_theta_tensor.clone()),
        )
        self.register_parameter(
            "raw_power_exponent",
            torch.nn.Parameter(initial_power_tensor.clone()),
        )
        self.register_constraint("raw_theta", GreaterThan(1e-5))
        self.register_constraint("raw_power_exponent", Interval(1.5, 2.0))

    def _coerce_vector(self, value: float | Sequence[float], name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.get_default_dtype()).flatten()
        if tensor.numel() == 1:
            tensor = tensor.repeat(self.input_dim)
        elif tensor.numel() != self.input_dim:
            raise ValueError(f"{name} must be a scalar or have one value per input dimension ({self.input_dim}).")
        return tensor

    @property
    def theta(self) -> torch.Tensor:
        return self.raw_theta_constraint.transform(self.raw_theta)

    @property
    def power_exponent(self) -> torch.Tensor:
        return self.raw_power_exponent_constraint.transform(self.raw_power_exponent)

    def _anisotropic_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        diff = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).abs()
        scaled = self.theta * diff.pow(self.power_exponent)
        return torch.sum(scaled, dim=-1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **kwargs) -> torch.Tensor:
        if diag:
            base_diag = torch.ones(x1.shape[:-1], device=x1.device, dtype=x1.dtype)
            return base_diag

        distance = self._anisotropic_distance(x1, x2)
        covariance = torch.exp(-distance)

        return covariance


def build_kernel(
    kernel_name: str,
    num_dims: int,
    config: Dict[str, Any],
) -> gpytorch.kernels.Kernel:
    """Factory for creating kernels from config."""
    kernel_name = str(kernel_name).lower() if kernel_name else "exponential"

    if kernel_name == "exponential":
        initial_theta = config.get("theta", 1.0)
        initial_power_exponent = config.get("power_exponent", 2.0)
        base_kernel = ExponentialKernel(
            input_dim=num_dims,
            initial_theta=initial_theta,
            initial_power_exponent=initial_power_exponent,
        )
        return ScaleKernel(base_kernel)

    elif kernel_name == "rbf":
        return ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=num_dims))

    elif kernel_name == "matern":
        nu = float(config.get("matern_nu", 2.5))
        return ScaleKernel(gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=num_dims))

    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")


