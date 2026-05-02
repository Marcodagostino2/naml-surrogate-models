"""
Gaussian Process (Kriging) model for aerodynamic coefficient prediction.

Implemented using GPyTorch with:
- Exact inference (ExactGP)
- Polynomial trend mean
- Exponential covariance kernel with learnable theta, power exponent
- Global variance scale tau^2 via ScaleKernel
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Dict, List, Optional, Sequence, Tuple

import gpytorch
import torch
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.means import Mean
from gpytorch.models import ExactGP

from src.utils import setup_logger

logger = setup_logger(__name__)


def _format_param_block(name: str, values: Sequence[float], precision: str = ".4g") -> str:
    """Render a vector parameter as a labelled block, one element per line.

    Scalar (single-value) parameters render inline as ``name=value`` so that
    short hyperparameters do not consume extra log lines.
    """

    seq = list(values)
    if len(seq) == 1:
        return f"{name}={seq[0]:{precision}}"
    lines = [f"{name} (n_dims={len(seq)}):"]
    lines.extend(f"    [{i}] {v:{precision}}" for i, v in enumerate(seq))
    return "\n  ".join(lines)


def _build_gaussian_likelihood(initial_noise: float) -> gpytorch.likelihoods.GaussianLikelihood:
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-10))
    likelihood.initialize(noise=float(initial_noise))
    return likelihood


def _require_config_value(config: Dict, key: str, context: str):
    if key not in config:
        raise KeyError(f"Missing required config key '{key}' in {context}.")
    return config[key]


class PolynomialTrendMean(Mean):
    """Polynomial trend mean with all monomials up to a given degree."""

    def __init__(self, input_dim: int, degree: int = 1) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.degree = int(degree)
        self.exponents: List[Tuple[int, ...]] = self._build_exponents(self.input_dim, self.degree)
        self.register_parameter(
            "coefficients",
            torch.nn.Parameter(torch.zeros(len(self.exponents), dtype=torch.get_default_dtype())),
        )

    @staticmethod
    def _build_exponents(input_dim: int, degree: int) -> List[Tuple[int, ...]]:
        exponents: List[Tuple[int, ...]] = [tuple()]
        for current_degree in range(1, degree + 1):
            for combo in combinations_with_replacement(range(input_dim), current_degree):
                exponents.append(combo)
        return exponents

    def _design_matrix(self, x: torch.Tensor) -> torch.Tensor:
        features: List[torch.Tensor] = [torch.ones(x.shape[0], device=x.device, dtype=x.dtype)]

        for exponent in self.exponents[1:]:
            term = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
            for feature_index in exponent:
                term = term * x[:, feature_index]
            features.append(term)

        return torch.stack(features, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._design_matrix(x) @ self.coefficients


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


class GPRegressionModel(ExactGP):
    """Exact Gaussian Process regression model for aerodynamic prediction."""

    def __init__(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        model_cfg: Dict,
        likelihood: Optional[gpytorch.likelihoods.Likelihood] = None,
    ) -> None:
        kernel_cfg = model_cfg.get("kernel", model_cfg)
        initial_likelihood_noise = float(_require_config_value(kernel_cfg, "likelihood_noise", "GP kernel config"))

        if likelihood is None:
            likelihood = _build_gaussian_likelihood(initial_likelihood_noise)

        super().__init__(X_train, y_train, likelihood)

        kernel_cfg = model_cfg.get("kernel", model_cfg)
        trend_degree = int(_require_config_value(model_cfg, "trend_degree", "GP model config"))
        self.mean_module = PolynomialTrendMean(input_dim=X_train.shape[-1], degree=trend_degree)

        kernel_type = str(kernel_cfg.get("type", kernel_cfg.get("kernel_type", "exponential"))).lower()
        if kernel_type != "exponential":
            raise ValueError("Only the exponential kernel is supported in this refactor.")

        initial_theta = _require_config_value(kernel_cfg, "theta", "GP kernel config")
        initial_p = _require_config_value(kernel_cfg, "power_exponent", "GP kernel config")
        initial_tau2 = float(_require_config_value(kernel_cfg, "tau2", "GP kernel config"))

        self.likelihood.initialize(noise=initial_likelihood_noise)

        base_kernel = ExponentialKernel(
            input_dim=X_train.shape[-1],
            initial_power_exponent=initial_p,
            initial_theta=initial_theta,
        )

        self.covar_module = ScaleKernel(base_kernel)
        self.covar_module.outputscale = initial_tau2
    

        theta_values = base_kernel.theta.flatten().detach().cpu().tolist()
        power_values = base_kernel.power_exponent.flatten().detach().cpu().tolist()
        logger.info(
            "GPRegressionModel initialized\n"
            f"  trend_degree={trend_degree}\n"
            f"  tau2={float(self.covar_module.outputscale.item()):.6f}\n"
            f"  likelihood_noise={float(self.likelihood.noise.item()):.6e}\n"
            f"  {_format_param_block('theta', theta_values)}\n"
            f"  {_format_param_block('power_exponent', power_values)}"
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
