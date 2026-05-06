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
from typing import Dict, List, Optional, Tuple

import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import Mean
from gpytorch.models import ExactGP

from src.models.gp_utils import build_kernel, clamp_to_bounds, extract_config_value, extract_kernel_name, extract_required_config_value
from src.utils import setup_logger

logger = setup_logger(__name__)


def _build_gaussian_likelihood(initial_noise: float, noise_floor: float, noise_ceiling: float) -> gpytorch.likelihoods.GaussianLikelihood:
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(noise_floor, noise_ceiling))
    clamped_noise = float(
        clamp_to_bounds(
            torch.as_tensor(initial_noise, dtype=torch.get_default_dtype()),
            noise_floor,
            noise_ceiling,
        ).item()
    )
    likelihood.initialize(noise=clamped_noise)
    return likelihood


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
        initial_likelihood_noise = float(
            extract_config_value(
                kernel_cfg,
                ("likelihood_noise_init"),
                "GP kernel config",
            )
        )
        noise_floor = float(
            extract_config_value(
                kernel_cfg,
                ("likelihood_noise_floor"),
                "GP kernel config",
                default=1.0e-6,
            )
        )
        noise_ceiling = float(
            extract_config_value(
                kernel_cfg,
                ("likelihood_noise_ceiling"),
                "GP kernel config",
                default=1.0,
            )
        )

        initial_likelihood_noise = float(
            clamp_to_bounds(
                torch.as_tensor(initial_likelihood_noise, dtype=torch.get_default_dtype()),
                noise_floor,
                noise_ceiling,
            ).item()
        )

        if likelihood is None:
            likelihood = _build_gaussian_likelihood(initial_likelihood_noise, noise_floor, noise_ceiling)

        super().__init__(X_train, y_train, likelihood)

        kernel_cfg = model_cfg.get("kernel", model_cfg)
        trend_degree = int(extract_required_config_value(model_cfg, "trend_degree", "GP model config"))
        input_dim = int(X_train.shape[-1])
        self.mean_module = PolynomialTrendMean(input_dim=input_dim, degree=trend_degree)

        # Extract kernel name; support both nested dict (old format) and direct string (new format)
        kernel_name = extract_kernel_name(kernel_cfg)

        self.likelihood.initialize(noise=initial_likelihood_noise)
        self.covar_module = build_kernel(kernel_name, input_dim, kernel_cfg)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
