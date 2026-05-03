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

from src.models.gp_utils import build_kernel
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
        input_dim = int(X_train.shape[-1])
        self.mean_module = PolynomialTrendMean(input_dim=input_dim, degree=trend_degree)

        # Extract kernel name; support both nested dict (old format) and direct string (new format)
        if isinstance(kernel_cfg, dict):
            kernel_name = str(kernel_cfg.get("type", kernel_cfg.get("kernel", "exponential"))).lower()
        else:
            kernel_name = str(kernel_cfg).lower()

        # Extract kernel hyperparameters (required for exponential, optional for others)
        if kernel_name == "exponential":
            initial_theta = _require_config_value(kernel_cfg, "theta", "GP kernel config")
            initial_p = _require_config_value(kernel_cfg, "power_exponent", "GP kernel config")
            initial_tau2 = float(_require_config_value(kernel_cfg, "tau2", "GP kernel config"))
        else:
            # For RBF/Matern, tau2 is handled via ScaleKernel from factory
            initial_tau2 = None

        self.likelihood.initialize(noise=initial_likelihood_noise)

        self.covar_module = build_kernel(kernel_name, input_dim, kernel_cfg)

        # Set outputscale for exponential kernel (RBF/Matern already have ScaleKernel from factory)
        if kernel_name == "exponential" and initial_tau2 is not None:
            if isinstance(self.covar_module, ScaleKernel):
                self.covar_module.outputscale = initial_tau2

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
