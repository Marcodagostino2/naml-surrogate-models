"""GP regression models: ``GPRegressionModel`` (exact) and ``SVGPRegressionModel`` (sparse variational).

Both share a polynomial trend mean and a kernel built by :func:`build_kernel`.
The variant is selected via ``model_cfg["inference"]`` in the pipeline.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Any, Dict, List, Optional, Tuple

import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import Mean
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from src.models.gp_utils import (
    build_kernel,
    clamp_to_bounds,
    extract_config_value,
    extract_kernel_name,
    extract_required_config_value,
)
from src.utils import setup_logger

logger = setup_logger(__name__)


def _build_gaussian_likelihood(initial_noise: float, noise_floor: float, noise_ceiling: float) -> gpytorch.likelihoods.GaussianLikelihood:
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(noise_floor, noise_ceiling))
    clamped = float(clamp_to_bounds(
        torch.as_tensor(initial_noise, dtype=torch.get_default_dtype()),
        noise_floor,
        noise_ceiling,
    ).item())
    likelihood.initialize(noise=clamped)
    return likelihood


def _parse_noise_cfg(kernel_cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    """Return ``(initial_noise_clamped, floor, ceiling)`` from a kernel config block."""
    initial = float(extract_config_value(kernel_cfg, "likelihood_noise_init", "GP kernel config"))
    floor = float(extract_config_value(kernel_cfg, "likelihood_noise_floor", "GP kernel config", default=1.0e-6))
    ceiling = float(extract_config_value(kernel_cfg, "likelihood_noise_ceiling", "GP kernel config", default=1.0))
    initial_clamped = float(clamp_to_bounds(
        torch.as_tensor(initial, dtype=torch.get_default_dtype()),
        floor,
        ceiling,
    ).item())
    return initial_clamped, floor, ceiling


def _build_likelihood_from_cfg(model_cfg: Dict[str, Any]) -> gpytorch.likelihoods.GaussianLikelihood:
    kernel_cfg = model_cfg.get("kernel", model_cfg)
    initial, floor, ceiling = _parse_noise_cfg(kernel_cfg)
    return _build_gaussian_likelihood(initial, floor, ceiling)


def _select_inducing_points(X_train: torch.Tensor, n_inducing: int) -> torch.Tensor:
    """Random subset of ``X_train`` for SVGP inducing-point initialization."""
    n_train = int(X_train.shape[0])
    n_inducing = int(n_inducing)
    if n_inducing < 1:
        raise ValueError(f"n_inducing_points must be >= 1, got {n_inducing}")
    if n_inducing >= n_train:
        logger.warning(
            f"n_inducing_points ({n_inducing}) >= n_train ({n_train}); using full training set."
        )
        return X_train.detach().clone()
    perm = torch.randperm(n_train, device=X_train.device)[:n_inducing]
    return X_train[perm].detach().clone()


class PolynomialTrendMean(Mean):
    """Polynomial trend mean with all monomials up to ``degree``."""

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


class _GPRegressionMixin:
    """Shared mean+kernel setup and ``forward`` for the exact and SV variants."""

    def _setup_mean_and_kernel(self, X_train: torch.Tensor, model_cfg: Dict[str, Any]) -> None:
        kernel_cfg = model_cfg.get("kernel", model_cfg)
        trend_degree = int(extract_required_config_value(model_cfg, "trend_degree", "GP model config"))
        input_dim = int(X_train.shape[-1])
        self.mean_module = PolynomialTrendMean(input_dim=input_dim, degree=trend_degree)
        self.covar_module = build_kernel(extract_kernel_name(kernel_cfg), input_dim, kernel_cfg)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class GPRegressionModel(_GPRegressionMixin, ExactGP):
    """Exact GP regression model.

    Mixin precedes ``ExactGP`` in the bases so the mixin's ``forward`` overrides
    ``ExactGP.forward`` (which raises ``NotImplementedError``) via MRO.
    """

    def __init__(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        model_cfg: Dict,
        likelihood: Optional[gpytorch.likelihoods.Likelihood] = None,
    ) -> None:
        if likelihood is None:
            likelihood = _build_likelihood_from_cfg(model_cfg)
        super().__init__(X_train, y_train, likelihood)
        self._setup_mean_and_kernel(X_train, model_cfg)


class SVGPRegressionModel(_GPRegressionMixin, ApproximateGP):
    """Sparse Variational GP (Hensman 2013/2015) with inducing points from a subset of ``X_train``.

    ``ApproximateGP`` does not own a likelihood; we attach it as a submodule so
    its parameters are picked up by ``self.parameters()``.
    """

    def __init__(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        model_cfg: Dict,
        likelihood: Optional[gpytorch.likelihoods.Likelihood] = None,
    ) -> None:
        del y_train  # targets enter via the ELBO at training time

        if likelihood is None:
            likelihood = _build_likelihood_from_cfg(model_cfg)

        svgp_cfg = model_cfg.get("svgp", {}) or {}
        n_inducing = int(svgp_cfg.get("n_inducing_points", 256))
        learn_inducing_locations = bool(svgp_cfg.get("learn_inducing_locations", True))

        inducing_points = _select_inducing_points(X_train, n_inducing)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            CholeskyVariationalDistribution(inducing_points.size(0)),
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)
        self.likelihood = likelihood
        self._setup_mean_and_kernel(X_train, model_cfg)

        logger.info(
            f"SVGPRegressionModel initialized with {inducing_points.size(0)} inducing points "
            f"(learn_inducing_locations={learn_inducing_locations})"
        )
