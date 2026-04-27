"""
Gaussian Process (Kriging) model for aerodynamic coefficient prediction.

Implemented using GPyTorch with:
- Exact inference (ExactGP)
- ConstantMean
- ScaleKernel wrapping a spatial kernel (Matérn or generalized exponential)
- Learnable hyperparameters: lengthscale, power exponent, nugget
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP

from src.utils import setup_logger

logger = setup_logger(__name__)


class GeneralizedExponentialKernel(gpytorch.kernels.Kernel):
    """
    Generalized exponential kernel with power exponent p.

    The kernel is: k(x, x') = τ² * exp(-||x - x'||^p / θ²)

    where:
    - τ² is the scale (global variance)
    - θ is the lengthscale
    - p is the power exponent (1.5 to 2.0)
    - λ is the nugget (added to diagonal)
    """

    is_stationary = True
    has_lengthscale = True

    def __init__(self, power_exponent: float = 2.0, **kwargs) -> None:
        """
        Initialize GeneralizedExponentialKernel.

        Parameters
        ----------
        power_exponent : float, optional
            Power exponent p. Default: 2.0 (squared exponential).
        """
        super().__init__(**kwargs)
        # Register power exponent as a parameter (bounded)
        self.register_parameter(
            "raw_power_exponent",
            torch.nn.Parameter(torch.tensor(power_exponent)),
        )

    @property
    def power_exponent(self) -> torch.Tensor:
        """Get power exponent with constraints (1.5 to 2.0)."""
        return torch.clamp(self.raw_power_exponent, min=1.5, max=2.0)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **kwargs) -> torch.Tensor:
        """
        Compute kernel matrix.

        Parameters
        ----------
        x1 : torch.Tensor
            First set of points.
        x2 : torch.Tensor
            Second set of points.
        diag : bool, optional
            If True, only return diagonal. Default: False.

        Returns
        -------
        torch.Tensor
            Kernel matrix or diagonal.
        """
        # Compute distance scaled by lengthscale
        x1_ = x1 / self.lengthscale
        x2_ = x2 / self.lengthscale

        if diag:
            diff = x1_ - x2_
            distance = torch.norm(diff, p=2, dim=-1)
        else:
            diff = x1_.unsqueeze(-2) - x2_.unsqueeze(-3)
            distance = torch.norm(diff, p=2, dim=-1)

        # Generalized exponential: exp(-distance^p)
        return torch.exp(-(distance ** self.power_exponent))


class GPRegressionModel(ExactGP):
    """
    Exact Gaussian Process regression model for aerodynamic prediction.

    Features:
    - ConstantMean (μ)
    - ScaleKernel(GeneralizedExponentialKernel)
    - Marginal log-likelihood optimization
    - Supports uncertainty quantification

    Parameters
    ----------
    X_train : torch.Tensor
        Training inputs (n_train, n_features).
    y_train : torch.Tensor
        Training targets (n_train, n_outputs).
    model_cfg : Dict
        Configuration with keys:
        - kernel_type: "matern" or "exponential"
        - initial_lengthscale: float
        - initial_nugget: float
        - likelihood_noise: float
    """

    def __init__(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        model_cfg: Dict,
        likelihood: Optional[gpytorch.likelihoods.Likelihood] = None,
    ) -> None:
        """
        Initialize GPRegressionModel.

        Parameters
        ----------
        X_train : torch.Tensor
            Training inputs.
        y_train : torch.Tensor
            Training targets.
        model_cfg : Dict
            Model configuration.
        likelihood : gpytorch.likelihoods.Likelihood, optional
            Custom likelihood. If None, Gaussian likelihood is created.
        """
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

        super().__init__(X_train, y_train, likelihood)

        # Mean
        self.mean_module = ConstantMean()

        # Covariance kernel
        kernel_type = model_cfg.get("kernel_type", "matern").lower()

        if kernel_type == "exponential":
            base_kernel = GeneralizedExponentialKernel(power_exponent=2.0)
        elif kernel_type == "matern":
            base_kernel = MaternKernel(nu=2.5)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        # Initialize lengthscale
        initial_lengthscale = float(model_cfg.get("initial_lengthscale", 1.0))
        base_kernel.lengthscale = initial_lengthscale

        self.covar_module = ScaleKernel(base_kernel)

        # Set likelihood noise
        likelihood_noise = float(model_cfg.get("likelihood_noise", 1e-4))
        self.likelihood.noise = likelihood_noise

        logger.info(f"GPRegressionModel initialized with kernel_type={kernel_type}")
        logger.info(f"Initial lengthscale={initial_lengthscale}, noise={likelihood_noise}")

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Forward pass through GP.

        Parameters
        ----------
        x : torch.Tensor
            Input points (n, n_features).

        Returns
        -------
        MultivariateNormal
            Multivariate normal distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
