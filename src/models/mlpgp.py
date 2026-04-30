"""
End-to-end MLPGP model with nonstationary variance and lengthscale.

This implementation follows the "A Framework for Nonstationary Gaussian
Processes with Neural Network Parameters" idea by using an MLP to predict
input-dependent local kernel parameters.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gpytorch
import torch
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP

from src.models.gp import ExponentialKernel, PolynomialTrendMean
from src.models.mlp import MLPRegressor
from src.utils import setup_logger

logger = setup_logger(__name__)


def _require_config_value(config: Dict[str, Any], key: str, context: str) -> Any:
    if key not in config:
        raise KeyError(f"Missing required config key '{key}' in {context}.")
    return config[key]


class MLPGPFeatureExtractor(torch.nn.Module):
    """MLP feature extractor producing positive local sigma(x) and l(x)."""

    def __init__(self, input_dim: int, output_lengthscale_dim: int, model_cfg: Dict[str, Any]) -> None:
        super().__init__()

        feature_cfg = model_cfg.get("feature_extractor", model_cfg)
        
        self.sigma_floor = float(feature_cfg.get("sigma_floor", 1e-8))
        self.lengthscale_floor = float(feature_cfg.get("lengthscale_floor", 1e-8))
        self.output_lengthscale_dim = int(output_lengthscale_dim)

        # Output head predicts [sigma(x), l_1(x), ..., l_d(x)] and positivity
        # is enforced by the configured output activation (Softplus by default).
        self.backbone = MLPRegressor(
            input_dim=input_dim,
            output_dim=1 + self.output_lengthscale_dim,
            model_cfg=feature_cfg,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_out = self.backbone(x)
        sigma_x = feature_out[..., :1].clamp_min(self.sigma_floor)
        lengthscale_x = feature_out[..., 1:].clamp_min(self.lengthscale_floor)
        return sigma_x, lengthscale_x


class NonstationaryExponentialKernel(gpytorch.kernels.Kernel):
    """
    Nonstationary wrapper around the existing stationary ExponentialKernel.

    The covariance is computed as:
        K(x, x') = sigma(x) * sigma(x') * k_stationary(z(x), z(x'))

    with lengthscale-warped coordinates:
        z_j(x) = x_j / l_j(x)

    This keeps direct reuse of the existing stationary kernel while injecting
    input-dependent lengthscales into the distance computation.
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(
        self,
        base_kernel: ExponentialKernel,
        min_lengthscale: float = 1e-8,
        min_sigma: float = 1e-8,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.min_lengthscale = float(min_lengthscale)
        self.min_sigma = float(min_sigma)
        self.parameterizer: Optional[torch.nn.Module] = None

    @property
    def nugget(self) -> torch.Tensor:
        return self.base_kernel.nugget

    def set_parameterizer(self, parameterizer: torch.nn.Module) -> None:
        """Attach feature extractor used when external params are not provided."""
        self.parameterizer = parameterizer

    def _resolve_local_parameters(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        sigma_x1: Optional[torch.Tensor],
        sigma_x2: Optional[torch.Tensor],
        lengthscale_x1: Optional[torch.Tensor],
        lengthscale_x2: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if sigma_x1 is None or lengthscale_x1 is None:
            if self.parameterizer is None:
                raise ValueError("sigma_x1/lengthscale_x1 must be provided when parameterizer is unset.")
            sigma_x1, lengthscale_x1 = self.parameterizer(x1)
        if sigma_x2 is None or lengthscale_x2 is None:
            if self.parameterizer is None:
                raise ValueError("sigma_x2/lengthscale_x2 must be provided when parameterizer is unset.")
            sigma_x2, lengthscale_x2 = self.parameterizer(x2)

        sigma_x1 = sigma_x1.squeeze(-1).clamp_min(self.min_sigma)
        sigma_x2 = sigma_x2.squeeze(-1).clamp_min(self.min_sigma)
        lengthscale_x1 = lengthscale_x1.clamp_min(self.min_lengthscale)
        lengthscale_x2 = lengthscale_x2.clamp_min(self.min_lengthscale)
        return sigma_x1, sigma_x2, lengthscale_x1, lengthscale_x2

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        sigma_x1: Optional[torch.Tensor] = None,
        sigma_x2: Optional[torch.Tensor] = None,
        lengthscale_x1: Optional[torch.Tensor] = None,
        lengthscale_x2: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute Gibbs (Paciorek & Schervish 2006) non-stationary kernel.

        Implementation of:
            K(x_i, x_j) = sigma_i * sigma_j
                          * prod_d [ sqrt( 2 * ell_i_d * ell_j_d / (ell_i_d^2 + ell_j_d^2) ) ]
                          * R0( delta(x_i, x_j) )

        where delta_d(x_i, x_j) = theta_d * ( |x_i_d - x_j_d| / sqrt( (ell_i_d^2 + ell_j_d^2) / 2 ) )^p

        Parameters
        ----------
        x1 : Tensor (n, d)
            First set of input points.
        x2 : Tensor (m, d)
            Second set of input points.
        diag : bool
            If True, only return diagonal of kernel matrix (n,).
        sigma_x1, sigma_x2 : Optional[Tensor]
            Pre-computed local standard deviations. If None, computed from parameterizer.
        lengthscale_x1, lengthscale_x2 : Optional[Tensor]
            Pre-computed local length scales. If None, computed from parameterizer.

        Returns
        -------
        Tensor
            If diag=True: (n,) diagonal elements
            If diag=False: (n, m) full kernel matrix
        """
        sigma_x1, sigma_x2, lengthscale_x1, lengthscale_x2 = self._resolve_local_parameters(
            x1=x1,
            x2=x2,
            sigma_x1=sigma_x1,
            sigma_x2=sigma_x2,
            lengthscale_x1=lengthscale_x1,
            lengthscale_x2=lengthscale_x2,
        )

        if diag:
            # For diagonal: diff=0 so dist=0 and R0_diag=1.
            # norm_factor_diag = prod_d(sqrt(2*L^2 / 2L^2)) = 1.
            # So diag output = sigma1^2 + nugget
            base_diag = sigma_x1.pow(2)
            return base_diag + self.nugget

        # Compute lengthscale squared (n, d) and (m, d)
        L1_sq = lengthscale_x1.pow(2)
        L2_sq = lengthscale_x2.pow(2)

        # Unsqueeze for broadcasting: (n, 1, d) and (1, m, d) -> (n, m, d)
        L1_sq_broad = L1_sq.unsqueeze(-2)  # (n, 1, d)
        L2_sq_broad = L2_sq.unsqueeze(-3)  # (1, m, d)

        # Compute norm_factor (n, m):
        # prod_d [ sqrt( 2 * L1_d * L2_d / (L1_sq_d + L2_sq_d) ) ]
        L1_broad = lengthscale_x1.unsqueeze(-2)  # (n, 1, d)
        L2_broad = lengthscale_x2.unsqueeze(-3)  # (1, m, d)

        per_dim_factor = torch.sqrt(2.0 * L1_broad * L2_broad / (L1_sq_broad + L2_sq_broad))
        # Shape: (n, m, d)
        norm_factor = torch.prod(per_dim_factor, dim=-1)  # (n, m)

        # Debug assertion: norm_factor should be in (0, 1]
        # (only in debug mode to avoid runtime cost)
        if False:  # Set to True for debugging
            assert (norm_factor > 0).all() and (norm_factor <= 1.0 + 1e-5).all(), \
                f"norm_factor out of bounds: min={norm_factor.min()}, max={norm_factor.max()}"

        # Compute harmonic mean squared length scale: (n, m, d)
        ell_harm_sq = (L1_sq_broad + L2_sq_broad) / 2.0

        # Compute differences: (n, m, d)
        diff = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).abs()  # (n, 1, d) - (1, m, d)

        # Get theta and p from base kernel
        theta = self.base_kernel.theta  # (d,)
        p = self.base_kernel.power_exponent  # scalar or (d,)

        # Compute scaled difference fed into R0:
        # delta_d = theta_d * ( diff_d / sqrt(ell_harm_sq_d) )^p
        ell_harm_std = torch.sqrt(ell_harm_sq)  # (n, m, d)
        scaled_diff = theta * (diff / ell_harm_std).pow(p)  # (n, m, d)

        # Sum across dimensions to get distance
        dist = torch.sum(scaled_diff, dim=-1)  # (n, m)

        # Evaluate base kernel on the scaled distance
        R0 = torch.exp(-dist)  # (n, m)

        # Compute sigma_outer: (n, 1) * (1, m) -> (n, m)
        sigma_outer = sigma_x1.unsqueeze(-1) * sigma_x2.unsqueeze(-2)

        # Full Gibbs kernel: sigma_outer * norm_factor * R0
        covariance = sigma_outer * norm_factor * R0

        # Add nugget only when evaluating at same points (x1 is x2)
        same_inputs = x1.shape == x2.shape and torch.equal(x1, x2)
        if same_inputs:
            eye = torch.eye(x1.shape[-2], device=x1.device, dtype=x1.dtype)
            covariance = covariance + self.nugget * eye

        return covariance


class MLPGPModel(ExactGP):
    """ExactGP model with MLP-parameterized nonstationary covariance."""

    def __init__(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        model_cfg: Dict,
        likelihood: Optional[gpytorch.likelihoods.Likelihood] = None,
    ) -> None:
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-10))

        super().__init__(X_train, y_train, likelihood)

        input_dim = int(X_train.shape[-1])
        
        # Extract configuration values
        feature_cfg = model_cfg.get("feature_extractor", model_cfg)
        kernel_cfg = model_cfg.get("kernel", model_cfg)
        
        lengthscale_mode = str(feature_cfg.get("lengthscale_mode", "per_dimension")).lower()
        if lengthscale_mode not in {"scalar", "per_dimension"}:
            raise ValueError("lengthscale_mode must be either 'scalar' or 'per_dimension'.")
        
        trend_degree = int(_require_config_value(model_cfg, "trend_degree", "MLPGP model config"))
        kernel_type = str(kernel_cfg.get("type", kernel_cfg.get("kernel_type", "exponential"))).lower()
        if kernel_type != "exponential":
            raise ValueError("Only the exponential kernel is supported in this MLPGP implementation.")
        
        local_lengthscale_dim = 1 if lengthscale_mode == "scalar" else input_dim

        self.feature_extractor = MLPGPFeatureExtractor(
            input_dim=input_dim,
            output_lengthscale_dim=local_lengthscale_dim,
            model_cfg=model_cfg,
        )
        self.mean_module = PolynomialTrendMean(input_dim=input_dim, degree=trend_degree)

        initial_theta = _require_config_value(kernel_cfg, "theta", "MLPGP kernel config")
        initial_p = _require_config_value(kernel_cfg, "power_exponent", "MLPGP kernel config")
        initial_nugget = float(_require_config_value(kernel_cfg, "nugget", "MLPGP kernel config"))

        # Fix likelihood noise to minimal constant; all regularization comes from nugget
        self.likelihood.noise = float(_require_config_value(kernel_cfg, "likelihood_noise", "MLPGP kernel config"))
        
        base_kernel = ExponentialKernel(
            input_dim=input_dim,
            initial_theta=initial_theta,
            initial_power_exponent=initial_p,
            initial_nugget=initial_nugget,
        )

        nonstationary_kernel = NonstationaryExponentialKernel(
            base_kernel=base_kernel,
            min_lengthscale=float(feature_cfg.get("lengthscale_floor", 1e-8)),
            min_sigma=float(feature_cfg.get("sigma_floor", 1e-8)),
        )
        nonstationary_kernel.set_parameterizer(self._predict_local_parameters)

        # No ScaleKernel wrapping — use nonstationary kernel directly
        # All variance scaling is handled internally by sigma(x)
        self.covar_module = nonstationary_kernel

        self.likelihood.raw_noise.requires_grad_(False)

        logger.info(
            "MLPGPModel initialized: "
            f"lengthscale_mode={lengthscale_mode}, "
            f"trend_degree={trend_degree}, "
        )

    def _predict_local_parameters(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma_x, lengthscale_x = self.feature_extractor(x)
        if lengthscale_x.shape[-1] == 1 and x.shape[-1] > 1:
            raise ValueError("Scalar lengthscale output is not compatible with multi-dimensional input.")
        return sigma_x, lengthscale_x

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
