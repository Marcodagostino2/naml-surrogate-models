"""
End-to-end MLPGP model with nonstationary variance and lengthscale.

This implementation follows the "A Framework for Nonstationary Gaussian
Processes with Neural Network Parameters" idea by using an MLP to predict
input-dependent local kernel parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gpytorch
import torch
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.models import ExactGP

from src.models.gp import ExponentialKernel, PolynomialTrendMean
from src.models.mlp import MLPRegressor
from src.utils import setup_logger

logger = setup_logger(__name__)


def _require_config_value(config: Dict[str, Any], key: str, context: str) -> Any:
    if key not in config:
        raise KeyError(f"Missing required config key '{key}' in {context}.")
    return config[key]


@dataclass(slots=True)
class MLPGPModelConfig:
    """Typed config for MLPGP model hyperparameters."""

    mlp_architecture: List[Any]
    mlp_dropout: float
    mlp_batch_norm: bool
    mlp_output_activation: str
    sigma_floor: float
    lengthscale_floor: float
    lengthscale_mode: str
    kernel_type: str
    kernel_theta: float | Sequence[float]
    kernel_power_exponent: float | Sequence[float]
    kernel_nugget: float
    kernel_tau2: float
    kernel_likelihood_noise: float
    trend_degree: int

    @classmethod
    def from_dict(cls, model_cfg: Dict[str, Any]) -> "MLPGPModelConfig":
        feature_cfg = model_cfg.get("feature_extractor", model_cfg)
        kernel_cfg = model_cfg.get("kernel", model_cfg)
        cfg = cls(
            mlp_architecture=list(_require_config_value(feature_cfg, "architecture", "MLPGP feature_extractor config")),
            mlp_dropout=float(feature_cfg.get("dropout", 0.0)),
            mlp_batch_norm=bool(feature_cfg.get("batch_norm", False)),
            mlp_output_activation=str(feature_cfg.get("output_activation", "softplus")),
            sigma_floor=float(feature_cfg.get("sigma_floor", 1e-8)),
            lengthscale_floor=float(feature_cfg.get("lengthscale_floor", 1e-8)),
            lengthscale_mode=str(feature_cfg.get("lengthscale_mode", "per_dimension")).lower(),
            kernel_type=str(kernel_cfg.get("type", kernel_cfg.get("kernel_type", "exponential"))).lower(),
            kernel_theta=_require_config_value(kernel_cfg, "theta", "MLPGP kernel config"),
            kernel_power_exponent=_require_config_value(kernel_cfg, "power_exponent", "MLPGP kernel config"),
            kernel_nugget=float(_require_config_value(kernel_cfg, "nugget", "MLPGP kernel config")),
            kernel_tau2=float(_require_config_value(kernel_cfg, "tau2", "MLPGP kernel config")),
            kernel_likelihood_noise=float(_require_config_value(kernel_cfg, "likelihood_noise", "MLPGP kernel config")),
            trend_degree=int(_require_config_value(model_cfg, "trend_degree", "MLPGP model config")),
        )
        if cfg.kernel_type != "exponential":
            raise ValueError("Only the exponential kernel is supported in this MLPGP implementation.")
        if cfg.lengthscale_mode not in {"scalar", "per_dimension"}:
            raise ValueError("lengthscale_mode must be either 'scalar' or 'per_dimension'.")
        return cfg


class MLPGPFeatureExtractor(torch.nn.Module):
    """MLP feature extractor producing positive local sigma(x) and l(x)."""

    def __init__(self, input_dim: int, output_lengthscale_dim: int, model_cfg: MLPGPModelConfig) -> None:
        super().__init__()

        self.sigma_floor = float(model_cfg.sigma_floor)
        self.lengthscale_floor = float(model_cfg.lengthscale_floor)
        self.output_lengthscale_dim = int(output_lengthscale_dim)

        feature_cfg: Dict[str, object] = {
            "architecture": model_cfg.mlp_architecture,
            "dropout": model_cfg.mlp_dropout,
            "batch_norm": model_cfg.mlp_batch_norm,
            "output_activation": model_cfg.mlp_output_activation,
        }

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
        sigma_x1, sigma_x2, lengthscale_x1, lengthscale_x2 = self._resolve_local_parameters(
            x1=x1,
            x2=x2,
            sigma_x1=sigma_x1,
            sigma_x2=sigma_x2,
            lengthscale_x1=lengthscale_x1,
            lengthscale_x2=lengthscale_x2,
        )

        if diag:
            base_diag = sigma_x1.pow(2)
            return base_diag + self.nugget

        # Reuse the existing stationary kernel by evaluating it on warped inputs.
        z1 = x1 / lengthscale_x1
        z2 = x2 / lengthscale_x2

        distance = self.base_kernel._anisotropic_distance(z1, z2)
        stationary_part = torch.exp(-distance)

        sigma_outer = sigma_x1.unsqueeze(-1) * sigma_x2.unsqueeze(-2)
        covariance = sigma_outer * stationary_part

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
        model_config = MLPGPModelConfig.from_dict(model_cfg)

        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-10))

        super().__init__(X_train, y_train, likelihood)

        input_dim = int(X_train.shape[-1])
        local_lengthscale_dim = 1 if model_config.lengthscale_mode == "scalar" else input_dim

        self.feature_extractor = MLPGPFeatureExtractor(
            input_dim=input_dim,
            output_lengthscale_dim=local_lengthscale_dim,
            model_cfg=model_config,
        )
        self.mean_module = PolynomialTrendMean(input_dim=input_dim, degree=model_config.trend_degree)

        base_kernel = ExponentialKernel(
            input_dim=input_dim,
            initial_theta=model_config.kernel_theta,
            initial_power_exponent=model_config.kernel_power_exponent,
            initial_nugget=model_config.kernel_nugget,
        )

        nonstationary_kernel = NonstationaryExponentialKernel(
            base_kernel=base_kernel,
            min_lengthscale=model_config.lengthscale_floor,
            min_sigma=model_config.sigma_floor,
        )
        nonstationary_kernel.set_parameterizer(self._predict_local_parameters)

        self.covar_module = ScaleKernel(nonstationary_kernel)
        self.covar_module.outputscale = float(model_config.kernel_tau2)

        self.likelihood.noise = float(model_config.kernel_likelihood_noise)
        self.likelihood.raw_noise.requires_grad_(False)

        logger.info(
            "MLPGPModel initialized: "
            f"lengthscale_mode={model_config.lengthscale_mode}, "
            f"trend_degree={model_config.trend_degree}, "
            f"tau2={float(self.covar_module.outputscale.item()):.6f}, "
            f"noise={float(self.likelihood.noise.item()):.6e}"
        )

    def _predict_local_parameters(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma_x, lengthscale_x = self.feature_extractor(x)
        if lengthscale_x.shape[-1] == 1 and x.shape[-1] > 1:
            lengthscale_x = lengthscale_x.expand(-1, x.shape[-1])
        return sigma_x, lengthscale_x

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
