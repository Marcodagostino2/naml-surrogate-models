"""MLPGP models: ``MLPGPModel`` (exact) and ``MLPSVGPModel`` (sparse variational).

Both architectures share an MLP-driven feature/parameter extractor, the same
covariance modules, and the same ``forward`` (factored into ``_MLPGPMixin``).
Selection is driven by ``model_cfg["inference"]`` in the pipeline.

Two architecture variants:

- ``non_stationary``: Gibbs (Paciorek & Schervish 2006) kernel with an MLP
  predicting per-input ``sigma(x)`` and ``l(x)``. Requires the exponential base.
- ``dkl``: Deep Kernel Learning. MLP maps inputs to a small feature space,
  then a stationary kernel is evaluated on the features.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Tuple

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from src.models.gp import (
    PolynomialTrendMean,
    _build_gaussian_likelihood,
    _parse_noise_cfg,
    _select_inducing_points,
)
from src.models.gp_utils import (
    ExponentialKernel,
    build_kernel,
    extract_config_value,
    extract_kernel_name,
    extract_required_config_value,
)
from src.models.mlp import MLPRegressor
from src.utils import setup_logger

logger = setup_logger(__name__)


class NonstationaryFeatureExtractor(torch.nn.Module):
    """MLP that produces positive local ``sigma(x)`` and ``l(x)`` for the Gibbs kernel.

    Outputs are mapped to ``[10**floor, 10**ceiling]`` via a sigmoid-of-MLP.
    """

    def __init__(self, input_dim: int, output_lengthscale_dim: int, feature_cfg: Dict[str, Any]) -> None:
        super().__init__()

        self.log_sigma_floor = float(feature_cfg.get("log_sigma_floor", -5))
        self.log_sigma_ceiling = float(feature_cfg.get("log_sigma_ceiling", 3))
        self.log_lengthscale_floor = float(feature_cfg.get("log_lengthscale_floor", -5))
        self.log_lengthscale_ceiling = float(feature_cfg.get("log_lengthscale_ceiling", 3))
        self.output_lengthscale_dim = int(output_lengthscale_dim)

        if self.log_sigma_ceiling <= self.log_sigma_floor:
            raise ValueError(
                f"log_sigma_ceiling ({self.log_sigma_ceiling}) must be > log_sigma_floor ({self.log_sigma_floor})"
            )
        if self.log_lengthscale_ceiling <= self.log_lengthscale_floor:
            raise ValueError(
                f"log_lengthscale_ceiling ({self.log_lengthscale_ceiling}) must be > "
                f"log_lengthscale_floor ({self.log_lengthscale_floor})"
            )

        self.backbone = MLPRegressor(
            input_dim=input_dim,
            output_dim=1 + self.output_lengthscale_dim,
            model_cfg=feature_cfg,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_out = torch.sigmoid(self.backbone(x))
        sigma_x = torch.pow(
            10,
            self.log_sigma_floor
            + (self.log_sigma_ceiling - self.log_sigma_floor) * feature_out[..., :1],
        )
        lengthscale_x = torch.pow(
            10,
            self.log_lengthscale_floor
            + (self.log_lengthscale_ceiling - self.log_lengthscale_floor) * feature_out[..., 1:],
        )
        return sigma_x, lengthscale_x


class NonstationaryExponentialKernel(gpytorch.kernels.Kernel):
    """Gibbs (Paciorek & Schervish 2006) non-stationary kernel.

    .. math::

        K(x_i, x_j) = \\sigma_i \\sigma_j
                      \\prod_d \\sqrt{\\frac{2 \\, \\ell_{i,d} \\ell_{j,d}}{\\ell_{i,d}^2 + \\ell_{j,d}^2}}
                      \\, R_0(\\delta(x_i, x_j))

    with :math:`\\delta_d = \\theta_d \\bigl(|x_{i,d} - x_{j,d}| /
    \\sqrt{(\\ell_{i,d}^2 + \\ell_{j,d}^2)/2}\\bigr)^p` and :math:`R_0` the
    base ExponentialKernel.
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

    def set_parameterizer(self, parameterizer: torch.nn.Module) -> None:
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

    def _compute_norm_factor(self, lengthscale_x1: torch.Tensor, lengthscale_x2: torch.Tensor) -> torch.Tensor:
        L1_sq = lengthscale_x1.pow(2).unsqueeze(-2)
        L2_sq = lengthscale_x2.pow(2).unsqueeze(-3)
        L1 = lengthscale_x1.unsqueeze(-2)
        L2 = lengthscale_x2.unsqueeze(-3)
        per_dim = torch.sqrt(2.0 * L1 * L2 / (L1_sq + L2_sq))
        return torch.prod(per_dim, dim=-1)

    def _compute_scaled_distance(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        lengthscale_x1: torch.Tensor,
        lengthscale_x2: torch.Tensor,
    ) -> torch.Tensor:
        L1_sq = lengthscale_x1.pow(2).unsqueeze(-2)
        L2_sq = lengthscale_x2.pow(2).unsqueeze(-3)
        ell_harm_std = torch.sqrt((L1_sq + L2_sq) / 2.0)
        diff = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).abs()
        scaled = self.base_kernel.theta * (diff / ell_harm_std).pow(self.base_kernel.power_exponent)
        return torch.sum(scaled, dim=-1)

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
            x1, x2, sigma_x1, sigma_x2, lengthscale_x1, lengthscale_x2
        )

        if diag:
            return sigma_x1.pow(2)

        norm_factor = self._compute_norm_factor(lengthscale_x1, lengthscale_x2)
        dist = self._compute_scaled_distance(x1, x2, lengthscale_x1, lengthscale_x2)
        sigma_outer = sigma_x1.unsqueeze(-1) * sigma_x2.unsqueeze(-2)
        result = sigma_outer * norm_factor * torch.exp(-dist)

        # Different broadcast/reduction orders in _compute_norm_factor and
        # _compute_scaled_distance leak ~1e-7 asymmetries into K(X, X);
        # psd_safe_cholesky needs exact symmetry, so symmetrize and force the
        # diagonal to sigma(x_i)^2 (which forward(diag=True) returns).
        is_self_kernel = (x1 is x2) or (x1.shape == x2.shape and torch.equal(x1, x2))
        if is_self_kernel:
            result = 0.5 * (result + result.transpose(-2, -1))
            eye = torch.eye(result.shape[-1], dtype=torch.bool, device=result.device)
            result = torch.where(eye, torch.diag_embed(sigma_x1.pow(2)), result)
        return result


class DKLFeatureExtractor(torch.nn.Module):
    """MLP mapping ``x`` to a small feature space ``phi(x)`` for use as ``k(phi(x), phi(x'))``."""

    def __init__(self, input_dim: int, output_dim: int, feature_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        if self.output_dim > 10:
            logger.warning(
                f"DKL feature extractor output_dim={self.output_dim} is large; GP may be intractable."
            )
        self.backbone = MLPRegressor(input_dim=input_dim, output_dim=self.output_dim, model_cfg=feature_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DKLKernel(gpytorch.kernels.Kernel):
    """``k_DKL(x, x') = k_base(phi(x), phi(x'))`` with feature extractor ``phi``."""

    is_stationary = True
    has_lengthscale = False

    def __init__(self, base_kernel: gpytorch.kernels.Kernel, feature_extractor: DKLFeatureExtractor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.feature_extractor = feature_extractor

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **kwargs) -> torch.Tensor:
        return self.base_kernel(self.feature_extractor(x1), self.feature_extractor(x2), diag=diag, **kwargs)


class _MLPGPMixin:
    """Shared kernel/feature setup, parameter helpers, and ``forward`` for the
    exact and SV variants. No ``__init__`` so each variant calls its own parent
    (``ExactGP`` / ``ApproximateGP``) initializer.
    """

    def _init_mlpgp(
        self,
        X_train: torch.Tensor,
        model_cfg: Dict[str, Any],
        initial_likelihood_noise: float,
    ) -> None:
        input_dim = int(X_train.shape[-1])
        kernel_cfg = model_cfg.get("kernel", model_cfg)

        self.architecture_name = str(model_cfg.get("architecture_name", "non_stationary")).lower()
        if self.architecture_name not in {"non_stationary", "dkl"}:
            raise ValueError(
                f"Unknown architecture: {self.architecture_name}. Must be 'non_stationary' or 'dkl'."
            )
        if self.architecture_name == "non_stationary" and extract_kernel_name(kernel_cfg) != "exponential":
            raise ValueError("Non-stationary architecture requires kernel type='exponential'.")

        trend_degree = int(extract_required_config_value(model_cfg, "trend_degree", "MLPGP model config"))
        self.likelihood.initialize(noise=initial_likelihood_noise)
        self.mean_module = PolynomialTrendMean(input_dim=input_dim, degree=trend_degree)

        if self.architecture_name == "non_stationary":
            self._setup_nonstationary(input_dim, model_cfg)
        else:
            self._setup_dkl(input_dim, model_cfg)

        logger.info(
            f"{type(self).__name__} initialized\n"
            f"  architecture_name={self.architecture_name}\n"
            f"  trend_degree={trend_degree}\n"
            f"  likelihood_noise={float(self.likelihood.noise.item()):.6e}"
        )

    def _setup_nonstationary(self, input_dim: int, model_cfg: Dict) -> None:
        feature_cfg = model_cfg.get(
            "non_stationary_feature_extractor",
            model_cfg.get("feature_extractor", {}),
        )
        if not feature_cfg:
            raise KeyError("Missing 'non_stationary_feature_extractor' section in MLPGP model config.")

        lengthscale_mode = str(feature_cfg.get("lengthscale_mode", "per_dimension")).lower()
        if lengthscale_mode not in {"scalar", "per_dimension"}:
            raise ValueError("lengthscale_mode must be either 'scalar' or 'per_dimension'.")
        local_lengthscale_dim = 1 if lengthscale_mode == "scalar" else input_dim

        kernel_cfg = model_cfg.get("kernel", model_cfg)
        base_kernel = ExponentialKernel(
            input_dim=input_dim,
            initial_theta=extract_config_value(kernel_cfg, ("theta_init", "theta"), "MLPGP kernel config"),
            initial_power_exponent=extract_config_value(
                kernel_cfg, ("power_exponent_init", "power_exponent"), "MLPGP kernel config"
            ),
            theta_floor=float(extract_config_value(kernel_cfg, "theta_floor", "MLPGP kernel config", default=1.0e-5)),
            theta_ceiling=float(extract_config_value(kernel_cfg, "theta_ceiling", "MLPGP kernel config", default=1.0e3)),
            power_exponent_floor=float(extract_config_value(kernel_cfg, "power_exponent_floor", "MLPGP kernel config", default=1.5)),
            power_exponent_ceiling=float(extract_config_value(kernel_cfg, "power_exponent_ceiling", "MLPGP kernel config", default=2.0)),
        )

        self.feature_extractor = NonstationaryFeatureExtractor(
            input_dim=input_dim,
            output_lengthscale_dim=local_lengthscale_dim,
            feature_cfg=feature_cfg,
        )

        nonstationary_kernel = NonstationaryExponentialKernel(
            base_kernel=base_kernel,
            min_lengthscale=float(feature_cfg.get("lengthscale_floor", 1e-8)),
            min_sigma=float(feature_cfg.get("sigma_floor", 1e-8)),
        )
        nonstationary_kernel.set_parameterizer(self._predict_local_parameters)
        self.covar_module = nonstationary_kernel

    def _setup_dkl(self, input_dim: int, model_cfg: Dict) -> None:
        dkl_cfg = model_cfg.get("dkl_feature_extractor", model_cfg.get("dkl", {}))
        if not dkl_cfg:
            raise KeyError("Missing 'dkl_feature_extractor' section in MLPGP model config.")
        output_dim = int(dkl_cfg.get("output_width", dkl_cfg.get("output_dim", 3)))
        kernel_cfg = model_cfg.get("kernel", model_cfg)

        self.dkl_feature_extractor = DKLFeatureExtractor(
            input_dim=input_dim, output_dim=output_dim, feature_cfg=dkl_cfg,
        )
        self.covar_module = build_kernel(extract_kernel_name(kernel_cfg), output_dim, kernel_cfg)

    def _predict_local_parameters(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.architecture_name != "non_stationary":
            raise RuntimeError("_predict_local_parameters only valid for non_stationary architecture.")
        sigma_x, lengthscale_x = self.feature_extractor(x)
        if lengthscale_x.shape[-1] == 1 and x.shape[-1] > 1:
            lengthscale_x = lengthscale_x.expand(*lengthscale_x.shape[:-1], x.shape[-1])
        return sigma_x, lengthscale_x

    def feature_extractor_module(self) -> torch.nn.Module:
        return self.dkl_feature_extractor if self.architecture_name == "dkl" else self.feature_extractor

    def feature_extractor_parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.feature_extractor_module().parameters()

    def gp_parameters(self) -> Iterator[torch.nn.Parameter]:
        """All parameters not belonging to the MLP feature extractor."""
        feature_param_ids = {id(p) for p in self.feature_extractor_parameters()}
        for param in self.parameters():
            if id(param) not in feature_param_ids:
                yield param

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        covar_input = self.dkl_feature_extractor(x) if self.architecture_name == "dkl" else x
        return MultivariateNormal(self.mean_module(x), self.covar_module(covar_input))


class MLPGPModel(_MLPGPMixin, ExactGP):
    """Exact-inference MLPGP (non-stationary or DKL).

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
        kernel_cfg = model_cfg.get("kernel", model_cfg)
        initial_likelihood_noise, noise_floor, noise_ceiling = _parse_noise_cfg(kernel_cfg)
        if likelihood is None:
            likelihood = _build_gaussian_likelihood(initial_likelihood_noise, noise_floor, noise_ceiling)
        super().__init__(X_train, y_train, likelihood)
        self._init_mlpgp(X_train, model_cfg, initial_likelihood_noise)


class MLPSVGPModel(_MLPGPMixin, ApproximateGP):
    """Sparse Variational MLPGP. Inducing points are kept in raw input space so
    the existing kernel modules apply the MLP feature/parameter extractor on
    them via their normal forward pass.
    """

    def __init__(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        model_cfg: Dict,
        likelihood: Optional[gpytorch.likelihoods.Likelihood] = None,
    ) -> None:
        del y_train  # targets enter via the ELBO at training time

        kernel_cfg = model_cfg.get("kernel", model_cfg)
        initial_likelihood_noise, noise_floor, noise_ceiling = _parse_noise_cfg(kernel_cfg)
        if likelihood is None:
            likelihood = _build_gaussian_likelihood(initial_likelihood_noise, noise_floor, noise_ceiling)

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
        # ApproximateGP doesn't own a likelihood — attach it so its parameters
        # are picked up by self.parameters() and trained jointly.
        self.likelihood = likelihood

        self._init_mlpgp(X_train, model_cfg, initial_likelihood_noise)

        logger.info(
            f"MLPSVGPModel: {inducing_points.size(0)} inducing points "
            f"(learn_inducing_locations={learn_inducing_locations})"
        )
