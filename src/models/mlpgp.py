"""
End-to-end MLPGP model with nonstationary variance and lengthscale.

This implementation follows the "A Framework for Nonstationary Gaussian
Processes with Neural Network Parameters" idea by using an MLP to predict
input-dependent local kernel parameters.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Tuple

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.models import ExactGP

from src.models.gp import PolynomialTrendMean, _build_gaussian_likelihood
from src.models.gp_utils import ExponentialKernel, build_kernel, extract_kernel_name, extract_required_config_value
from src.models.mlp import MLPRegressor
from src.utils import setup_logger

logger = setup_logger(__name__)


def _require_config_value(config: Dict[str, Any], key: str, context: str) -> Any:
    """Legacy wrapper for extract_required_config_value (mlpgp-specific usage patterns)."""
    return extract_required_config_value(config, key, context)


class NonstationaryFeatureExtractor(torch.nn.Module):
    """MLP feature extractor producing positive local sigma(x) and l(x).

    Expects ``feature_cfg`` to be the resolved sub-config that contains
    ``architecture``, ``output_activation``, ``sigma_floor``, ``lengthscale_floor``
    directly at the top level (e.g. the dict under
    ``model_cfg["non_stationary_feature_extractor"]``).
    """

    def __init__(self, input_dim: int, output_lengthscale_dim: int, feature_cfg: Dict[str, Any]) -> None:
        super().__init__()

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
        force_p2: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.min_lengthscale = float(min_lengthscale)
        self.min_sigma = float(min_sigma)
        # When True, _compute_scaled_distance ignores base_kernel.power_exponent
        # and uses p=2 exactly. This is required by the Gibbs (Paciorek &
        # Schervish 2006) PSD theorem, which holds only for the squared-
        # exponential base. Using base_kernel.power_exponent directly is unsafe
        # when the kernel's Interval(1.5, 2.0) constraint cannot reach exactly
        # 2.0 from the raw parameter side.
        self.force_p2 = bool(force_p2)
        self.parameterizer: Optional[torch.nn.Module] = None

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

    def _compute_norm_factor(
        self, 
        lengthscale_x1: torch.Tensor, 
        lengthscale_x2: torch.Tensor
    ) -> torch.Tensor:
        """Compute Gibbs normalization factor: prod_d [ sqrt( 2 * L1_d * L2_d / (L1_sq_d + L2_sq_d) ) ].
        
        Parameters
        ----------
        lengthscale_x1 : Tensor (n, d)
            First set of lengthscales.
        lengthscale_x2 : Tensor (m, d)
            Second set of lengthscales.
        
        Returns
        -------
        Tensor (n, m)
            Normalization factor for all pairwise combinations.
        """
        L1_sq = lengthscale_x1.pow(2)
        L2_sq = lengthscale_x2.pow(2)
        L1_sq_broad = L1_sq.unsqueeze(-2)  # (n, 1, d)
        L2_sq_broad = L2_sq.unsqueeze(-3)  # (1, m, d)
        
        L1_broad = lengthscale_x1.unsqueeze(-2)  # (n, 1, d)
        L2_broad = lengthscale_x2.unsqueeze(-3)  # (1, m, d)
        
        per_dim_factor = torch.sqrt(2.0 * L1_broad * L2_broad / (L1_sq_broad + L2_sq_broad))
        return torch.prod(per_dim_factor, dim=-1)  # (n, m)

    def _compute_scaled_distance(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        lengthscale_x1: torch.Tensor,
        lengthscale_x2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scaled distance for Gibbs kernel evaluation.
        
        Implements: delta_d = theta_d * ( diff_d / sqrt(ell_harm_sq_d) )^p
        
        Parameters
        ----------
        x1, x2 : Tensor
            Input points.
        lengthscale_x1, lengthscale_x2 : Tensor
            Local lengthscales.
        
        Returns
        -------
        Tensor (n, m)
            Sum of scaled differences across dimensions.
        """
        L1_sq = lengthscale_x1.pow(2)
        L2_sq = lengthscale_x2.pow(2)
        L1_sq_broad = L1_sq.unsqueeze(-2)
        L2_sq_broad = L2_sq.unsqueeze(-3)
        
        ell_harm_sq = (L1_sq_broad + L2_sq_broad) / 2.0
        ell_harm_std = torch.sqrt(ell_harm_sq)
        
        diff = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).abs()
        
        theta = self.base_kernel.theta
        # Use exact p=2 when this kernel was constructed with force_p2 (the
        # non-stationary Gibbs PSD requirement). Otherwise read the learnable
        # power_exponent from the base kernel (DKL path - not currently using
        # this kernel class, but kept for symmetry).
        p = 2.0 if self.force_p2 else self.base_kernel.power_exponent
        scaled_diff = theta * (diff / ell_harm_std).pow(p)
        
        return torch.sum(scaled_diff, dim=-1)

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
            # So diag output = sigma1^2
            return sigma_x1.pow(2)

        # Compute normalization factor and scaled distance
        norm_factor = self._compute_norm_factor(lengthscale_x1, lengthscale_x2)
        dist = self._compute_scaled_distance(x1, x2, lengthscale_x1, lengthscale_x2)

        # Evaluate base kernel on the scaled distance
        R0 = torch.exp(-dist)

        # Compute sigma_outer: (n, 1) * (1, m) -> (n, m)
        sigma_outer = sigma_x1.unsqueeze(-1) * sigma_x2.unsqueeze(-2)

        # Full Gibbs kernel: sigma_outer * norm_factor * R0
        result = sigma_outer * norm_factor * R0

        # When evaluating K(X, X), enforce exact symmetry and a clean diagonal.
        # The Gibbs formula is mathematically symmetric in (i, j), but the
        # different broadcast/reduction orders along axes -2 vs -3 in
        # _compute_norm_factor and _compute_scaled_distance cause floating-point
        # asymmetries of ~1e-7 between K[i,j] and K[j,i]. psd_safe_cholesky
        # needs an exactly symmetric input, so we symmetrize. The diagonal is
        # forced to sigma(x_i)^2 to match what forward(diag=True) would
        # return; the off-diagonal formula evaluated at i==j computes the same
        # quantity but accumulates rounding through norm_factor and R0.
        is_self_kernel = (x1 is x2) or (
            x1.shape == x2.shape and torch.equal(x1, x2)
        )
        if is_self_kernel:
            result = 0.5 * (result + result.transpose(-2, -1))
            n = result.shape[-1]
            diag_idx = torch.arange(n, device=result.device)
            result[..., diag_idx, diag_idx] = sigma_x1.pow(2)

        return result


class DKLFeatureExtractor(torch.nn.Module):
    """
    Deep Kernel Learning feature extractor.

    Maps input x from R^input_dim to R^output_dim (typically small, 2-5).
    Used as phi(x) in k_DKL(x, x') = k_base(phi(x), phi(x')).

    Expects ``feature_cfg`` to be the resolved sub-config (the dict under
    ``model_cfg["dkl_feature_extractor"]``).
    """

    def __init__(self, input_dim: int, output_dim: int, feature_cfg: Dict[str, Any]) -> None:
        super().__init__()

        self.output_dim = int(output_dim)

        # Warn if output_dim is too large for tractable GP
        if self.output_dim > 10:
            logger.warning(
                f"DKL feature extractor output_dim={self.output_dim} is large; "
                "GP may become numerically intractable. Consider reducing to < 10."
            )

        # Reuse the block-based MLP used elsewhere in the codebase.
        # BatchNorm is controlled by config and should remain disabled for DKL.
        self.backbone = MLPRegressor(
            input_dim=input_dim,
            output_dim=self.output_dim,
            model_cfg=feature_cfg,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input through feature extractor.

        Parameters
        ----------
        x : Tensor (n, input_dim)
            Input data.

        Returns
        -------
        Tensor (n, output_dim)
            Transformed features phi(x).
        """
        return self.backbone(x)


class DKLKernel(gpytorch.kernels.Kernel):
    """
    Deep Kernel Learning kernel wrapper.

    Applies feature extractor phi(.) to both inputs, then evaluates
    base kernel on transformed features: k_DKL(x, x') = k_base(phi(x), phi(x')).
    """

    is_stationary = True
    has_lengthscale = False

    def __init__(
        self,
        base_kernel: gpytorch.kernels.Kernel,
        feature_extractor: DKLFeatureExtractor,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.feature_extractor = feature_extractor

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **kwargs) -> torch.Tensor:
        """
        Evaluate DKL kernel on raw inputs.

        Parameters
        ----------
        x1 : Tensor (n, d_in)
            First set of input points (raw input space).
        x2 : Tensor (m, d_in)
            Second set of input points (raw input space).
        diag : bool
            If True, return only diagonal of kernel matrix.

        Returns
        -------
        Tensor
            If diag=True: (n,) diagonal elements.
            If diag=False: (n, m) full kernel matrix.
        """
        # Transform inputs to feature space
        phi_x1 = self.feature_extractor(x1)  # (n, d_out)
        phi_x2 = self.feature_extractor(x2)  # (m, d_out)

        # Evaluate base kernel on transformed features
        return self.base_kernel(phi_x1, phi_x2, diag=diag, **kwargs)


class MLPGPModel(ExactGP):
    """ExactGP model supporting multiple architectures: "non_stationary" or "dkl"."""

    def __init__(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        model_cfg: Dict,
        likelihood: Optional[gpytorch.likelihoods.Likelihood] = None,
    ) -> None:
        kernel_cfg = model_cfg.get("kernel", model_cfg)
        initial_likelihood_noise = float(_require_config_value(kernel_cfg, "likelihood_noise", "MLPGP kernel config"))

        if likelihood is None:
            likelihood = _build_gaussian_likelihood(initial_likelihood_noise)

        super().__init__(X_train, y_train, likelihood)

        input_dim = int(X_train.shape[-1])

        # Extract architecture_name
        self.architecture_name = str(model_cfg.get("architecture_name", "non_stationary")).lower()
        if self.architecture_name not in {"non_stationary", "dkl"}:
            raise ValueError(f"Unknown architecture: {self.architecture_name}. Must be 'non_stationary' or 'dkl'.")

        # Extract kernel type; support both nested dict and direct string
        kernel_type = extract_kernel_name(kernel_cfg)

        # Validate kernel compatibility with architecture
        if self.architecture_name == "non_stationary" and kernel_type != "exponential":
            raise ValueError("Non-stationary architecture requires kernel type='exponential' (Gibbs kernel).")

        # Common setup shared by all architectures
        trend_degree = int(_require_config_value(model_cfg, "trend_degree", "MLPGP model config"))

        self.likelihood.initialize(noise=initial_likelihood_noise)
        self.mean_module = PolynomialTrendMean(input_dim=input_dim, degree=trend_degree)

        # Architecture-specific setup builds the kernel and logs its initial state.
        if self.architecture_name == "non_stationary":
            self._setup_nonstationary(input_dim, model_cfg)
        elif self.architecture_name == "dkl":
            self._setup_dkl(input_dim, model_cfg)

        logger.info(
            "MLPGPModel initialized\n"
            f"  architecture_name={self.architecture_name}\n"
            f"  trend_degree={trend_degree}\n"
            f"  likelihood_noise={float(self.likelihood.noise.item()):.6e}"
        )

    def _setup_nonstationary(self, input_dim: int, model_cfg: Dict) -> None:
        """Build the non-stationary covariance: K(x,x') = sigma(x) sigma(x') R0(z(x), z(x')).

        For the Gibbs / non-stationary architecture we hard-pin power_exponent
        to 2 (squared exponential) and freeze it. The Gibbs construction with
        the sqrt(2 L_i L_j / (L_i^2 + L_j^2)) normalization is a PSD kernel
        only for the squared-exponential base (Paciorek & Schervish, 2006);
        for p != 2 the resulting matrix has no PSD guarantee and Cholesky
        will fail with no amount of jitter able to save it. The DKL
        architecture keeps p learnable since it doesn't use the Gibbs
        construction.
        """
        feature_cfg = model_cfg.get(
            "non_stationary_feature_extractor",
            model_cfg.get("feature_extractor", {}),
        )
        if not feature_cfg:
            raise KeyError(
                "Missing 'non_stationary_feature_extractor' section in MLPGP model config."
            )

        lengthscale_mode = str(feature_cfg.get("lengthscale_mode", "per_dimension")).lower()
        if lengthscale_mode not in {"scalar", "per_dimension"}:
            raise ValueError("lengthscale_mode must be either 'scalar' or 'per_dimension'.")

        local_lengthscale_dim = 1 if lengthscale_mode == "scalar" else input_dim

        kernel_cfg = model_cfg.get("kernel", model_cfg)
        initial_theta = _require_config_value(kernel_cfg, "theta", "MLPGP kernel config")

        # Warn if the config requests a non-2 power_exponent. The kernel below
        # will use p=2 regardless (via force_p2), but flagging the override
        # makes user (and Optuna) intent visible. If you have power_exponent
        # in your hyperparameter search space for non_stationary, remove it
        # there as well — the dimension is frozen and would waste budget.
        configured_p = kernel_cfg.get("power_exponent", 2.0)
        configured_p_scalar = (
            float(configured_p[0]) if isinstance(configured_p, (list, tuple))
            else float(configured_p)
        )
        if abs(configured_p_scalar - 2.0) > 1e-6:
            logger.warning(
                "Non-stationary MLPGP requires power_exponent=2 (Gibbs PSD "
                f"requirement). Configured value {configured_p_scalar} is "
                "ignored; using p=2.0."
            )

        # base_kernel is built normally (its raw_power_exponent stays a
        # nn.Parameter), but the wrapping NonstationaryExponentialKernel will
        # use p=2 directly via force_p2. We also freeze raw_power_exponent so
        # the optimizer can't drift it — even though force_p2 makes the value
        # unused, freezing keeps things tidy and prevents confusion if a
        # future maintainer inspects parameter histories.
        base_kernel = ExponentialKernel(
            input_dim=input_dim,
            initial_theta=initial_theta,
            initial_power_exponent=2.0,
        )
        base_kernel.raw_power_exponent.requires_grad_(False)

        self.feature_extractor = NonstationaryFeatureExtractor(
            input_dim=input_dim,
            output_lengthscale_dim=local_lengthscale_dim,
            feature_cfg=feature_cfg,
        )

        nonstationary_kernel = NonstationaryExponentialKernel(
            base_kernel=base_kernel,
            min_lengthscale=float(feature_cfg.get("lengthscale_floor", 1e-8)),
            min_sigma=float(feature_cfg.get("sigma_floor", 1e-8)),
            force_p2=True,
        )
        nonstationary_kernel.set_parameterizer(self._predict_local_parameters)
        self.covar_module = nonstationary_kernel

        logger.info(
            f"Non-stationary kernel ready (lengthscale_mode={lengthscale_mode}, "
            f"power_exponent=2.0 [frozen via force_p2])"
        )

    def _setup_dkl(self, input_dim: int, model_cfg: Dict) -> None:
        """Build the DKL covariance: K(x,x') = tau^2 * R0(phi(x), phi(x')).

        The amplitude ``tau^2`` is exposed as a learnable ``ScaleKernel`` outputscale,
        mirroring the standard GP and giving the model a learnable prior variance.
        
        For exponential kernels: theta and power_exponent are learnable parameters initialized
        from config. Theta is re-parameterized to the feature space dimensionality (output_dim).
        """
        dkl_cfg = model_cfg.get("dkl_feature_extractor", model_cfg.get("dkl", {}))
        if not dkl_cfg:
            raise KeyError("Missing 'dkl_feature_extractor' section in MLPGP model config.")
        output_dim = int(dkl_cfg.get("output_width", dkl_cfg.get("output_dim", 3)))

        kernel_cfg = model_cfg.get("kernel", model_cfg)
        
        # Extract kernel name; support both nested dict and direct string
        kernel_name = extract_kernel_name(kernel_cfg)
        
        initial_tau2 = float(kernel_cfg.get("tau2", 1.0))

        # Feature extractor maps raw input -> small feature space used by the GP kernel.
        self.dkl_feature_extractor = DKLFeatureExtractor(
            input_dim=input_dim,
            output_dim=output_dim,
            feature_cfg=dkl_cfg,
        )

        # Base kernel is sized to the feature space, not the raw input.
        # For exponential, need to re-parameterize theta to feature space dimensions
        if kernel_name == "exponential":
            initial_theta = _require_config_value(kernel_cfg, "theta", "DKL kernel config")
            initial_p = _require_config_value(kernel_cfg, "power_exponent", "DKL kernel config")
            
            # theta lives in the feature space; expand a scalar / take first element of a vector.
            # This ensures the exponential kernel operates on output_dim features, not raw input_dim.
            if isinstance(initial_theta, (int, float)):
                theta_value = float(initial_theta)
            else:
                theta_value = float(initial_theta[0])
            dkl_theta = [theta_value] * output_dim
            
            # Create exponential kernel with feature-space dimensions
            dkl_base_kernel = ExponentialKernel(
                input_dim=output_dim,
                initial_theta=dkl_theta,
                initial_power_exponent=initial_p,
            )
        else:
            # For RBF/Matern, use the factory (which wraps in ScaleKernel)
            # Extract the inner kernel from the factory's ScaleKernel wrapper
            scaled_kernel = build_kernel(kernel_name, output_dim, kernel_cfg)
            if isinstance(scaled_kernel, ScaleKernel):
                dkl_base_kernel = scaled_kernel.base_kernel
            else:
                dkl_base_kernel = scaled_kernel

        # ScaleKernel adds the learnable outputscale tau^2 (matches GPRegressionModel).
        self.covar_module = ScaleKernel(dkl_base_kernel)
        self.covar_module.outputscale = initial_tau2

    def _predict_local_parameters(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper for nonstationary architecture."""
        if self.architecture_name != "non_stationary":
            raise RuntimeError("_predict_local_parameters only valid for non_stationary architecture.")
        sigma_x, lengthscale_x = self.feature_extractor(x)
        if lengthscale_x.shape[-1] == 1 and x.shape[-1] > 1:
            raise ValueError("Scalar lengthscale output is not compatible with multi-dimensional input.")
        return sigma_x, lengthscale_x

    def feature_extractor_module(self) -> torch.nn.Module:
        """Return the active feature-extractor MLP for the configured architecture.

        Centralizing this lookup lets the trainer treat both architectures uniformly,
        without branching on ``architecture_name`` or knowing the attribute name.
        """
        if self.architecture_name == "dkl":
            return self.dkl_feature_extractor
        return self.feature_extractor

    def feature_extractor_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Iterate over feature-extractor parameters (MLP weights/biases)."""
        return self.feature_extractor_module().parameters()

    def gp_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Iterate over GP-side parameters (kernel hyperparams, mean trend, likelihood).

        Defined as ``self.parameters()`` minus the feature-extractor parameters,
        so any future kernel parameter (e.g. ``raw_outputscale`` of a ScaleKernel)
        is picked up automatically without changing the trainer.
        """
        feature_param_ids = {id(p) for p in self.feature_extractor_parameters()}
        for param in self.parameters():
            if id(param) not in feature_param_ids:
                yield param

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_input = self.dkl_feature_extractor(x) if self.architecture_name == "dkl" else x
        covar_x = self.covar_module(covar_input)
        return MultivariateNormal(mean_x, covar_x)