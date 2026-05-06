from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import gpytorch
import torch
from gpytorch.constraints import GreaterThan, Interval, LessThan
from gpytorch.kernels import ScaleKernel

from src.utils import setup_logger

logger = setup_logger(__name__)

_MISSING = object()


def extract_config_value(config: Dict[str, Any], key: str | Sequence[str], context: str, default: Any = _MISSING) -> Any:
    """Return the first matching config value from one or more alias keys."""
    key_names = (key,) if isinstance(key, str) else tuple(key)
    for candidate in key_names:
        if candidate in config:
            return config[candidate]
    if default is not _MISSING:
        return default
    if len(key_names) == 1:
        raise KeyError(f"Missing required config key '{key_names[0]}' in {context}.")
    alias_text = " or ".join(f"'{candidate}'" for candidate in key_names)
    raise KeyError(f"Missing required config key {alias_text} in {context}.")


def extract_required_config_value(config: Dict[str, Any], key: str | Sequence[str], context: str) -> Any:
    """Extract a required config value, accepting a single key or aliases."""
    return extract_config_value(config, key, context)


def clamp_to_bounds(value: torch.Tensor, lower: Optional[float], upper: Optional[float]) -> torch.Tensor:
    """Clamp a tensor while keeping inverse transforms finite for bounded parameters."""
    if lower is None and upper is None:
        return value

    if lower is not None and upper is not None:
        lower_value = float(lower)
        upper_value = float(upper)
        if upper_value <= lower_value:
            raise ValueError(f"Invalid bounds: low={lower_value} must be < high={upper_value}")

        epsilon = torch.finfo(value.dtype).eps
        if upper_value - lower_value <= 2.0 * epsilon:
            raise ValueError(f"Bounds too tight: low={lower_value}, high={upper_value}")

        lower_tensor = torch.as_tensor(lower_value + epsilon, dtype=value.dtype, device=value.device)
        upper_tensor = torch.as_tensor(upper_value - epsilon, dtype=value.dtype, device=value.device)
        return value.clamp(min=lower_tensor, max=upper_tensor)

    if lower is not None:
        lower_tensor = torch.as_tensor(float(lower), dtype=value.dtype, device=value.device)
        return value.clamp_min(lower_tensor)

    upper_tensor = torch.as_tensor(float(upper), dtype=value.dtype, device=value.device)
    return value.clamp_max(upper_tensor)


class ExponentialKernel(gpytorch.kernels.Kernel):
    """Anisotropic generalized exponential covariance with learnable theta and power exponent.
    
    Parameters theta and power_exponent are learnable (constrained) parameters that are
    initialized from config values and optimized during training.
    
    Constraints:
    - theta: Interval(theta_floor, theta_ceiling)
    - power_exponent: Interval(power_exponent_floor, power_exponent_ceiling)
    """

    is_stationary = True
    has_lengthscale = False

    def __init__(
        self,
        input_dim: int,
        initial_theta: float | Sequence[float],
        initial_power_exponent: float | Sequence[float],
        theta_floor: float = 1.0e-5,
        theta_ceiling: float = 1.0e3,
        power_exponent_floor: float = 1.5,
        power_exponent_ceiling: float = 2.0,
        **kwargs,
    ) -> None:
        """
        Initialize ExponentialKernel.
        
        Parameters
        ----------
        input_dim : int
            Number of input dimensions (for ARD).
        initial_theta : float or Sequence[float]
            Initial value for theta (learnable parameter). If scalar, expanded to all dimensions.
        initial_power_exponent : float or Sequence[float]
            Initial value for power exponent (learnable parameter). If scalar, expanded to all dimensions.
        """
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)

        theta_floor = float(theta_floor)
        theta_ceiling = float(theta_ceiling)
        power_exponent_floor = float(power_exponent_floor)
        power_exponent_ceiling = float(power_exponent_ceiling)

        if theta_ceiling <= theta_floor:
            raise ValueError(f"theta_ceiling ({theta_ceiling}) must be greater than theta_floor ({theta_floor}).")
        if power_exponent_ceiling <= power_exponent_floor:
            raise ValueError(
                f"power_exponent_ceiling ({power_exponent_ceiling}) must be greater than "
                f"power_exponent_floor ({power_exponent_floor})."
            )

        initial_theta_tensor = self._coerce_vector(initial_theta, "initial_theta")
        initial_power_tensor = self._coerce_vector(initial_power_exponent, "initial_power_exponent")

        initial_theta_tensor = clamp_to_bounds(initial_theta_tensor, theta_floor, theta_ceiling)
        initial_power_tensor = clamp_to_bounds(initial_power_tensor, power_exponent_floor, power_exponent_ceiling)

        theta_constraint = Interval(theta_floor, theta_ceiling)
        power_constraint = Interval(power_exponent_floor, power_exponent_ceiling)

        self.register_parameter(
            "raw_theta",
            torch.nn.Parameter(torch.zeros_like(initial_theta_tensor)),
        )
        self.register_parameter(
            "raw_power_exponent",
            torch.nn.Parameter(torch.zeros_like(initial_power_tensor)),
        )
        self.register_constraint("raw_theta", theta_constraint)
        self.register_constraint("raw_power_exponent", power_constraint)

        with torch.no_grad():
            self.raw_theta.copy_(theta_constraint.inverse_transform(initial_theta_tensor))
            self.raw_power_exponent.copy_(power_constraint.inverse_transform(initial_power_tensor))

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

def _build_base_kernel(
    kernel_name: str,
    num_dims: int,
    config: Dict[str, Any],
) -> gpytorch.kernels.Kernel:
    kernel_name = str(kernel_name).lower() if kernel_name else "exponential"

    if kernel_name == "exponential":
        initial_theta = extract_config_value(config, ("theta_init", "theta"), "GP kernel config", default=1.0)
        initial_power_exponent = extract_config_value(
            config,
            ("power_exponent_init", "power_exponent"),
            "GP kernel config",
            default=2.0,
        )
        theta_floor = float(extract_config_value(config, "theta_floor", "GP kernel config", default=1.0e-5))
        theta_ceiling = float(extract_config_value(config, "theta_ceiling", "GP kernel config", default=1.0e3))
        power_exponent_floor = float(
            extract_config_value(config, "power_exponent_floor", "GP kernel config", default=1.5)
        )
        power_exponent_ceiling = float(
            extract_config_value(config, "power_exponent_ceiling", "GP kernel config", default=2.0)
        )
        return ExponentialKernel(
            input_dim=num_dims,
            initial_theta=initial_theta,
            initial_power_exponent=initial_power_exponent,
            theta_floor=theta_floor,
            theta_ceiling=theta_ceiling,
            power_exponent_floor=power_exponent_floor,
            power_exponent_ceiling=power_exponent_ceiling,
        )

    if kernel_name == "rbf":
        return gpytorch.kernels.RBFKernel(ard_num_dims=num_dims)

    if kernel_name == "matern":
        nu = float(config.get("matern_nu", 2.5))
        return gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=num_dims)

    raise ValueError(f"Unknown kernel: {kernel_name}")

def build_kernel(
    kernel_name: str,
    num_dims: int,
    config: Dict[str, Any],
) -> gpytorch.kernels.Kernel:
    """Factory for creating kernels from config.
    
    For exponential kernels, theta and power_exponent from config are used as
    initial values for learnable parameters (constrained via register_constraint).
    
    Parameters
    ----------
    kernel_name : str
        Kernel type: "exponential", "rbf", or "matern".
    num_dims : int
        Input dimensionality (for ARD).
    config : Dict[str, Any]
        Config dict with kernel-specific keys (theta, power_exponent, tau2, matern_nu, etc.).
    
    Returns
    -------
    gpytorch.kernels.Kernel
        Kernel wrapped in ScaleKernel for learnable outputscale (tau2).
    """
    kernel_name = str(kernel_name).lower() if kernel_name else "exponential"

    base_kernel = _build_base_kernel(kernel_name, num_dims, config)

    scaled_kernel = ScaleKernel(base_kernel)
    tau2 = float(
        extract_config_value(
            config,
            ("tau2_init"),
            "GP kernel config",
            default=1.0,
        )
    )
    outputscale_floor = extract_config_value(
        config,
        ("tau2_floor"),
        "GP kernel config",
        default=None,
    )
    outputscale_ceiling = extract_config_value(
        config,
        ("tau2_ceiling"),
        "GP kernel config",
        default=None,
    )

    if outputscale_floor is not None or outputscale_ceiling is not None:
        if outputscale_floor is not None and outputscale_ceiling is not None:
            if float(outputscale_ceiling) <= float(outputscale_floor):
                raise ValueError(
                    f"Invalid outputscale bounds: low={outputscale_floor} must be < high={outputscale_ceiling}"
                )
            outputscale_constraint = Interval(float(outputscale_floor), float(outputscale_ceiling))
        elif outputscale_floor is not None:
            outputscale_constraint = GreaterThan(float(outputscale_floor))
        else:
            outputscale_constraint = LessThan(float(outputscale_ceiling))
        scaled_kernel.register_constraint("raw_outputscale", outputscale_constraint)
        tau2 = float(
            clamp_to_bounds(
                torch.as_tensor(tau2, dtype=torch.get_default_dtype()),
                outputscale_floor,
                outputscale_ceiling,
            ).item()
        )
    else:
        tau2 = max(float(tau2), 1.0e-6)

    scaled_kernel.outputscale = tau2
    return scaled_kernel

def extract_kernel_name(kernel_cfg: Dict[str, Any] | str) -> str:
    """Extract kernel name from config, supporting both nested dict and direct string formats.
    
    Parameters
    ----------
    kernel_cfg : Dict[str, Any] | str
        Kernel config (dict with optional "type"/"kernel" keys or direct string).
    
    Returns
    -------
    str
        Kernel name in lowercase. Defaults to "exponential" if not found.
    """
    if isinstance(kernel_cfg, dict):
        return str(kernel_cfg.get("type", kernel_cfg.get("kernel", "exponential"))).lower()
    return str(kernel_cfg).lower() if kernel_cfg else "exponential"