"""Optuna helpers, search-space translation, and result persistence for HPO.

Plotting lives in ``src.visualization.suites.optimization`` — this module is
plot-free.
"""

from __future__ import annotations

import csv
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import gpytorch
import torch
import yaml

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency for optimization workflows
    optuna = None

if TYPE_CHECKING:
    from optuna import Study, Trial
    from optuna.pruners import BasePruner
    from optuna.samplers import BaseSampler
    from optuna.trial import FrozenTrial

from src.pipeline import save_run_artifacts, train_model
from src.utils import OutputPathResolver, RunType, set_global_seed, setup_logger

logger = setup_logger(__name__)


# Static hyperparameters describing the single hidden block, shared by mlp
# and the feature extractors of mlpgp. ``layer_size`` is a *template*: at
# trial time exactly ``n_layers`` width params (``layer1_size`` ..
# ``layer{n_layers}_size``) are suggested dynamically, so the search-space
# dimensionality reflects the actual depth — no fixed maximum.
_BLOCK_HPS = ("n_layers", "activation", "layer_size")
_LAYER_SIZE_TEMPLATE = "layer_size"
_LAYER_SIZE_RE = re.compile(r"layer\d+_size")


def _is_dynamic_layer_size(name: str) -> bool:
    return bool(_LAYER_SIZE_RE.fullmatch(name))


_MLPGP_HP_DESTINATIONS: Dict[str, str] = {
    **{hp: "feature_extractor" for hp in _BLOCK_HPS},
    "dropout": "feature_extractor",
    "batch_norm": "feature_extractor",
    "output_activation": "feature_extractor",
    "output_width": "feature_extractor",
    "sigma_floor": "feature_extractor",
    "lengthscale_floor": "feature_extractor",
    "lengthscale_mode": "feature_extractor",
    "tau2_init": "kernel",
    "theta_init": "kernel",
    "power_exponent_init": "kernel",
    "likelihood_noise_init": "kernel",
    "trend_degree": "root",
}

_HP_TO_MODEL_CFG: Dict[tuple[str, str], str] = {
    ("mlp", "dropout"): "dropout",
    **{("mlp", hp): "architecture" for hp in _BLOCK_HPS},

    ("gp", "tau2_init"): "kernel",
    ("gp", "theta_init"): "kernel",
    ("gp", "power_exponent_init"): "kernel",
    ("gp", "likelihood_noise_init"): "kernel",
    ("gp", "trend_degree"): "root",

    **{("mlpgp_ns", hp): dest for hp, dest in _MLPGP_HP_DESTINATIONS.items()},
    **{("mlpgp_dkl", hp): dest for hp, dest in _MLPGP_HP_DESTINATIONS.items()},
}


def _hp_destination(model_name: str, hp_name: str) -> Optional[str]:
    """Return the registry destination, mapping ``layer{i}_size`` through the template."""
    if _is_dynamic_layer_size(hp_name):
        return _HP_TO_MODEL_CFG.get((model_name, _LAYER_SIZE_TEMPLATE))
    return _HP_TO_MODEL_CFG.get((model_name, hp_name))

# Type coercion for hyperparameters mapped to the "root" destination
# (top-level model_cfg keys, not nested under kernel/feature_extractor).
_ROOT_COERCERS: Dict[str, Callable[[Any], Any]] = {
    "trend_degree": int,
}


def _resolve_architecture_name(model_name: str, opt_cfg: Dict[str, Any], base_model_cfg: Dict[str, Any]) -> str:
    """Resolve the MLPGP architecture for ``model_name`` (canonical names are authoritative)."""
    if model_name == "mlpgp_dkl":
        return "dkl"
    if model_name == "mlpgp_ns":
        return "non_stationary"
    architecture_name = opt_cfg.get("architecture_name", base_model_cfg.get("architecture_name", "non_stationary"))
    return str(architecture_name).lower()


def _resolve_active_search_space(
    model_name: str,
    opt_cfg: Dict[str, Any],
    base_model_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Select the active search space for the current architecture."""
    architecture_name = _resolve_architecture_name(model_name, opt_cfg, base_model_cfg)

    search_spaces = opt_cfg.get("search_spaces")
    if isinstance(search_spaces, dict):
        if architecture_name in search_spaces:
            return search_spaces[architecture_name]

    search_space = opt_cfg.get("search_space", {})
    if isinstance(search_space, dict):
        if "model" in search_space or "training" in search_space:
            return search_space
        if architecture_name in search_space and isinstance(search_space[architecture_name], dict):
            return search_space[architecture_name]

    raise KeyError(
        f"No search space found for architecture '{architecture_name}'. "
        "Expected opt_cfg['search_spaces'][architecture_name] or a flat opt_cfg['search_space']."
    )


def _suggest_one(trial: Trial, name: str, spec: Dict[str, Any]) -> Any:
    param_type = spec.get("type")
    if param_type == "int":
        return trial.suggest_int(
            name,
            int(spec.get("low", 1)),
            int(spec.get("high", 10)),
            step=int(spec.get("step", 1)),
        )
    if param_type == "float":
        return trial.suggest_float(
            name,
            float(spec.get("low", 0.0)),
            float(spec.get("high", 1.0)),
            log=bool(spec.get("log", False)),
        )
    if param_type == "categorical":
        return trial.suggest_categorical(name, spec.get("choices", []))
    raise ValueError(f"Unknown parameter type: {param_type}")


def suggest_from_config(trial: Trial, search_space: Dict[str, Any], target: str) -> Dict[str, Any]:
    """Convert a YAML search space section into Optuna suggestions.

    A ``layer_size`` entry is treated as a template: after sampling
    ``n_layers``, exactly that many ``layer{i}_size`` params are suggested
    dynamically using the same spec, so the search-space dimensionality
    matches the chosen depth.
    """
    if target not in search_space:
        return {}

    target_space = search_space[target]
    params: Dict[str, Any] = {}
    for param_name, param_spec in target_space.items():
        if param_name == _LAYER_SIZE_TEMPLATE:
            continue
        params[param_name] = _suggest_one(trial, param_name, param_spec)

    layer_template = target_space.get(_LAYER_SIZE_TEMPLATE)
    n_layers = params.get("n_layers")
    if layer_template is not None and n_layers is not None:
        for i in range(1, int(n_layers) + 1):
            name = f"layer{i}_size"
            params[name] = _suggest_one(trial, name, layer_template)

    return params


def overlay(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge patch into base without mutating either dictionary."""
    result = dict(base)
    for key, value in patch.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = overlay(result[key], value)
        else:
            result[key] = value
    return result


def _format_duration(seconds: float) -> str:
    """Render a duration as ``Hh MMm SSs`` (compact, drops leading zeros)."""
    if not (seconds == seconds) or seconds < 0:  # nan or negative
        return "?"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _make_logging_callback(
    model_name: str,
    n_trials: int,
    metric_name: str,
    log_every: int = 5,
) -> Callable[[Study, FrozenTrial], None]:
    """Create a compact Optuna callback that prints periodic progress + ETA.

    ETA is a moving estimate: ``(n_trials - completed) * (elapsed / completed)``.
    Anchored on the first trial so the first log line still prints something.
    """
    start_time: Dict[str, float] = {}

    def _compact_param_summary(params: Dict[str, Any]) -> str:
        if not params:
            return "{}"

        if model_name in {"mlp", "mlpgp_ns", "mlpgp_dkl"}:
            activation = params.get("activation")
            layer_keys = sorted([key for key in params if _is_dynamic_layer_size(key)])
            layers = [str(int(params[key])) for key in layer_keys]
            lr = params.get("learning_rate")
            wd = params.get("weight_decay")
            parts: List[str] = []
            if layers:
                parts.append(f"layers=[{','.join(layers)}]")
            if activation is not None:
                parts.append(f"act={activation}")
            if lr is not None:
                parts.append(f"lr={float(lr):.2e}")
            if wd is not None:
                parts.append(f"wd={float(wd):.2e}")
            return "{" + ", ".join(parts) + "}" if parts else str(params)

        if model_name == "gp":
            kernel_keys = ["tau2_init", "theta_init", "power_exponent_init", "likelihood_noise_init"]
            parts = [
                f"{key}={float(params[key]):.2e}" for key in kernel_keys if key in params
            ]
            return "{" + ", ".join(parts) + "}" if parts else str(params)

        return str(params)

    def callback(study: Study, trial: FrozenTrial) -> None:
        start_time.setdefault("t0", time.perf_counter())
        try:
            best = study.best_value
            best_trial = study.best_trial
        except ValueError:
            best = float("nan")
            best_trial = None
        current = trial.value if trial.value is not None else float("nan")
        completed = trial.number + 1
        elapsed = time.perf_counter() - start_time["t0"]
        eta = (n_trials - completed) * (elapsed / completed) if completed > 0 else float("nan")
        pct = completed / max(n_trials, 1) * 100

        logger.info(
            f"[{model_name.upper()}] trial {completed:>3}/{n_trials} ({pct:4.0f}%) | "
            f"current={current:.5f} | best={best:.5f} | params={_compact_param_summary(trial.params)}"
        )

        if log_every <= 0 or trial.number % log_every == 0:
            best_params_summary = _compact_param_summary(best_trial.params) if best_trial is not None else "{}"
            best_trial_number = (best_trial.number + 1) if best_trial is not None else "?"
            logger.info(
                f"[{model_name.upper()}] best@trial={best_trial_number} | "
                f"metric={metric_name} | params={best_params_summary} | "
                f"elapsed {_format_duration(elapsed)} | ETA {_format_duration(eta)}"
            )

    return callback


def compute_val_nll(model: Any, X_val_t: torch.Tensor, y_val_t: torch.Tensor) -> float:
    """Compute negative log-likelihood on a validation set."""
    model.eval()
    model_device = next(model.parameters()).device
    X_val_t = X_val_t.to(model_device)
    y_val_t = y_val_t.to(model_device)
    with torch.no_grad(), torch.inference_mode():
        f_pred = model(X_val_t)
        likelihood = model.likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        nll = -mll(f_pred, y_val_t)
    return float(nll.item())


def build_sampler(study_cfg: Dict[str, Any]) -> BaseSampler:
    """Create an Optuna sampler from study configuration."""
    if optuna is None:
        raise ModuleNotFoundError("optuna is required to build optimization samplers.")

    sampler_name = str(study_cfg.get("sampler", "tpe")).lower()
    seed = int(study_cfg.get("seed", 42))

    from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

    if sampler_name == "tpe":
        return TPESampler(seed=seed, multivariate=study_cfg.get("multivariate", False), n_startup_trials=study_cfg.get("n_startup_trials", 10))
    if sampler_name == "gp":
        from optuna.samplers import GPSampler
        return GPSampler(seed=seed)
    if sampler_name == "random":
        return RandomSampler(seed=seed)
    if sampler_name == "cmaes":
        return CmaEsSampler(seed=seed, restart_strategy=study_cfg.get("cmaes_restart", "ipop"))
    if sampler_name == "qmc":
        from optuna.samplers import QMCSampler
        return QMCSampler(seed=seed, qmc_type="sobol")
    if sampler_name == "grid":
        from optuna.samplers import GridSampler
        grid = study_cfg.get("grid_search_space")
        if grid is None:
            raise ValueError("GridSampler richiede 'grid_search_space' nel config")
        return GridSampler(grid)
    raise ValueError(f"Unknown sampler: {sampler_name}")


def build_pruner(study_cfg: Dict[str, Any]) -> BasePruner:
    """Create an Optuna pruner from study configuration."""
    if optuna is None:
        raise ModuleNotFoundError("optuna is required to build optimization pruners.")

    pruner_name = study_cfg.get("pruner", "median")

    from optuna.pruners import MedianPruner, NopPruner
    if pruner_name is None or pruner_name == "null":
        return NopPruner()
    if pruner_name == "median":
        return MedianPruner(
            n_startup_trials=study_cfg.get("n_startup_trials", 5),
            n_warmup_steps=study_cfg.get("n_warmup_steps", 0),
            interval_steps=study_cfg.get("pruner_interval", 1),
        )
    raise ValueError(f"Unknown pruner: {pruner_name!r}. Valid: 'median', 'null', None.")


def _build_architecture(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a *partial* architecture patch from the sampled HPs.

    Emits ``layers`` only when ``n_layers`` and the matching
    ``layer{i}_size`` are all present (always coupled by the dynamic
    sampler); emits ``activation`` independently. Fields not in the patch
    are inherited from the base ``architecture`` via ``overlay``'s
    deep-merge — so the user can vary, e.g., only ``activation`` and
    keep the layer sizes from ``{model}_config.yaml``.
    """
    arch: Dict[str, Any] = {}
    n_layers = int(params.get("n_layers", 0))
    if n_layers > 0:
        layer_sizes = [params.get(f"layer{i}_size") for i in range(1, n_layers + 1)]
        if all(size is not None for size in layer_sizes):
            arch["layers"] = [int(size) for size in layer_sizes]
    if (activation := params.get("activation")) is not None:
        arch["activation"] = str(activation)
    return arch or None


def _apply_hp(model_name: str, params: Dict[str, Any], architecture_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Translate sampled hyperparameters into a config patch dict.

    Dispatch is fully driven by ``_HP_TO_MODEL_CFG``: each (model_name, hp_name)
    pair maps to a destination scope string, which determines how the value is
    written into the patch.  No model-specific branching lives in this function.

    Supported destination scopes:
    - ``"architecture"``       → patch["architecture"]           (mlp)
    - ``"dropout"``            → patch["dropout"]                (mlp)
    - ``"feature_extractor"``  → patch["feature_extractor"]      (mlpgp)
    - ``"kernel"``             → patch["kernel"]                 (gp, mlpgp)

    Parameters
    ----------
    model_name : str
        Model identifier used for registry lookup (e.g. ``"mlp"``, ``"gp"``, ``"mlpgp_ns"``, ``"mlpgp_dkl"``).
    scope : str
        Optimisation scope (``"model"`` or ``"training"``); currently unused but
        kept for forward compatibility.
    params : Dict[str, Any]
        Raw hyperparameter dict as returned by the Optuna sampler.

    Returns
    -------
    Dict[str, Any]
        Nested config patch ready to be merged into the model/training config.
    """
    scoped: Dict[str, Dict[str, Any]] = {}
    for hp_name, hp_value in params.items():
        dest = _hp_destination(model_name, hp_name)
        if dest is not None:
            scoped.setdefault(dest, {})[hp_name] = hp_value

    patch: Dict[str, Any] = {}

    if arch_params := scoped.get("architecture"):
        if (arch := _build_architecture(arch_params)) is not None:
            patch["architecture"] = arch

    if model_name in {"mlpgp_ns", "mlpgp_dkl"} and scoped.get("feature_extractor"):
        fe_params = scoped["feature_extractor"]
        feature_patch: Dict[str, Any] = {}

        if (arch := _build_architecture(fe_params)) is not None:
            feature_patch["architecture"] = arch

        for key in ["dropout", "batch_norm", "output_activation", "output_width", "sigma_floor", "lengthscale_floor", "lengthscale_mode"]:
            if key in fe_params:
                value = fe_params[key]
                if key in {"dropout", "sigma_floor", "lengthscale_floor"}:
                    value = float(value)
                elif key == "batch_norm":
                    value = bool(value)
                elif key == "output_width":
                    value = int(value)
                else:
                    value = str(value)
                feature_patch[key] = value

        if feature_patch:
            patch["dkl_feature_extractor" if architecture_name == "dkl" else "non_stationary_feature_extractor"] = feature_patch

    if kernel_params := scoped.get("kernel"):
        kernel = {k: float(v) for k, v in kernel_params.items()}
        if kernel:
            patch["kernel"] = kernel

    if root_params := scoped.get("root"):
        for key, value in root_params.items():
            coerce = _ROOT_COERCERS.get(key, lambda x: x)
            patch[key] = coerce(value)

    return patch


def _extract_metric(results: Dict[str, Any], metric_name: str, data_bundle: Any, model: Any) -> float:
    """Extract the target metric from validation metrics or validation NLL."""
    if metric_name == "val_nll":
        X_val_t = torch.from_numpy(data_bundle.X_val)
        y_val_t = torch.from_numpy(data_bundle.y_val)
        if y_val_t.ndim == 2 and y_val_t.shape[1] == 1:
            y_val_t = y_val_t.squeeze(-1)
        return compute_val_nll(model, X_val_t, y_val_t)

    metric_key = metric_name.replace("val_", "")
    value = results["val_metrics"].get(metric_key)
    if value is None:
        raise ValueError(f"Metric '{metric_key}' not in val_metrics")
    if metric_key == "r2":
        return 1.0 - float(value)
    return float(value)


def make_objective(
    model_name: str,
    data_bundle: Any,
    base_model_cfg: Dict[str, Any],
    base_training_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    opt_cfg: Dict[str, Any],
    device: torch.device,
    seed: Optional[int] = None,
    n_seeds: int = 3,
) -> Callable[[Trial], float]:
    """Create an Optuna objective bound to the shared training pipeline.

    When ``seed`` is provided, every trial re-seeds Python/NumPy/PyTorch
    before training so that only the sampled HPs vary across trials —
    init RNG state is not inherited from the previous trial.

    Multi-seed robustness: each trial runs the model ``n_seeds`` times with
    different seeds and returns mean + std (penalizing high variance). For
    GP-only training (pure exact GP, no MLP feature extractor) every seed
    produces an identical result — set ``n_seeds=1`` to avoid wasted compute.
    """
    active_search_space = _resolve_active_search_space(model_name, opt_cfg, base_model_cfg)
    architecture_name = _resolve_architecture_name(model_name, opt_cfg, base_model_cfg)
    n_seeds = max(1, int(n_seeds))

    def objective(trial: Trial) -> float:
        model_params = suggest_from_config(trial, active_search_space, "model")
        training_params = suggest_from_config(trial, active_search_space, "training")

        model_cfg = overlay(base_model_cfg, {"architecture_name": architecture_name})
        model_cfg = overlay(model_cfg, _apply_hp(model_name, model_params, architecture_name=architecture_name))
        training_cfg = overlay(base_training_cfg, training_params)
        training_cfg = overlay(training_cfg, opt_cfg.get("training_overrides", {}))

        if model_name == "gp":
            optimizer_name = str(training_cfg.get("optimizer", "adamw")).lower()
            if optimizer_name not in {"adam", "adamw", "sgd"}:
                logger.warning(f"GP optimizer '{optimizer_name}' is not one of adam, adamw, sgd")

        # Multi-seed robustness evaluation: run the model n_seeds times
        scores: List[float] = []
        base_seed = seed if seed is not None else 0
        
        for seed_idx in range(n_seeds):
            current_seed = base_seed + seed_idx
            set_global_seed(current_seed)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                run_dir = Path(tmpdir)
                try:
                    results = train_model(
                        model_name=model_name,
                        data_bundle=data_bundle,
                        dataset_cfg=dataset_cfg,
                        model_cfg=model_cfg,
                        training_cfg=training_cfg,
                        output_resolver=OutputPathResolver(
                            dataset_name="tmp",
                            run_type=RunType.SINGLE_RUN,
                            model_name=model_name,
                            output_base_dir=run_dir,
                        ),
                        device=device,
                        trial=(trial if seed_idx == 0 else None),
                    )

                    metric_name = str(opt_cfg["study"]["metric"])
                    metric_value = _extract_metric(results, metric_name, data_bundle, results["model"])
                    scores.append(float(metric_value))
                    
                    # Store metrics from last seed run as trial attributes
                    if seed_idx == n_seeds - 1:
                        trial.set_user_attr("objective_metric", metric_name)
                        trial.set_user_attr("objective_value", float(metric_value))
                        trial.set_user_attr(
                            "val_metrics",
                            {key: float(value) for key, value in results.get("val_metrics", {}).items()},
                        )
                        trial.set_user_attr(
                            "test_metrics",
                            {key: float(value) for key, value in results.get("test_metrics", {}).items()},
                        )

                        for metric_key, metric_val in results.get("val_metrics", {}).items():
                            trial.set_user_attr(f"val_{metric_key}", float(metric_val))
                        for metric_key, metric_val in results.get("test_metrics", {}).items():
                            trial.set_user_attr(f"test_{metric_key}", float(metric_val))

                        history = results.get("history", {})
                        if isinstance(history.get("train_losses"), list):
                            trial.set_user_attr("train_losses", [float(x) for x in history["train_losses"]])
                        if isinstance(history.get("val_losses"), list):
                            trial.set_user_attr("val_losses", [float(x) for x in history["val_losses"]])
                            
                except optuna.TrialPruned:
                    raise
                except Exception as exc:
                    logger.warning(f"Trial {trial.number} seed_idx={seed_idx} failed: {exc}")
                    scores.append(float("inf"))
        
        # Compute mean and std across seeds; return mean + std to penalize variance
        if not scores or all(s == float("inf") for s in scores):
            logger.warning(f"Trial {trial.number}: all seed runs failed")
            return float("inf")
        
        valid_scores = [s for s in scores if s != float("inf")]
        if not valid_scores:
            logger.warning(f"Trial {trial.number}: no valid seed runs")
            return float("inf")
        
        score_mean = float(np.mean(valid_scores))
        score_std = float(np.std(valid_scores))
        robust_score = score_mean + score_std  # Penalize high variance
        
        trial.set_user_attr("score_mean", score_mean)
        trial.set_user_attr("score_std", score_std)
        trial.set_user_attr("robust_score", robust_score)
        trial.set_user_attr("n_valid_seeds", len(valid_scores))
        
        logger.info(f"Trial {trial.number}: {n_seeds} seeds → mean={score_mean:.6f}, std={score_std:.6f}, robust={robust_score:.6f}")
        
        return robust_score

    return objective


def _extract_best_params(best_params: Dict[str, Any], search_space: Dict[str, Any], target: str) -> Dict[str, Any]:
    """Extract best parameters for a specific target.

    The ``layer_size`` template is not a real param; the dynamic
    ``layer{i}_size`` keys actually present in ``best_params`` are pulled
    in instead.
    """
    if target not in search_space:
        return {}
    target_space = search_space[target]
    extracted = {
        name: best_params[name]
        for name in target_space
        if name != _LAYER_SIZE_TEMPLATE and name in best_params
    }
    if _LAYER_SIZE_TEMPLATE in target_space:
        for name, value in best_params.items():
            if _is_dynamic_layer_size(name):
                extracted[name] = value
    return extracted


def save_optimization_results(
    study: Study,
    output_resolver: OutputPathResolver,
    opt_cfg: Dict[str, Any],
    base_model_cfg: Dict[str, Any],
    base_training_cfg: Dict[str, Any],
    model_name: str,
) -> None:
    """Persist trials, best params and the optimized configs."""
    active_search_space = _resolve_active_search_space(model_name, opt_cfg, base_model_cfg)
    architecture_name = _resolve_architecture_name(model_name, opt_cfg, base_model_cfg)
    run_dir = output_resolver.root
    run_dir.mkdir(parents=True, exist_ok=True)

    trials_data = []
    for trial in study.trials:
        trials_data.append(
            {
                "trial_number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                **trial.params,
                **trial.user_attrs,
            }
        )

    trials_csv = output_resolver.all_trials_csv
    if trials_data:
        fieldnames = sorted({key for row in trials_data for key in row.keys()})
        with open(trials_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trials_data)
        logger.debug(f"Trials saved to {trials_csv}")

    if study.best_trial is not None:
        best_params_file = output_resolver.best_params
        with open(best_params_file, "w", encoding="utf-8") as f:
            json.dump(study.best_params, f, indent=2)
        logger.debug(f"Best params saved to {best_params_file}")

        model_params = _extract_best_params(study.best_params, active_search_space, "model")
        training_params = _extract_best_params(study.best_params, active_search_space, "training")
        best_model_cfg = overlay(base_model_cfg, {"architecture_name": architecture_name})
        best_model_cfg = overlay(best_model_cfg, _apply_hp(model_name, model_params, architecture_name=architecture_name))
        best_training_cfg = overlay(overlay(base_training_cfg, training_params), opt_cfg.get("training_overrides", {}))

        config_dir = output_resolver.config_dir
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "model").mkdir(exist_ok=True)
        (config_dir / "training").mkdir(exist_ok=True)

        with open(config_dir / "optimize_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(opt_cfg, f, default_flow_style=False, allow_unicode=True)
        with open(config_dir / "model" / f"{model_name}_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(best_model_cfg, f, default_flow_style=False, allow_unicode=True)
        with open(config_dir / "training" / f"training_{model_name}_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(best_training_cfg, f, default_flow_style=False, allow_unicode=True)

        logger.debug(f"Best configs saved to {config_dir}")

        save_run_artifacts(
            output_resolver,
            mode="optimization",
            model_name=model_name,
            history={},
            metrics={
                "best_value": float(study.best_value),
                "n_trials_completed": len(study.trials),
                "objective_metric": opt_cfg.get("study", {}).get("metric"),
            },
            timings={},
            extra={
                "best_params": study.best_params,
                "direction": opt_cfg.get("study", {}).get("direction"),
            },
        )


__all__ = [
    "suggest_from_config",
    "overlay",
    "_make_logging_callback",
    "compute_val_nll",
    "build_sampler",
    "build_pruner",
    "make_objective",
    "save_optimization_results",
]