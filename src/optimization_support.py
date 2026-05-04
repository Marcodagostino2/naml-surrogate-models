"""
Shared helpers for hyperparameter optimization.

This module keeps optimize.py thin by hosting the reusable Optuna helpers,
configuration translation, result persistence and plotting.
"""

from __future__ import annotations

import csv
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency for optimization workflows
    optuna = None

from src.pipeline import train_model
from src.utils import OutputPathResolver, setup_logger

logger = setup_logger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


_HP_TO_MODEL_CFG: Dict[tuple[str, str], str] = {
    ("mlp", "dropout"): "dropout",
    ("mlp", "block1_layers"): "architecture",
    ("mlp", "block1_hidden_size"): "architecture",
    ("mlp", "block1_activation"): "architecture",
    ("mlp", "block2_layers"): "architecture",
    ("mlp", "block2_hidden_size"): "architecture",
    ("mlp", "block2_activation"): "architecture",
    ("mlp", "block3_layers"): "architecture",
    ("mlp", "block3_hidden_size"): "architecture",
    ("mlp", "block3_activation"): "architecture",
    ("gp", "outputscale_init"): "kernel",
    ("gp", "lengthscale_init"): "kernel",
    ("gp", "noise_var"): "kernel",
    ("mlpgp", "block1_layers"): "feature_extractor",
    ("mlpgp", "block1_hidden_size"): "feature_extractor",
    ("mlpgp", "block1_activation"): "feature_extractor",
    ("mlpgp", "block2_layers"): "feature_extractor",
    ("mlpgp", "block2_hidden_size"): "feature_extractor",
    ("mlpgp", "block2_activation"): "feature_extractor",
    ("mlpgp", "block3_layers"): "feature_extractor",
    ("mlpgp", "block3_hidden_size"): "feature_extractor",
    ("mlpgp", "block3_activation"): "feature_extractor",
    ("mlpgp", "dropout"): "feature_extractor",
    ("mlpgp", "batch_norm"): "feature_extractor",
    ("mlpgp", "output_activation"): "feature_extractor",
    ("mlpgp", "output_width"): "feature_extractor",
    ("mlpgp", "sigma_floor"): "feature_extractor",
    ("mlpgp", "lengthscale_floor"): "feature_extractor",
    ("mlpgp", "lengthscale_mode"): "feature_extractor",
    ("mlpgp", "theta"): "kernel",
    ("mlpgp", "power_exponent"): "kernel",
    ("mlpgp", "outputscale_init"): "kernel",
    ("mlpgp", "tau2"): "kernel",
    ("mlpgp", "lengthscale_init"): "kernel",
    ("mlpgp", "noise_var"): "kernel",
}

_KERNEL_PARAM_MAP: Dict[str, str] = {
    "outputscale_init": "tau2",
    "tau2": "tau2",
    "lengthscale_init": "theta",
    "noise_var": "likelihood_noise",
    "theta": "theta",
    "power_exponent": "power_exponent",
}


def _resolve_architecture_name(opt_cfg: Dict[str, Any], base_model_cfg: Dict[str, Any]) -> str:
    """Resolve the active MLPGP architecture name."""
    architecture_name = opt_cfg.get("architecture_name", base_model_cfg.get("architecture_name", "non_stationary"))
    return str(architecture_name).lower()


def _resolve_active_search_space(
    opt_cfg: Dict[str, Any],
    base_model_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Select the active search space for the current architecture."""
    architecture_name = _resolve_architecture_name(opt_cfg, base_model_cfg)

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


def suggest_from_config(trial: optuna.Trial, search_space: Dict[str, Any], target: str) -> Dict[str, Any]:
    """Convert a YAML search space section into Optuna suggestions."""
    if target not in search_space:
        return {}

    params: Dict[str, Any] = {}
    for param_name, param_spec in search_space[target].items():
        param_type = param_spec.get("type")
        if param_type == "int":
            params[param_name] = trial.suggest_int(
                param_name,
                int(param_spec.get("low", 1)),
                int(param_spec.get("high", 10)),
                step=int(param_spec.get("step", 1)),
            )
        elif param_type == "float":
            params[param_name] = trial.suggest_float(
                param_name,
                float(param_spec.get("low", 0.0)),
                float(param_spec.get("high", 1.0)),
                log=bool(param_spec.get("log", False)),
            )
        elif param_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, param_spec.get("choices", []))
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
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


def _make_logging_callback(
    model_name: str,
    n_trials: int,
    metric_name: str,
    log_every: int = 5,
) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """Create a compact Optuna callback that prints periodic progress."""
    # removed: per-trial logger spam, replaced by coarse-grained callback logging.
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if log_every <= 0 or trial.number % log_every != 0:
            return
        best = study.best_value if study.best_trial else float("nan")
        current = trial.value if trial.value is not None else float("nan")
        pct = (trial.number + 1) / max(n_trials, 1) * 100
        logger.info(
            f"[{model_name.upper()}] {trial.number + 1:>3}/{n_trials} ({pct:4.0f}%) | "
            f"current={current:.5f}  best={best:.5f}"
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
        return TPESampler(seed=seed, multivariate=study_cfg.get("multivariate", True), n_startup_trials=study_cfg.get("n_startup_trials", 10))
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

    from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
    if pruner_name is None or pruner_name == "null":
        return NopPruner()
    if pruner_name == "median":
        return MedianPruner(
            n_startup_trials=study_cfg.get("n_startup_trials", 5),
            n_warmup_steps=study_cfg.get("n_warmup_steps", 0),
            interval_steps=study_cfg.get("pruner_interval", 1),
        )
    if pruner_name == "hyperband":
        return HyperbandPruner(
            min_resource=study_cfg.get("min_resource", 10),
            max_resource=study_cfg.get("max_resource", "auto"),
            reduction_factor=study_cfg.get("reduction_factor", 3),
        )
    raise ValueError(f"Unknown pruner: {pruner_name}")


def _build_block_architecture(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build an architecture list from blockN_layers / blockN_hidden_size /
    blockN_activation entries in params.

    Each block produces one dict: {"layers": [hidden_size] * n_layers, "activation": ...}.
    Blocks with n_layers == 0 or missing size/activation are skipped.
    """
    block_indices = sorted({
        int(m.group(1))
        for key in params
        if (m := re.fullmatch(r"block(\d+)_(layers|hidden_size|activation)", key))
    })
    architecture = []
    for idx in block_indices:
        n_layers    = int(params.get(f"block{idx}_layers", 0))
        hidden_size = params.get(f"block{idx}_hidden_size")
        activation  = params.get(f"block{idx}_activation")
        if n_layers > 0 and hidden_size is not None and activation is not None:
            architecture.append({
                "layers":     [int(hidden_size)] * n_layers,
                "activation": str(activation),
            })
    return architecture


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
        Model identifier used for registry lookup (e.g. ``"mlp"``, ``"gp"``, ``"mlpgp"``).
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
        dest = _HP_TO_MODEL_CFG.get((model_name, hp_name))
        if dest is not None:
            scoped.setdefault(dest, {})[hp_name] = hp_value

    patch: Dict[str, Any] = {}

    if arch_params := scoped.get("architecture"):
        arch = _build_block_architecture(arch_params)
        if arch:
            patch["architecture"] = arch

    if model_name == "mlpgp" and scoped.get("feature_extractor"):
        fe_params = scoped["feature_extractor"]
        feature_patch: Dict[str, Any] = {}

        arch = _build_block_architecture(fe_params)
        if arch:
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
        kernel = {
            _KERNEL_PARAM_MAP[k]: float(v)
            for k, v in kernel_params.items()
            if k in _KERNEL_PARAM_MAP
        }
        if kernel:
            patch["kernel"] = kernel

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
) -> Callable[[optuna.Trial], float]:
    """Create an Optuna objective bound to the shared training pipeline."""
    active_search_space = _resolve_active_search_space(opt_cfg, base_model_cfg)
    architecture_name = _resolve_architecture_name(opt_cfg, base_model_cfg)

    def objective(trial: optuna.Trial) -> float:
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
                        run_type="single_run",
                        model_name=model_name,
                        output_base_dir=run_dir,
                    ),
                    device=device,
                    trial=trial,
                )

                metric_name = str(opt_cfg["study"]["metric"])
                metric_value = _extract_metric(results, metric_name, data_bundle, results["model"])
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
                
                return float(metric_value)
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                logger.warning(f"Trial {trial.number} failed: {exc}")
                return float("inf")

    return objective


def _extract_best_params(best_params: Dict[str, Any], search_space: Dict[str, Any], target: str) -> Dict[str, Any]:
    """Extract best parameters for a specific target."""
    if target not in search_space:
        return {}
    return {param_name: best_params[param_name] for param_name in search_space[target] if param_name in best_params}


def save_optimization_results(
    study: optuna.Study,
    output_resolver: OutputPathResolver,
    opt_cfg: Dict[str, Any],
    base_model_cfg: Dict[str, Any],
    base_training_cfg: Dict[str, Any],
    model_name: str,
) -> None:
    """Persist trials, best params and the optimized configs."""
    active_search_space = _resolve_active_search_space(opt_cfg, base_model_cfg)
    architecture_name = _resolve_architecture_name(opt_cfg, base_model_cfg)
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

# Plotting helpers are kept here so optimize.py can stay thin.

def _generate_summary_dashboard(study: optuna.Study, plots_dir: Path) -> None:
    """Generate a 2x3 dashboard with optimization and multi-metric diagnostics."""
    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and np.isfinite(float(t.value))  
    ]
    if not completed_trials:
        logger.warning("Skipping summary dashboard: no completed trials with finite objective.")
        return

    trial_numbers    = np.asarray([t.number    for t in completed_trials], dtype=int)
    objective_values = np.asarray([float(t.value) for t in completed_trials], dtype=float)
    # objective_values è già garantito finito, accumulate è sicuro
    best_so_far = np.minimum.accumulate(objective_values)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax_convergence, ax_metrics, ax_hist = axes[0]
    ax_improvement, ax_top10, ax_losses = axes[1]

    ax_convergence.plot(trial_numbers, best_so_far, color="tab:blue", linewidth=2)
    ax_convergence.scatter(trial_numbers, objective_values, color="tab:blue", s=20, alpha=0.45)
    ax_convergence.set_title("Convergence (best so far)")
    ax_convergence.set_xlabel("Trial")
    ax_convergence.set_ylabel("Objective")
    ax_convergence.grid(alpha=0.3)

    metric_specs = [
        ("val_rmse", "tab:red",    "RMSE"),
        ("val_mae",  "tab:orange", "MAE"),
        ("val_mape", "tab:green",  "MAPE"),
    ]
    metric_r2 = np.asarray(
        [float(t.user_attrs.get("val_r2", np.nan)) for t in completed_trials], dtype=float
    )

    has_primary = False
    for key, color, label in metric_specs:
        vals = np.asarray(
            [float(t.user_attrs.get(key, np.nan)) for t in completed_trials], dtype=float
        )
        finite_mask = np.isfinite(vals)  
        if finite_mask.any():
            has_primary = True
            ax_metrics.plot(
                trial_numbers[finite_mask], vals[finite_mask],
                marker="o", linewidth=1.5, color=color, label=label,
            )

    ax_metrics_r2 = ax_metrics.twinx()
    r2_mask = np.isfinite(metric_r2)
    has_r2 = r2_mask.any()
    if has_r2:
        ax_metrics_r2.plot(
            trial_numbers[r2_mask], metric_r2[r2_mask],
            marker="x", linewidth=1.4, color="tab:purple", label="R2",
        )
        ax_metrics_r2.set_ylabel("R2")

    ax_metrics.set_title("Val metrics over trials")
    ax_metrics.set_xlabel("Trial")
    ax_metrics.set_ylabel("Error metrics")
    ax_metrics.grid(alpha=0.3)
    if has_primary or has_r2:
        handles_1, labels_1 = ax_metrics.get_legend_handles_labels()
        handles_2, labels_2 = ax_metrics_r2.get_legend_handles_labels()
        ax_metrics.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best", fontsize=8)
    else:
        ax_metrics.text(0.5, 0.5, "No val_* user attrs", ha="center", va="center",
                        transform=ax_metrics.transAxes)

    ax_hist.hist(objective_values, bins=min(20, len(objective_values)),
                 color="tab:cyan", edgecolor="black")
    best_finite = float(np.min(objective_values))
    ax_hist.axvline(best_finite, color="tab:red", linestyle="--", linewidth=2)
    ax_hist.set_title("Distribution of values")
    ax_hist.set_xlabel("Objective")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(alpha=0.3)

    improvement = np.concatenate([[np.nan], np.diff(best_so_far)])
    ax_improvement.plot(trial_numbers, improvement, color="tab:green", marker="o", linewidth=1.5)
    ax_improvement.axhline(0.0, color="black", linewidth=1, linestyle=":")
    ax_improvement.set_title("Incremental improvement")
    ax_improvement.set_xlabel("Trial")
    ax_improvement.set_ylabel("Delta best")
    ax_improvement.grid(alpha=0.3)

    top_trials = sorted(completed_trials, key=lambda t: float(t.value))[:10]
    table_rows = [
        [
            str(t.number),
            f"{float(t.value):.5f}",
            f"{float(t.user_attrs.get('val_rmse', np.nan)):.4f}",
            f"{float(t.user_attrs.get('val_r2',   np.nan)):.4f}",
        ]
        for t in top_trials
    ]
    ax_top10.axis("off")
    ax_top10.set_title("Top-10 trials")
    table = ax_top10.table(
        cellText=table_rows,
        colLabels=["Trial", "Objective", "Val RMSE", "Val R2"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    best_trial    = study.best_trial if study.best_trial is not None else None
    train_losses  = best_trial.user_attrs.get("train_losses") if best_trial is not None else None
    val_losses    = best_trial.user_attrs.get("val_losses")   if best_trial is not None else None

    if isinstance(train_losses, list) and len(train_losses) > 0:
        train_arr = np.asarray(train_losses, dtype=float)
        train_finite = np.isfinite(train_arr)
        epochs = np.arange(1, len(train_arr) + 1)
        ax_losses.plot(epochs[train_finite], train_arr[train_finite],
                       color="tab:blue", linewidth=1.8, label="Train")
        if isinstance(val_losses, list) and len(val_losses) > 0:
            val_arr    = np.asarray(val_losses, dtype=float)
            val_finite = np.isfinite(val_arr)
            ax_losses.plot(
                np.arange(1, len(val_arr) + 1)[val_finite], val_arr[val_finite],
                color="tab:orange", linewidth=1.8, label="Val",
            )
        ax_losses.set_title("Train vs Val loss (best trial)")
        ax_losses.set_xlabel("Epoch")
        ax_losses.set_ylabel("Loss")
        ax_losses.grid(alpha=0.3)
        ax_losses.legend(loc="best", fontsize=8)
    else:
        ax_losses.text(0.5, 0.5, "No train/val loss history available",
                       ha="center", va="center", transform=ax_losses.transAxes)
        ax_losses.set_title("Train vs Val loss (best trial)")
        ax_losses.set_xticks([])
        ax_losses.set_yticks([])

    fig.suptitle("Optimization Summary Dashboard", fontsize=14)
    fig.tight_layout()
    dashboard_path = plots_dir / "summary_dashboard.png"
    fig.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Summary dashboard saved to {dashboard_path}")


def generate_optimization_plots(study: optuna.Study, plots_dir: Path) -> None:
    """Generate the optimization dashboard."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    _generate_summary_dashboard(study, plots_dir)


__all__ = [
    "suggest_from_config",
    "overlay",
    "_make_logging_callback",
    "compute_val_nll",
    "build_sampler",
    "build_pruner",
    "make_objective",
    "save_optimization_results",
    "generate_optimization_plots",
]
