"""Optuna-based hyperparameter optimization entrypoint."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.data import load_data_from_config
from src.optimization_support import (
    _make_logging_callback,
    build_pruner,
    build_sampler,
    make_objective,
    save_optimization_results,
)
from src.pipeline import save_summary_row
from src.utils import OutputPathResolver, RunType, load_run_configs, resolve_device, setup_logger
from src.utils.entrypoints import load_and_initialize_run_configs  # noqa: F401
from src.visualization import run_visualization
from src.visualization.config import VisualizationConfig

logger = setup_logger(__name__)


def run_optimization(
    config_dir: Path,
    model_name: str,
    output_dir: Path,
    n_trials_override: Optional[int] = None,
    timeout_override: Optional[int] = None,
) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise ModuleNotFoundError("optuna is required to run optimization workflows.") from exc

    logger.info("=" * 70)
    logger.info(f"Hyperparameter Optimization for {model_name.upper()}")
    logger.info("=" * 70)

    loaded = load_run_configs(config_dir, model_name, include_optimize=True)
    main_cfg = loaded["main_cfg"]
    dataset_cfg = loaded["dataset_cfg"]
    base_model_cfg = loaded["model_cfg"]
    base_training_cfg = loaded["training_cfg"]
    opt_cfg = loaded.get("opt_cfg", {})
    global_seed = int(main_cfg.get("global_seed", 42))
    dataset_name = loaded["dataset_name"]
    logger.info(f"Dataset: {dataset_name} - {dataset_cfg.get('description', '')}")

    if n_trials_override is not None:
        opt_cfg.setdefault("study", {})["n_trials"] = n_trials_override
    if timeout_override is not None:
        opt_cfg.setdefault("study", {})["timeout"] = timeout_override

    logger.info("Loading and preprocessing data...")
    data_bundle = load_data_from_config(
        config_dir.parent,
        dataset_cfg,
        global_seed,
        str(main_cfg.get("precision", "float32")),
    )

    device = resolve_device(base_training_cfg.get("device", main_cfg.get("device", "auto")))

    study_cfg = opt_cfg.get("study", {})
    sampler = build_sampler(study_cfg)
    pruner = build_pruner(study_cfg)
    study = optuna.create_study(
        direction=study_cfg.get("direction", "minimize"),
        sampler=sampler,
        pruner=pruner,
    )

    objective = make_objective(
        model_name=model_name,
        data_bundle=data_bundle,
        base_model_cfg=base_model_cfg,
        base_training_cfg=base_training_cfg,
        dataset_cfg=dataset_cfg,
        opt_cfg=opt_cfg,
        device=device,
        seed=global_seed,
    )

    n_trials = int(study_cfg.get("n_trials", 1))
    timeout = study_cfg.get("timeout")
    metric_name = str(study_cfg.get("metric", ""))
    sampler_name = str(study_cfg.get("sampler", "tpe"))
    log_every = int(study_cfg.get("log_every_n_trials", 5))

    logger.info(f"Starting HPO: {model_name} | {n_trials} trials | metric={metric_name} | sampler={sampler_name}")

    start_time = time.perf_counter()
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
        callbacks=[_make_logging_callback(model_name, n_trials, metric_name, log_every)],
    )
    duration = time.perf_counter() - start_time

    if study.best_trial is None:
        raise RuntimeError("No completed trial found in study.")
    logger.info(
        f"HPO complete | best={study.best_value:.6f} | params={study.best_params} | duration={duration:.0f}s"
    )

    output_resolver = OutputPathResolver(
        dataset_name=dataset_name,
        run_type=RunType.OPTIMIZATION,
        model_name=model_name,
        output_base_dir=output_dir,
    )
    run_dir = output_resolver.root
    run_dir.mkdir(parents=True, exist_ok=True)

    save_optimization_results(
        study=study,
        output_resolver=output_resolver,
        opt_cfg=opt_cfg,
        base_model_cfg=base_model_cfg,
        base_training_cfg=base_training_cfg,
        model_name=model_name,
    )

    viz_cfg_raw = loaded.get("viz_cfg", {})
    viz_config = VisualizationConfig.from_mapping(viz_cfg_raw, output_dir=str(output_resolver.plots_dir))
    run_visualization("optimization", config=viz_config, study=study)

    save_summary_row(
        output_resolver.summary_csv,
        {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "run_dir": str(run_dir),
            "model": model_name,
            "n_trials_completed": len(study.trials),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "metric": study_cfg.get("metric"),
            "duration_seconds": duration,
        },
    )

    logger.debug(f"Optimization results saved to {run_dir}")
