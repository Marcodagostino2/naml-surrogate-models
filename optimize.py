"""
Hyperparameter optimization entrypoint for aerodynamic surrogate models.

This file intentionally stays thin and delegates the heavy lifting to
src.optimization_support and src.pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import optuna

from src.data import load_data_from_config
from src.optimization_support import (
    _make_logging_callback,
    build_pruner,
    build_sampler,
    generate_optimization_plots,
    make_objective,
    save_optimization_results,
)
from src.pipeline import save_summary_row
from src.utils import (
    OutputPathResolver,
    RunType,
    load_run_configs,
    resolve_device,
    set_global_seed,
    setup_logger,
    setup_torch_dtype,
)

logger = setup_logger(__name__)


def run_optimization(
    config_dir: Path,
    model_name: str,
    output_dir: Path,
    n_trials_override: Optional[int] = None,
    timeout_override: Optional[int] = None,
) -> None:
    """Run hyperparameter optimization for a single model."""
    logger.info("=" * 70)
    logger.info(f"Hyperparameter Optimization for {model_name.upper()}")
    logger.info("=" * 70)

    loaded = load_run_configs(config_dir, model_name, include_optimize=True)
    main_cfg = loaded["main_cfg"]
    dataset_cfg = loaded["dataset_cfg"]
    base_model_cfg = loaded["model_cfg"]
    base_training_cfg = loaded["training_cfg"]
    opt_cfg = loaded["opt_cfg"]
    global_seed = int(main_cfg.get("global_seed", 42))
    dataset_name = loaded["dataset_name"]
    logger.info(f"Dataset: {dataset_name} - {dataset_cfg.get('description', '')}")

    if n_trials_override is not None:
        opt_cfg["study"]["n_trials"] = n_trials_override
    if timeout_override is not None:
        opt_cfg["study"]["timeout"] = timeout_override

    logger.info("Loading and preprocessing data...")
    data_bundle = load_data_from_config(
        config_dir.parent,
        dataset_cfg,
        global_seed,
        str(main_cfg.get("precision", "float32")),
    )
    logger.debug(f"Data loaded: {data_bundle.X_train.shape[0]} train samples")

    device = resolve_device(base_training_cfg.get("device", main_cfg.get("device", "auto")))
    logger.debug(f"Resolved optimization device: {device}")

    sampler = build_sampler(opt_cfg["study"])
    pruner = build_pruner(opt_cfg["study"])
    study = optuna.create_study(
        direction=opt_cfg["study"]["direction"],
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
    )

    n_trials = opt_cfg["study"]["n_trials"]
    timeout = opt_cfg["study"].get("timeout")
    metric_name = str(opt_cfg["study"]["metric"])
    sampler_name = str(opt_cfg["study"].get("sampler", "tpe"))
    log_every = int(opt_cfg["study"].get("log_every_n_trials", 5))

    logger.info(
        f"Starting HPO: {model_name} | {n_trials} trials | metric={metric_name} | sampler={sampler_name}"
    )

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

    generate_optimization_plots(study, output_resolver.plots_dir)

    save_summary_row(
        output_resolver.summary_csv,
        {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "run_dir": str(run_dir),
            "model": model_name,
            "n_trials_completed": len(study.trials),
            "best_value": study.best_value,
            "best_params": json.dumps(study.best_params),
            "metric": opt_cfg["study"]["metric"],
            "duration_seconds": duration,
        },
    )

    logger.debug(f"Optimization results saved to {run_dir}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for surrogate models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-dir", type=str, default="config", help="Path to configuration directory")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "gp", "mlpgp", "all"], help="Model to optimize")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for results")
    parser.add_argument("--n-trials", type=int, default=None, help="Override number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Override timeout in seconds")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir)

    if not config_dir.exists():
        logger.error(f"Config directory not found: {config_dir}")
        return
    if not (config_dir / "optimization").exists() and not (config_dir / "optimize_config").exists():
        logger.error(f"Optimization config directory not found in: {config_dir}")
        return

    main_cfg = load_run_configs(config_dir, "mlp", include_optimize=False)["main_cfg"]
    set_global_seed(int(main_cfg.get("global_seed", 42)))
    setup_torch_dtype(str(main_cfg.get("precision", "float32")))

    if args.model == "all":
        for model_name in ["mlp", "gp", "mlpgp"]:
            run_optimization(config_dir, model_name, output_dir, args.n_trials, args.timeout)
    else:
        run_optimization(config_dir, args.model, output_dir, args.n_trials, args.timeout)

    logger.info("All optimizations completed!")


if __name__ == "__main__":
    main()
