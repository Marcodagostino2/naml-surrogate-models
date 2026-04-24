from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data import build_data_bundle
from src.models import MLPRegressor
from src.plot import plot_diagnostics, plot_physical_sweeps
from src.training import evaluate_model, train_model
from src.training.trainer import _resolve_device
from src.utils import load_yaml, save_json, save_pickle, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline MLP on CFD aerodynamic data.")
    parser.add_argument("--dataset-config", default="config/dataset_config.yaml")
    parser.add_argument("--model-config", default="config/model_config.yaml")
    parser.add_argument("--training-config", default="config/training_config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    dataset_cfg = load_yaml(project_root / args.dataset_config)
    model_cfg = load_yaml(project_root / args.model_config)
    training_cfg = load_yaml(project_root / args.training_config)

    seed = int(training_cfg.get("random_seed", dataset_cfg.get("random_seed", 42)))
    set_global_seed(seed)

    data_bundle = build_data_bundle(dataset_cfg, project_root)
    model = MLPRegressor(
        input_dim=data_bundle.input_dim,
        output_dim=data_bundle.output_dim,
        model_cfg=model_cfg,
    )

    training_result, best_state = train_model(model, data_bundle, training_cfg)
    model.load_state_dict(best_state)

    device = _resolve_device(str(training_cfg.get("device", "auto")))
    model.to(device)

    val_metrics = evaluate_model(
        model,
        data_bundle.val_loader,
        device,
        target_scaler=data_bundle.target_scaler,
    )
    test_metrics = evaluate_model(
        model,
        data_bundle.test_loader,
        device,
        target_scaler=data_bundle.target_scaler,
    )

    output_dir = project_root / str(training_cfg.get("output_dir", "outputs/mlp_baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / str(training_cfg.get("checkpoint_name", "best_model.pt"))
    torch.save(model.state_dict(), checkpoint_path)

    artifacts_path = output_dir / str(training_cfg.get("artifacts_name", "artifacts.pkl"))
    save_pickle(
        artifacts_path,
        {
            "feature_scaler": data_bundle.feature_scaler,
            "target_scaler": data_bundle.target_scaler,
            "input_columns": data_bundle.input_columns,
            "target_columns": data_bundle.target_columns,
            "feature_bounds": data_bundle.feature_bounds,
            "target_bounds": data_bundle.target_bounds,
        },
    )

    history_path = output_dir / str(training_cfg.get("history_name", "history.json"))
    report = {
        "seed": seed,
        "input_columns": data_bundle.input_columns,
        "target_columns": data_bundle.target_columns,
        "best_epoch": training_result.best_epoch,
        "best_val_loss": training_result.best_val_loss,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": training_result.history,
    }
    save_json(history_path, report)

    if bool(training_cfg.get("generate_plots", True)):
        plots_dir = output_dir / "plots"
        diagnostic_files = plot_diagnostics(
            model=model,
            loader=data_bundle.test_loader,
            device=device,
            target_scaler=data_bundle.target_scaler,
            target_names=data_bundle.target_columns,
            output_dir=plots_dir / "diagnostics",
        )
        reference_point = np.median(data_bundle.train_features_raw, axis=0)
        sweep_files = plot_physical_sweeps(
            model=model,
            device=device,
            feature_scaler=data_bundle.feature_scaler,
            target_scaler=data_bundle.target_scaler,
            input_columns=data_bundle.input_columns,
            target_name=data_bundle.target_columns[0],
            feature_bounds=data_bundle.feature_bounds,
            reference_point=reference_point,
            output_dir=plots_dir / "physical_sweeps",
            num_points=int(training_cfg.get("diagnostics_num_points", 200)),
        )
        report["diagnostic_plots"] = {
            key: str(path.relative_to(project_root)) for key, path in diagnostic_files.items()
        }
        report["physical_sweep_plots"] = {
            key: str(path.relative_to(project_root)) for key, path in sweep_files.items()
        }
        save_json(history_path, report)

    print("Training completed.")
    print(f"Best epoch: {training_result.best_epoch}")
    print(f"Best val loss: {training_result.best_val_loss:.6f}")
    print(f"Validation metrics: {val_metrics}")
    print(f"Test metrics: {test_metrics}")
    print(f"Model checkpoint: {checkpoint_path}")
    print(f"Training artifacts: {artifacts_path}")
    print(f"Training report: {history_path}")


if __name__ == "__main__":
    main()