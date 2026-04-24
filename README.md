# naml-surrogate-models

MATLAB implementation of Kriging and Co-Kriging surrogate models for rocket aerodynamic coefficients in the context of the Numerical Analysis for Machine Learning (NAML) course.

The project combines low-fidelity DATCOM data and high-fidelity CFD data to build fast surrogate models that can be queried inside simulation workflows.

## Project goals

- Estimate aerodynamic coefficients as a function of flight conditions (Mach, angle variables, altitude, airbrake/protuberance parameter).
- Reduce computational cost relative to direct CFD evaluations.
- Preserve enough accuracy for repeated evaluations in trajectory and dynamics loops.
- Compare single-fidelity and multi-fidelity surrogate strategies.

## Main features

- Single-fidelity Kriging mode:
	- CFD-only or DATCOM-only training.
- Multi-fidelity Co-Kriging mode:
	- DATCOM as low-fidelity support.
	- CFD as high-fidelity reference.
- Multi-output training (per aerodynamic coefficient).
- Hyperparameter tuning with analytical gradients and MATLAB fmincon multi-start optimization.
- Diagnostic pipeline for:
	- timing,
	- regression metrics,
	- residual and parity plots,
	- response surfaces/slices.

## Repository structure

- [mainCoefficientModel.m](mainCoefficientModel.m): entry-point workflow.
- [config/](config): dataset and optimization settings.
- [data/](data): persisted datasets and trained surrogate models.
- [src/classes/](src/classes): object-oriented core:
	- [src/classes/Dataset.m](src/classes/Dataset.m)
	- [src/classes/SurrogateModel.m](src/classes/SurrogateModel.m)
	- [src/classes/CoefficientModel.m](src/classes/CoefficientModel.m)
	- [src/classes/ModelDiagnostics.m](src/classes/ModelDiagnostics.m)
- [src/datasetGeneration/](src/datasetGeneration): dataset build/parsing utilities.
- [src/surrogateModelFunctions/](src/surrogateModelFunctions): Kriging/Co-Kriging math, optimization helpers, and predictors.
- [src/plot/](src/plot): plotting and metric utilities.

## Quick start

Open MATLAB in the repository root and run:

```matlab
coeffModel = mainCoefficientModel( ...
		"mode", "multiFidelity", ...
		"splitPerc", 0.85, ...
		"reBuild", false, ...
		"reTrain", false, ...
		"plots", true, ...
		"exportFormat", "png", ...
		"plotSurface", true);
```

Available modes:

- "multiFidelity"
- "cfd"
- "datcom"

## Configuration

Edit these files to customize experiments:

- [config/datasets/cfdDatasetConfig.m](config/datasets/cfdDatasetConfig.m)
- [config/datasets/datcomDatasetConfig.m](config/datasets/datcomDatasetConfig.m)
- [config/models/coefficientOptimizationConfig.m](config/models/coefficientOptimizationConfig.m)

Typical configurable items include:

- variable bounds,
- selected outputs,
- train/test split,
- polynomial trend degree,
- bounds for theta, p, and nugget (lambda),
- optimizer settings (algorithm, tolerances, iterations, multi-start count).

## Diagnostics and outputs

When plotting is enabled, diagnostics are exported under [results/](results) by mode and timestamp.

The diagnostic pipeline reports:

- training and inference timing,
- RMSE, R2, MAPE, MaxAE,
- residual diagnostics,
- parity plots,
- 1D/3D surrogate visualizations.

## Notes on persistence

By default in [mainCoefficientModel.m](mainCoefficientModel.m), these are currently set to false:

- saveDataset
- saveModel

Set them to true if you want to persist rebuilt datasets and retrained models in [data/](data).

## Dependencies

The path setup in [mainCoefficientModel.m](mainCoefficientModel.m) currently references external folders:

- ../common
- ../../dissileMatcom

Make sure your local environment includes these paths, or adapt the setupPaths section to your workspace layout.

## Suggested workflow for experiments

- Start with multiFidelity mode and fixed split ratio.
- Tune optimization bounds and polynomial trend degree via config files.
- Run convergence sweeps by increasing CFD training budget.
- Compare against CFD-only Kriging baseline with same test set.
- Collect timing and error metrics in a reproducible table.

## License

No license file is currently provided in this repository.
