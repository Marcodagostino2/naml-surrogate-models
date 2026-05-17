# Session handoff

Living document for cross-session continuity. Update at the end of each working session: append new entries to **Recent work**, refresh **Open items** and **In escrow**, and edit **Conventions** when the team agrees on a new pattern.

---

## Repo at a glance

Aerodynamic surrogate-model framework. Three workflow modes (`single_run`, `optimization`, `convergence`) routed by `main.py`; the heavy lifting lives under `src/`.

```
naml-surrogate-models/
├── main.py                       # CLI dispatcher (--mode flag or main_config.mode)
├── config/
│   ├── main_config.yaml          # mode + dataset + global seed + precision + model_to_train
│   ├── dataset/dataset_config.yaml
│   ├── model/{mlp,gp,mlpgp}_config.yaml
│   ├── training/{training_common,training_mlp,training_gp,training_mlpgp}_config.yaml
│   ├── optimization/optimize_{mlp,gp,mlpgp}_config.yaml
│   ├── convergence/convergence_config.yaml
│   └── visualization/visualization_config.yaml
├── src/
│   ├── data/                     # CSV loader, normalization, train/val/test split, DataLoaders
│   │   ├── dataset.py            # DataBundle + load_and_preprocess_data
│   │   └── __init__.py           # load_data_from_config (config -> DataBundle)
│   ├── models/
│   │   ├── mlp.py                # MLPRegressor (single hidden block, activation-driven init)
│   │   ├── gp.py                 # GPRegressionModel (exact) + SVGPRegressionModel (svgp, in escrow)
│   │   ├── gp_utils.py           # ExponentialKernel, build_kernel, config helpers
│   │   ├── mlpgp.py              # MLPGPModel + MLPSVGPModel; non_stationary (Gibbs) and dkl
│   │   └── init.py               # _BLOCK_INIT registry: relu/leaky_relu/selu/tanh/mish
│   ├── training/
│   │   ├── trainer.py            # MLPTrainer, GPTrainer, MLPGPTrainer, SVGPTrainer; ConvergenceMonitor
│   │   └── metrics.py
│   ├── inference/predictor.py    # unified predict() dispatching on model type
│   ├── pipeline.py               # train_model() + model/trainer factories + run-dir layout
│   ├── optimization_support.py   # Optuna helpers; _HP_TO_MODEL_CFG; _build_architecture
│   ├── convergence_analysis.py   # repeated split/seed analysis
│   ├── entrypoints/{optimize,convergence}.py
│   ├── visualization/
│   │   ├── runner.py             # per-mode dispatcher (single_run | optimization | convergence)
│   │   ├── config.py, labels.py, utils.py
│   │   ├── plots/                # leaf plot funcs: diagnostics, metrics, surface_generic, uncertainty, table, optimization, convergence
│   │   ├── suites/               # per-mode orchestration: single_run, optimization, convergence
│   │   └── datasets/hyddb.py     # HYDDB physics suite (5 plots)
│   └── utils/
│       ├── config.py             # load_run_configs, save_*, set_global_seed, resolve_device, setup_logger
│       ├── paths.py              # OutputPathResolver (dataset/run_type/model_name layout)
│       └── entrypoints.py        # load_and_initialize_run_configs
└── outputs/                      # auto-created: <dataset>/<run_type>/<model>/
```

### Key flows

- `python main.py` reads `config/main_config.yaml`, routes to one of:
  - `_run_single_run` → `train_model` per model (`pipeline.train_model`)
  - `_run_optimization` → `src/entrypoints/optimize.run_optimization` → Optuna study
  - `_run_convergence` → `src/entrypoints/convergence.run_analysis`
- `pipeline.train_model` builds the model + trainer via the registries `_MODEL_REGISTRY` and `_SVGP_CLASSES` and the `inference: "exact" | "svgp"` switch.

---

## Conventions

Style and design rules established with the user:

- **Cut hard on docs**: no multi-paragraph or NumPy-style `Parameters/Returns` docstrings on private helpers. Keep one-line docstrings only when they add real WHY (constraints, invariants, surprising design choices). User explicitly does not want code that reads as "obviously AI-written" — drop hedging phrases (`Note that…`, `Importantly…`, `Centralizing this lookup…`, `Single source of truth`), em-dashes in narration, and commentary that re-states what the code does.
- **No dead code, no duplication, no work-arounds.** When refactoring, delete the old path; don't leave shims.
- **Explicit over implicit for risky knobs.** GP `force_cholesky` warnings are intentionally NOT silenced right now (debug knob in `_BaseGPTrainer._cholesky_context`).
- **Smoke-test cadence.** Any non-trivial structural change is verified by `python main.py --config-dir config_smoke --output-dir outputs_smoke` with reduced step budgets (`n_training_steps: 30`, `n_epochs: 30`) — see "Reproducing the smoke test" below.
- **Italian/English mix is fine** in informal asides; YAML and code stay English.

---

## Recent work (latest first)

### Run persistence (no-overwrite) + checkpoint-based testing/resume wiring

Implemented the latest requested workflow hardening so every execution is preserved and prior runs can be reused for testing.

- **Run-id aware output layout** (`src/utils/paths.py`):
  - Added `RunType.TESTING`.
  - `OutputPathResolver` now accepts optional `run_id` and exposes run-aware roots:
    - `model_root`: `<output>/<dataset>/<run_type>/<model>/`
    - `runs_root`: `<model_root>/runs/`
    - `root`: `<runs_root>/<run_id>` when `run_id` is set, otherwise legacy `model_root`.
  - Added `make_run_id(prefix="run")` helper (timestamp-based id generation).
- **Pipeline testing entrypoint from checkpoint** (`src/pipeline.py`):
  - Added `_evaluate_model(...)` shared by training and test-only workflows.
  - Added `test_model_from_checkpoint(...)` to load a checkpoint and run evaluation/summary/plots artifacts without re-training.
  - Standardized artifact writing for testing runs too (`history.json`, `results_manifest.json` where applicable).
- **Main workflow wiring** (`main.py`):
  - Added checkpoint source resolver `_resolve_checkpoint_source(...)` with priority:
    1. explicit `--resume-run-id`,
    2. latest run under `runs/`,
    3. legacy checkpoint fallback.
  - Single-run path now supports explicit run ids and test-only execution backed by checkpoints.
  - Added CLI flags:
    - `--run-id` (force output run id),
    - `--resume-run-id` (select prior run for checkpoint loading).
  - Optimization flow now forwards `run_id`.
- **Optimization and convergence propagation**:
  - `src/entrypoints/optimize.py`: `run_optimization(..., run_id=None)` now uses run-specific roots when provided.
  - `src/convergence_analysis.py`: convergence outputs are grouped under analysis run ids (`runs/<analysis_run_id>/...`) to avoid overwrite between analyses.
- **Static validation**:
  - Checked modified files with diagnostics (`get_errors`) after patching:
    - `main.py`
    - `src/utils/paths.py`
    - `src/utils/__init__.py`
    - `src/pipeline.py`
    - `src/entrypoints/optimize.py`
    - `src/convergence_analysis.py`
  - Result: no static errors reported.

### Visualization refactor wrap-up (stages 5, 6, 10) + dynamic-depth MLP HPO

Closed out the deferred plan stages and removed the last hardcoded depth cap.

- **Stage 5** — moved the Optuna summary dashboard out of `src/optimization_support.py` into `src/visualization/plots/optimization.py` (leaf) + `src/visualization/suites/optimization.py` (orchestrator). `src/entrypoints/optimize.py` now calls `run_visualization("optimization", config=viz_config, study=study)`; the `optimization_support.py` `__all__` no longer exports `generate_optimization_plots` (deleted). `optimization_support.py` is now plot-free.
- **Stage 6** — moved `_plot_metric_curve*` / `_plot_training_histories` from `src/convergence_analysis.py` into `src/visualization/plots/convergence.py` + `src/visualization/suites/convergence.py`. The convergence orchestrator calls `run_visualization("convergence", config=viz_config, convergence_records=(metrics_df, histories, model_sequence))`. The CSV/aggregation logic stays in `convergence_analysis.py`. `convergence_analysis.py` no longer imports matplotlib.
- **Stage 10** — `src/visualization/labels.py` holds shared TeX constants (`TEX_R2`, `TEX_RMSE`, `TEX_MAE`, `TEX_MAPE`, `TEX_PRED`, `TEX_TRUE`, `TEX_RESIDUAL`, `TEX_SIGMA`). `axes.titleweight` switched from "bold" to "normal" globally; the residual `fontweight="bold"` callsites in `metrics.py`, `surface_generic.py`, `diagnostics.py`, and `apply_style()` are gone. `finalize_figure(...)` gained an optional `model_name=` kwarg that draws a faint bottom-right footer so figures are self-identifying when shared out of context.
- **MLP HPO dynamic depth** — dropped the `_MAX_LAYERS = 6` cap. `optimize_mlp_config.yaml` and `optimize_mlpgp_config.yaml` now declare a single `layer_size` template (no more `layer1_size`...`layer6_size` anchors). At trial time `suggest_from_config` samples `n_layers` first, then suggests exactly `layer1_size`...`layer{n_layers}_size` using the template, so Optuna sees a search space whose dimensionality matches the chosen depth. Helpers added: `_LAYER_SIZE_TEMPLATE`, `_LAYER_SIZE_RE`, `_is_dynamic_layer_size`, `_hp_destination` (routes any `layer{i}_size` to the same registry slot as the template). `_extract_best_params` skips the template and pulls in whatever dynamic `layer{i}_size` the best trial actually sampled. Verified end-to-end: a trial with `n_layers=8` produces `layer1_size`...`layer8_size`; a trial with `n_layers=3` produces only `layer1_size`...`layer3_size`.

Smoke verification: all three modes (`single_run` on `model_to_train=all` + HYDDB_v3, `optimization` on mlp with `n_trials=4`, `convergence` analysis on mlpgp with two splits × two seeds) run end-to-end against `config_smoke` with reduced budgets. Single-run plot tree: 55 PNGs across `diagnostics/`, `metrics/`, `surface/`, `uncertainty/` (GP-family), `hyddb/` (5 files for GP-family, 4 for MLP since 3D+σ skips when σ is unavailable), `table/`. Optimization tree: `summary_dashboard.png` per model. Convergence tree: per-model `r2_vs_split.png`, `rmse_vs_split.png`, `mae_vs_split.png`, `mape_vs_split.png` plus `runs.csv` / `metrics_vs_split.csv`, and a `complete/` overlay when more than one model is present.

Note: `src/convergence_analysis.py` still hardcodes `config_dir = PROJECT_ROOT / "config"` and `analysis_root = PROJECT_ROOT / "outputs" / dataset_name / ANALYSIS_ROOT_NAME`, so convergence-mode smoke runs ignore `--config-dir config_smoke` / `--output-dir outputs_smoke` for those reads/writes. Pre-existing limitation; not in scope of this refactor.

### Visualization refactor + HYDDB physics-driven plot suite (stages 0-4, 7-9)

Restructured `src/visualization/` from a flat layout into three explicit layers:

- **`plots/`** (was `modules/` + the loose `diagnostics.py`) — leaf plot functions, one figure each. Files: `metrics.py` (parity / residuals / mape, was `parity.py`), `surface_generic.py` (was `surface.py`), `uncertainty.py` (was `uncertainty_gp.py`), `table.py`, `diagnostics.py`.
- **`suites/single_run.py`** — orchestrates which leaves run for the single-run workflow; iterates models, runs metrics + uncertainty (GP-only) + surface + dataset suite (if any), then the comparison table when all 3 model families are present.
- **`datasets/hyddb.py`** — physics suite for HYDDB_v{2,3}: `Cfx vs Mach` parametrized on airbrake opening (alpha=0), `Cfx vs alpha` parametrized on Mach (abk=0), phi sensitivity, 2D `(Mach, alpha) → (Cfx, σ)` heatmaps, and a 3D `Cfx` surface paired with a σ heatmap (3D skipped for MLP since σ is unavailable).

Other changes that landed together:

- `runner.py` is now a thin per-mode dispatcher: `run_visualization(workflow_mode, *, config, dataset_name="", models=..., data=..., study=..., convergence_records=...)`. Only `single_run` is wired through it for now; `optimization` and `convergence` raise on entry but their old plotting paths still produce output (see "Open items").
- Output now lands in `<run-dir>/plots/<category>/<file>.png` (categories: `diagnostics`, `metrics`, `uncertainty`, `surface`, `hyddb`, `table`). `finalize_figure(fig, filename, config, *, subdir="")` is the single seam.
- The `mode="train"/"test"` magic argument was removed from every plot signature; display-after-save is now `config.show_after_save: bool` (default `False`).
- `VisualizationConfig.HYDDBConfig` block: ranges for each axis, `abk_curves` and `mach_curves` lists, `n_grid_points` / `n_sweep_points`, `enable_3d`, and **`locked_values`** (denormalized {column → value} map). Per the user's decision, missing locked values for any input that is neither swept nor parameterized **raise `ValueError`** with the exact missing column names — no median fallback.
- `config/visualization/visualization_config.yaml` ships a working `hyddb:` block with locked_values for every HYDDB_v3 input. Adjust the locked values to taste.
- `main.py` passes `dataset_name` into the dispatcher so HYDDB_v2 / HYDDB_v3 routes through `datasets.hyddb`. The post-loop "table-only" call disables `parity`, `surface`, `uncertainty_gp`, **and `hyddb`** (otherwise the physics plots run again per output dir).

Verified end-to-end: `python main.py --config-dir config_smoke --output-dir outputs_smoke` on HYDDB_v3 / 30-step budget produces 55 plots — 5 HYDDB plots per GP-family model (4 for MLP, since the 3D + σ panel is skipped) plus all generic categories, no errors.

Stages 5 and 6 of the original plan (move optimization/convergence plotting *into* `suites/optimization.py` and `suites/convergence.py`) were intentionally **deferred**: they touch heavy non-viz code (`src/optimization_support.py`, `src/convergence_analysis.py`) and are organizationally independent of HYDDB. Stage 10 (full aesthetic / TeX-label sweep on the non-HYDDB plots) was also deferred. See "Open items" for what is still pending.

### MLP structural change → single hidden block + activation-driven init
- `src/models/init.py` — registry `_BLOCK_INIT[activation] = init_fn` for `{relu, leaky_relu, selu, tanh, mish}`. `init_block_linear(layer, activation)` looks up and applies; `init_output_linear(layer)` is activation-agnostic (Xavier-normal, gain=1). Adding a new activation = one line in the registry.
- `src/models/mlp.py` — single-block `MLPRegressor`. Architecture is now a single dict `{layers: [int, ...], activation: str}` (no more list of blocks). Hidden-layer init is driven by the block activation; output layer uses the agnostic init. Output activation factory (`_resolve_activation`) covers block activations + `sigmoid` + `identity`.
- `src/optimization_support.py` — replaced `_build_block_architecture` with `_build_architecture` (single block). `_HP_TO_MODEL_CFG` now maps `n_layers`, `layer{i}_size` (i ∈ 1..`_MAX_LAYERS`=6), and `activation` to the architecture scope; `_BLOCK_HPS` is the canonical tuple, aliased per model. Old `block{N}_*` keys removed.
- `config/model/mlp_config.yaml` and `config/model/mlpgp_config.yaml` — `architecture` is now a single dict (both feature extractors of MLPGP follow the same shape). The standalone `init` field has been removed (init is fixed per activation).
- `config/optimization/optimize_mlp_config.yaml` and `optimize_mlpgp_config.yaml` — search spaces refactored to `n_layers ∈ [1, 6]`, `layer{i}_size` (categorical, anchored via `&layer_size`), `activation ∈ [relu, leaky_relu, selu, tanh, mish]`. The mlpgp file uses a hidden `_shared.block` anchor merged into both architectures via `<<: *block` to avoid duplication.

### `trend_degree` exposed to HPO
- `_HP_TO_MODEL_CFG` gained `("gp", "trend_degree"): "root"` and `("mlpgp", "trend_degree"): "root"`. New `"root"` destination in `_apply_hp` writes top-level `model_cfg` keys with per-key coercion via `_ROOT_COERCERS = {"trend_degree": int}` (extensible).
- Added to `optimize_gp_config.yaml` (search_space.model) and `optimize_mlpgp_config.yaml` (both arch sub-spaces). Range: int [0, 3] (user-edited from initial [0, 2]).

### SVGP support added (in escrow — may be removed)
- `SVGPRegressionModel` and `MLPSVGPModel` (mixin-based: `_GPRegressionMixin`, `_MLPGPMixin` provide shared `forward` and setup; mixin precedes the GPyTorch parent in MRO so the mixin's `forward` overrides `*GP.forward`'s `NotImplementedError`).
- `SVGPTrainer` in `trainer.py`: mini-batch ELBO via `gpytorch.mlls.VariationalELBO`, no `_cholesky_context` (inducing block is small). Optional joint MLP+GP parameter groups when the model exposes `feature_extractor_parameters`.
- Pipeline routes via `model_cfg["inference"] = "exact" | "svgp"` (model classes) and `isinstance(model, (SVGPRegressionModel, MLPSVGPModel))` (trainer). Predictor recognizes the new classes.
- Configs: `inference` + `svgp.{n_inducing_points, learn_inducing_locations}` in model YAMLs; `svgp_n_epochs` + `svgp_batch_size` in training YAMLs.

### Repo-wide cleanup pass
- Trimmed verbose docstrings, removed narrative comments, factored duplicated DataLoader construction in `DataBundle._make_loader`, factored `_gp_train_tensors` in pipeline. Replaced `assert` ratio check with `raise ValueError` in `dataset.py` (assertions strip under `python -O`). Removed `_require_config_value` legacy alias in `mlpgp.py`.
- All four batches verified bit-identical metrics on the smoke test before SVGP/init changes intentionally diverged the RNG sequence.

---

## Open items

- **Repo refactor totale APPROVED, NOT IMPLEMENTED (2026-05-17).** Plan in [PLAN_repo_refactor.md](PLAN_repo_refactor.md). 10 stage, ~3500 righe nette. User-confirmed decisions: (1) SVGP rimosso completamente; (2) modelli canonici `mlp`, `gp`, `mlpgp_ns`, `mlpgp_dkl` (split `mlpgp_config.yaml` in due files); (3) HPO solo via `convergence/optimization` sub-mode, top-level `optimization` rimosso; (4) try/except a 2 livelli (main + per-model); (5) `models_to_train: [list]` + CLI `--models` con `all` come alias; (6) regression via smoke + soglie R² ±0.005 / RMSE rel ±1%; (7) logging tag `[Stage X/Y | split A/B | model M | run R/N]`; (8) early-stopping params hoist in `training_common_config.yaml`; (9) `kernel_init_bounds` rimosso da convergence config. Restore points: questo handoff item + [memory/project_repo_refactor.md](C:\Users\marco\.claude\projects\c--Users-marco-Desktop-Poli-NAML-project-repo-naml-surrogate-models\memory\project_repo_refactor.md) + Claude plan file `C:\Users\marco\.claude\plans\analizza-il-codice-e-dazzling-duckling.md`. Procedere su feature branch `refactor-2026` da `refactoring-repo`.

- **HYDDB_v3 report figures — DONE (2026-05-17).** Plan in [PLAN_hyddb_v3_report_figures.md](PLAN_hyddb_v3_report_figures.md) implementato. Bug normalizzazione fixato (`normalize_features` al call site di `_plot_slice_coverage`), viridis colormap sub-range con colorbar discreta esterna, sample dots con bordo scuro, Plot 2 layout quadrato con ideal Gaussian più visibile. Annotations under/over-confident e freccia adattiva rimosse su richiesta dell'utente. Output in `report/gfx/hyddb_v3/`, run_id `single_20260515_190104`.

- **Runtime smoke verification for new run-id/test flow still pending.**
  - Need quick matrix validation:
    - two consecutive `single_run` trainings create distinct `runs/<id>` folders,
    - test-only execution resolves checkpoint via `--resume-run-id` and via latest-run fallback,
    - optimization and convergence invocations write into distinct run-id trees without overwrite.
  - Static checks are clean, but this runtime matrix has not been fully executed yet.

- **Convergence orchestrator paths are still hardcoded.** `src/convergence_analysis.py` reads `main_config.yaml` from `PROJECT_ROOT / "config"` and writes to `PROJECT_ROOT / "outputs" / <dataset>` regardless of `--config-dir` / `--output-dir`. Smoke runs in convergence mode therefore can't be redirected. Threading the CLI args through `run_analysis(config_path)` would close this — but it changes a public-ish surface, so left for later.
- **Aesthetic pass coverage.** Stage 10 dropped global bold titles and added the self-identifying footer + `labels.py`, but only the optimization dashboard and convergence curves are wired to the new constants. Parity/residual/surface/uncertainty leaf plots still build labels inline (mostly already TeX where it matters). Pure cosmetic — pick up only if a future pass needs it.
- **SVGP review pending.** Treated as in-escrow. Decision to keep or delete is the user's; if deleted, removing it is delete-only (no entanglement with mainline code), but the mixins (`_GPRegressionMixin` / `_MLPGPMixin`) introduced for it would also become unnecessary and could be inlined back.
- **`optimization_support.py` and `convergence_analysis.py` got only a top-docstring trim.** They still contain heavier narrative comments compared to the rest of the codebase; if a presentation pass is needed there, do it next.
- **Cholesky/CG warnings are NOT silenced** in `_BaseGPTrainer._cholesky_context` — debug-only state. Re-enable the `NumericalWarning` filter once GP numerics are validated on the real datasets.
- **MLP smoke metrics will not match prior baselines anymore** because Xavier-tanh init now diverges the RNG sequence vs. the prior PyTorch-default Kaiming-uniform-`a=√5`. The smoke test should now treat "all models train + reasonable R²" as the contract, not bit-identical.
- **Optuna search-space side-effect on `optimize_mlpgp_config.yaml`.** The hidden `_shared` key is loaded into `opt_cfg` but never read — harmless. If a YAML linter ever objects, gate the loader to drop keys starting with `_`.

## In escrow / be careful when removing

- **SVGP code paths**: `SVGPRegressionModel`, `MLPSVGPModel`, `SVGPTrainer`, the `inference` switch, and the `svgp` config blocks. If you delete, also drop `_select_inducing_points`, `_build_likelihood_from_cfg`, the `ApproximateGP` imports, and the `_SVGP_CLASSES` map.
- **Mixins** (`_GPRegressionMixin`, `_MLPGPMixin`): exist primarily to share `forward` + setup between exact and SV variants. If SVGP goes, the mixin can be inlined back into the exact class — just move the `_setup_*` / `forward` methods back, drop the mixin, and the parent-list ordering becomes uncritical again.

---

## Reproducing the smoke test

The user's authoritative test runner is `python main.py`. To iterate quickly on structural changes without touching the live `config/`:

```bash
cp -r config config_smoke
# Edit config_smoke/main_config.yaml: mode=single_run, model_to_train=all, dataset=sml
# Edit config_smoke/training/training_*_config.yaml: n_training_steps=30, n_epochs=30
rm -rf outputs_smoke
python main.py --config-dir config_smoke --output-dir outputs_smoke
# Per-model R² and RMSE: outputs_smoke/<dataset>/single_run/<model>/summary.csv
```

After verifying, `rm -rf config_smoke outputs_smoke`. Do not commit either.

Active datasets in `config/dataset/dataset_config.yaml`: `sml`, `airfoil_self_noise`, `HYDDB_v3` (the live config currently uses `HYDDB_v3`). The smoke run uses `sml` because it converges fast on a 30-step budget.

---

## Known external state

- Active branch: `refactoring-repo`. Main is `main`. The user has not asked for a commit yet — uncommitted edits are accumulating; surface this when ending a session if a logical chunk is ready.
- The user has live edits to `main_config.yaml` (mode and dataset toggle frequently). Treat that file as user-owned; don't restore it without asking.