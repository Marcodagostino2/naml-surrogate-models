# PLAN — Refactor totale del repo `naml-surrogate-models`

**Status:** APPROVATO il 2026-05-17 — non ancora implementato.
**Branch consigliato:** nuovo da `refactoring-repo` (es. `refactor-2026`).
**Effort stimato:** ~3500 righe nette modificate, 10 stage, ~1-2 settimane di lavoro a tempo pieno.
**Companion plan** (Claude plan dir, contenuto identico): `C:\Users\marco\.claude\plans\analizza-il-codice-e-dazzling-duckling.md`.

---

## Context

Il repo è cresciuto in modo organico attraverso più refactor (vedi `SESSION_HANDOFF.md`). L'utente vuole una pulizia totale: rimuovere duplicazioni, accentrare try/except sparsi, eliminare modalità ridondanti, supportare entrambe le varianti di MLPGP simultaneamente, rendere il codice "meno-AI" e più leggibile, mantenendo i risultati numerici invariati entro tolleranza.

I 12 requisiti utente espressi:
1. Rimuovere logiche duplicate (classi astratte/mixins).
2. Try/except sparsi → pochi global try/except.
3. Ridurre funzioni "one-liner" inutili.
4. Eliminare top-level `optimization` mode; far confluire HPO in `convergence`.
5. Rimuovere `kernel_inits` da `convergence_config` (no re-randomizzazione kernel per run).
6. Lista esplicita di modelli (oggi `"all"` esclude una variante MLPGP).
7. Pulire override config non funzionali, spostare logiche comuni.
8. Risultati identici (entro tolleranza) agli attuali.
9. Logging migliorato con tag di stage.
10. Documentazione minima, no AI-tells.
11. Aggiornare README.
12. Flusso chiaro, niente dead code.

---

## Decisioni utente (confermate durante planning)

| # | Decisione |
|---|---|
| 1 | **SVGP rimosso completamente** (SVGPRegressionModel, MLPSVGPModel, SVGPTrainer + config blocks `svgp.*`, `svgp_n_epochs`, `svgp_batch_size`, `_resolve_gp_inference`, `_SVGP_CLASSES`). |
| 2 | **Modelli canonici**: `mlp`, `gp`, `mlpgp_ns`, `mlpgp_dkl`. `architecture_name` config diventa ridondante (dedotto dal nome classe). |
| 3 | **HPO via convergence** con `mode: optimization`. Top-level `optimization` mode rimosso. `RunType.OPTIMIZATION` rimosso. |
| 4 | **Try/except a 2 livelli**: top-level in `main.py` + per-model (continua col modello successivo se uno fallisce). Try interni a Cholesky e Optuna pruning restano (load-bearing). |
| 5 | **`models_to_train: [list]`** nel config + CLI `--models mlp,gp,mlpgp_ns`. `all` resta come alias del set canonico. |
| 6 | **Verifica regressione**: smoke-test + soglie metriche (R² ±0.005, RMSE relativo ±1%). Bit-identical NON richiesto. |
| 7 | **Logging**: INFO con tag `[Stage X/Y \| split A/B \| model M \| run R/N]`. Niente tqdm. |
| 8 | **Common params**: spostare early-stopping (`rel_tol`, `abs_tol`, `train_patience`, `val_patience`, `warmup_epochs`) + log intervals in `training_common_config.yaml`. |

Vincoli aggiuntivi dichiarati:
- Checkpoint vecchi (sotto `outputs/<dataset>/single_run/mlpgp/...`) diventano inaccessibili dopo Stage 5. Documentare nel README; nessuno script di migration.
- `optimization` mode legacy → errore esplicito con messaggio di migrazione, non silent fallback.

---

## Stage-by-stage

### Stage 0 — Test harness baseline (prerequisito)

**Cosa cambia (additivo, no edit a codice di produzione):**
- Nuovo file `tests/test_regression_baseline.py` (smoke + soglia metriche tramite pytest).
- Nuovo file `scripts/capture_baseline_metrics.py` (esegue ogni combo `mode/model` su `config_smoke_quick` e scrive `tests/baselines/<mode>_<model>.json` con R², RMSE, MAE).
- Tolleranze: `|R²_new − R²_baseline| ≤ 0.005`; `|RMSE_new − RMSE_baseline| / RMSE_baseline ≤ 0.01`.
- Combo da catturare: `single_run × {mlp, gp, mlpgp (architecture_name=non_stationary), mlpgp (architecture_name=dkl)}`, `convergence/analysis`, `convergence/optimization`, `optimization` (legacy, prima della rimozione).

**Dipendenze:** nessuna.
**Verifica:** harness gira green su `main` prima di toccare codice.
**Rischio:** basso (additivo).
**Effort:** small (~150 righe).

---

### Stage 1 — Rimozione SVGP

**Cosa cambia:**
- [src/models/gp.py](src/models/gp.py): elimina `SVGPRegressionModel`, `_select_inducing_points`. Mantieni `_build_gaussian_likelihood`, `_parse_noise_cfg`, `PolynomialTrendMean`, `_GPRegressionMixin`, `GPRegressionModel`.
- [src/models/mlpgp.py](src/models/mlpgp.py): elimina `MLPSVGPModel` + imports relativi (`VariationalStrategy`, `CholeskyVariationalDistribution`, `ApproximateGP`).
- [src/models/__init__.py](src/models/__init__.py): rimuovi export SVGP.
- [src/training/trainer.py](src/training/trainer.py): elimina classe `SVGPTrainer` (righe ~1005-1245) e imports SVGP.
- [src/training/__init__.py](src/training/__init__.py): rimuovi export.
- [src/pipeline.py](src/pipeline.py): elimina `_SVGP_CLASSES`, `_resolve_gp_inference`. In `_build_model`: drop branch `inference == "svgp"`. In `_build_trainer`: drop branch `isinstance(model, (SVGPRegressionModel, MLPSVGPModel))`. Rimuovi imports.
- [src/inference/predictor.py](src/inference/predictor.py): rimuovi SVGP da type union e da `predict` dispatch.
- Configs (delete only):
  - `config/model/gp_config.yaml`: rimuovi linea `inference: "exact"` e blocco `svgp:`.
  - `config/model/mlpgp_config.yaml`: idem.
  - `config/training/training_gp_config.yaml`: rimuovi `svgp_n_epochs`, `svgp_batch_size`.
  - `config/training/training_mlpgp_config.yaml`: idem.
  - Mirror in `config_smoke_quick/`.

**Dipendenze:** Stage 0.
**Verifica:** smoke harness su `single_run × {mlp, gp, mlpgp}` + `convergence/analysis`; metriche entro tolleranza. `python scripts/import_smoke_test.py` deve passare.
**Rischio:** basso (SVGP non è esercitato dai config di default, `inference: "exact"` ovunque).
**Effort:** medium (~400 righe rimosse, ~20 modificate).

---

### Stage 2 — Cleanup dead code

**Cosa cambia:**
- [src/models/gp_utils.py](src/models/gp_utils.py): elimina `extract_required_config_value` (alias di `extract_config_value`).
- [src/models/gp.py](src/models/gp.py) + [src/models/mlpgp.py](src/models/mlpgp.py): sostituisci i due callsites di `extract_required_config_value` con `extract_config_value`.
- [src/training/trainer.py](src/training/trainer.py): elimina `GPTrainer._optimizable_parameters` (righe 613-614), inline `list(self.model.parameters())` in `_build_optimizer`.
- [src/optimization_support.py](src/optimization_support.py): elimina handling `min_resource`/`max_resource`/`reduction_factor` in `build_pruner`. Se `pruner == "hyperband"`, raise `ValueError("hyperband pruner not supported")`.
- `config/optimization/optimize_mlp_config.yaml`: rimuovi `min_resource`, `max_resource`, `reduction_factor`.
- `config/training/training_mlp_config.yaml`: rimuovi `save_checkpoint_every_n_epochs`.
- `config/training/training_gp_config.yaml`: rimuovi `save_checkpoint_every_n_steps`.
- `config/training/training_mlpgp_config.yaml`: idem.

**Dipendenze:** Stage 1 (stessi file YAML, evita merge churn).
**Verifica:** smoke harness deve match baseline (no semantic change). Grep audit: zero match per le chiavi rimosse in `src/`.
**Rischio:** basso (target dimostrabilmente unused).
**Effort:** small (~40 righe nette).

---

### Stage 3 — Estrazione `BaseTrainer`

**Cosa cambia:**
- Nuovo file `src/training/base_trainer.py` con classe `BaseTrainer`:
  - `__init__`: device, `n_epochs/n_training_steps`, intervalli validation/log, `ConvergenceMonitor`, `_best_model_state`, `_last_reported_step`, `log_prefix`.
  - `_step_scheduler(self)` — scheduler.step() + auto-disable on failure (rimpiazza 3 duplicazioni).
  - `_maybe_save_best(self, val_loss)` — deep-copy state_dict se nuovo best.
  - `_maybe_report_to_trial(self, step, val_loss)` — `trial.report` + `should_prune` + `TrialPruned`.
  - `_restore_best(self)` — load `_best_model_state` a fine training.
  - `_log_progress(self, ...)` — hook di logging condiviso con format tag-prefisso.
- [src/training/trainer.py](src/training/trainer.py):
  - `MLPTrainer(BaseTrainer)`: rimuovi codice duplicato da `train()`.
  - `_BaseGPTrainer(BaseTrainer)`: eredita anche da BaseTrainer, mantiene `_cholesky_context`.
  - `GPTrainer(_BaseGPTrainer)`, `MLPGPTrainer(_BaseGPTrainer)`: usano metodi ereditati.
- `ConvergenceMonitor`: lascia in `trainer.py` (o sposta in `base_trainer.py`, indifferente).

**Critico per ordering interno al loop**: in GP/MLPGP, le 3 history list (`train_losses`, `val_losses`, `mll_history`) devono essere appese **prima** di `_maybe_save_best` / `_maybe_report_to_trial`. Se BaseTrainer rilocasse questi step, una Cholesky failure potrebbe lasciare liste di lunghezza diversa e rompere plotting downstream.

**Dipendenze:** Stage 1, 2.
**Verifica:** smoke harness; per `single_run/mlp` bit-identical atteso; per GP/MLPGP entro tolleranza. Unit test in `tests/test_base_trainer_invariants.py`: (a) `_maybe_save_best` salva solo quando `val_loss < best`; (b) `_step_scheduler` disabilita on exception; (c) Cholesky failure simulato lascia history lengths uguali.
**Rischio:** medium (hot path). Mitigazione: estrarre un metodo alla volta con smoke fra ognuno.
**Effort:** large (~300-400 righe ristrutturate, ~150 nette eliminate).

---

### Stage 4 — HPO via convergence (additivo; legacy `optimization` ancora attivo)

**Cosa cambia:**
- [src/convergence_analysis.py](src/convergence_analysis.py):
  - Elimina `_kernel_init_for_model` (righe 140-158) e `_set_kernel_seeded_values` (righe 161-178).
  - In `_run_analysis_mode`: sostituisci i call site con `model_cfg = deepcopy(model_cfg_map[model_name])`.
  - Rimuovi argomento `preserve_kernel_if_present` e il suo read.
  - In `_run_optimization_mode`: elimina dead-code branch alle righe 473-477 (logger.debug su `kernel_init_bounds`).
- `config/convergence/convergence_config.yaml`:
  - Elimina blocco `kernel_init_bounds:` (righe 19-34).
  - Elimina `preserve_kernel_if_present: true`.
- Mirror in `config_smoke_quick/`.

**Dipendenze:** Stage 1.
**Verifica:** `python main.py --mode convergence` con `convergence_config.yaml: mode: optimization` su smoke. Confronto best_value, n_trials, retrain metrics entro tolleranza. `python main.py --mode optimization` (legacy) deve ANCORA funzionare invariato — invariante chiave di questo stage.
**Rischio:** basso.
**Effort:** small (~80 righe rimosse).

---

### Stage 5 — Split `mlpgp` in `mlpgp_ns` e `mlpgp_dkl`

Stage più invasivo. Cambia il "nome modello surface area" in configs, code, run-dir layout, registry.

**Cosa cambia:**

Modelli:
- [src/models/mlpgp.py](src/models/mlpgp.py): introduci `MLPGPNonStationaryModel(MLPGPModel)` e `MLPGPDeepKernelModel(MLPGPModel)` come sottoclassi che impostano `self.architecture_name` in `__init__` prima di delegare. Elimina branch dispatcher `architecture_name in {...}` da `_MLPGPMixin._init_mlpgp`.
- [src/models/__init__.py](src/models/__init__.py): export delle due nuove classi. Rimuovi `MLPGPModel` da export.

Pipeline + paths:
- [src/pipeline.py](src/pipeline.py): `_MODEL_REGISTRY`:
  ```
  "mlp":        (MLPRegressor, MLPTrainer, False),
  "gp":         (GPRegressionModel, GPTrainer, True),
  "mlpgp_ns":   (MLPGPNonStationaryModel, MLPGPTrainer, True),
  "mlpgp_dkl":  (MLPGPDeepKernelModel,    MLPGPTrainer, True),
  ```
- [src/utils/paths.py](src/utils/paths.py): `ModelName.MLPGP_NS = "mlpgp_ns"`, `ModelName.MLPGP_DKL = "mlpgp_dkl"`; rimuovi `MLPGP`.
- [src/convergence_analysis.py](src/convergence_analysis.py): `MODEL_SEQUENCE = ("mlp", "gp", "mlpgp_ns", "mlpgp_dkl")`.

Configs (split + cancellazione `mlpgp_config.yaml`):
- Nuovi files:
  - `config/model/mlpgp_ns_config.yaml`: copia `mlpgp_config.yaml`, mantieni solo `non_stationary_feature_extractor:`, drop `dkl_feature_extractor:`, rimuovi `architecture_name`.
  - `config/model/mlpgp_dkl_config.yaml`: idem ma con `dkl_feature_extractor:`.
  - `config/training/training_mlpgp_ns_config.yaml` e `training_mlpgp_dkl_config.yaml`: copie identiche di `training_mlpgp_config.yaml`.
  - `config/optimization/optimize_mlpgp_ns_config.yaml`: estrai `search_spaces.non_stationary` di `optimize_mlpgp_config.yaml`, hoist a top-level `search_space:`. Drop `architecture_name`.
  - `config/optimization/optimize_mlpgp_dkl_config.yaml`: idem con `dkl`.
- Cancella: `config/model/mlpgp_config.yaml`, `config/training/training_mlpgp_config.yaml`, `config/optimization/optimize_mlpgp_config.yaml`.
- Mirror tutto in `config_smoke_quick/`.

Optuna support:
- [src/optimization_support.py](src/optimization_support.py):
  - Elimina `_resolve_architecture_name` e il branch `search_spaces` di `_resolve_active_search_space`.
  - `_HP_TO_MODEL_CFG`: aggiungi entry per `mlpgp_ns` e `mlpgp_dkl`.
  - In `_apply_hp`: rimpiazza test `architecture_name == "dkl"` con `model_name == "mlpgp_dkl"`.
  - Passa `model_name` (non `architecture_name`) a `_apply_hp` ovunque.

Main + dispatcher:
- [config/main_config.yaml](config/main_config.yaml): aggiorna commento valori validi per `model_to_train`.
- [main.py](main.py): aggiorna `argparse choices`, validation sets, literal `["mlp","gp","mlpgp"]`.
- `config/convergence/convergence_config.yaml`: `runs_per_model`, `optimization_overrides.study.n_trials/n_seeds` con chiavi `mlpgp_ns` e `mlpgp_dkl`.
- [src/visualization/](src/visualization/): grep `"mlpgp"` literal, aggiorna se presente.

**Dipendenze:** Stage 1, 2, 4.
**Verifica:** smoke `single_run --model mlpgp_ns`, `--model mlpgp_dkl`, `--model all` (4 modelli). `convergence/analysis` su tutti e 4. `convergence/optimization` su almeno `mlpgp_dkl`. Tolleranza vs baseline. Grep audit: zero match `architecture_name` in `src/`.
**Rischio:** alto. Mitigazioni: sub-step come commit separati; test schema config; error message esplicito.
**Effort:** large (~600 righe modificate + 6 YAML nuovi + 3 cancellati).

---

### Stage 6 — `model_to_train` → `models_to_train` (lista) + CLI `--models`

**Cosa cambia:**
- [config/main_config.yaml](config/main_config.yaml): `models_to_train: ["gp"]` (lista; `"all"` allowed come alias).
- [main.py](main.py):
  - CLI `--model` → `--models` (comma-separated).
  - Helper `_parse_models_list(main_cfg, cli_arg)` ritorna lista canonical.
  - Itera lista; rimuovi validation `model_to_train not in {...}`.
- [src/pipeline.py](src/pipeline.py) `save_run_configs` riga 75: `patched_main_cfg["models_to_train"] = [selected_model]`.
- [src/convergence_analysis.py](src/convergence_analysis.py) `_get_models_to_train`: legge `models_to_train`, espande `"all"`, valida, ritorna tupla.
- Mirror in `config_smoke_quick/main_config.yaml`.
- Migration error: vecchio `model_to_train` → `ValueError` chiaro.

**Dipendenze:** Stage 5.
**Verifica:** smoke `--models mlp`, `--models mlp,gp`, `--models all`. Convergence con `models_to_train: ["mlpgp_ns", "mlpgp_dkl"]` runna solo le due MLPGP.
**Rischio:** medium.
**Effort:** medium (~120 righe).

---

### Stage 7 — Drop top-level `optimization` mode

**Cosa cambia:**
- **Cancella file** `src/entrypoints/optimize.py`.
- [src/utils/paths.py](src/utils/paths.py): elimina `RunType.OPTIMIZATION`.
- [src/pipeline.py](src/pipeline.py) `save_run_artifacts` riga 164: cambia gate da `"optimization"` a `"convergence_analysis"`; gestisci optional `best_params`/`all_trials_csv`.
- [main.py](main.py):
  - Elimina `_run_optimization` e dispatch branch.
  - Elimina CLI args `--model`, `--n-trials`, `--timeout`.
  - `--mode` choices: `("single_run", "convergence")`.
  - Config `mode: optimization` → `ValueError` con migration message.

**Dipendenze:** Stage 4, 5, 6.
**Verifica:** smoke + audit run-dir (no nuovi `outputs/<dataset>/optimization/`).
**Rischio:** medium.
**Effort:** medium (~250 righe cancellate).

---

### Stage 8 — Consolidamento common params

**Cosa cambia:**
- [config/training/training_common_config.yaml](config/training/training_common_config.yaml) — append:
  ```yaml
  rel_tol: 1.0e-4
  abs_tol: 1.0e-6
  train_patience: 200
  val_patience: 400
  warmup_epochs: 400
  train_metrics_log_interval: 10
  validation_check_interval: 10
  ```
- Cancella le stesse chiavi da: `training_mlp_config.yaml`, `training_gp_config.yaml`, `training_mlpgp_ns_config.yaml`, `training_mlpgp_dkl_config.yaml`.
- Mirror in `config_smoke_quick/training/`.

**Dipendenze:** Stage 5, 6.
**Verifica:** smoke harness, bit-identical atteso.
**Rischio:** basso.
**Effort:** small (~35 righe nette).

---

### Stage 9 — Try/except a 2 livelli + structured logging tags

**Cosa cambia:**
- [main.py](main.py):
  - Top-level `try/except Exception`: log `exc_info=True`, exit 1. Re-raise `KeyboardInterrupt`, `MemoryError`.
  - In `_run_single_run` e `_run_convergence`: wrappa body model loop. Log + continua col modello successivo.
- [src/convergence_analysis.py](src/convergence_analysis.py): stesso pattern in loop `(split, model)`. Failure → riga NaN nel metrics dataframe.
- Logging helper `_build_run_log_prefix` esteso a `[Stage X/Y | split A/B | model M | run R/N]`. Centralizzato in `BaseTrainer` (Stage 3).
- Audit: `Grep "tqdm"` deve dare zero match.

**Dipendenze:** Stage 7.
**Verifica:** inject failure (es. `learning_rate` negativo per `gp`): altri modelli runnano, summary CSV con NaN row, exit code parziale ma no crash.
**Rischio:** medium.
**Effort:** medium (~100 righe).

---

### Stage 10 — README + docs refresh

**Cosa cambia:**
- [README.md](README.md):
  - Workflow list: `single_run`, `convergence`.
  - Model list: `mlp`, `gp`, `mlpgp_ns`, `mlpgp_dkl`.
  - Quick-start aggiornato.
  - Migration note per `optimization` → `convergence`, `mlpgp` → `mlpgp_ns`/`mlpgp_dkl`.
  - Project structure tree aggiornato.
- Docstring headers YAML se referenziano nomi vecchi.

**Dipendenze:** Stage 1-9.
**Verifica:** human readthrough; comandi consistenti con `main.py --help`.
**Rischio:** basso.
**Effort:** small (~80 righe markdown).

---

## Critical files (riepilogo per stage)

| File | Stage che lo toccano |
|---|---|
| [main.py](main.py) | 5, 6, 7, 9 |
| [src/pipeline.py](src/pipeline.py) | 1, 5, 7 |
| [src/training/trainer.py](src/training/trainer.py) | 1, 2, 3 |
| `src/training/base_trainer.py` | 3 (new) |
| [src/optimization_support.py](src/optimization_support.py) | 2, 5 |
| [src/convergence_analysis.py](src/convergence_analysis.py) | 4, 5, 6, 9 |
| [src/models/mlpgp.py](src/models/mlpgp.py) | 1, 5 |
| [src/models/gp.py](src/models/gp.py) | 1, 2 |
| [src/utils/paths.py](src/utils/paths.py) | 5, 7 |
| `config/main_config.yaml` | 5, 6, 7 |
| `config/convergence/convergence_config.yaml` | 4, 5 |
| `config/model/mlpgp_*.yaml` | 5 |
| `config/training/training_*.yaml` | 1, 2, 5, 8 |
| `config/optimization/*.yaml` | 2, 5 |
| [README.md](README.md) | 10 |
| `tests/test_regression_baseline.py` | 0 (new) |
| `tests/test_base_trainer_invariants.py` | 3 (new) |
| `scripts/capture_baseline_metrics.py` | 0 (new) |

---

## Dependency graph

```
Stage 0 (baseline harness)
   |
   v
Stage 1 (SVGP removal) --> Stage 2 (dead-code cleanup)
   |                            |
   |                            v
   +--------------------> Stage 3 (BaseTrainer)
   |                            |
   |                            v
   +--------------------> Stage 4 (HPO via convergence, additive)
                                |
                                v
                          Stage 5 (mlpgp_ns / mlpgp_dkl split)
                                |
                                v
                          Stage 6 (models_to_train list + --models)
                                |
                                v
                          Stage 7 (drop top-level optimization)
                                |
                                v
                          Stage 8 (common-params consolidation)
                                |
                                v
                          Stage 9 (try/except + log tags)
                                |
                                v
                          Stage 10 (README + docs)
```

Critical path: **0 → 1 → 3 → 5 → 6 → 7 → 9 → 10** (8 stage). Stage 2, 4, 8 parallelizzabili se più persone.

---

## Critical points of attention

1. **Pipeline factory `_build_trainer` ([src/pipeline.py:217](src/pipeline.py#L217))** — Stage 1 (rimozione SVGP `isinstance`) e Stage 5 (split MLPGP) toccano entrambi questa funzione. Dopo Stage 5, trainer dispatch deve essere puramente `_MODEL_REGISTRY[model_name][1]`. Acceptance criterion: zero `isinstance` checks superstiti.

2. **HPO `_apply_hp` patch destination per MLPGP variants ([src/optimization_support.py:362-436](src/optimization_support.py#L362))** — Stage 5 deve atomicamente migrare il test `architecture_name == "dkl"` e le chiavi `_HP_TO_MODEL_CFG`. Unit test obbligatorio: `_apply_hp("mlpgp_dkl", {...})` deve asserire patch sotto `dkl_feature_extractor.architecture.layers`. Idem per `mlpgp_ns`.

3. **BaseTrainer MRO + scheduler-step ordering in GP/MLPGP loops (Stage 3)** — il codice attuale commits all per-step appends together per mantenere history a `step-1` dopo una Cholesky failure. Se BaseTrainer rilocasse `_maybe_save_best` / `_maybe_report_to_trial`, una failure nello stesso step potrebbe lasciare history partial-populated. Invariant test: simulare `NotPSDError` post-`_evaluate_regression_model` e asserire `len(train_losses) == len(val_losses) == len(mll_history)`.

---

## Verification approach

**Baseline capture (Stage 0):** prima di toccare codice, eseguire `scripts/capture_baseline_metrics.py` su `main` branch e committere `tests/baselines/*.json`. Combo necessarie:
- `single_run × {mlp, gp, mlpgp_ns_equivalent, mlpgp_dkl_equivalent}` (oggi `mlpgp` con `architecture_name` toggled).
- `convergence/analysis`.
- `convergence/optimization`.
- `optimization` (legacy — necessario per Stage 4 e 7).

**Per-stage smoke:** dopo ogni stage, `pytest tests/test_regression_baseline.py`. Tolleranza: R² ±0.005, RMSE rel ±1%. Se fallisce, rollback dello stage o investigare root cause.

**Static checks:** dopo ogni stage:
- `python -m compileall src/`
- `python scripts/import_smoke_test.py`
- Grep audit specifici per stage.

---

## Out of scope (deliberato)

- `src/visualization/` — già refactored. Solo grep + sostituzione di literal `"mlpgp"` se presenti (Stage 5).
- HYDDB-specific plots in `src/visualization/datasets/hyddb.py` — invariati.
- `PLAN_hyddb_v3_report_figures.md` e altre figure di report — lavoro precedente, già implementato.
- Migration script per checkpoint vecchi — l'utente accetta che i vecchi `outputs/<dataset>/single_run/mlpgp/...` diventino inaccessibili.

---

## Next steps

1. Aprire feature branch `refactor-2026` da `refactoring-repo`.
2. Eseguire Stage 0 (baseline capture). Committere baseline.
3. Stage 1 → Stage 10, ognuno come commit (o più commit) distinti per facilitare bisect.
4. Smoke test dopo ogni stage; rollback locale se la soglia metrica fallisce.
5. PR finale `refactor-2026 → refactoring-repo` (o `main`, da decidere).
