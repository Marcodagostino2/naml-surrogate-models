# PLAN — Fix + restyle delle figure HYDDB_v3 per il report LaTeX

**Status:** NOT IMPLEMENTED — piano scritto durante session planning del 2026-05-16, da discutere con l'utente nella prossima sessione prima di implementare.
**Branch:** `refactoring-repo`
**Target file da modificare:** `scripts/generate_hyddb_v3_report_figures.py` (unico file modificato).
**Companion plan file** (Claude plan dir, identico contenuto): `C:\Users\marco\.claude\plans\analizza-il-codice-e-dazzling-duckling.md`.

---

## Context

Le due figure del report LaTeX HYDDB_v3 in [report/gfx/hyddb_v3/](report/gfx/hyddb_v3/) hanno problemi causati da un session del copilot precedente:

- **Plot 1** `hyddb_v3_slice_coverage_gp_mlpgp.{pdf,png}` (1D slice di C_A vs Mach, parametrizzato su delta_abk): nel subplot MLPGP le curve appaiono costanti a ≈30, e nel subplot GP appaiono schiacciate a fondo asse Y. Lo script è in [scripts/generate_hyddb_v3_report_figures.py](scripts/generate_hyddb_v3_report_figures.py).
- **Plot 2** `hyddb_v3_uncertainty_reliability_gp_mlpgp.{pdf,png}` (reliability diagram): valori corretti ma stile poco paper-like.

Lo script ha anche dead code accumulato (funzione `_slice_distance` mai chiamata, tolleranze CLI mai applicate, `predictions_many`/`many_abk` calcolati ma mai usati) — da ripulire concomitantemente.

## Root cause Plot 1

`_predict()` ([scripts/generate_hyddb_v3_report_figures.py:236](scripts/generate_hyddb_v3_report_figures.py#L236)) costruisce un `Predictor` e gli passa `features` in coordinate **fisiche raw** (Mach=0.3-1.8, alpha=0, phi=0, alt=5000, abk=0-100). Ma `Predictor.predict_gp` ([src/inference/predictor.py:56-68](src/inference/predictor.py#L56-L68)) **non normalizza gli input** — il modello è stato allenato su input normalizzati `(X - bundle.input_scaler_mean) / bundle.input_scaler_std`.

- Per **GP** la `PolynomialTrendMean` ha `trend_degree=0` (costante) e il kernel su input fuori range "reverte alla media" → output ~0.5-1.5 plausibili.
- Per **MLPGP** la `PolynomialTrendMean` ha `trend_degree=1` ed esplode con `altitude=5000`; l'MLP feature extractor riceve input completamente fuori dal range di training → output costante ~30.

L'asse Y è poi calcolato globalmente sui due modelli ([scripts/generate_hyddb_v3_report_figures.py:300-311](scripts/generate_hyddb_v3_report_figures.py#L300-L311)), schiacciando il GP a fondo asse.

`_plot_uncertainty_metrics` e `_plot_uncertainty_reliability` invece passano `bundle.X_test` che **è già normalizzato**, quindi funzionano correttamente.

## Decisioni di design (confermate dall'utente durante planning)

| Aspetto | Scelta utente | Note |
|---|---|---|
| Sample point filter (Plot 1) | Tutti con ABK matching | Non applicare tolleranze su alpha/phi/alt; mostra tutti i sample con ABK esatto |
| Curve color (Plot 1) | Colormap viridis sub-range + freccia | `viridis(np.linspace(0.15, 0.85, n))` per evitare giallo poco visibile e blu navy poco contrastato |
| Banda incertezza (Plot 1) | 2σ | Conferma dell'attuale, ~95% coverage |
| Y-range (Plot 1) | Condiviso tra GP e MLPGP | Una volta fixato il bug, le curve cadono nel range fisico corretto |
| Run ID | `latest common` (default) | Auto-pick tra `outputs/HYDDB_v3/single_run/{gp,mlpgp}/runs/` |
| Plot 2 paper-like | Linea ideal Gaussian più visibile + annotations over/under-confident + layout quadrato | Niente banda di tolleranza |
| Output dir | `report/gfx/hyddb_v3` invariato | Già referenziato in [report/sections/06_results_for_HYDDB.tex](report/sections/06_results_for_HYDDB.tex) righe 119, 145 |

## Implementation plan

### A. Fix bug normalizzazione (Plot 1)

Normalizzare al **call site** dentro `_plot_slice_coverage`, non dentro `_predict()` — `_predict()` è chiamato anche dagli altri due plot con `bundle.X_test` già normalizzato.

Riutilizzare `normalize_features` da [src/visualization/utils.py:136](src/visualization/utils.py#L136) (gestisce edge-case std≈0).

```python
from src.visualization.utils import normalize_features  # add to imports

# inside _plot_slice_coverage, replacing lines 290-298:
for model_name, model in models.items():
    for abk in abk_values:
        inputs_phys = _build_slice_inputs(reference, mach_grid, abk)
        inputs_norm = normalize_features(inputs_phys, bundle.input_scaler_mean, bundle.input_scaler_std)
        mean, sigma = _predict(model, inputs_norm, bundle)
        predictions[model_name][abk] = (mean, sigma)
```

`_build_slice_inputs` resta invariato (continua a restituire coordinate fisiche, usate per X-axis e arrow placement).

### B. Plot 1 — restyle

- **Colormap viridis sub-range**:
  ```python
  abk_colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(abk_values)))
  ```
- **Per-curve coloring**: ogni curva e la sua banda 2σ usano `abk_colors[i]`. Linestyle `"-"` per GP, `"--"` per MLPGP (invariato).
- **Banda 2σ**: alpha da `0.08` → `0.14`.
- **Sample points** (replacing lines 338-346):
  ```python
  ax.scatter(
      raw["X"][split_idx, 0], raw["y"][split_idx],
      s=28, alpha=0.95, color=color_split,
      edgecolor="#222222", linewidth=0.6, zorder=5,
  )
  ```
  Eliminato bordo bianco e gating `marker_alpha` basato su `dist` (sempre zero, dead logic).
- **Freccia delta_abk adattiva** (replacing lines 348-368):
  ```python
  stack = np.stack([predictions[model_name][abk][0] for abk in abk_values], axis=0)
  spread = stack.max(axis=0) - stack.min(axis=0)
  mid_idx = int(np.argmax(spread))
  mach_mid = float(mach_grid[mid_idx])
  y_top, y_bot = float(stack[:, mid_idx].max()), float(stack[:, mid_idx].min())
  ax.annotate("", xy=(mach_mid, y_top), xytext=(mach_mid, y_bot),
              arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.1})
  ```
  Eliminato il testo `$\delta_{abk}$` accanto alla freccia (ridondante con la colorbar).
- **Colorbar discreta** a destra dei subplot:
  ```python
  from matplotlib.colors import BoundaryNorm, ListedColormap
  boundaries = np.array([0, 12.5, 37.5, 62.5, 87.5, 100])  # midpoints between abk_values
  sm = plt.cm.ScalarMappable(cmap=ListedColormap(abk_colors),
                              norm=BoundaryNorm(boundaries, ncolors=len(abk_values)))
  sm.set_array([])
  cbar = fig.colorbar(sm, ax=axes.tolist(), ticks=list(abk_values),
                       fraction=0.025, pad=0.02, aspect=24)
  cbar.set_label(r"$\delta_{abk}\,[\%]$")
  ```
  Figsize `(10.5, 4.2)` → `(11.4, 4.2)` per accogliere la colorbar.
- **Y-range condiviso**: invariato (calcolo `y_min`/`y_max` su tutte le predizioni). Una volta fixato il bug, le curve di entrambi i modelli stanno nel range fisico corretto e il range condiviso funziona naturalmente.

### C. Plot 2 — restyle paper-like (`_plot_uncertainty_reliability`)

- **Layout**: figsize `(5.7, 4.4)` → `(4.8, 4.8)` (quadrato).
- **Linea ideal Gaussian** (riga 498): `linestyle="--", color=PALETTE["ref"], linewidth=1.1` → `linestyle="-", color="#444444", linewidth=1.8, zorder=3`.
- **Annotations over/under-confident** (prima della legend):
  ```python
  ax.text(0.06, 0.78, "underconfident",
          transform=ax.transAxes, fontsize=9, color="#444444",
          style="italic", ha="left", va="center")
  ax.text(0.55, 0.22, "overconfident",
          transform=ax.transAxes, fontsize=9, color="#444444",
          style="italic", ha="left", va="center")
  ```
  Convenzione: coverage empirico SOPRA l'ideal = intervalli troppo larghi → "underconfident"; SOTTO = troppo stretti → "overconfident".
- **Grid** (riga 519): `ax.grid(alpha=0.22)` → `ax.grid(alpha=0.18, linewidth=0.6)`.

### D. Code cleanup concomitante

- `_slice_distance` (righe 179-186): mai chiamata → rimuovere.
- `predictions_many` e `many_abk` (righe 285-289, 295-298): calcolati ma mai plottati → rimuovere.
- Ritorno `distances` di `_select_slice_points` (sempre zeri): rimuovere; semplificare a un solo `np.ndarray` di indici. Eliminare di conseguenza la logica `marker_alpha = 0.78 if ... else 0.58`.
- Parametri `alpha_tol`, `phi_tol`, `altitude_tol`: trascinati attraverso `_plot_slice_coverage` → `_select_slice_points` ma mai usati. Rimuovere dalle firme.
- CLI flags `--alpha-tol`, `--phi-tol`, `--altitude-tol` (righe 532-534) e i loro pass-through (righe 565-567): rimuovere.
- `reference` argomento di `_select_slice_points`: mai usato → rimuovere.
- `neutral_color = "#7f7f7f"`: dopo viridis non serve più → rimuovere.
- `abk_handles` (riga 314): lista creata ma mai usata → rimuovere.
- Commenti inline non-WHY che diventerebbero falsi (es. riga 276 "Use a paper-friendly qualitative palette"): rimuovere.

## Critical files

- **Da modificare**: [scripts/generate_hyddb_v3_report_figures.py](scripts/generate_hyddb_v3_report_figures.py) — unico file modificato.
- **Da riutilizzare** (no modifiche):
  - [src/visualization/utils.py](src/visualization/utils.py) — `normalize_features` da importare.
  - [src/inference/predictor.py](src/inference/predictor.py) — interfaccia invariata.
  - [src/data/dataset.py](src/data/dataset.py) — riferimento per la normalizzazione.
  - [report/sections/06_results_for_HYDDB.tex](report/sections/06_results_for_HYDDB.tex) — i path `gfx/hyddb_v3/` restano validi.

## Out of scope

- `_plot_uncertainty_metrics` (Plot intermedio `hyddb_v3_uncertainty_gp_mlpgp.{pdf,png}`): non richiesto nel report. Lasciato invariato (regression guard: dopo il fix deve risultare identico).
- `Predictor` in `src/inference/predictor.py`: interfaccia invariata.
- Cambi al LaTeX: nomi file e path output mantengono `report/gfx/hyddb_v3/`.

## Verification

Dopo l'edit, dalla root del repo:

```powershell
python scripts/generate_hyddb_v3_report_figures.py
```

Ispezioni:

1. `report/gfx/hyddb_v3/hyddb_v3_slice_coverage_gp_mlpgp.pdf`:
   - GP e MLPGP nel range fisico corretto (~0.3-1.6 C_A), non più costanti ≈30.
   - 5 curve in gradiente viridis (scuro → chiaro) per delta_abk = 0, 25, 50, 75, 100.
   - Colorbar discreta a destra con tick e label `$\delta_{abk}\,[\%]$`.
   - Banda 2σ visibile, colorata come la curva.
   - Sample points con bordo scuro, ben distinguibili sopra le bande.
   - Freccia in un punto Mach di massima separazione delle curve.
2. `report/gfx/hyddb_v3/hyddb_v3_uncertainty_reliability_gp_mlpgp.pdf`:
   - Layout quadrato.
   - Linea ideal Gaussian solida, grigio scuro, ben visibile.
   - Etichette italiche "underconfident" in alto-sx e "overconfident" in basso-dx.
   - Curve GP/MLPGP invariati nei valori.
3. `report/gfx/hyddb_v3/hyddb_v3_uncertainty_gp_mlpgp.pdf`: **deve restare identico** all'attuale (regression guard).
4. Sanity numerica: prima del fix MLPGP@Mach=1.0, ABK=0 produce ~30; dopo deve essere ~0.5-1.5 C_A.

## Risks e mitigazioni

- **dtype mismatch**: `Predictor` usa `torch.get_default_dtype()`. Il path è già usato con successo per X_test → safe.
- **`BoundaryNorm` non-monotonico**: se l'utente passa `--abk-panels` non ordinato, la colorbar fallisce. Mitigazione: `abk_values = tuple(sorted(abk_values))` subito dopo il parsing.
- **`normalize_features` su array 2D**: verificare alla prima esecuzione; se no, fallback inline `(X - mean) / std` con guard `std=np.where(np.abs(std)<1e-12, 1.0, std)`.

## Prossimi passi

1. Discutere il piano insieme all'utente (eventuali aggiustamenti).
2. Implementare le modifiche (probabilmente un singolo passaggio di Edit/Write su `scripts/generate_hyddb_v3_report_figures.py`).
3. Rigenerare le figure con `python scripts/generate_hyddb_v3_report_figures.py`.
4. Verifica visuale dei 3 PDF (Plot 1, Plot 2, regression guard Plot intermedio).
5. Aggiornare `SESSION_HANDOFF.md` rimuovendo l'open item.
