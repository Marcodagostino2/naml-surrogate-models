"""Capture baseline metrics for the Stage 0 regression harness.

For each combo in CONFIG_MATRIX, copies config_smoke_quick into a temp dir,
patches main_config.yaml (and optionally model/mlpgp_config.yaml or
convergence/convergence_config.yaml) to match the combo, runs main.py with
the smoke output_dir, locates the produced results_manifest.json, and writes
a baseline JSON under tests/baselines/<combo_id>.json.

Used by tests/test_regression_baseline.py to detect regressions during the
Stage 1-10 refactor.

Usage:
    python scripts/capture_baseline_metrics.py --combo single_run.mlp
    python scripts/capture_baseline_metrics.py --all
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG_DIR = PROJECT_ROOT / "config_smoke_quick"
BASELINE_DIR = PROJECT_ROOT / "tests" / "baselines"
TIMEOUT_SECONDS = 600  # 10 min hard cap per combo


# Smoke baselines must run quickly without exhausting GPU memory. yacht_hydrodynamics
# (308 samples, 6 inputs) keeps full Cholesky GP cheap; float32 halves the working set.
_COMMON_MAIN_PATCH = {"dataset": "yacht_hydrodynamics", "precision": "float32"}


def _main_patch(extra: dict[str, Any]) -> dict[str, Any]:
    return {**_COMMON_MAIN_PATCH, **extra}


# Each combo describes: workflow mode, model, optional patches to YAML files.
CONFIG_MATRIX: dict[str, dict[str, Any]] = {
    "single_run.mlp": {
        "mode": "single_run",
        "model": "mlp",
        "main_patch": _main_patch({"mode": "single_run", "model_to_train": "mlp", "run_mode": "training"}),
    },
    "single_run.gp": {
        "mode": "single_run",
        "model": "gp",
        "main_patch": _main_patch({"mode": "single_run", "model_to_train": "gp", "run_mode": "training"}),
    },
    "single_run.mlpgp_ns": {
        "mode": "single_run",
        "model": "mlpgp_ns",
        "main_patch": _main_patch({"mode": "single_run", "model_to_train": "mlpgp_ns", "run_mode": "training"}),
    },
    "single_run.mlpgp_dkl": {
        "mode": "single_run",
        "model": "mlpgp_dkl",
        "main_patch": _main_patch({"mode": "single_run", "model_to_train": "mlpgp_dkl", "run_mode": "training"}),
    },
    # Optimization + convergence baselines deliberately omitted from --all:
    # optimization legacy mode is going away in Stage 7 (no long-term baseline value),
    # and convergence_analysis.py writes its own all_runs.csv layout — when Stages 4
    # and 9 land, validate convergence behaviour against a manual smoke run rather
    # than this manifest-based harness.
}


def _patch_yaml(path: Path, patch: dict[str, Any]) -> None:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    data.update(patch)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _prepare_config_dir(combo_id: str, spec: dict[str, Any]) -> Path:
    target = PROJECT_ROOT / f"config_baseline_{combo_id.replace('.', '_')}"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(SMOKE_CONFIG_DIR, target)

    _patch_yaml(target / "main_config.yaml", spec["main_patch"])
    if "convergence_patch" in spec:
        _patch_yaml(target / "convergence" / "convergence_config.yaml", spec["convergence_patch"])
    return target


def _run_main(config_dir: Path, output_dir: Path, mode: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "main.py", "--config-dir", str(config_dir), "--output-dir", str(output_dir)]
    if mode == "convergence":
        cmd.extend(["--config-path", str(config_dir / "convergence" / "convergence_config.yaml")])
    return subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=TIMEOUT_SECONDS)


def _find_run_manifest(output_dir: Path) -> Path | None:
    """Return the most recently created results_manifest.json in output_dir."""
    manifests = sorted(output_dir.rglob("results_manifest.json"), key=lambda p: p.stat().st_mtime)
    return manifests[-1] if manifests else None


def _extract_metrics(manifest_path: Path) -> dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    return {
        "dataset": manifest.get("dataset"),
        "mode": manifest.get("mode"),
        "model": manifest.get("model"),
        "metrics": manifest.get("metrics", {}),
    }


def capture(combo_id: str) -> dict[str, Any]:
    if combo_id not in CONFIG_MATRIX:
        raise ValueError(f"Unknown combo: {combo_id}. Available: {sorted(CONFIG_MATRIX)}")
    spec = CONFIG_MATRIX[combo_id]

    config_dir = _prepare_config_dir(combo_id, spec)
    output_dir = PROJECT_ROOT / f"outputs_baseline_{combo_id.replace('.', '_')}"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    try:
        proc = _run_main(config_dir, output_dir, spec["mode"])
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout)[-2000:]
            raise RuntimeError(f"main.py failed for {combo_id} (exit {proc.returncode}):\n{tail}")
        manifest = _find_run_manifest(output_dir)
        if manifest is None:
            raise RuntimeError(f"No results_manifest.json produced for {combo_id} under {output_dir}")
        baseline = _extract_metrics(manifest)
        baseline["combo_id"] = combo_id
        baseline["spec"] = spec

        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = BASELINE_DIR / f"{combo_id.replace('.', '_')}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(baseline, fh, indent=2)
        return {"combo": combo_id, "baseline_path": str(out_path.relative_to(PROJECT_ROOT)), "metrics": baseline["metrics"]}
    finally:
        shutil.rmtree(config_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combo", type=str, default=None, help="Single combo id to capture")
    parser.add_argument("--all", action="store_true", help="Capture all combos in CONFIG_MATRIX")
    parser.add_argument("--list", action="store_true", help="List available combo ids and exit")
    args = parser.parse_args()

    if args.list:
        for combo in sorted(CONFIG_MATRIX):
            print(combo)
        return

    if args.all:
        combos = list(CONFIG_MATRIX)
    elif args.combo:
        combos = [args.combo]
    else:
        parser.error("Pass --combo NAME, --all, or --list")

    results = []
    for combo_id in combos:
        print(f"[capture] {combo_id} ...", flush=True)
        try:
            result = capture(combo_id)
            results.append(result)
            print(f"[capture] {combo_id} -> {result['baseline_path']}")
        except Exception as exc:
            print(f"[capture] {combo_id} FAILED: {exc}", file=sys.stderr)
            results.append({"combo": combo_id, "error": str(exc)})

    print("\n=== Summary ===")
    for r in results:
        if "error" in r:
            print(f"  FAIL  {r['combo']}: {r['error'][:200]}")
        else:
            test_r2 = r["metrics"].get("test", {}).get("r2")
            print(f"  OK    {r['combo']}: test R²={test_r2}")


if __name__ == "__main__":
    main()
