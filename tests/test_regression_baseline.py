"""Regression harness comparing fresh runs against committed baselines.

For each baseline JSON under tests/baselines/, re-runs the same combo via
scripts.capture_baseline_metrics.capture(...) and checks that R²/RMSE/MAE
stay within tolerance (R² ±0.005 absolute, RMSE/MAE ±1% relative).

Marked as ``regression`` so it's skipped by default (the runs are slow);
enable with ``pytest -m regression`` or ``pytest tests/test_regression_baseline.py -k <combo>``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "tests" / "baselines"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.capture_baseline_metrics import capture  # noqa: E402


R2_ABS_TOL = 0.005
RMSE_REL_TOL = 0.01
MAE_REL_TOL = 0.01


def _load_baselines() -> list[Path]:
    if not BASELINE_DIR.exists():
        return []
    return sorted(BASELINE_DIR.glob("*.json"))


@pytest.mark.regression
@pytest.mark.parametrize("baseline_path", _load_baselines(), ids=lambda p: p.stem)
def test_metrics_within_tolerance(baseline_path: Path) -> None:
    with open(baseline_path, "r", encoding="utf-8") as fh:
        baseline = json.load(fh)
    combo_id = baseline["combo_id"]

    fresh = capture(combo_id)
    fresh_metrics = fresh["metrics"]
    baseline_metrics = baseline["metrics"]

    for split in ("train", "val", "test"):
        b = baseline_metrics.get(split, {})
        f = fresh_metrics.get(split, {})
        if not b or not f:
            continue
        if "r2" in b and "r2" in f:
            assert abs(f["r2"] - b["r2"]) <= R2_ABS_TOL, f"{combo_id}/{split} R² drift: {b['r2']} -> {f['r2']}"
        for metric, tol in (("rmse", RMSE_REL_TOL), ("mae", MAE_REL_TOL)):
            if metric in b and metric in f:
                denom = max(abs(b[metric]), 1e-10)
                assert abs(f[metric] - b[metric]) / denom <= tol, (
                    f"{combo_id}/{split} {metric} drift: {b[metric]} -> {f[metric]}"
                )
