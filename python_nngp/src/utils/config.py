from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in config file: {path}")
    return payload
