from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def save_pickle(path: Path, payload: Any) -> None:
    with path.open("wb") as stream:
        pickle.dump(payload, stream)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)
