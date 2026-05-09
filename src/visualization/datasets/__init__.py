"""Dataset-specific plot suites.

Each module exposes a single ``run(model, model_name, data, config)`` function
that emits the suite's figures under ``config.output_dir / <suite>/``.
"""

DATASET_SUITES = {
    "HYDDB_v2": "hyddb",
    "HYDDB_v3": "hyddb",
}


def get_suite(name: str):
    """Resolve a suite by name. Raises ``KeyError`` if unknown."""
    if name == "hyddb":
        from . import hyddb
        return hyddb
    raise KeyError(f"Unknown dataset suite: '{name}'")
