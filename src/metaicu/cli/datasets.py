"""Map Hydra dataset names to their grid packages."""

from importlib import import_module
from types import ModuleType

GRID_DATASETS = {
    "aumcdb": "metaicu.aumcdb.grid",
    "mimic_iv": "metaicu.mimiciv.grid",
}


def load_grid_module(dataset_name: str, module: str) -> ModuleType:
    """Import the selected operation without loading unrelated dataset workflows."""
    if dataset_name not in GRID_DATASETS:
        raise ValueError(
            f"Unknown dataset_name={dataset_name!r}, expected one of {sorted(GRID_DATASETS)}"
        )
    return import_module(f"{GRID_DATASETS[dataset_name]}.{module}")
