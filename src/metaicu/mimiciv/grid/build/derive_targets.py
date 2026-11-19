"""Derived hourly targets for MIMIC-IV."""

from metaicu.common.grid.derive_targets import (
    DERIVED_TARGET_MATCHES as _COMMON_MATCHES,
    DERIVED_TARGET_SOURCES as DERIVED_TARGET_SOURCES,
    K35_TTE_TARGETS as MIMIC_K35_TTE_TARGETS,
    derive_targets,
)

# MIMIC manifest entries carry a list of source matches, empty for a derived feature.
DERIVED_TARGET_MATCHES = {
    tag: {**entry, "keep_matches": []} for tag, entry in _COMMON_MATCHES.items()
}

__all__ = [
    "DERIVED_TARGET_MATCHES",
    "DERIVED_TARGET_SOURCES",
    "MIMIC_K35_TTE_TARGETS",
    "add_derived_tte_targets",
]


def add_derived_tte_targets(grid, admissions):
    return derive_targets(grid, admissions, DERIVED_TARGET_MATCHES)
