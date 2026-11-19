"""Derived hourly targets for AmsterdamUMCdb."""

from metaicu.common.grid.derive_targets import (
    DERIVED_TARGET_MATCHES,
    DERIVED_TARGET_SOURCES as DERIVED_TARGET_SOURCES,
    K35_TTE_TARGETS,
    derive_targets,
)

# Kept for callers that used the historical list name before direct bilirubin was added.
K34_TTE_TARGETS = K35_TTE_TARGETS


def add_derived_tte_targets(grid, admissions):
    return derive_targets(grid, admissions, DERIVED_TARGET_MATCHES)
