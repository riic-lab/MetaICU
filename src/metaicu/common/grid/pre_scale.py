"""Data exchanged between source extraction, pooled fitting and grid output."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class PreScaleGrid:
    """Store one cohort before scaling, imputation and categorical grid encoding.

    The grid carries observation masks. Admissions carry raw static features and split
    assignments. Keep native matches separate from derived targets so joint builds can
    pad the native schema across cohorts before fitting scalers on training admissions.
    """

    grid: pl.DataFrame
    matches: dict
    matches_with_derived: dict
    derived_target_matches: dict
    admissions: pl.DataFrame
    train_admission_ids: list[int]
    demo_source: pl.DataFrame
    static_categorical_encoding: list[dict]
    next_categorical_pos: int
    presence_mask_cols: list[str]
    manifest_report: dict
    raw_shard_summary: dict
    admissions_before_inclusion: int
