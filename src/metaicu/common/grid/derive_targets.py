"""Time-to-event targets computed on the hourly grid before scaling or imputation."""

import logging

import polars as pl

log = logging.getLogger(__name__)

DERIVED_TARGET_MATCHES = {
    "pf_ratio": {"reconstruction_type": "direct_numeric", "target_unit": "mmHg"},
    "urine_rate_per_weight": {
        "reconstruction_type": "direct_numeric",
        "target_unit": "mL/h/kg",
    },
}

DERIVED_TARGET_SOURCES = {
    "pf_ratio": ["po2", "fio2"],
    "urine_rate_per_weight": ["urine_rate", "weight"],
}

# Order is part of the tte_targets.json contract.
K35_TTE_TARGETS = [
    "lact",
    "map",
    "sbp",
    "hr",
    "tnt",
    "po2",
    "pco2",
    "fio2",
    "spo2",
    "resp",
    "pf_ratio",
    "crea",
    "bun",
    "urine_rate",
    "urine_rate_per_weight",
    "bili",
    "bili_dir",
    "ast",
    "alt",
    "plt",
    "wbc",
    "rbc",
    "hct",
    "inr_pt",
    "temp",
    "crp",
    "ph",
    "na",
    "k",
    "ca",
    "mg",
    "cl",
    "glu",
    "ck",
    "ckmb",
]


def derive_targets(grid, admissions, target_matches):
    """Add available ratios and return their dataset-specific manifest entries.

    FiO2 is recorded as a percentage and weight in kg. Both ratios propagate
    missing measurements and require a positive denominator.
    """
    available = {}
    if "po2" in grid.columns and "fio2" in grid.columns:
        grid = grid.with_columns(
            pl.when(pl.col("fio2") > 0)
            .then(pl.col("po2") / (pl.col("fio2") / 100.0))
            .alias("pf_ratio")
        )
        available["pf_ratio"] = target_matches["pf_ratio"]
    else:
        log.warning("Skipping pf_ratio because po2 or fio2 is unavailable")

    if "urine_rate" in grid.columns and "weight" in admissions.columns:
        grid = grid.join(
            admissions.select("admissionid", "weight"), on="admissionid", how="left"
        )
        grid = grid.with_columns(
            pl.when(pl.col("weight") > 0)
            .then(pl.col("urine_rate") / pl.col("weight"))
            .alias("urine_rate_per_weight")
        ).drop("weight")
        available["urine_rate_per_weight"] = target_matches["urine_rate_per_weight"]
    else:
        log.warning(
            "Skipping urine_rate_per_weight because urine_rate or weight is unavailable"
        )

    return grid, available
