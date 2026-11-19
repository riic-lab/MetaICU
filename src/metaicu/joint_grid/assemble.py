"""Assemble cohort shards and metadata into one joint grid.

All cohorts share a feature schema. Prefixing admission and subject IDs with the
source name keeps identifiers unique when cohorts use the same numeric IDs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import polars as pl

from metaicu.common.grid.integrity import audit_grid_dataset

# These names remain available to users of the original assembly module.
from metaicu.joint_grid.debug_subset import (
    DEFAULT_DEBUG_DECILE_FRAC as DEFAULT_DEBUG_DECILE_FRAC,
    DEFAULT_DEBUG_MICROBATCH_SIZE as DEFAULT_DEBUG_MICROBATCH_SIZE,
    DEFAULT_DEBUG_N_DECILES as DEFAULT_DEBUG_N_DECILES,
    DEFAULT_DEBUG_TOP_PCT as DEFAULT_DEBUG_TOP_PCT,
    select_debug_admission_ids as select_debug_admission_ids,
    write_debug_subset as write_debug_subset,
)
from metaicu.joint_grid.shards import write_split_shards

ID_COLUMNS = ("admissionid", "subject_id")


def namespace_ids(df, cohort, id_columns=ID_COLUMNS):
    """df: any DataFrame carrying one or more of id_columns. Returns a NEW DataFrame with each
    present id column cast to String and prefixed f"{cohort}_{value}"."""
    present = [c for c in id_columns if c in df.columns]
    return df.with_columns(
        [(pl.lit(f"{cohort}_") + pl.col(c).cast(pl.Utf8)).alias(c) for c in present]
    )


def count_cross_cohort_id_collisions(raw_id_sets_by_cohort):
    """raw_id_sets_by_cohort: {cohort: set of raw (pre-namespacing) ID values}. Returns the total
    count of ID values shared by two or more cohorts -- purely diagnostic, reported in the joint
    summary for auditability. namespace_ids makes an actual collision structurally impossible
    regardless of this count (distinct cohort prefixes can never coincide)."""
    cohorts = list(raw_id_sets_by_cohort)
    collisions = set()
    for i, a in enumerate(cohorts):
        for b in cohorts[i + 1 :]:
            collisions |= raw_id_sets_by_cohort[a] & raw_id_sets_by_cohort[b]
    return len(collisions)


def assert_globally_unique(df, column):
    """df: a DataFrame already carrying every cohort's rows (post namespace_ids). Raises
    ValueError if `column` has any duplicate value -- expected to never fire once IDs are
    namespaced, unless a single cohort's own output already had an internal duplicate (a bug
    upstream of this module, not a cross-cohort collision)."""
    n_total, n_unique = df.height, df[column].n_unique()
    if n_unique != n_total:
        raise ValueError(
            f"{column}: {n_total - n_unique} duplicate values found after namespacing"
        )


def merge_tte_targets(tte_infos_by_cohort):
    """Combine target names and sources. Mark a target missing if every cohort lists it."""
    all_targets = set()
    all_missing = None
    derived = {}
    for info in tte_infos_by_cohort.values():
        all_targets.update(info.get("targets", []))
        missing = set(info.get("missing", []))
        all_missing = missing if all_missing is None else (all_missing & missing)
        derived.update(info.get("derived", {}))
    return {
        "targets": sorted(all_targets),
        "missing": sorted(all_missing or set()),
        "derived": derived,
    }


def _read_cohort_outputs(cohort, output_dir):
    output_dir = Path(output_dir)
    grid_paths = sorted(output_dir.glob("*/*.parquet"))
    if not grid_paths:
        raise ValueError(f"{cohort}: no parquet shards found under {output_dir}")
    grid = pl.concat([pl.read_parquet(path) for path in grid_paths], how="vertical")

    metadata = pl.read_csv(output_dir / "metadata.csv")
    if "admissionid" not in metadata.columns:
        raise ValueError(
            f"{cohort}: metadata.csv has no admissionid column -- only admission-grain per-cohort "
            f"outputs (unit_of_analysis='admission') are supported for a joint build"
        )

    schema = json.loads((output_dir / "feature_schema.json").read_text())
    tte = json.loads((output_dir / "tte_targets.json").read_text())
    encoding_path = output_dir / "categorical_encoding.csv"
    encoding = pl.read_csv(encoding_path) if encoding_path.exists() else None
    return grid, metadata, schema, tte, encoding


def write_joint_outputs(
    cohort_output_dirs,
    joint_output_dir,
    joint_audit_dir,
    patients_per_file,
    weights,
    n_train_admissions_by_cohort,
):
    """Write joint shards, admission metadata, schema, scalers and audit reports.

    Read completed admission-level cohort outputs and preserve native IDs alongside
    source-prefixed IDs. Return the paths to the written artifacts.
    """
    joint_output_dir = Path(joint_output_dir)
    joint_audit_dir = Path(joint_audit_dir)
    joint_output_dir.mkdir(parents=True, exist_ok=True)
    joint_audit_dir.mkdir(parents=True, exist_ok=True)

    grids, metadatas, schemas, ttes, encodings = {}, {}, {}, {}, {}
    raw_admissionid_sets, raw_subject_id_sets = {}, {}
    for cohort, output_dir in cohort_output_dirs.items():
        grid, metadata, schema, tte, encoding = _read_cohort_outputs(cohort, output_dir)
        raw_admissionid_sets[cohort] = set(metadata["admissionid"].to_list())
        if "subject_id" in metadata.columns:
            raw_subject_id_sets[cohort] = set(metadata["subject_id"].to_list())

        grids[cohort] = namespace_ids(grid, cohort, id_columns=("admissionid",))
        # Preserve native IDs so consumers can look up an admission in the source tables.
        metadata_with_native_ids = metadata.with_columns(
            [
                pl.col(c).alias(f"native_{c}")
                for c in ("admissionid", "subject_id")
                if c in metadata.columns
            ]
        )
        metadatas[cohort] = namespace_ids(
            metadata_with_native_ids, cohort
        ).with_columns(pl.lit(cohort).alias("source"))
        schemas[cohort] = schema
        ttes[cohort] = tte
        encodings[cohort] = encoding

    id_collisions = {
        "admissionid": count_cross_cohort_id_collisions(raw_admissionid_sets),
        "subject_id": count_cross_cohort_id_collisions(raw_subject_id_sets),
    }

    schema_values = list(schemas.values())
    if any(s != schema_values[0] for s in schema_values[1:]):
        raise ValueError(
            "feature_schema.json differs across cohorts -- expected identical by construction "
            "(data-content invariant schema, see metaicu.common.grid.schema_union)"
        )
    joint_schema = schema_values[0]

    encoding_values = [e for e in encodings.values() if e is not None]
    if len(encoding_values) > 1 and any(
        not e.equals(encoding_values[0]) for e in encoding_values[1:]
    ):
        raise ValueError(
            "categorical_encoding.csv differs across cohorts -- one-hot vocabularies must match "
            "for a joint build (see metaicu.common.grid.schema_union.compute_union_categorical_vocab)"
        )

    # Match grid columns by name because padded features can appear in a different order.
    joint_grid = pl.concat(list(grids.values()), how="diagonal_relaxed")
    joint_metadata = pl.concat(list(metadatas.values()), how="diagonal_relaxed")
    # A subject can have several stays. Only admission IDs must be unique here.
    assert_globally_unique(joint_metadata, "admissionid")

    metadata_path, split_counts = write_split_shards(
        joint_grid, joint_metadata, joint_output_dir, patients_per_file
    )

    schema_path = joint_output_dir / "feature_schema.json"
    schema_path.write_text(json.dumps(joint_schema, indent=2, sort_keys=True))

    tte_targets_path = joint_output_dir / "tte_targets.json"
    tte_targets_path.write_text(json.dumps(merge_tte_targets(ttes), indent=2))

    if encoding_values:
        encoding_values[0].write_csv(joint_output_dir / "categorical_encoding.csv")

    # Per-cohort staging is disposable. Pick a reproducible source for the pooled scalers.
    canonical_cohort = sorted(cohort_output_dirs)[0]
    canonical_scalers_dir = Path(cohort_output_dirs[canonical_cohort])
    scalers_path = canonical_scalers_dir / "scalers.pkl"
    scalers_summary_path = canonical_scalers_dir / "scalers.summary.json"
    if scalers_path.exists():
        shutil.copy2(scalers_path, joint_output_dir / "scalers.pkl")
    if scalers_summary_path.exists():
        shutil.copy2(scalers_summary_path, joint_output_dir / "scalers.summary.json")

    integrity_path = audit_grid_dataset(
        joint_output_dir,
        joint_audit_dir,
        joint_grid.columns,
        subject_column="subject_id",
    )

    summary_path = joint_audit_dir / "joint_grid_build_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "cohorts": sorted(cohort_output_dirs),
                "cohort_admission_counts": {
                    cohort: metadatas[cohort].height for cohort in cohort_output_dirs
                },
                "cohort_weights": weights,
                "n_train_admissions_by_cohort": n_train_admissions_by_cohort,
                "split_admission_counts": split_counts,
                "cross_cohort_id_collisions_pre_namespacing": id_collisions,
                "grid_rows": joint_grid.height,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )

    return {
        "output_dir": joint_output_dir,
        "metadata": metadata_path,
        "feature_schema": schema_path,
        "tte_targets": tte_targets_path,
        "summary": summary_path,
        "integrity": integrity_path,
        "scalers": joint_output_dir / "scalers.pkl" if scalers_path.exists() else None,
    }
