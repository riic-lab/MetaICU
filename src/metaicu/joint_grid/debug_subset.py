"""Sample a joint grid by source and length of stay for smaller development runs."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import polars as pl

from metaicu.common.grid.integrity import audit_grid_dataset
from metaicu.joint_grid.shards import write_split_shards

log = logging.getLogger(__name__)

DEFAULT_DEBUG_TOP_PCT = 0.01
DEFAULT_DEBUG_DECILE_FRAC = 0.1
DEFAULT_DEBUG_N_DECILES = 10
DEFAULT_DEBUG_MICROBATCH_SIZE = 64


def select_debug_admission_ids(
    metadata,
    top_pct=DEFAULT_DEBUG_TOP_PCT,
    decile_frac=DEFAULT_DEBUG_DECILE_FRAC,
    n_deciles=DEFAULT_DEBUG_N_DECILES,
    seed=42,
):
    """Select long stays and sample shorter stays by length within each source.

    Include every admission at or above the source-specific tail quantile.
    Sample `decile_frac` of each remaining length-of-stay bin using `seed`.
    Return admission IDs and the per-source counts used in the audit summary.
    """
    selected: list = []
    summary: dict = {}
    for (source,), group in metadata.sort("source").group_by(
        ["source"], maintain_order=True
    ):
        los = group["los_hours"]
        threshold = float(los.quantile(1 - top_pct, interpolation="higher"))
        tail = group.filter(pl.col("los_hours") >= threshold)
        rest = group.filter(pl.col("los_hours") < threshold)

        decile_counts: dict = {}
        sampled_rest_ids: list = []
        if rest.height:
            rest = rest.with_columns(
                pl.col("los_hours")
                .qcut(n_deciles, allow_duplicates=True)
                .alias("_decile")
            )
            for (decile,), decile_group in rest.sort("_decile").group_by(
                ["_decile"], maintain_order=True
            ):
                # max(1, ...) only when decile_frac itself asks for a nonzero sample -- guards
                # against rounding a small decile*decile_frac down to 0, without overriding an
                # explicit decile_frac=0.0 ("tail only, no stratified remainder").
                n_sample = (
                    max(1, round(decile_group.height * decile_frac))
                    if decile_frac > 0
                    else 0
                )
                sampled = decile_group.sample(
                    n=min(n_sample, decile_group.height), seed=seed, shuffle=True
                )
                sampled_rest_ids.extend(sampled["admissionid"].to_list())
                decile_counts[str(decile)] = {
                    "n_total": decile_group.height,
                    "n_sampled": sampled.height,
                }

        source_ids = tail["admissionid"].to_list() + sampled_rest_ids
        selected.extend(source_ids)
        summary[source] = {
            "los_hours_tail_threshold": threshold,
            "n_admissions": group.height,
            "n_tail": tail.height,
            "n_rest": rest.height,
            "n_sampled_from_rest": len(sampled_rest_ids),
            "n_selected": len(source_ids),
            "decile_counts": decile_counts,
        }
    return selected, summary


def write_debug_subset(
    joint_output_dir,
    output_dir,
    audit_dir,
    top_pct=DEFAULT_DEBUG_TOP_PCT,
    decile_frac=DEFAULT_DEBUG_DECILE_FRAC,
    n_deciles=DEFAULT_DEBUG_N_DECILES,
    microbatch_size=DEFAULT_DEBUG_MICROBATCH_SIZE,
    seed=42,
):
    """Write sampled grid rows, metadata and sidecars from an existing joint build."""
    joint_output_dir = Path(joint_output_dir)
    output_dir = Path(output_dir)
    audit_dir = Path(audit_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    metadata = pl.read_csv(joint_output_dir / "metadata.csv")
    for required in ("admissionid", "los_hours", "source", "split", "shard_file"):
        if required not in metadata.columns:
            raise ValueError(
                f"write_debug_subset requires joint metadata.csv; missing column {required!r}"
            )

    selected_ids, summary = select_debug_admission_ids(
        metadata,
        top_pct=top_pct,
        decile_frac=decile_frac,
        n_deciles=n_deciles,
        seed=seed,
    )
    selected_metadata = metadata.filter(pl.col("admissionid").is_in(selected_ids))

    shard_files = sorted(set(selected_metadata["shard_file"].drop_nulls().to_list()))
    grid = pl.concat(
        [pl.read_parquet(joint_output_dir / shard_file) for shard_file in shard_files],
        how="vertical",
    ).filter(pl.col("admissionid").is_in(selected_ids))

    # One shard per nonempty split in the debug layout.
    metadata_path, split_counts = write_split_shards(
        grid,
        selected_metadata,
        output_dir,
        admissions_per_file=selected_metadata.height,
    )

    for sidecar in (
        "feature_schema.json",
        "tte_targets.json",
        "categorical_encoding.csv",
        "scalers.pkl",
        "scalers.summary.json",
    ):
        src = joint_output_dir / sidecar
        if src.exists():
            shutil.copy2(src, output_dir / sidecar)

    integrity_path = audit_grid_dataset(
        output_dir, audit_dir, grid.columns, subject_column="subject_id"
    )

    smallest_bucket = min(
        (
            decile["n_sampled"]
            for source_summary in summary.values()
            for decile in source_summary["decile_counts"].values()
        ),
        default=0,
    )
    if smallest_bucket < microbatch_size:
        log.warning(
            f"debug subset: smallest per-source LOS decile has only {smallest_bucket} sampled "
            f"admissions, below the requested microbatch_size={microbatch_size} -- a microbatch "
            f"drawn purely from the shortest bucket won't be full"
        )

    summary_path = audit_dir / "debug_sample_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source_joint_output_dir": str(joint_output_dir),
                "top_pct": top_pct,
                "decile_frac": decile_frac,
                "n_deciles": n_deciles,
                "microbatch_size": microbatch_size,
                "seed": seed,
                "split_admission_counts": split_counts,
                "total_admissions": len(selected_ids),
                "total_rows": grid.height,
                "per_source": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )

    return {
        "output_dir": output_dir,
        "metadata": metadata_path,
        "feature_schema": output_dir / "feature_schema.json",
        "tte_targets": output_dir / "tte_targets.json",
        "summary": summary_path,
        "integrity": integrity_path,
    }
