"""Write split shards and their admission index for a joint grid."""

import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)


def write_split_shards(
    grid: pl.DataFrame,
    metadata: pl.DataFrame,
    output_dir: Path,
    admissions_per_file: int,
) -> tuple[Path, dict[str, int]]:
    """Write admissions in ID order and record each one's file and row count."""
    files_by_admission = {}
    rows_by_admission = {}
    split_counts = {}
    for split in ("train", "val", "test"):
        ids = sorted(metadata.filter(pl.col("split") == split)["admissionid"].to_list())
        split_counts[split] = len(ids)
        if not ids:
            continue
        (output_dir / split).mkdir(parents=True, exist_ok=True)
        for index, start in enumerate(range(0, len(ids), admissions_per_file)):
            batch = ids[start : start + admissions_per_file]
            shard = grid.filter(pl.col("admissionid").is_in(batch)).sort(
                "admissionid", "hour"
            )
            filename = f"{split}/{index}.parquet"
            shard.write_parquet(output_dir / filename)
            for admission_id, count in shard.group_by("admissionid").len().iter_rows():
                files_by_admission[admission_id] = filename
                rows_by_admission[admission_id] = count
            log.info(
                "Wrote %s: %d admissions, %d rows", filename, len(batch), shard.height
            )

    metadata_path = output_dir / "metadata.csv"
    metadata.with_columns(
        pl.col("admissionid")
        .replace_strict(files_by_admission, default=None)
        .alias("shard_file"),
        pl.col("admissionid")
        .replace_strict(rows_by_admission, default=0)
        .alias("n_rows"),
    ).write_csv(metadata_path)
    return metadata_path, split_counts
