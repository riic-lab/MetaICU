"""Chunked readers and partitioned parquet writers for large Amsterdam tables.

Handles numericitems, listitems, and drugitems — tables too large to hold in
memory at once. Source-preserving CSV sharding is provided by
``metaicu.aumcdb.common.raw_shards`` and shared with the grid pipeline.
"""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from metaicu.aumcdb.common.raw_schema import LARGE_TABLE_RAW_SCHEMAS
from metaicu.aumcdb.common.raw_shards import (
    polars_dtypes as _polars_dtypes,
    raw_shards_exist,
    read_latin1_csv_batches as _read_latin1_csv_batches,
    read_raw_shard_batches as _read_raw_shard_batches,
)
from metaicu.aumcdb.tokenized.pre_meds.common import (
    admission_anchor_columns,
    interval_time_anomalies,
    measurement_time_anomalies,
    temporal_phase_counts,
)
from metaicu.aumcdb.tokenized.pre_meds.interval_tables import transform_drugitems
from metaicu.aumcdb.tokenized.pre_meds.measured import (
    transform_listitems,
    transform_numericitems,
)


@dataclass
class TableAccumulator:
    """Incremental audit state for one large-table extraction."""

    table: str
    input_mode: str = "raw_csv_chunks"
    rows_read: int = 0
    rows_excluded_sentinel: int = 0
    rows_emitted: int = 0
    missing_join_rows: int = 0
    partition_count: int = 0
    split_partition_counts: Counter = field(default_factory=Counter)
    split_rows_emitted: Counter = field(default_factory=Counter)
    raw_dtypes: dict[str, str] = field(default_factory=dict)
    output_dtypes: dict[str, str] = field(default_factory=dict)
    anomaly_counts: Counter = field(default_factory=Counter)
    phase_counts: Counter = field(default_factory=Counter)

    def as_summary(self, output_dir: Path, max_rows: int | None) -> dict[str, Any]:
        return {
            "table": self.table,
            "output_dataset": str(output_dir / self.table),
            "input_mode": self.input_mode,
            "max_rows": max_rows,
            "partition_count": self.partition_count,
            "split_partition_counts": dict(self.split_partition_counts),
            "split_rows_emitted": dict(self.split_rows_emitted),
            "raw_dtypes": self.raw_dtypes,
            "output_dtypes": self.output_dtypes,
            "row_counts": {
                "rows_read": self.rows_read,
                "rows_excluded_measuredat_minus_1899": self.rows_excluded_sentinel,
                "rows_after_exclusions": self.rows_read - self.rows_excluded_sentinel,
                "rows_emitted": self.rows_emitted,
                "missing_admission_join_rows": self.missing_join_rows,
            },
            "time_anomalies": dict(self.anomaly_counts),
            "temporal_phase_counts": dict(self.phase_counts),
        }


def _prepare_output_dir(output_dir: Path, table: str, overwrite: bool) -> Path:
    table_dir = output_dir / table
    if table_dir.exists() and any(table_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{table_dir} already contains files. "
                "Set run.overwrite=true to replace."
            )
        shutil.rmtree(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    return table_dir


def _accumulate_anomalies(
    table: str,
    transformed: pl.DataFrame,
    acc: TableAccumulator,
) -> None:
    if table in {"numericitems", "listitems"}:
        acc.anomaly_counts.update(measurement_time_anomalies(transformed))
    else:
        acc.anomaly_counts.update(interval_time_anomalies(transformed))
    acc.phase_counts.update(temporal_phase_counts(transformed))


def _transform_batch(
    table: str, raw: pl.DataFrame, anchors: pl.DataFrame
) -> tuple[pl.DataFrame, int, int]:
    if table == "numericitems":
        return transform_numericitems(raw, anchors)
    if table == "listitems":
        return transform_listitems(raw, anchors)
    transformed, missing = transform_drugitems(raw, anchors)
    return transformed, 0, missing


def _write_split_parts(
    transformed: pl.DataFrame,
    split_dirs: dict[str, Path],
    acc: TableAccumulator,
) -> None:
    if not split_dirs or "split" not in transformed.columns:
        return
    for split, split_dir in split_dirs.items():
        part = transformed.filter(pl.col("split") == split)
        if part.is_empty():
            continue
        path = split_dir / f"part-{acc.split_partition_counts[split]:05d}.parquet"
        part.write_parquet(path)
        acc.split_partition_counts[split] += 1
        acc.split_rows_emitted[split] += part.height


def _raw_batches(
    table: str,
    raw_dir: Path,
    raw_shards_dir: Path | None,
    partition_rows: int,
    max_rows: int | None,
    admission_ids: set[int] | None,
):
    use_shards = raw_shards_dir is not None and raw_shards_exist(raw_shards_dir, table)
    if use_shards:
        batches = _read_raw_shard_batches(
            table, raw_shards_dir, max_rows, admission_ids
        )
    else:
        batches = _read_latin1_csv_batches(
            table, raw_dir / f"{table}.csv", partition_rows, max_rows
        )
    return use_shards, batches


def transform_table(
    table: str,
    raw_dir: Path,
    output_dir: Path,
    anchors: pl.DataFrame,
    partition_rows: int,
    max_rows: int | None,
    overwrite: bool,
    admission_ids: set[int] | None = None,
    split_values: list[str] | None = None,
    raw_shards_dir: Path | None = None,
) -> TableAccumulator:
    """Read, transform, and write one large table as a partitioned parquet dataset.

    When admission_ids is supplied (bounded mode), each batch is pre-filtered
    to the selected admissions before joining and transforming.
    """
    if table not in LARGE_TABLE_RAW_SCHEMAS:
        raise ValueError(f"Unsupported large table: {table!r}")

    # Anchors for bounded mode contain only the selected admissions; for full
    # mode they contain all admissions. Split-aware runs carry the split label
    # through the admission join when present.
    bounded_anchors = anchors.select(admission_anchor_columns(anchors))

    table_dir = _prepare_output_dir(output_dir, table, overwrite)
    split_dirs: dict[str, Path] = {}
    if split_values:
        for split in split_values:
            split_dirs[split] = _prepare_output_dir(
                output_dir / split, table, overwrite
            )

    use_raw_shards, raw_batches = _raw_batches(
        table, raw_dir, raw_shards_dir, partition_rows, max_rows, admission_ids
    )
    input_mode = "raw_parquet_shards" if use_raw_shards else "raw_csv_chunks"
    acc = TableAccumulator(table=table, input_mode=input_mode)

    for raw in raw_batches:
        if raw.is_empty():
            continue

        if admission_ids is not None and not use_raw_shards:
            raw = raw.filter(pl.col("admissionid").is_in(list(admission_ids)))
        if raw.is_empty():
            continue

        acc.rows_read += raw.height
        if not acc.raw_dtypes:
            acc.raw_dtypes = _polars_dtypes(raw)

        transformed, n_excl, n_miss = _transform_batch(table, raw, bounded_anchors)

        acc.rows_excluded_sentinel += n_excl
        acc.rows_emitted += transformed.height
        acc.missing_join_rows += n_miss

        if not acc.output_dtypes:
            acc.output_dtypes = _polars_dtypes(transformed)

        _accumulate_anomalies(table, transformed, acc)

        part_path = table_dir / f"part-{acc.partition_count:05d}.parquet"
        transformed.write_parquet(part_path)
        acc.partition_count += 1

        _write_split_parts(transformed, split_dirs, acc)

    return acc
