"""Post-write integrity checks shared by AUMCdb and MIMIC-IV hourly grids."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def _audit_inventory(
    output_dir: Path, metadata: pl.DataFrame, shard_paths: list[Path]
) -> list[str]:
    failures = []
    if not shard_paths:
        failures.append("no parquet shards written")
    actual = {str(path.relative_to(output_dir)) for path in shard_paths}
    referenced = set(metadata["shard_file"].drop_nulls().to_list())
    if actual != referenced:
        failures.append(
            f"metadata/shard-file mismatch: unreferenced={sorted(actual - referenced)}, "
            f"missing={sorted(referenced - actual)}"
        )
    return failures


def _audit_split_isolation(metadata: pl.DataFrame, subject_column: str) -> list[str]:
    subjects = {
        split: set(metadata.filter(pl.col("split") == split)[subject_column].to_list())
        for split in ("train", "val", "test")
    }
    failures = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = subjects[left] & subjects[right]
        if overlap:
            failures.append(f"{left}/{right} subject overlap: {len(overlap)}")
    return failures


def _one_hot_groups(encoding: pl.DataFrame | None) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    if encoding is not None:
        for row in encoding.iter_rows(named=True):
            groups.setdefault(row["feature"], []).append(row["column_name"])
    return groups


def _audit_frame_values(
    frame: pl.DataFrame, relative_path: Path, groups: dict[str, list[str]]
) -> list[str]:
    failures = []
    for column in (name for name in frame.columns if name.endswith("__observed")):
        invalid = len(
            frame[column].drop_nulls().filter(~frame[column].drop_nulls().is_in([0, 1]))
        )
        if invalid:
            failures.append(f"{relative_path}:{column} has {invalid} non-binary values")
    for feature, columns in groups.items():
        present = [column for column in columns if column in frame.columns]
        if present and frame.select(pl.sum_horizontal(present).ne(1).sum()).item():
            failures.append(f"{relative_path}:{feature} one-hot rows are not exclusive")
    for column, dtype in frame.schema.items():
        if dtype.is_float() and (~frame[column].drop_nulls().is_finite()).sum():
            failures.append(f"{relative_path}:{column} contains non-finite values")
    return failures


def _audit_frame(
    frame: pl.DataFrame,
    relative_path: Path,
    expected_columns: list[str],
    groups: dict[str, list[str]],
) -> tuple[list[str], dict[int | str, int]]:
    if frame.columns != expected_columns:
        return [f"{relative_path} column order differs from canonical schema"], {}
    failures = []
    keys = frame.select(["admissionid", "hour"])
    if keys.n_unique() != frame.height:
        failures.append(f"{relative_path} has duplicate (admissionid,hour) keys")
    per_admission = frame.group_by("admissionid").agg(
        pl.len().alias("rows"),
        pl.col("hour").n_unique().alias("unique_hours"),
        pl.col("hour").min().alias("min_hour"),
        pl.col("hour").max().alias("max_hour"),
    )
    bad_hours = per_admission.filter(
        (pl.col("rows") != pl.col("unique_hours"))
        | (pl.col("min_hour") != 0)
        | (pl.col("max_hour") != pl.col("rows") - 1)
    )
    if bad_hours.height:
        failures.append(
            f"{relative_path} has {bad_hours.height} non-dense admission timelines"
        )
    failures.extend(_audit_frame_values(frame, relative_path, groups))
    counts = dict(
        zip(per_admission["admissionid"].to_list(), per_admission["rows"].to_list())
    )
    return failures, counts


def _audit_shards(
    output_dir: Path, expected_columns: list[str], groups: dict[str, list[str]]
) -> tuple[list[str], dict[int | str, int], int]:
    failures = []
    counts: dict[int | str, int] = {}
    total_rows = 0
    for split in ("train", "val", "test"):
        paths = sorted(
            (output_dir / split).glob("*.parquet"), key=lambda path: int(path.stem)
        )
        if [path.name for path in paths] != [
            f"{index}.parquet" for index in range(len(paths))
        ]:
            failures.append(f"{split} shard numbering is not contiguous from zero")
        for path in paths:
            frame = pl.read_parquet(path)
            total_rows += frame.height
            frame_failures, frame_counts = _audit_frame(
                frame, path.relative_to(output_dir), expected_columns, groups
            )
            failures.extend(frame_failures)
            counts.update(frame_counts)
    return failures, counts, total_rows


def _audit_schema(schema: dict, tte: dict, expected_columns: list[str]) -> list[str]:
    failures = []
    available = set(expected_columns)
    for tag, info in schema.items():
        physical = info.get("physical_columns")
        if physical is None:
            failures.append(f"feature schema lacks physical_columns for {tag}")
        elif not set(physical).issubset(available):
            failures.append(
                f"feature schema points to absent columns for {tag}: {physical}"
            )
    missing_tte = set(tte.get("missing", []))
    for tag in tte.get("targets", []):
        if tag not in missing_tte and not schema.get(tag, {}).get("physical_columns"):
            failures.append(
                f"TTE target {tag} is declared available but has no physical column"
            )
    return failures


def audit_grid_dataset(
    output_dir: Path,
    audit_dir: Path,
    expected_columns: list[str],
    subject_column: str,
) -> Path:
    """Validate written shards and sidecars, write a JSON audit, and fail on violations."""
    metadata = pl.read_csv(output_dir / "metadata.csv")
    schema = json.loads((output_dir / "feature_schema.json").read_text())
    tte = json.loads((output_dir / "tte_targets.json").read_text())
    encoding_path = output_dir / "categorical_encoding.csv"
    encoding = pl.read_csv(encoding_path) if encoding_path.exists() else None
    shard_paths = sorted(output_dir.glob("*/*.parquet"))
    failures = _audit_inventory(output_dir, metadata, shard_paths)
    failures.extend(_audit_split_isolation(metadata, subject_column))
    shard_failures, observed_counts, total_rows = _audit_shards(
        output_dir, expected_columns, _one_hot_groups(encoding)
    )
    failures.extend(shard_failures)

    expected_counts = dict(
        zip(metadata["admissionid"].to_list(), metadata["n_rows"].to_list())
    )
    if observed_counts != expected_counts:
        failures.append("metadata admission IDs or n_rows differ from parquet shards")

    failures.extend(_audit_schema(schema, tte, expected_columns))

    summary = {
        "passed": not failures,
        "failures": failures,
        "shards": len(shard_paths),
        "rows": total_rows,
        "admissions": len(observed_counts),
        "subjects": metadata[subject_column].n_unique(),
        "columns": len(expected_columns),
    }
    audit_dir.mkdir(parents=True, exist_ok=True)
    output_path = audit_dir / "grid_integrity_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise ValueError(f"Grid integrity audit failed; see {output_path}: {failures}")
    return output_path
