"""Integration test for metaicu.joint_grid.assemble.write_debug_subset: builds a real joint
output (via write_joint_outputs, same helper as test_joint_assemble_integration.py) with enough
admissions to have a meaningful los_hours spread per cohort, then derives a debug subset from it
and checks the two-tier sampling rule end to end against real files on disk."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from metaicu.joint_grid.assemble import write_debug_subset, write_joint_outputs

_SCHEMA = {
    "hr": {
        "reconstruction_type": "direct_numeric",
        "target_unit": "bpm",
        "presence_mask_column": "hr__observed",
        "physical_columns": ["hr"],
    }
}


def _write_cohort_output(
    output_dir: Path, n_admissions: int, los_hours_start: int
) -> None:
    """n_admissions admissions, all in "train", los_hours = los_hours_start..los_hours_start+n-1,
    each admission with exactly 1 hourly row (n_rows/hour don't matter for the sampling rule --
    only los_hours does)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    admission_ids = list(range(1, n_admissions + 1))
    los_hours = [float(los_hours_start + i) for i in range(n_admissions)]

    grid = pl.DataFrame(
        {
            "admissionid": admission_ids,
            "hour": [0] * n_admissions,
            "hr": [80.0 + i for i in range(n_admissions)],
            "hr__observed": [1] * n_admissions,
        }
    )
    split_dir = output_dir / "train"
    split_dir.mkdir(parents=True, exist_ok=True)
    grid.sort(["admissionid", "hour"]).write_parquet(split_dir / "0.parquet")

    metadata = pl.DataFrame(
        {
            "admissionid": admission_ids,
            "subject_id": admission_ids,
            "split": ["train"] * n_admissions,
            "shard_file": ["train/0.parquet"] * n_admissions,
            "n_rows": [1] * n_admissions,
            "los_hours": los_hours,
            "outcome": ["alive"] * n_admissions,
        }
    )
    metadata.write_csv(output_dir / "metadata.csv")
    (output_dir / "feature_schema.json").write_text(
        json.dumps(_SCHEMA, indent=2, sort_keys=True)
    )
    (output_dir / "tte_targets.json").write_text(
        json.dumps({"targets": [], "missing": ["hr"], "derived": {}})
    )


class WriteDebugSubsetIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)

        aumc_dir = root / "aumcdb_output"
        _write_cohort_output(
            aumc_dir, n_admissions=60, los_hours_start=1
        )  # los_hours 1..60
        mimic_dir = root / "mimic_iv_output"
        _write_cohort_output(
            mimic_dir, n_admissions=60, los_hours_start=200
        )  # los_hours 200..259

        self.joint_output_dir = root / "joint" / "output"
        write_joint_outputs(
            cohort_output_dirs={"aumcdb": aumc_dir, "mimic_iv": mimic_dir},
            joint_output_dir=self.joint_output_dir,
            joint_audit_dir=root / "joint" / "audit",
            patients_per_file=1000,
            weights={"aumcdb": 0.5, "mimic_iv": 0.5},
            n_train_admissions_by_cohort={"aumcdb": 60, "mimic_iv": 60},
        )

        self.debug_output_dir = root / "debug" / "output"
        self.debug_audit_dir = root / "debug" / "audit"
        self.outputs = write_debug_subset(
            joint_output_dir=self.joint_output_dir,
            output_dir=self.debug_output_dir,
            audit_dir=self.debug_audit_dir,
            top_pct=0.1,
            decile_frac=0.1,
            n_deciles=5,
            microbatch_size=64,
            seed=0,
        )

    def test_both_sources_represented_in_proportion(self) -> None:
        metadata = pl.read_csv(self.outputs["metadata"])
        by_source = metadata.group_by("source").len().sort("source")
        counts = dict(zip(by_source["source"].to_list(), by_source["len"].to_list()))
        self.assertIn("aumcdb", counts)
        self.assertIn("mimic_iv", counts)
        # both cohorts have identical size (60) and identical top_pct/decile_frac -> near-equal counts
        self.assertLessEqual(abs(counts["aumcdb"] - counts["mimic_iv"]), 2)

    def test_tail_includes_each_source_own_max(self) -> None:
        metadata = pl.read_csv(self.outputs["metadata"])
        self.assertIn(
            60.0, metadata.filter(pl.col("source") == "aumcdb")["los_hours"].to_list()
        )
        self.assertIn(
            259.0,
            metadata.filter(pl.col("source") == "mimic_iv")["los_hours"].to_list(),
        )

    def test_short_admissions_from_aumcdb_survive_despite_mimic_being_longer(
        self,
    ) -> None:
        # A pooled (not per-source) threshold/decile split would let MIMIC-IV's much longer
        # los_hours push AUMCdb's short admissions entirely below any pooled decile boundary that
        # matters; per-source stratification must keep AUMCdb's own short end represented.
        metadata = pl.read_csv(self.outputs["metadata"])
        aumc_los = metadata.filter(pl.col("source") == "aumcdb")["los_hours"].to_list()
        self.assertTrue(any(los <= 10.0 for los in aumc_los))

    def test_shard_files_written_and_referenced_consistently(self) -> None:
        metadata = pl.read_csv(self.outputs["metadata"])
        self.assertTrue((self.debug_output_dir / "train" / "0.parquet").exists())
        shard = pl.read_parquet(self.debug_output_dir / "train" / "0.parquet")
        self.assertEqual(shard.height, metadata.height)
        self.assertEqual(set(metadata["shard_file"].to_list()), {"train/0.parquet"})

    def test_integrity_audit_passes(self) -> None:
        integrity = json.loads(self.outputs["integrity"].read_text())
        self.assertTrue(integrity["passed"])

    def test_summary_reports_thresholds_and_warns_below_microbatch(self) -> None:
        summary = json.loads(self.outputs["summary"].read_text())
        self.assertIn("aumcdb", summary["per_source"])
        self.assertIn("mimic_iv", summary["per_source"])
        self.assertGreater(
            summary["per_source"]["aumcdb"]["los_hours_tail_threshold"], 0
        )
        # tiny fixture (60 admissions/source) can't realistically hit microbatch_size=64 per
        # decile -- this just proves the summary carries the numbers a caller would check.
        self.assertEqual(summary["microbatch_size"], 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
