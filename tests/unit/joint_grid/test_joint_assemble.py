"""Unit tests for metaicu.joint_grid.assemble's pure helpers -- ID namespacing, cross-cohort
collision counting, uniqueness assertion, and TTE-target-list merging for a joint multi-dataset
grid build. write_joint_outputs itself (disk I/O against real per-cohort outputs) is covered by
tests/integration/joint_grid/test_joint_assemble_integration.py instead."""

from __future__ import annotations

import unittest

import polars as pl

from metaicu.joint_grid.assemble import (
    assert_globally_unique,
    count_cross_cohort_id_collisions,
    merge_tte_targets,
    namespace_ids,
    select_debug_admission_ids,
)


class NamespaceIdsTests(unittest.TestCase):
    def test_prefixes_and_casts_present_id_columns(self) -> None:
        df = pl.DataFrame(
            {"admissionid": [1, 2], "subject_id": [10, 20], "hour": [0, 1]}
        )
        namespaced = namespace_ids(df, "aumcdb")
        self.assertEqual(namespaced["admissionid"].to_list(), ["aumcdb_1", "aumcdb_2"])
        self.assertEqual(namespaced["subject_id"].to_list(), ["aumcdb_10", "aumcdb_20"])
        self.assertEqual(namespaced["hour"].to_list(), [0, 1])
        self.assertEqual(df["admissionid"].to_list(), [1, 2])
        self.assertEqual(df.schema["admissionid"], pl.Int64)
        self.assertEqual(namespaced.schema["admissionid"], pl.Utf8)

    def test_only_namespaces_columns_that_are_present(self) -> None:
        df = pl.DataFrame({"admissionid": [1, 2], "hour": [0, 1]})
        namespaced = namespace_ids(df, "mimic_iv")
        self.assertEqual(set(namespaced.columns), {"admissionid", "hour"})
        self.assertEqual(
            namespaced["admissionid"].to_list(), ["mimic_iv_1", "mimic_iv_2"]
        )


class CountCrossCohortIdCollisionsTests(unittest.TestCase):
    def test_no_overlap_is_zero(self) -> None:
        self.assertEqual(
            count_cross_cohort_id_collisions({"a": {1, 2}, "b": {3, 4}}), 0
        )

    def test_counts_shared_values(self) -> None:
        self.assertEqual(
            count_cross_cohort_id_collisions({"a": {1, 2, 3}, "b": {2, 3, 4}}), 2
        )

    def test_single_cohort_is_zero(self) -> None:
        self.assertEqual(count_cross_cohort_id_collisions({"a": {1, 2, 3}}), 0)


class AssertGloballyUniqueTests(unittest.TestCase):
    def test_duplicate_raises(self) -> None:
        df = pl.DataFrame({"admissionid": ["aumcdb_1", "aumcdb_1", "mimic_iv_1"]})
        with self.assertRaises(ValueError):
            assert_globally_unique(df, "admissionid")


class MergeTteTargetsTests(unittest.TestCase):
    def test_union_targets_intersect_missing_merge_derived(self) -> None:
        # Synthetic asymmetry (both cohorts' real tte_targets.json are K35, bili_dir included in
        # both -- see aumcdb/grid/build/derive_targets.py) exercising the general rule: a target
        # real in only one cohort must survive into the joint target list regardless.
        aumc = {
            "targets": ["lact", "map"],
            "missing": ["tri", "absent"],
            "derived": {"pf_ratio": ["po2", "fio2"]},
        }
        mimic = {
            "targets": ["lact", "map", "bili_dir"],
            "missing": ["absent"],
            "derived": {
                "pf_ratio": ["po2", "fio2"],
                "urine_rate_per_weight": ["urine_rate", "weight"],
            },
        }
        merged = merge_tte_targets({"aumcdb": aumc, "mimic_iv": mimic})
        self.assertEqual(merged["targets"], ["bili_dir", "lact", "map"])
        # "tri" is missing for aumcdb but not declared missing for mimic_iv -- only genuinely
        # globally-missing tags (missing in EVERY cohort) belong in the joint "missing" list.
        self.assertEqual(merged["missing"], ["absent"])
        self.assertEqual(
            merged["derived"],
            {
                "pf_ratio": ["po2", "fio2"],
                "urine_rate_per_weight": ["urine_rate", "weight"],
            },
        )


class SelectDebugAdmissionIdsTests(unittest.TestCase):
    def _metadata(self, source, admission_ids, los_hours):
        return pl.DataFrame(
            {
                "admissionid": admission_ids,
                "los_hours": los_hours,
                "source": [source] * len(admission_ids),
            }
        )

    def test_tail_includes_the_max_and_is_above_threshold(self) -> None:
        # 100 admissions, los_hours = 1..100 -- p99 (interpolation="higher") is exactly 100, so
        # only the single max-LOS admission should land in the tail.
        los = list(range(1, 101))
        ids = [f"a{i}" for i in range(100)]
        metadata = self._metadata("aumcdb", ids, los)
        selected, summary = select_debug_admission_ids(
            metadata, top_pct=0.01, decile_frac=0.0, n_deciles=10, seed=0
        )
        self.assertEqual(summary["aumcdb"]["n_tail"], 1)
        self.assertEqual(summary["aumcdb"]["los_hours_tail_threshold"], 100.0)
        self.assertEqual(selected, [ids[los.index(100)]])
        self.assertEqual(summary["aumcdb"]["n_sampled_from_rest"], 0)

    def test_every_decile_of_the_remainder_contributes_at_least_one(self) -> None:
        # 100 admissions below threshold, spread 1..95 -- with decile_frac=0.1 every one of the
        # 10 deciles should contribute >= 1 admission (max(1, round(...)) floor), so short-LOS
        # buckets are never silently starved out.
        los = list(range(1, 96))
        ids = [f"a{i}" for i in range(95)]
        metadata = self._metadata("mimic_iv", ids, los)
        selected, summary = select_debug_admission_ids(
            metadata, top_pct=0.5, decile_frac=0.1, n_deciles=10, seed=1
        )
        decile_counts = summary["mimic_iv"]["decile_counts"]
        self.assertTrue(decile_counts)
        for stats in decile_counts.values():
            self.assertGreaterEqual(stats["n_sampled"], 1)
        self.assertEqual(len(selected), len(set(selected)))  # no duplicate admissions

    def test_each_source_thresholded_and_sampled_independently(self) -> None:
        # AUMCdb admissions are all short (1..50); MIMIC-IV admissions are all long (500..550) --
        # a pooled threshold would put ALL of MIMIC-IV in the tail and none of AUMCdb's own long
        # end represented; per-source thresholds must instead give each cohort its own tail.
        aumc = self._metadata(
            "aumcdb", [f"a{i}" for i in range(50)], list(range(1, 51))
        )
        mimic = self._metadata(
            "mimic_iv", [f"m{i}" for i in range(50)], list(range(500, 550))
        )
        metadata = pl.concat([aumc, mimic])
        _, summary = select_debug_admission_ids(
            metadata, top_pct=0.1, decile_frac=0.1, n_deciles=5, seed=0
        )
        self.assertGreater(summary["aumcdb"]["n_tail"], 0)
        self.assertGreater(summary["mimic_iv"]["n_tail"], 0)
        self.assertLess(summary["aumcdb"]["los_hours_tail_threshold"], 100)
        self.assertGreaterEqual(summary["mimic_iv"]["los_hours_tail_threshold"], 500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
