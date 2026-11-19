"""Clinical ratios and manifest entries produced by the dataset target adapters."""

import unittest

import polars as pl

from metaicu.aumcdb.grid.build import derive_targets as aumcdb
from metaicu.mimiciv.grid.build import derive_targets as mimiciv


class DerivedTargetTests(unittest.TestCase):
    def test_ratios_use_raw_measurements_and_positive_denominators(self):
        grid = pl.DataFrame(
            {
                "admissionid": [1, 1, 1, 1, 2],
                "hour": [0, 1, 2, 3, 0],
                "po2": [90.0, None, 80.0, 100.0, 75.0],
                "fio2": [30.0, 40.0, 0.0, 50.0, 50.0],
                "urine_rate": [40.0, None, 20.0, 60.0, 50.0],
            }
        )
        admissions = pl.DataFrame({"admissionid": [1, 2], "weight": [80.0, 0.0]})
        for dataset in (aumcdb, mimiciv):
            with self.subTest(dataset=dataset.__name__):
                result, matches = dataset.add_derived_tte_targets(grid, admissions)
                result = result.sort("admissionid", "hour")
                self.assertEqual(
                    result["pf_ratio"].to_list(), [300.0, None, None, 200.0, 150.0]
                )
                self.assertEqual(
                    result["urine_rate_per_weight"].to_list(),
                    [0.5, None, 0.25, 0.75, None],
                )
                self.assertNotIn("weight", result.columns)
                for tag, unit in (
                    ("pf_ratio", "mmHg"),
                    ("urine_rate_per_weight", "mL/h/kg"),
                ):
                    expected = {
                        "reconstruction_type": "direct_numeric",
                        "target_unit": unit,
                    }
                    if dataset is mimiciv:
                        expected["keep_matches"] = []
                    self.assertEqual(matches[tag], expected)

    def test_missing_source_columns_skip_only_the_unavailable_target(self):
        admissions = pl.DataFrame({"admissionid": [1], "weight": [80.0]})
        grid = pl.DataFrame({"admissionid": [1], "hour": [0], "urine_rate": [40.0]})
        for columns, expected in (
            (["admissionid", "hour"], set()),
            (grid.columns, {"urine_rate_per_weight"}),
        ):
            with self.subTest(input_columns=columns):
                result, matches = aumcdb.add_derived_tte_targets(
                    grid.select(columns), admissions
                )
                self.assertEqual(set(matches), expected)
                self.assertEqual(set(result.columns) - set(columns), expected)
