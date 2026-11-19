"""Unit tests for shared grid ordering, rerun safety, and post-write integrity."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import polars as pl

from metaicu.common.grid.assemble import canonical_column_order
from metaicu.common.grid.impute import (
    capture_presence_mask,
    materialize_structural_zero_columns,
)
from metaicu.mimiciv.grid.build.build_workflow import _prepare_output_dir


class GridOutputContractTests(unittest.TestCase):
    def test_structural_zeros_preserve_numeric_masks_and_treatment_types(self) -> None:
        matches = {
            "tri": {"reconstruction_type": "direct_numeric", "structural_zero": True},
            "hba1c": {"reconstruction_type": "direct_numeric"},
            "adh": {"reconstruction_type": "treatment_rate", "structural_zero": True},
            "adh_ind": {
                "reconstruction_type": "treatment_indicator",
                "structural_zero": True,
            },
        }
        source = pl.DataFrame({"admissionid": [1, 1], "hour": [0, 1]})
        grid = materialize_structural_zero_columns(source, matches)
        grid, masks = capture_presence_mask(grid, matches)
        self.assertEqual(masks, ["tri__observed", "hba1c__observed"])
        self.assertEqual(grid["tri"].to_list(), [0.0, 0.0])
        self.assertEqual(grid["tri__observed"].to_list(), [0, 0])
        self.assertEqual(grid["hba1c"].to_list(), [None, None])
        self.assertEqual(grid["hba1c__observed"].to_list(), [0, 0])
        self.assertEqual(grid["adh"].to_list(), [0.0, 0.0])
        self.assertEqual(grid["adh_ind"].to_list(), [0, 0])
        self.assertNotIn("adh__observed", grid.columns)
        self.assertNotIn("adh_ind__observed", grid.columns)
        self.assertEqual(grid.schema["tri"], pl.Float64)
        self.assertEqual(grid.schema["adh"], pl.Float64)
        self.assertEqual(grid.schema["adh_ind"], pl.Int32)

    def test_column_order_follows_the_manifest_order(self) -> None:
        columns = ["b", "hour", "a__observed", "admissionid", "a"]
        matches = {
            "a": {"reconstruction_type": "direct_numeric"},
            "b": {"reconstruction_type": "direct_numeric"},
        }
        expected = ["admissionid", "hour", "a", "b", "a__observed"]
        physical, mapping = canonical_column_order(
            columns, matches, [], ["a__observed"], []
        )
        self.assertEqual(physical, expected)
        self.assertEqual(mapping, {"a": ["a"], "b": ["b"]})

    def test_overwrite_refuses_stale_outputs_and_preserves_unrelated_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            (root / "train" / "0.parquet").write_bytes(b"stale")
            (root / "note.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                _prepare_output_dir(root, overwrite=False)
            _prepare_output_dir(root, overwrite=True)
            self.assertFalse((root / "train").exists())
            self.assertEqual((root / "note.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main(verbosity=2)
