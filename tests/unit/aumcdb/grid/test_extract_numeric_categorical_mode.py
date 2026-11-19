"""Hourly categorical extraction uses the most frequent label, with alphabetical ties."""

import unittest
from unittest.mock import patch

import polars as pl

from metaicu.aumcdb.grid.build import extract_numeric


class ModeTieBreakTests(unittest.TestCase):
    def test_hourly_mode_and_alphabetical_tie_break(self):
        rows = pl.DataFrame(
            {
                "admissionid": [1] * 7,
                "tag": ["rass"] * 7,
                "hour": [0, 0, 0, 0, 1, 1, 1],
                "label": ["b", "a", "b", "a", "a", "b", "b"],
            }
        )
        for order, raw in (("forward", rows), ("reverse", rows.reverse())):
            with (
                self.subTest(order=order),
                patch.object(extract_numeric, "_build_categorical", return_value=raw),
            ):
                _, hourly = extract_numeric.extract_numeric_categorical({}, None, None)
                self.assertEqual(hourly.sort("hour")["agg_label"].to_list(), ["a", "b"])
