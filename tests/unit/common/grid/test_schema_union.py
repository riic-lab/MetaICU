"""Unit tests for metaicu.common.grid.schema_union -- the cross-cohort matches/vocab union used to make
a joint multi-dataset grid's schema data-content invariant (same tag shape regardless of which
cohort's raw data actually supports it)."""

from __future__ import annotations

import unittest

from metaicu.common.grid.schema_union import (
    compute_union_categorical_vocab,
    compute_union_matches,
    pad_matches_for_cohort,
)


class ComputeUnionMatchesTests(unittest.TestCase):
    def test_union_of_disjoint_tags(self) -> None:
        aumc = {"hr": {"reconstruction_type": "direct_numeric", "target_unit": "bpm"}}
        mimic = {"pt": {"reconstruction_type": "direct_numeric", "target_unit": "sec"}}
        registry = compute_union_matches({"aumcdb": aumc, "mimic_iv": mimic})
        self.assertEqual(
            registry,
            {
                "hr": {"reconstruction_type": "direct_numeric", "target_unit": "bpm"},
                "pt": {"reconstruction_type": "direct_numeric", "target_unit": "sec"},
            },
        )

    def test_shared_tag_with_conflicting_reconstruction_type_raises(self) -> None:
        aumc = {
            "tri": {"reconstruction_type": "direct_numeric", "target_unit": "ng/mL"}
        }
        mimic = {"tri": {"reconstruction_type": "categorical", "target_unit": "ng/mL"}}
        with self.assertRaises(ValueError):
            compute_union_matches({"aumcdb": aumc, "mimic_iv": mimic})

    def test_shared_tag_with_conflicting_target_unit_raises(self) -> None:
        aumc = {
            "pf_ratio": {
                "reconstruction_type": "direct_numeric",
                "target_unit": "ratio",
            }
        }
        mimic = {
            "pf_ratio": {"reconstruction_type": "direct_numeric", "target_unit": "mmHg"}
        }
        with self.assertRaises(ValueError):
            compute_union_matches({"aumcdb": aumc, "mimic_iv": mimic})

    def test_ignores_extra_keys_like_keep_matches_when_comparing(self) -> None:
        aumc = {
            "hr": {
                "reconstruction_type": "direct_numeric",
                "target_unit": "bpm",
                "keep_matches": [{"table": "numericitems", "itemid": "1"}],
                "n_keep": 1,
            }
        }
        mimic = {
            "hr": {
                "reconstruction_type": "direct_numeric",
                "target_unit": "bpm",
                "keep_matches": [{"table": "chartevents", "itemid": "220045"}],
                "n_keep": 1,
            }
        }
        registry = compute_union_matches({"aumcdb": aumc, "mimic_iv": mimic})
        self.assertEqual(
            registry,
            {"hr": {"reconstruction_type": "direct_numeric", "target_unit": "bpm"}},
        )


class PadMatchesForCohortTests(unittest.TestCase):
    def test_pads_missing_tag_as_structural_zero(self) -> None:
        own = {
            "hr": {
                "reconstruction_type": "direct_numeric",
                "target_unit": "bpm",
                "keep_matches": [{"itemid": "1"}],
                "n_keep": 1,
            }
        }
        registry = {
            "hr": {"reconstruction_type": "direct_numeric", "target_unit": "bpm"},
            "pt": {"reconstruction_type": "direct_numeric", "target_unit": "sec"},
        }
        padded = pad_matches_for_cohort(own, registry)
        self.assertEqual(set(own), {"hr"})
        self.assertEqual(padded["hr"], own["hr"])
        self.assertEqual(
            padded["pt"],
            {
                "reconstruction_type": "direct_numeric",
                "target_unit": "sec",
                "keep_matches": [],
                "n_keep": 0,
                "structural_zero": True,
            },
        )


class ComputeUnionCategoricalVocabTests(unittest.TestCase):
    def test_unions_and_sorts_categories_per_tag(self) -> None:
        aumc_vocab = {
            "rass": ["0 Alert and calm", "-1 Drowsy"],
            "airway": ["Tracheostomy"],
        }
        mimic_vocab = {"rass": ["0 Alert and calm", "+1 Restless"]}
        union = compute_union_categorical_vocab(
            {"aumcdb": aumc_vocab, "mimic_iv": mimic_vocab}
        )
        self.assertEqual(
            union,
            {
                "rass": sorted(["0 Alert and calm", "-1 Drowsy", "+1 Restless"]),
                "airway": ["Tracheostomy"],
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
