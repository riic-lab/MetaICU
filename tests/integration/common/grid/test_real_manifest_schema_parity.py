"""Proves the "data-content invariant schema" contract (metaicu.common.grid.schema_union) against the
real, packaged AUMCdb/MIMIC-IV manifests -- not a synthetic fixture. feature_schema.json is a
pure function of a cohort's own `matches` (+ derived targets and presence-mask flags, both
themselves derived from `matches`) plus categorical vocab; if `matches`/vocab are already union-
stable for both cohorts, every solo run's feature_schema.json is *guaranteed* identical to the
padded joint run's, without needing three separate full-pipeline builds to check file bytes.

Operationalizes the Phase 1 design note: "today's 120/120 tag parity ... write the schema-
identity test to prove it, not assume it."
"""

from __future__ import annotations

import unittest

from metaicu.aumcdb.grid.build.encode import (
    get_categorical_vocab as aumcdb_categorical_vocab,
)
from metaicu.aumcdb.grid.build.manifest_parser import (
    parse_manifest as parse_aumcdb_manifest,
)
from metaicu.mimiciv.grid.build.encode import (
    get_categorical_vocab as mimiciv_categorical_vocab,
)
from metaicu.mimiciv.grid.build.manifest_parser import (
    parse_manifest as parse_mimiciv_manifest,
)
from metaicu.common.grid.schema_union import (
    compute_union_categorical_vocab,
    compute_union_matches,
    pad_matches_for_cohort,
)


class RealManifestSchemaParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.aumc_matches, _ = parse_aumcdb_manifest()
        cls.mimic_matches, _ = parse_mimiciv_manifest()
        cls.union = compute_union_matches(
            {"aumcdb": cls.aumc_matches, "mimic_iv": cls.mimic_matches}
        )

    def test_padding_is_a_no_op_for_both_cohorts(self) -> None:
        self.assertEqual(len(self.union), 120)
        self.assertEqual(list(self.aumc_matches), list(self.mimic_matches))
        self.assertEqual(set(self.union), set(self.aumc_matches))
        self.assertEqual(
            pad_matches_for_cohort(self.aumc_matches, self.union), self.aumc_matches
        )
        self.assertEqual(
            pad_matches_for_cohort(self.mimic_matches, self.union), self.mimic_matches
        )

    def test_categorical_vocab_union_is_a_no_op_for_both_cohorts(self) -> None:
        aumc_vocab = aumcdb_categorical_vocab(self.aumc_matches)
        mimic_vocab = mimiciv_categorical_vocab(self.mimic_matches)
        union_vocab = compute_union_categorical_vocab(
            {"aumcdb": aumc_vocab, "mimic_iv": mimic_vocab}
        )
        for tag, categories in aumc_vocab.items():
            self.assertEqual(
                categories, union_vocab[tag], f"{tag}: AUMCdb vocab differs from union"
            )
        for tag, categories in mimic_vocab.items():
            self.assertEqual(
                categories,
                union_vocab[tag],
                f"{tag}: MIMIC-IV vocab differs from union",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
