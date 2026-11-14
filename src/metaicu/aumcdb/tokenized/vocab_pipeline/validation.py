"""Structural and semantic contract checks for a constructed supplied vocabulary.

Mirrors the invariants in docs/aumc_vocab_rebuild_handoff.md ("Validation Contract") and
tests/test_supplied_vocab_contract.py. Raises ``AssertionError`` with every violation listed
together (not just the first one found) so a failed build reports its full scope in one pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from metaicu.aumcdb.tokenized.vocab_pipeline.omop_validation import (
    lookup_concept_vocabulary,
)

LEGACY_NAMESPACE_PREFIX = "OMOP//OMOP_CONCEPT//"
LAB_PREFIX = "LAB//"
LAB_ROLE = "dynamic_event/lab"
GCS_COMPONENT_IDS = {"3016335", "3026019", "3008223", "3026549", "3009094", "3013144"}
SCORE_COMPONENT_ROLE = "dynamic_event/score_component"
RA_VERBAL_CORRECT_CONCEPT_ID = "3013144"


def _is_true(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().isin(["true", "1", "yes"])


def _identity_violations(source: pd.DataFrame, final: pd.DataFrame) -> list[str]:
    violations = []
    if len(final) != len(source):
        violations.append(
            f"row count changed: {len(source)} source tokens -> {len(final)} final rows"
        )
    if final["source_token"].nunique() != len(final):
        violations.append("source_token is not unique")
    if final["source_token"].eq("").any():
        violations.append("empty source_token present")
    columns = [
        column
        for column in (
            "source_table",
            "source_itemid",
            "source_valueid",
            "source_unitid",
            "source_ordercategoryid",
            "row_count",
        )
        if column in source.columns and column in final.columns
    ]
    joined = source.set_index("source_token")[columns].join(
        final.set_index("source_token")[columns],
        lsuffix="_source",
        rsuffix="_final",
        how="inner",
    )
    for column in columns:
        mismatched = joined[f"{column}_source"].astype(str) != joined[
            f"{column}_final"
        ].astype(str)
        if mismatched.any():
            violations.append(
                f"{int(mismatched.sum())} rows changed identity column {column!r}"
            )
    return violations


def _emission_violations(final: pd.DataFrame, emit: pd.Series) -> list[str]:
    emitted = final[emit]
    violations = []
    for column, message in (
        ("harmonized_token", "emitted row(s) with empty harmonized_token"),
        ("token_role", "emitted row(s) with empty token_role"),
    ):
        if emitted[column].eq("").any():
            violations.append(message)
    target = final["target_concept_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    if (emit & target.eq("0")).any():
        violations.append("emitted row(s) with target_concept_id==0")
    legacy = (
        emitted["harmonized_token"].astype(str).str.startswith(LEGACY_NAMESPACE_PREFIX)
    )
    if legacy.any():
        violations.append(
            f"{int(legacy.sum())} emitted row(s) still in legacy OMOP//OMOP_CONCEPT// namespace"
        )
    excluded = final["source_table"].isin(["freetextitems", "procedureorderitems"])
    if (emit & excluded).any():
        violations.append("emitted freetextitems or procedureorderitems row(s)")
    return violations


def _clinical_violations(
    final: pd.DataFrame, emit: pd.Series, strict: bool
) -> list[str]:
    violations = []
    is_lab = final["source_token"].str.startswith(LAB_PREFIX, na=False)
    lab_role = final["token_role"].eq(LAB_ROLE)
    if (emit & is_lab & ~lab_role).any():
        violations.append("emitted LAB// row(s) missing dynamic_event/lab role")
    if (lab_role & ~is_lab).any():
        violations.append("non-LAB// row(s) carry dynamic_event/lab role")
    target = final["target_concept_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    is_gcs = final["source_table"].eq("listitems") & target.isin(GCS_COMPONENT_IDS)
    if (is_gcs & (~emit | final["token_role"].ne(SCORE_COMPONENT_ROLE))).any():
        violations.append(
            "accepted GCS component row(s) not emitted as dynamic_event/score_component"
        )
    ra_verbal = final["source_label"].eq("RA_Verbal")
    if ra_verbal.any() and not target[ra_verbal].eq(RA_VERBAL_CORRECT_CONCEPT_ID).all():
        violations.append("RA_Verbal does not resolve to concept 3013144")
    conflicts = final[
        final["source_label"].str.contains("Metoprolol|Selokeen", case=False, na=False)
    ]
    emitted_conflicts = conflicts[_is_true(conflicts["emit_as_model_token"])]
    if emitted_conflicts.empty and strict:
        violations.append("no emitted Metoprolol row(s) found (expected at least one)")
    elif not emitted_conflicts.empty:
        text = "|".join(
            emitted_conflicts["harmonized_token"].tolist()
            + emitted_conflicts["target_label"].tolist()
        )
        if "pantoprazole" in text.lower():
            violations.append("Metoprolol/pantoprazole conflict present")
    return violations


def _omop_violations(
    final: pd.DataFrame, emit: pd.Series, omop_vocab_dir: Path
) -> list[str]:
    target = final["target_concept_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    numeric_ids = (
        target[emit & target.str.match(r"^\d+$", na=False)]
        .astype(int)
        .unique()
        .tolist()
    )
    resolved = lookup_concept_vocabulary(numeric_ids, omop_vocab_dir)
    unresolved = sorted(set(numeric_ids) - set(resolved))
    if unresolved:
        return [
            f"{len(unresolved)} emitted target_concept_id(s) not found in local Athena export: {unresolved[:20]}"
        ]
    return []


def validate_supplied_vocab(
    source_vocab: pd.DataFrame,
    final_vocab: pd.DataFrame,
    omop_vocab_dir: Path,
    strict: bool = True,
) -> dict[str, Any]:
    """Check the constructed vocabulary against the supplied-vocabulary contract.

    Returns a JSON-serializable report on success; raises AssertionError listing every violation
    on failure. ``strict=False`` skips checks that only make sense for a full official
    AmsterdamUMCdb release (e.g. "at least one Metoprolol row is emitted") -- use it for bounded
    or synthetic-fixture builds, matching ``allow_unresolved_source_tokens``.
    """

    emit = _is_true(final_vocab["emit_as_model_token"])
    emitted = final_vocab[emit]
    violations = _identity_violations(source_vocab, final_vocab)
    violations.extend(_emission_violations(final_vocab, emit))
    violations.extend(_clinical_violations(final_vocab, emit, strict))
    violations.extend(_omop_violations(final_vocab, emit, omop_vocab_dir))

    if violations:
        raise AssertionError(
            "Supplied-vocabulary contract violations:\n- " + "\n- ".join(violations)
        )

    return {
        "source_tokens": int(len(final_vocab)),
        "emitted_source_tokens": int(emit.sum()),
        "emitted_rows": int(
            pd.to_numeric(emitted["row_count"], errors="coerce").fillna(0).sum()
        ),
        "unique_emitted_destinations": int(emitted["harmonized_token"].nunique()),
        "violations": [],
    }
