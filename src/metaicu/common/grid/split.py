"""Assign all admissions from one patient to the same train, validation or test split."""

import logging
import random

import polars as pl

log = logging.getLogger(__name__)


def assign_patient_splits(
    admissions, train_frac, val_frac, test_frac, seed, *, patient_column
):
    """Return admissionid and split, using a seeded shuffle of sorted patient IDs."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, (
        f"train/val/test fracs must sum to 1.0, got {train_frac}+{val_frac}+{test_frac}"
    )
    patients = sorted(admissions[patient_column].unique().to_list())
    random.Random(seed).shuffle(patients)
    n_patients = len(patients)
    train_end = round(n_patients * train_frac)
    val_end = train_end + round(n_patients * val_frac)

    labels = []
    for index in range(n_patients):
        if index < train_end:
            labels.append("train")
        elif index < val_end:
            labels.append("val")
        else:
            labels.append("test")
    log.info(
        "Split %d patients (seed=%s): train=%d, val=%d, test=%d",
        n_patients,
        seed,
        train_end,
        val_end - train_end,
        n_patients - val_end,
    )
    splits = pl.DataFrame({patient_column: patients, "split": labels})
    return (
        admissions.select("admissionid", patient_column)
        .join(splits, on=patient_column)
        .select("admissionid", "split")
    )
