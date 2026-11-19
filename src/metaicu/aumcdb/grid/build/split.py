"""Patient-level grid split assignment for AmsterdamUMCdb."""

from metaicu.common.grid.split import assign_patient_splits


def assign_splits(admissions, train_frac, val_frac, test_frac, seed):
    return assign_patient_splits(
        admissions, train_frac, val_frac, test_frac, seed, patient_column="patientid"
    )
