"""Dataset-independent contracts and mechanics for hourly grid pipelines."""

from pathlib import Path


FEATURE_LIST_PATH = (
    Path(__file__).resolve().parent / "data/icarefm_table_s3_features.csv"
)
