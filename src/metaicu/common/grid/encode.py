"""Encode grid and static categories in vocabulary order, with one column for nulls."""

import logging
import re

import polars as pl

log = logging.getLogger(__name__)
MISSING_CATEGORY_LABEL = "(missing)"


def get_categorical_vocab(matches):
    """Return sorted labels declared in the manifest, including absent categories."""
    vocab = {}
    for tag, info in matches.items():
        if info["reconstruction_type"] == "categorical":
            labels = (match["standardized_label"] for match in info["keep_matches"])
            vocab[tag] = sorted({label for label in labels if label})
    return vocab


def _sanitize(label):
    return re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_")


def one_hot_encode_columns(df, vocab, start_pos=0):
    """Replace each present tag with Int8 columns and return (frame, schema, next_pos).

    The schema records each column's position within the feature and across features.
    Pass next_pos as start_pos to continue numbering in a second frame.
    """
    encoding_schema = []
    for tag, categories in vocab.items():
        if tag not in df.columns:
            continue
        names = [f"{tag}__{_sanitize(category)}" for category in categories]
        if len(set(names)) != len(names):
            raise ValueError(f"{tag}: sanitized category names collide ({names})")

        column = pl.col(tag)
        indicators = [
            (column == category).fill_null(False).cast(pl.Int8).alias(name)
            for category, name in zip(categories, names)
        ]
        missing_name = f"{tag}__missing"
        indicators.append(column.is_null().cast(pl.Int8).alias(missing_name))
        df = df.with_columns(indicators).drop(tag)

        columns = [*zip(categories, names), (MISSING_CATEGORY_LABEL, missing_name)]
        for position, (category, name) in enumerate(columns):
            encoding_schema.append(
                {
                    "feature": tag,
                    "category": category,
                    "column_name": name,
                    "position_in_feature": position,
                    "position_global": start_pos + len(encoding_schema),
                }
            )
        log.info("%s: one-hot encoded into %d columns", tag, len(columns))
    return df, encoding_schema, start_pos + len(encoding_schema)


def save_categorical_encoding(encoding_schema, output_path):
    """Write the feature, category, column names and positions to CSV in encoding order."""
    pl.DataFrame(encoding_schema).write_csv(output_path)
    log.info("Wrote %s (%d one-hot columns)", output_path, len(encoding_schema))


def verify_one_hot_exclusive(grid, encoding_schema):
    """Return the count of rows whose indicators do not sum to one for each failing tag."""
    by_feature = {}
    for row in encoding_schema:
        by_feature.setdefault(row["feature"], []).append(row["column_name"])
    bad = {}
    for feature, columns in by_feature.items():
        row_sums = grid.select(pl.sum_horizontal(columns).alias("s"))["s"]
        n_bad = int((row_sums != 1).sum())
        if n_bad:
            bad[feature] = n_bad
    return bad
