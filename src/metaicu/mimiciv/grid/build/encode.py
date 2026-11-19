"""Encode MIMIC-IV labels using the grid's eight-level RASS scale."""

import polars as pl

from metaicu.common.grid.encode import (
    MISSING_CATEGORY_LABEL,
    get_categorical_vocab as _manifest_vocab,
    one_hot_encode_columns,
    save_categorical_encoding,
    verify_one_hot_exclusive,
)

__all__ = [
    "CATEGORY_COLLAPSE",
    "MISSING_CATEGORY_LABEL",
    "get_categorical_vocab",
    "one_hot_encode_categorical",
    "one_hot_encode_columns",
    "save_categorical_encoding",
    "verify_one_hot_exclusive",
]

# The reviewed manifest retains the original RASS labels for future use.
CATEGORY_COLLAPSE = {
    "rass": {"+3 Very agitated": "+2 Agitated", "+4 Combative": "+2 Agitated"},
}


def get_categorical_vocab(matches):
    """Return manifest labels after applying MIMIC-IV's category grouping."""
    vocab = _manifest_vocab(matches)
    for tag, collapse in CATEGORY_COLLAPSE.items():
        if tag in vocab:
            vocab[tag] = sorted({collapse.get(label, label) for label in vocab[tag]})
    return vocab


def one_hot_encode_categorical(grid, matches, start_pos=0, external_vocab=None):
    """Group source labels and return the encoded frame and encoding schema."""
    vocab = (
        external_vocab if external_vocab is not None else get_categorical_vocab(matches)
    )
    for tag, collapse in CATEGORY_COLLAPSE.items():
        if tag in grid.columns:
            grid = grid.with_columns(pl.col(tag).replace(collapse).alias(tag))
    grid, encoding_schema, _ = one_hot_encode_columns(grid, vocab, start_pos=start_pos)
    return grid, encoding_schema
