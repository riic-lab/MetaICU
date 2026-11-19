"""Encode AmsterdamUMCdb categories from the reviewed manifest or a supplied vocabulary."""

from metaicu.common.grid.encode import (
    MISSING_CATEGORY_LABEL,
    get_categorical_vocab,
    one_hot_encode_columns,
    save_categorical_encoding,
    verify_one_hot_exclusive,
)

__all__ = [
    "MISSING_CATEGORY_LABEL",
    "get_categorical_vocab",
    "one_hot_encode_categorical",
    "one_hot_encode_columns",
    "save_categorical_encoding",
    "verify_one_hot_exclusive",
]


def one_hot_encode_categorical(grid, matches, start_pos=0, external_vocab=None):
    """Return the encoded frame, encoding schema and next global position."""
    vocab = (
        external_vocab if external_vocab is not None else get_categorical_vocab(matches)
    )
    return one_hot_encode_columns(grid, vocab, start_pos=start_pos)
