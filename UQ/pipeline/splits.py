"""Per-seed data splits that reproduce the conformal pipeline's partition.

The conformal run uses, for each seed, a stratified 50/25/25 split of
``train.csv`` with ``random_state = seed`` (stratified by sex), and the external
``test.csv`` as the held-out TEST set. ``data_loader.load_data`` does the same
split but hard-codes ``random_state = RANDOM_STATE`` (42), so we re-implement the
split here parameterized by seed while reusing the unchanged source readers and
normalization helpers from ``data_loader``.

For seed ``s`` this yields exactly the train / val / calib / test dataframes the
conformal pipeline produced for its seed ``s`` (calib here = the conformal
scale+cal union), which is what makes the comparison fair.
"""
from __future__ import annotations

from sklearn.model_selection import train_test_split

from . import pconfig as C  # noqa: F401  (ensures sys.path is set up)

# Reuse the project's source readers / normalizers unchanged.
from data_loader import _read_source, _add_norm_cols  # noqa: E402
from config import DATA_DIR  # noqa: E402


def _read_test_set():
    """Held-out TEST set = the official source validation set (``val.csv``).

    The conformal config defines TEST as the "official source validation set";
    in this dataset that is ``data/val.csv`` (schema ``Image ID, male,
    Bone Age (months)``) with images under ``data/val/`` — 1425 cases, matching
    the conformal ``n_test``. ``data_loader._read_test_source`` instead looks for
    a ``test.csv`` that does not exist here, so we read ``val.csv`` directly.
    """
    return _read_source(DATA_DIR / "val.csv", "Image ID", "Bone Age (months)",
                        "male", "val")


def _stratify_values(df):
    col = C.SPLIT_STRATIFY_COL
    if col not in df.columns:
        return None
    vals = df[col]
    # Need at least two members per class for a stratified split.
    return vals if vals.value_counts().min() >= 2 else None


def make_seed_split(seed: int, data_fraction: float = 1.0):
    """Return ``(train_df, val_df, calib_df, test_df, max_age)`` for ``seed``.

    50 % train / 25 % val / 12.5 % calib (the conformal cal half; the matching
    12.5 % scale half is dropped), stratified by sex, random_state=seed; test is
    the external ``test.csv``. Targets are normalized against the train split's
    max bone age (same convention as ``data_loader.load_data``).
    """
    rs = C.split_random_state(seed)

    train_full = _read_source(DATA_DIR / "train.csv", "id", "boneage", "male", "train")
    test_df = _read_test_set()

    if data_fraction < 1.0:
        train_full = train_full.sample(frac=data_fraction, random_state=rs).reset_index(drop=True)

    # 50 / 50 then 50 / 50 -> train 50 / val 25 / calib-block 25.
    train_df, temp_df = train_test_split(
        train_full, test_size=0.5, random_state=rs,
        stratify=_stratify_values(train_full),
    )
    val_df, calib_block = train_test_split(
        temp_df, test_size=0.5, random_state=rs,
        stratify=_stratify_values(temp_df),
    )
    # Conformal splits the 25% calib-block into scale (12.5%) + cal (12.5%) and
    # only calibrates on the cal half. Mirror that: use cal as calib, drop scale.
    _scale_df, calib_df = train_test_split(
        calib_block, test_size=0.5, random_state=rs,
        stratify=_stratify_values(calib_block),
    )

    max_age = train_df["boneage"].max()
    train_df = _add_norm_cols(train_df, max_age).reset_index(drop=True)
    val_df = _add_norm_cols(val_df, max_age).reset_index(drop=True)
    calib_df = _add_norm_cols(calib_df, max_age).reset_index(drop=True)
    test_df = _add_norm_cols(test_df, max_age).reset_index(drop=True)

    return train_df, val_df, calib_df, test_df, max_age
