import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import cps_config as C
from data_loader import load_data


def _sample_eval_split(df, seed):
    if C.DATA_FRACTION >= 1.0 or len(df) == 0:
        return df

    n = max(1, int(round(len(df) * C.DATA_FRACTION)))
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def _can_stratify(series):
    counts = pd.Series(series).value_counts()
    return len(counts) > 1 and counts.min() >= 2


def _split_random_state(seed):
    """Return the sklearn random_state used for this seed's data partitions."""
    if C.PER_SEED_SPLITS:
        return seed
    return C.SPLIT_RANDOM_STATE


def _split_existing_calibration(calib_df, split_random_state):
    if len(calib_df) < 2:
        scale_df = calib_df.copy()
        cal_df = calib_df.copy()
        return scale_df.reset_index(drop=True), cal_df.reset_index(drop=True)

    stratify = calib_df["male"] if _can_stratify(calib_df["male"]) else None
    scale_df, cal_df = train_test_split(
        calib_df,
        test_size=0.5,
        random_state=split_random_state,
        stratify=stratify,
    )
    return scale_df.reset_index(drop=True), cal_df.reset_index(drop=True)


def _split_frame(df, split_name):
    return pd.DataFrame({
        "id": df["id"].astype(int).values,
        "split": split_name,
        "y_true": df["boneage"].astype(float).values,
        "sex": df["male"].astype(int).values,
        "image_path": df["filepath"].astype(str).values,
    })


def create_split(seed):
    """Create the CP split file for one seed."""
    seed_dir = C.RUN_OUTPUT_DIR / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    split_random_state = _split_random_state(seed)
    train_df, val_df, calib_df, test_df, max_age = load_data(
        sample_frac=C.DATA_FRACTION,
        split_random_state=split_random_state,
    )
    test_df = _sample_eval_split(test_df, seed)
    scale_df, cal_df = _split_existing_calibration(calib_df, split_random_state)

    split_df = pd.concat(
        [
            _split_frame(train_df, "train"),
            _split_frame(val_df, "val"),
            _split_frame(scale_df, "scale"),
            _split_frame(cal_df, "cal"),
            _split_frame(test_df, "test"),
        ],
        ignore_index=True,
    )

    split_path = seed_dir / "split.csv"
    split_df.to_csv(split_path, index=False)

    meta = {
        "seed": seed,
        "max_age": float(max_age),
        "data_fraction": C.DATA_FRACTION,
        "split_note": C.SPLIT_NOTE,
        "per_seed_splits": C.PER_SEED_SPLITS,
        "split_random_state": int(split_random_state),
        "n_train": int((split_df["split"] == "train").sum()),
        "n_val": int((split_df["split"] == "val").sum()),
        "n_scale": int((split_df["split"] == "scale").sum()),
        "n_cal": int((split_df["split"] == "cal").sum()),
        "n_test": int((split_df["split"] == "test").sum()),
    }
    with open(seed_dir / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return split_path


def _with_dataset_columns(df, max_age):
    out = pd.DataFrame({
        "id": df["id"].astype(int),
        "boneage": df["y_true"].astype(float),
        "male": df["sex"].astype(int).astype(bool),
        "filepath": df["image_path"].astype(str),
    })
    out["boneage_norm"] = out["boneage"] / max_age
    out["sex_norm"] = out["male"].astype("float32")
    return out


def load_seed_split(seed):
    seed_dir = C.RUN_OUTPUT_DIR / f"seed_{seed:02d}"
    split_path = seed_dir / "split.csv"
    meta_path = seed_dir / "split_meta.json"

    if not split_path.exists():
        create_split(seed)

    split_df = pd.read_csv(split_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    max_age = float(meta["max_age"])
    dfs = {}
    for name in ["train", "val", "scale", "cal", "test"]:
        part = split_df[split_df["split"] == name].reset_index(drop=True)
        dfs[name] = _with_dataset_columns(part, max_age)

    return dfs, max_age


def split_sizes(seed):
    seed_dir = C.RUN_OUTPUT_DIR / f"seed_{seed:02d}"
    path = seed_dir / "split.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return df["split"].value_counts().to_dict()
