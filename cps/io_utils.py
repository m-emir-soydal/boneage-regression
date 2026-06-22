import json
from pathlib import Path

import pandas as pd


def write_table(df, path):
    """Write a dataframe to parquet, falling back to pickle if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        fmt = "parquet"
    except Exception:
        df.to_pickle(path)
        fmt = "pickle"

    with open(path.with_suffix(path.suffix + ".format.json"), "w", encoding="utf-8") as f:
        json.dump({"format": fmt}, f)


def read_table(path):
    """Read a table written by write_table."""
    path = Path(path)
    fmt_path = path.with_suffix(path.suffix + ".format.json")
    if fmt_path.exists():
        with open(fmt_path, "r", encoding="utf-8") as f:
            fmt = json.load(f).get("format")
        if fmt == "pickle":
            return pd.read_pickle(path)

    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_pickle(path)
