import pandas as pd

import cps_config as C
from cps.io_utils import read_table, write_table


def collect_predictions():
    rows = []
    for seed in C.SEEDS:
        path = C.RUN_OUTPUT_DIR / f"seed_{seed:02d}" / "predictions.parquet"
        rows.append(read_table(path))

    all_predictions = pd.concat(rows, ignore_index=True)
    out_path = C.RUN_OUTPUT_DIR / "all_predictions.parquet"
    write_table(all_predictions, out_path)
    print(f"Saved all predictions to {out_path}")
    return out_path


if __name__ == "__main__":
    collect_predictions()
