import numpy as np
import pandas as pd
import torch

import cps_config as C
from cps.device import get_device
from cps.io_utils import write_table
from cps.loaders import build_eval_loader
from cps.split_data import load_seed_split
from model import build_multi_input_model


def _predict_with_embeddings(model, loader, df, max_age, device):
    model.eval()
    preds = []
    embeddings = []

    with torch.no_grad():
        for inputs, _ in loader:
            img = inputs["image_input"].to(device)
            sex = inputs["sex_input"].to(device)

            img_feat = model.base_model(img)
            sex_feat = model.relu(model.sex_fc(sex))
            fused = torch.cat((img_feat, sex_feat), dim=1)
            penultimate = model.relu(model.fc1(fused))
            out = model.out(model.dropout(penultimate))

            preds.append(out.cpu().numpy())
            embeddings.append(penultimate.cpu().numpy())

    y_pred = np.vstack(preds).reshape(-1) * max_age
    z = np.vstack(embeddings)
    rows = pd.DataFrame({
        "id": df["id"].astype(int).values,
        "y_true": df["boneage"].astype(float).values,
        "y_pred": y_pred,
        "sex": df["male"].astype(int).values,
    })
    return rows, z


def predict_one_seed(seed):
    seed_dir = C.RUN_OUTPUT_DIR / f"seed_{seed:02d}"
    checkpoint_path = seed_dir / "model_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    dfs, max_age = load_seed_split(seed)
    device = get_device()

    model = build_multi_input_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    all_rows = []
    for split in ["scale", "cal", "test"]:
        loader = build_eval_loader(dfs[split], device)
        rows, embeddings = _predict_with_embeddings(model, loader, dfs[split], max_age, device)
        rows.insert(1, "seed", seed)
        rows.insert(2, "split", split)
        all_rows.append(rows)

        if C.SAVE_EMBEDDINGS:
            np.save(seed_dir / f"embeddings_{split}.npy", embeddings)
            np.save(seed_dir / f"ids_{split}.npy", rows["id"].astype(int).values)

    predictions = pd.concat(all_rows, ignore_index=True)
    write_table(predictions, seed_dir / "predictions.parquet")
    print(f"[seed {seed:02d}] saved predictions and embeddings")
    return seed_dir / "predictions.parquet"


if __name__ == "__main__":
    for seed_value in C.SEEDS:
        predict_one_seed(seed_value)
