"""Train + evaluate one UQ method on one seed, the fair way.

For a given ``(method, seed)`` this:

  1. rebuilds the conformal per-seed split (``splits.make_seed_split``),
  2. trains the model for ``EPOCHS`` with UQ-aware checkpoint selection
     (same ``combo`` criterion as the existing UQ trainers: minimise calibrated
     MPIW@90 traded against calib MAE),
  3. fits the definitive post-hoc temperature (std-scale) on the calib split,
  4. runs MC inference on TEST and builds Gaussian intervals at every
     configured confidence,
  5. writes the per-seed folder
     ``comparison/<Method>/seed_XX/outputs/{metrics.csv,intervals.csv,
     calibration.json,model_best.pt,history.pkl}``.

The deterministic backbone (EfficientNet-B3 + sex branch) and the MC sampling
strategy are reused unchanged from the project's UQ package.
"""
from __future__ import annotations

import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from . import pconfig as C
from . import pmetrics as M
from .splits import make_seed_split

# Project / UQ modules (sys.path was wired in pconfig).
from data_loader import build_datasets, build_eval_loader  # noqa: E402
from model import build_multi_input_model  # noqa: E402
from train import select_device, move_training_to_cpu  # noqa: E402
from common.inference import enable_dropout, mc_forward_passes, denormalize  # noqa: E402
from common.calibration import fit_temperature  # noqa: E402
from common.uq_metrics import picp as picp90, mpiw as mpiw90  # noqa: E402
from bnn.bnn_model import build_bnn_model, enable_bnn_sampling  # noqa: E402

Z90 = C.Z_SCORES[0.90]


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
def set_all_seeds(seed: int) -> None:
    """Match the conformal pipeline's seeding for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# --------------------------------------------------------------------------- #
# Method dispatch
# --------------------------------------------------------------------------- #
def _build_model(method: str):
    if method == "bnn":
        return build_bnn_model(dropout=C.DROPOUT, prior_sigma=C.BNN_PRIOR_SIGMA)
    if method == "mc_dropout":
        return build_multi_input_model(dropout=C.DROPOUT)
    raise ValueError(f"Unknown method '{method}'")


def _enable_sampling(method: str, model):
    return enable_bnn_sampling(model) if method == "bnn" else enable_dropout(model)


def _train_loss(method: str, model, criterion, outputs, targets, n_train):
    """Plain Huber for MC-Dropout; ELBO (Huber + KL/N) for the BNN."""
    nll = criterion(outputs, targets)
    if method == "bnn":
        kl = model.kl_loss()
        loss = nll + C.BNN_KL_WEIGHT * (kl / n_train)
        return loss, float(nll.item()), float(kl.item())
    return nll, float(nll.item()), 0.0


def _mc_eval_calib(method, model, calib_loader, y_true_calib, max_age, device, samples):
    """Calibrated UQ stats on the calib split (PICP/MPIW@90 + std-scale + MAE)."""
    _enable_sampling(method, model)
    preds_norm, _ = mc_forward_passes(model, calib_loader, device, samples)
    model.train()
    pred_mean = denormalize(preds_norm.mean(axis=0), max_age)
    pred_std = denormalize(preds_norm.std(axis=0), max_age)
    scale = fit_temperature(y_true_calib, pred_mean, pred_std)
    std_cal = pred_std * scale
    lower, upper = pred_mean - Z90 * std_cal, pred_mean + Z90 * std_cal
    return {
        "std_scale": scale,
        "picp90": picp90(y_true_calib, lower, upper),
        "mpiw90": mpiw90(lower, upper),
        "mae": float(np.mean(np.abs(pred_mean - y_true_calib))),
    }


def _selection_score(val_mae, uq):
    """Lower-is-better ``combo`` score (calibrated MPIW@90 + w*MAE)."""
    if C.SELECT_BY == "val_mae" or uq is None:
        return val_mae if C.SELECT_BY == "val_mae" else None
    if C.SELECT_BY == "picp90":
        return abs(uq["picp90"] - 0.90)
    if C.SELECT_BY == "mpiw90":
        return uq["mpiw90"]
    if C.SELECT_BY == "combo":
        return uq["mpiw90"] + C.MAE_WEIGHT * uq["mae"]
    raise ValueError(f"Unknown SELECT_BY '{C.SELECT_BY}'")


# --------------------------------------------------------------------------- #
# Train one (method, seed)
# --------------------------------------------------------------------------- #
def train_method_seed(method: str, seed: int, dfs, device_hint=None):
    """Train, calibrate, and return ``(model, std_scale, max_age, history)``."""
    train_df, val_df, calib_df, test_df, max_age = dfs

    train_loader, val_loader = build_datasets(train_df, val_df)
    calib_loader = build_eval_loader(calib_df)
    y_true_calib = calib_df["boneage_norm"].values * max_age
    n_train = len(train_loader.dataset)

    model = _build_model(method)
    criterion = nn.SmoothL1Loss()
    mae_metric = nn.L1Loss()
    device, model = select_device(model, train_loader, criterion)
    print(f"    device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=C.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_score = float("inf")
    best_state = None
    history = {"loss": [], "val_mae": [], "calib_picp90": [],
               "calib_mpiw90": [], "calib_std_scale": [], "score": []}

    for epoch in range(C.EPOCHS):
        model.train()
        running = 0.0
        bar = tqdm(train_loader, desc=f"    [{method}] seed{seed} ep{epoch+1}/{C.EPOCHS}",
                   leave=False)
        for inputs, targets in bar:
            img = inputs["image_input"].to(device)
            sex = inputs["sex_input"].to(device)
            targets = targets.to(device)
            try:
                optimizer.zero_grad()
                outputs = model(img, sex)
                loss, _, _ = _train_loss(method, model, criterion, outputs, targets, n_train)
                loss.backward()
                optimizer.step()
            except RuntimeError as exc:
                if device.type != "cuda" or "out of memory" not in str(exc).lower():
                    raise
                print("    warning: GPU OOM; switching to CPU.")
                device = torch.device("cpu")
                model, optimizer = move_training_to_cpu(model, optimizer)
                img, sex, targets = (img.to(device), sex.to(device), targets.to(device))
                optimizer.zero_grad()
                outputs = model(img, sex)
                loss, _, _ = _train_loss(method, model, criterion, outputs, targets, n_train)
                loss.backward()
                optimizer.step()
            running += loss.item() * img.size(0)
            bar.set_postfix({"loss": f"{loss.item():.4f}"})
        train_loss = running / n_train

        # Deterministic val MAE (months). tqdm bar so the post-training phase
        # is visibly active (this is the silent gap users hit "after epoch 0").
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"      val ep{epoch+1}",
                                        leave=False):
                img = inputs["image_input"].to(device)
                sex = inputs["sex_input"].to(device)
                targets = targets.to(device)
                vloss += mae_metric(model(img, sex), targets).item() * img.size(0)
        val_mae = (vloss / len(val_loader.dataset)) * max_age

        uq = None
        if C.SELECT_BY != "val_mae" and ((epoch + 1) % C.MC_EVAL_EVERY == 0):
            print(f"      [{method}] seed{seed} ep{epoch+1}: MC calib eval "
                  f"({C.MC_EVAL_SAMPLES} passes over {len(calib_loader.dataset)} calib)...",
                  flush=True)
            uq = _mc_eval_calib(method, model, calib_loader, y_true_calib,
                                max_age, device, C.MC_EVAL_SAMPLES)
        score = _selection_score(val_mae, uq)

        msg = (f"    [{method}] seed{seed} ep{epoch+1}/{C.EPOCHS} "
               f"loss={train_loss:.4f} val_mae={val_mae:.2f}mo")
        if uq is not None:
            msg += (f" | calib PICP90={uq['picp90']*100:.1f}% "
                    f"MPIW90={uq['mpiw90']:.2f} s={uq['std_scale']:.3f}")
        if score is not None:
            msg += f" | score={score:.4f}"
        print(msg, flush=True)

        history["loss"].append(train_loss)
        history["val_mae"].append(val_mae)
        history["calib_picp90"].append(uq["picp90"] if uq else None)
        history["calib_mpiw90"].append(uq["mpiw90"] if uq else None)
        history["calib_std_scale"].append(uq["std_scale"] if uq else None)
        history["score"].append(score)

        scheduler.step(val_mae / max_age)

        if score is not None and score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Definitive calibration on calib with more MC passes.
    final = _mc_eval_calib(method, model, calib_loader, y_true_calib,
                           max_age, device, C.FINAL_SAMPLES)
    model.to(device)
    return model, float(final["std_scale"]), max_age, history, device, final


# --------------------------------------------------------------------------- #
# Evaluate on TEST
# --------------------------------------------------------------------------- #
def evaluate_test(method, seed, model, test_df, max_age, std_scale, device,
                  samples=None):
    """MC inference on TEST -> metric rows (all confidences) + per-sample intervals."""
    samples = samples or C.FINAL_SAMPLES
    test_loader = build_eval_loader(test_df)
    _enable_sampling(method, model)
    preds_norm, _ = mc_forward_passes(model, test_loader, device, samples)

    pred_mean = denormalize(preds_norm.mean(axis=0), max_age)
    pred_std = denormalize(preds_norm.std(axis=0), max_age)
    std_eff = pred_std * std_scale
    y_true = test_df["boneage_norm"].values * max_age

    display = C.METHOD_DISPLAY_NAMES[method]
    rows = []
    intervals = {
        "id": test_df["id"].values,
        "sex": test_df["male"].values.astype(int),
        "true_age": y_true,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "std_scale": std_scale,
    }
    for conf in C.CONFIDENCES:
        z = C.Z_SCORES[conf]
        row, (lower, upper) = M.metrics_at_confidence(
            seed, display, conf, z, y_true, pred_mean, std_eff, C.Y_MIN, C.Y_MAX
        )
        rows.append(row)
        tag = int(round(conf * 100))
        intervals[f"lower{tag}"] = lower
        intervals[f"upper{tag}"] = upper
        intervals[f"covered{tag}"] = ((y_true >= lower) & (y_true <= upper)).astype(int)

    return pd.DataFrame(rows)[M.METRIC_COLUMNS], pd.DataFrame(intervals)


# --------------------------------------------------------------------------- #
# Per-seed driver
# --------------------------------------------------------------------------- #
def run_method_seed(method: str, seed: int) -> pd.DataFrame:
    """Full train+eval for one (method, seed); writes the seed folder; returns metrics."""
    display = C.METHOD_DISPLAY_NAMES[method]
    seed_dir = C.OUTPUT_ROOT / display.replace(" ", "_") / f"seed_{seed:02d}" / "outputs"
    seed_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(seed)
    dfs = make_seed_split(seed, C.DATA_FRACTION)
    train_df, val_df, calib_df, test_df, max_age = dfs
    if C.MAX_TEST is not None and len(test_df) > C.MAX_TEST:
        test_df = test_df.iloc[: C.MAX_TEST].reset_index(drop=True)
        dfs = (train_df, val_df, calib_df, test_df, max_age)
    print(f"  [{method}] seed {seed}: train {len(train_df)} | val {len(val_df)} | "
          f"calib {len(calib_df)} | test {len(test_df)} | max_age {max_age}")

    model, std_scale, max_age, history, device, final = train_method_seed(method, seed, dfs)

    metrics_df, intervals_df = evaluate_test(
        method, seed, model, test_df, max_age, std_scale, device
    )
    metrics_df.to_csv(seed_dir / "metrics.csv", index=False)

    if C.SAVE_INTERVALS:
        intervals_df.to_csv(seed_dir / "intervals.csv", index=False)
    if C.SAVE_MODELS:
        torch.save(model.state_dict(), seed_dir / "model_best.pt")
    with open(seed_dir / "history.pkl", "wb") as f:
        pickle.dump(history, f)

    calib_payload = {
        "method": method, "seed": int(seed), "std_scale": float(std_scale),
        "n_calib": int(len(calib_df)), "max_age": float(max_age),
        "final_samples": C.FINAL_SAMPLES,
        "calib_picp90": final["picp90"], "calib_mpiw90": final["mpiw90"],
        "calib_mae": final["mae"], "split_note": C.SPLIT_NOTE,
    }
    (seed_dir / "calibration.json").write_text(json.dumps(calib_payload, indent=2),
                                               encoding="utf-8")

    # Free GPU memory before the next seed/method.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  [{method}] seed {seed} done -> {seed_dir}")
    return metrics_df
