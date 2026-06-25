"""Train the heteroscedastic regression model for UQ.

Sibling of the base ``train.py`` that trains a model with the *same* backbone
but two output heads (mean + log-variance) optimized with a Gaussian negative
log-likelihood. The result is a single-pass UQ model that emits per-sample
``mean +/- z * sigma(x)`` intervals, comparable on the same PICP/MPIW metrics
as the MC Dropout method.

Why a sibling script (not a flag on ``train.py``):
- Keeps the production point-regression pipeline (``train.py`` / ``model.py``)
  unchanged so the UQ layer is fully removable.
- Lets checkpoint selection optionally optimize a UQ-aware criterion
  (calibrated MPIW@90) without changing how the base trainer behaves.

Example:
    python -m UQ.heteroscedastic.train_hetero \\
        --output-name hetero --seed 42 --epochs 50

    # smoke test (1% data, 2 epochs)
    python -m UQ.heteroscedastic.train_hetero --quick-test

After training, fit calibration on the calib split and evaluate the test
split (see UQ/heteroscedastic/run_hetero.py).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# Make the project root and the UQ package importable whether this is run as
# a module or as a plain script. Mirrors run_mc_dropout.py.
_THIS = Path(__file__).resolve()
_UQ_DIR = _THIS.parent.parent
_PROJECT_ROOT = _UQ_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_UQ_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common.calibration import fit_temperature, save_calibration  # noqa: E402
from common.inference import denormalize  # noqa: E402
from common.paths import ensure_project_on_path, method_results_dir  # noqa: E402
from common.uq_metrics import mpiw, picp  # noqa: E402

ensure_project_on_path()

from config import OUTPUT_DIR  # noqa: E402
from data_loader import build_datasets, build_eval_loader, load_data  # noqa: E402
from train import move_training_to_cpu  # noqa: E402

from .hetero_model import build_hetero_model  # noqa: E402
from .losses import clamp_log_var, hetero_loss  # noqa: E402

METHOD_NAME = "heteroscedastic"

# z for a 90% two-sided Gaussian interval (matches common.inference.Z_SCORES).
Z90 = 1.645

# Headroom required before we attempt to put the model + Adam states on GPU.
# Mirrors train.MIN_CUDA_FREE_BYTES so behavior is consistent across trainers.
MIN_CUDA_FREE_BYTES = 4 * 1024 ** 3


def _select_device_hetero(model, train_loader, mae_weight):
    """Probe a single CUDA training step; fall back to CPU on OOM.

    Mirrors ``train.select_device`` but invokes the heteroscedastic forward
    (which returns ``(mean, log_var)``) so the probe actually exercises the
    real model + loss path.
    
    This function prevents GPU Out Of Memory (OOM) failures early in execution
    by verifying that the GPU has enough free memory (at least 4 GiB) and running
    a single test forward/backward/optimizer step. If any part of this fails
    due to OOM, it defaults to CPU training.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu"), model

    torch.backends.cudnn.enabled = False
    try:
        free, _total = torch.cuda.mem_get_info()
        if free < MIN_CUDA_FREE_BYTES:
            free_gb = free / (1024 ** 3)
            print(f"  warning: GPU has only {free_gb:.1f} GiB free; using CPU.")
            return torch.device("cpu"), model

        # Move to CUDA and do a single test optimization step
        model = model.cuda().train()
        probe_opt = optim.Adam(model.parameters(), lr=1e-4)
        inputs, targets = next(iter(train_loader))
        img = inputs["image_input"].cuda()
        sex = inputs["sex_input"].cuda()
        targets = targets.cuda()

        probe_opt.zero_grad()
        mean, log_var = model(img, sex)
        loss = hetero_loss(targets, mean, log_var, mae_weight=mae_weight)
        loss.backward()
        probe_opt.step()

        # Clean up temporary tensors to free GPU memory
        del probe_opt, mean, log_var, loss, img, sex, targets, inputs
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return torch.device("cuda"), model
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print("  warning: GPU ran out of memory during probe; using CPU.")
        torch.cuda.empty_cache()
        return torch.device("cpu"), model.cpu()


@torch.no_grad()
def _eval_val_mae(model, val_loader, device, max_age):
    """Return validation MAE in **months** from a single deterministic pass.

    Uses only the predicted mean head (`mean`), completely ignoring the
    predicted uncertainty log-variance, to calculate standard L1 error.
    """
    model.eval()
    total = 0.0
    n = 0
    for batch in val_loader:
        inputs, targets = batch
        img = inputs["image_input"].to(device)
        sex = inputs["sex_input"].to(device)
        targets = targets.to(device)
        mean, _ = model(img, sex)
        total += torch.abs(mean - targets).sum().item()
        n += img.size(0)
    return (total / max(n, 1)) * max_age


@torch.no_grad()
def _hetero_predict(model, loader, device):
    """Run one forward pass over ``loader`` and return ``(mean, std)`` arrays.

    Both arrays are in **normalized** target space; denormalize with
    ``common.inference.denormalize`` before computing metrics in months.
    
    Unlike Monte Carlo Dropout which requires running multiple stochastic passes,
    the heteroscedastic method produces the mean and variance in a single,
    deterministic forward pass, which makes inference highly efficient.
    """
    model.eval()
    means = []
    stds = []
    for batch in loader:
        inputs, _ = batch
        img = inputs["image_input"].to(device)
        sex = inputs["sex_input"].to(device)
        
        # 1. Forward pass extracts mean and log-variance
        mean, log_var = model(img, sex)
        
        # 2. Clamp log-variance to prevent numerical explosion
        log_var = clamp_log_var(log_var)
        
        # 3. Convert log-variance back to standard deviation: std = exp(0.5 * log_var)
        std = torch.exp(0.5 * log_var)
        
        means.append(mean.cpu().numpy().reshape(-1))
        stds.append(std.cpu().numpy().reshape(-1))
    return np.concatenate(means), np.concatenate(stds)


def hetero_eval_calib(model, calib_loader, y_true_calib, max_age, device):
    """Run hetero inference on calib, fit temperature, return calibrated stats.

    Returns a dict with the fitted temperature plus the calibrated PICP@90,
    MPIW@90 and MAE (all in months).
    
    Calibration ensures that the predicted uncertainty actually corresponds to
    the empirical error rate (i.e. a 90% prediction interval contains exactly 90%
    of the true targets).
    """
    # 1. Obtain normalized point predictions (mean) and uncertainty (std)
    mean_norm, std_norm = _hetero_predict(model, calib_loader, device)
    
    # 2. De-normalize to months
    pred_mean = denormalize(mean_norm, max_age)
    pred_std = denormalize(std_norm, max_age)

    # 3. Solve for standard deviation multiplier (temperature scale) `s`
    scale = fit_temperature(y_true_calib, pred_mean, pred_std)
    std_cal = pred_std * scale
    
    # 4. Form prediction intervals
    lower = pred_mean - Z90 * std_cal
    upper = pred_mean + Z90 * std_cal

    # 5. Measure coverage (PICP), mean interval width (MPIW), and MAE
    return {
        "std_scale": scale,
        "picp90": picp(y_true_calib, lower, upper),
        "mpiw90": mpiw(lower, upper),
        "mae": float(np.mean(np.abs(pred_mean - y_true_calib))),
    }


def selection_score(select_by, val_mae, uq, mae_weight):
    """Lower-is-better score for checkpoint selection.

    Calculates the scoring metric to choose the best epoch checkpoint.
    
    Args:
        select_by (str): Metric to use for checkpoint selection (e.g. combo).
        val_mae (float): Mean Absolute Error on validation set.
        uq (dict): Uncertainty metrics from hetero_eval_calib (or None if skipped).
        mae_weight (float): Multiplier for point prediction error in combined score.
    """
    if select_by == "val_mae":
        return val_mae
    if uq is None:
        return None
    if select_by == "picp90":
        # Target: PICP = 90% (minimize distance to 0.90)
        return abs(uq["picp90"] - 0.90)
    if select_by == "mpiw90":
        # Target: Minimize mean interval width (narrower intervals = higher precision)
        return uq["mpiw90"]
    if select_by == "combo":
        # Combined score: Minimize calibrated width (uq["mpiw90"]) while penalizing
        # point accuracy degradation (uq["mae"]) to ensure model remains accurate.
        return uq["mpiw90"] + mae_weight * uq["mae"]
    raise ValueError(f"Unknown --select-by '{select_by}'")


def main():
    parser = argparse.ArgumentParser(
        description="Heteroscedastic regression training for bone age UQ"
    )
    parser.add_argument("--quick-test", action="store_true",
                        help="Run with 1%% of data and 2 epochs for a smoke test")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-name", type=str, default="hetero",
                        help="Prefix for output files / checkpoint directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate for the shared head")
    parser.add_argument("--mae-weight", type=float, default=0.01,
                        help="SmoothL1 weight on the mean head added to NLL "
                             "(0 disables); also used as the MAE weight in "
                             "the 'combo' selection score")
    parser.add_argument("--select-by", type=str, default="val_mae",
                        choices=["val_mae", "picp90", "mpiw90", "combo"],
                        help="Checkpoint selection criterion (lower is better)")
    parser.add_argument("--calib-eval-every", type=int, default=1,
                        help="Run the calib eval every k epochs (only used "
                             "when --select-by != val_mae)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sample_frac = 0.01 if args.quick_test else 1.0
    epochs = 2 if args.quick_test else args.epochs
    run_name = args.output_name + f"_seed{args.seed}" + ("_quicktest" if args.quick_test else "")

    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting heteroscedastic run: {run_name}")
    print(f"Using {sample_frac*100}% of data for {epochs} epochs "
          f"(dropout={args.dropout}, select-by={args.select_by}, "
          f"mae_weight={args.mae_weight}).")

    print("Loading data...")
    train_df, val_df, calib_df, test_df, max_age = load_data(sample_frac=sample_frac)
    print(f"Splits -> train {len(train_df)} | val {len(val_df)} | "
          f"calibration {len(calib_df)} | test {len(test_df)}")

    print("Building DataLoaders...")
    train_loader, val_loader = build_datasets(train_df, val_df)
    calib_loader = build_eval_loader(calib_df)
    y_true_calib = calib_df["boneage_norm"].values * max_age

    print("Building model...")
    model = build_hetero_model(dropout=args.dropout)
    device, model = _select_device_hetero(model, train_loader, args.mae_weight)
    print(f"Using device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_score = float("inf")
    history = {
        "loss": [], "val_mae": [],
        "calib_picp90": [], "calib_mpiw90": [],
        "calib_std_scale": [], "score": [],
    }
    checkpoint_path = None

    print("Starting training...")
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            inputs, targets = batch
            img = inputs["image_input"].to(device)
            sex = inputs["sex_input"].to(device)
            targets = targets.to(device)

            try:
                # 1. Forward pass outputs both predicted mean and log-variance
                optimizer.zero_grad()
                mean, log_var = model(img, sex)
                
                # 2. Compute the heteroscedastic loss (NLL + optional Smooth L1 on the mean)
                loss = hetero_loss(targets, mean, log_var,
                                   mae_weight=args.mae_weight)
                
                # 3. Backpropagation and optimization step
                loss.backward()
                optimizer.step()
            except RuntimeError as exc:
                # OOM recovery: fall back to CPU if GPU runs out of memory mid-epoch
                if device.type != "cuda" or "out of memory" not in str(exc).lower():
                    raise
                print("  warning: GPU OOM during training; switching to CPU.")
                device = torch.device("cpu")
                model, optimizer = move_training_to_cpu(model, optimizer)
                img = inputs["image_input"].to(device)
                sex = inputs["sex_input"].to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                mean, log_var = model(img, sex)
                loss = hetero_loss(targets, mean, log_var,
                                   mae_weight=args.mae_weight)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * img.size(0)
            progress_bar.set_postfix({"loss": loss.item()})

        train_loss /= len(train_loader.dataset)

        # --- STANDARD VALIDATION PHASE ---
        # Run standard validation to check point prediction MAE (dropout-off)
        val_mae = _eval_val_mae(model, val_loader, device, max_age)

        # --- UQ-AWARE EVALUATION PHASE ---
        # If selection criterion is UQ-aware, evaluate the prediction intervals
        # on the calibration set every `calib_eval_every` epochs.
        run_calib = (args.select_by != "val_mae") and (
            (epoch + 1) % args.calib_eval_every == 0
        )
        uq = None
        if run_calib:
            uq = hetero_eval_calib(model, calib_loader, y_true_calib,
                                   max_age, device)

        # Calculate selection score (lower is better)
        score = selection_score(args.select_by, val_mae, uq, args.mae_weight)

        current_lr = optimizer.param_groups[0]["lr"]
        msg = (f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - "
               f"val_mae: {val_mae:.3f} mo - lr: {current_lr:.6f}")
        if uq is not None:
            msg += (f" | calib PICP@90: {uq['picp90']*100:.1f}% "
                    f"MPIW@90: {uq['mpiw90']:.2f} mo (s={uq['std_scale']:.3f})")
        if score is not None:
            msg += f" | score: {score:.4f}"
        print(msg)

        history["loss"].append(train_loss)
        history["val_mae"].append(val_mae)
        history["calib_picp90"].append(uq["picp90"] if uq else None)
        history["calib_mpiw90"].append(uq["mpiw90"] if uq else None)
        history["calib_std_scale"].append(uq["std_scale"] if uq else None)
        history["score"].append(score)

        # Step the learning rate scheduler based on point-prediction validation accuracy
        scheduler.step(val_mae)

        # Save checkpoint if selection score improved
        if score is not None and score < best_score:
            new_path = run_dir / f"best_{run_name}_ep{epoch+1:02d}_score{score:.4f}.pth"
            print(f"  score improved {best_score:.4f} -> {score:.4f}, saving {new_path}")
            best_score = score
            if checkpoint_path and checkpoint_path.exists():
                checkpoint_path.unlink()
            checkpoint_path = new_path
            torch.save(model.state_dict(), checkpoint_path)

    # Restore the best epoch's weights for final calibration
    if checkpoint_path and checkpoint_path.exists():
        print(f"Restoring best weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

    history_path = run_dir / f"history_{run_name}.pkl"
    with open(history_path, "wb") as f:
        pickle.dump(history, f)
    print(f"Saved training history to {history_path}")

    # --- FINAL TEMPERATURE CALIBRATION ---
    # Fit the definitive temperature calibration multiplier on the calibration split.
    # This scale factor adjusts the raw predicted log-variance outputs to align
    # interval coverage with nominal expectations (e.g. 90% or 95%).
    print("Fitting final calibration on calib split...")
    final_uq = hetero_eval_calib(model, calib_loader, y_true_calib,
                                 max_age, device)
    
    # Save the calibration multiplier to a JSON file so run_hetero.py can reuse it
    cal_dir = method_results_dir(METHOD_NAME)
    cal_path = save_calibration(
        cal_dir, final_uq["std_scale"], split="calib", n=len(calib_df),
        extra={
            "method": METHOD_NAME,
            "source": "hetero_train",
            "checkpoint": checkpoint_path.name if checkpoint_path else None,
            "run_name": run_name,
            "dropout": args.dropout,
            "mae_weight": args.mae_weight,
        },
    )
    print(f"Final calib temperature {final_uq['std_scale']:.4f} -> {cal_path}")
    print(f"Final calib PICP@90: {final_uq['picp90']*100:.1f}% | "
          f"MPIW@90: {final_uq['mpiw90']:.2f} mo | MAE: {final_uq['mae']:.2f} mo")

    print("\nHeteroscedastic training completed.")
    print("Next: evaluate the test split with the fitted calibration:")
    ckpt = checkpoint_path.name if checkpoint_path else "<checkpoint>.pth"
    print(f"  python -m UQ.heteroscedastic.run_hetero "
          f"--checkpoint {run_dir / ckpt} --split test --calibration auto")


if __name__ == "__main__":
    main()
