"""UQ-aware training loop for Bayesian Neural Network (BNN).

Trains the Bayesian multi-input regression model using the Evidence Lower Bound (ELBO)
loss (SmoothL1Loss + KL divergence). Checkpoint selection is UQ-aware: after each
epoch it runs Monte Carlo sampling on the held-out ``calib`` split, fits a post-hoc
temperature (see ``common.calibration``), and scores the epoch by the quality of
the calibrated prediction intervals (MPIW@90) traded off against point accuracy (MAE).

Example:
    python -m UQ.bnn.train_bnn \\
        --output-name bnntrain --dropout 0.5 --prior-sigma 1.0 --epochs 50 \\
        --select-by combo --mc-eval-samples 10 --mc-eval-every 1
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# Make the project root and the UQ package importable whether run as a module
# or as a plain script (mirrors train_mc_dropout.py).
_THIS = Path(__file__).resolve()
_UQ_DIR = _THIS.parent.parent
_PROJECT_ROOT = _UQ_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_UQ_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from common.paths import method_results_dir, ensure_project_on_path
from common.inference import mc_forward_passes, denormalize
from common.uq_metrics import picp, mpiw
from common.calibration import fit_temperature, save_calibration
from bnn.bnn_model import build_bnn_model, enable_bnn_sampling

ensure_project_on_path()

from config import OUTPUT_DIR
from data_loader import load_data, build_datasets, build_eval_loader
from train import select_device, move_training_to_cpu

METHOD_NAME = "bnn"

# z for a 90% two-sided Gaussian interval (matches common.inference.Z_SCORES).
Z90 = 1.645


def mc_eval_calib(model, calib_loader, y_true_calib, max_age, device, samples):
    """Run Monte Carlo sampling on the calib split and return calibrated UQ stats.

    Returns a dict with the fitted temperature plus the calibrated PICP@90,
    MPIW@90 and the MC-mean MAE (all in months). Restores ``model.train()``
    before returning so the caller can keep training.
    """
    enable_bnn_sampling(model)
    preds_norm, _ = mc_forward_passes(model, calib_loader, device, samples)
    model.train()

    pred_mean = denormalize(preds_norm.mean(axis=0), max_age)
    pred_std = denormalize(preds_norm.std(axis=0), max_age)

    scale = fit_temperature(y_true_calib, pred_mean, pred_std)
    std_cal = pred_std * scale
    lower = pred_mean - Z90 * std_cal
    upper = pred_mean + Z90 * std_cal

    return {
        "std_scale": scale,
        "picp90": picp(y_true_calib, lower, upper),
        "mpiw90": mpiw(lower, upper),
        "mae": float(np.mean(np.abs(pred_mean - y_true_calib))),
    }


def selection_score(select_by, val_mae, uq, mae_weight):
    """Lower-is-better score for checkpoint selection."""
    if select_by == "val_mae":
        return val_mae
    if uq is None:
        return None
    if select_by == "picp90":
        return abs(uq["picp90"] - 0.90)
    if select_by == "mpiw90":
        return uq["mpiw90"]
    if select_by == "combo":
        return uq["mpiw90"] + mae_weight * uq["mae"]
    raise ValueError(f"Unknown --select-by '{select_by}'")


def main():
    parser = argparse.ArgumentParser(
        description="UQ-aware BNN training for the bone age model"
    )
    parser.add_argument("--quick-test", action="store_true",
                        help="Run with 1%% of data and 2 epochs for a smoke test")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-name", type=str, default="bnn",
                        help="Prefix for output files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate for the model head")
    parser.add_argument("--prior-sigma", type=float, default=1.0,
                        help="Standard deviation for the Gaussian prior of variational weights")
    parser.add_argument("--kl-weight", type=float, default=1.0,
                        help="Multiplier for the KL divergence term in the ELBO loss")
    parser.add_argument("--select-by", type=str, default="combo",
                        choices=["val_mae", "picp90", "mpiw90", "combo"],
                        help="Checkpoint selection criterion (lower is better)")
    parser.add_argument("--mae-weight", type=float, default=0.01,
                        help="Weight on calib MAE (months) in the 'combo' score")
    parser.add_argument("--mc-eval-samples", type=int, default=10,
                        help="MC forward passes for the per-epoch calib eval")
    parser.add_argument("--mc-eval-every", type=int, default=1,
                        help="Run the MC calib eval every k epochs")
    parser.add_argument("--final-samples", type=int, default=30,
                        help="MC passes for the final calibration fit on calib")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sample_frac = 0.01 if args.quick_test else 1.0
    epochs = 2 if args.quick_test else args.epochs
    run_name = args.output_name + f"_seed{args.seed}" + ("_quicktest" if args.quick_test else "")
    if args.select_by != "val_mae" and (args.epochs % args.mc_eval_every != 0):
        print("  note: last epoch may skip MC eval; consider --mc-eval-every "
              "that divides --epochs so the final epoch is scored.")

    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting UQ-aware BNN run: {run_name}")
    print(f"Using {sample_frac*100}% of data for {epochs} epochs "
          f"(dropout={args.dropout}, prior_sigma={args.prior_sigma}, kl_weight={args.kl_weight}, select-by={args.select_by}).")

    print("Loading data...")
    train_df, val_df, calib_df, test_df, max_age = load_data(sample_frac=sample_frac)
    print(f"Splits -> train {len(train_df)} | val {len(val_df)} | "
          f"calibration {len(calib_df)} | test {len(test_df)}")

    print("Building DataLoaders...")
    train_loader, val_loader = build_datasets(train_df, val_df)
    calib_loader = build_eval_loader(calib_df)
    y_true_calib = calib_df["boneage_norm"].values * max_age
    num_train_samples = len(train_loader.dataset)

    print("Building model...")
    model = build_bnn_model(dropout=args.dropout, prior_sigma=args.prior_sigma)
    criterion = nn.SmoothL1Loss()
    device, model = select_device(model, train_loader, criterion)
    print(f"Using device: {device}")

    mae_metric = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    best_score = float('inf')
    best_uq = None
    history = {'loss': [], 'nll_loss': [], 'kl_loss': [], 'val_mae': [], 'calib_picp90': [],
               'calib_mpiw90': [], 'calib_std_scale': [], 'score': []}
    checkpoint_path = None

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_nll = 0.0
        train_kl = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            inputs, targets = batch
            img = inputs['image_input'].to(device)
            sex = inputs['sex_input'].to(device)
            targets = targets.to(device)

            try:
                optimizer.zero_grad()
                outputs = model(img, sex)
                nll = criterion(outputs, targets)
                kl = model.kl_loss()
                # ELBO loss = NLL + kl_weight * (KL / N)
                loss = nll + args.kl_weight * (kl / num_train_samples)
                loss.backward()
                optimizer.step()
            except RuntimeError as exc:
                if device.type != "cuda" or "out of memory" not in str(exc).lower():
                    raise
                print("  warning: GPU OOM during training; switching to CPU.")
                device = torch.device("cpu")
                model, optimizer = move_training_to_cpu(model, optimizer)
                img = inputs['image_input'].to(device)
                sex = inputs['sex_input'].to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(img, sex)
                nll = criterion(outputs, targets)
                kl = model.kl_loss()
                loss = nll + args.kl_weight * (kl / num_train_samples)
                loss.backward()
                optimizer.step()

            batch_size = img.size(0)
            train_loss += loss.item() * batch_size
            train_nll += nll.item() * batch_size
            train_kl += kl.item() * batch_size
            progress_bar.set_postfix({'loss': loss.item(), 'nll': nll.item(), 'kl': kl.item()})

        train_loss /= num_train_samples
        train_nll /= num_train_samples
        train_kl /= num_train_samples

        # Standard (deterministic mean) validation MAE.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                img = inputs['image_input'].to(device)
                sex = inputs['sex_input'].to(device)
                targets = targets.to(device)
                outputs = model(img, sex)
                val_loss += mae_metric(outputs, targets).item() * img.size(0)
        val_loss /= len(val_loader.dataset)
        val_mae = val_loss * max_age  # report in months

        # UQ-aware calib eval (BNN MC sampling + post-hoc temperature).
        run_mc = (args.select_by != "val_mae") and ((epoch + 1) % args.mc_eval_every == 0)
        uq = None
        if run_mc:
            uq = mc_eval_calib(model, calib_loader, y_true_calib, max_age,
                               device, args.mc_eval_samples)

        score = selection_score(args.select_by, val_mae, uq, args.mae_weight)

        current_lr = optimizer.param_groups[0]['lr']
        msg = (f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} (nll:{train_nll:.4f}, kl:{train_kl:.4f}) - "
               f"val_mae: {val_mae:.3f} mo - lr: {current_lr:.6f}")
        if uq is not None:
            msg += (f" | calib PICP@90: {uq['picp90']*100:.1f}% "
                    f"MPIW@90: {uq['mpiw90']:.2f} mo (s={uq['std_scale']:.3f})")
        if score is not None:
            msg += f" | score: {score:.4f}"
        print(msg)

        history['loss'].append(train_loss)
        history['nll_loss'].append(train_nll)
        history['kl_loss'].append(train_kl)
        history['val_mae'].append(val_mae)
        history['calib_picp90'].append(uq['picp90'] if uq else None)
        history['calib_mpiw90'].append(uq['mpiw90'] if uq else None)
        history['calib_std_scale'].append(uq['std_scale'] if uq else None)
        history['score'].append(score)

        scheduler.step(val_loss)

        if score is not None and score < best_score:
            new_path = run_dir / f"best_{run_name}_ep{epoch+1:02d}_score{score:.4f}.pth"
            print(f"  score improved {best_score:.4f} -> {score:.4f}, saving {new_path}")
            best_score = score
            best_uq = uq
            if checkpoint_path and checkpoint_path.exists():
                checkpoint_path.unlink()
            checkpoint_path = new_path
            torch.save(model.state_dict(), checkpoint_path)

    if checkpoint_path and checkpoint_path.exists():
        print(f"Restoring best weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

    history_path = run_dir / f"history_{run_name}.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Saved training history to {history_path}")

    # Definitive calibration on calib with more MC passes, saved where
    # run_bnn.py --calibration auto can find it.
    print(f"Fitting final calibration on calib ({args.final_samples} MC passes)...")
    final_uq = mc_eval_calib(model, calib_loader, y_true_calib, max_age,
                             device, args.final_samples)
    cal_dir = method_results_dir(METHOD_NAME)
    cal_path = save_calibration(
        cal_dir, final_uq["std_scale"], split="calib", n=len(calib_df),
        extra={
            "method": METHOD_NAME,
            "source": "uq_train",
            "samples": args.final_samples,
            "checkpoint": checkpoint_path.name if checkpoint_path else None,
            "run_name": run_name,
            "dropout": args.dropout,
            "prior_sigma": args.prior_sigma,
            "kl_weight": args.kl_weight,
        },
    )
    print(f"Final calib temperature {final_uq['std_scale']:.4f} -> {cal_path}")
    print(f"Final calib PICP@90: {final_uq['picp90']*100:.1f}% | "
          f"MPIW@90: {final_uq['mpiw90']:.2f} mo | MAE: {final_uq['mae']:.2f} mo")

    print("\nUQ-aware BNN training completed.")
    print("Next: evaluate the test split with the fitted calibration:")
    ckpt = checkpoint_path.name if checkpoint_path else "<checkpoint>.pth"
    print(f"  python -m UQ.bnn.run_bnn "
          f"--checkpoint {run_dir / ckpt} --split test "
          f"--samples {args.final_samples} --calibration auto")


if __name__ == "__main__":
    main()
