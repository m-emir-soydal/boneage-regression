import argparse
import json
import pickle
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from config import OUTPUT_DIR, CONFIDENCES, RANDOM_STATE
from data_loader import load_data, build_datasets, build_val_or_test_loader
from model import build_multi_input_model
from metrics import evaluate_and_save_metrics
from cqr_utils import pinball_loss, predict_quantiles, calibrate

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Input Bone Age Model with PyTorch")
    parser.add_argument("--quick-test", action="store_true", help="Run a quick test with 1% of data and 2 epochs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--output-name", type=str, default="run", help="Prefix for output files")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                        help="Seed for the per-seed data split and torch init/training")
    args = parser.parse_args()

    # Seed torch and the per-seed data split (see PER_SEED_SPLITS in config).
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sample_frac = 0.01 if args.quick_test else 1.0
    epochs = 2 if args.quick_test else args.epochs
    run_name = args.output_name + f"_seed{args.seed}" + ("_quicktest" if args.quick_test else "")

    # All outputs for this run live under OUTPUT_DIR/<run_name>/
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting run: {run_name}")
    print(f"Using {sample_frac*100}% of data for {epochs} epochs.")

    print("Loading data...")
    train_df, val_df, scale_df, cal_df, test_df, max_age = load_data(
        sample_frac=sample_frac, seed=args.seed
    )
    print(f"Splits -> train {len(train_df)} | val {len(val_df)} | "
          f"scale {len(scale_df)} | cal {len(cal_df)} | test {len(test_df)}")

    print("Building PyTorch DataLoaders...")
    train_loader, val_loader = build_datasets(train_df, val_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building CQR quantile model...")
    model = build_multi_input_model()
    model = model.to(device)

    # Pinball (quantile) loss for training; same loss tracked on val for
    # checkpointing the best quantile fit.
    criterion = pinball_loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )

    best_val_loss = float('inf')
    #early_stop_patience = 5
    #early_stop_counter = 0
    history = {'loss': [], 'val_loss': []}

    checkpoint_path = None

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Training loop
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            inputs, targets = batch
            img = inputs['image_input'].to(device)
            sex = inputs['sex_input'].to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(img, sex)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * img.size(0)
            progress_bar.set_postfix({'loss': loss.item()})

        train_loss /= len(train_loader.dataset)

        # Validation loop (pinball loss)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                img = inputs['image_input'].to(device)
                sex = inputs['sex_input'].to(device)
                targets = targets.to(device)

                outputs = model(img, sex)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * img.size(0)

        val_loss /= len(val_loader.dataset)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - loss (pinball): {train_loss:.4f} - val_loss (pinball): {val_loss:.4f} - lr: {current_lr:.6f}")

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        # Model Checkpointing
        if val_loss < best_val_loss:
            new_checkpoint_path = run_dir / f"best_{run_name}_ep{epoch+1:02d}_val{val_loss:.4f}.pth"
            print(f"val_loss improved from {best_val_loss:.4f} to {val_loss:.4f}, saving model to {new_checkpoint_path}")
            best_val_loss = val_loss

            if checkpoint_path and checkpoint_path.exists():
                checkpoint_path.unlink()

            checkpoint_path = new_checkpoint_path
            torch.save(model.state_dict(), checkpoint_path)
        #     early_stop_counter = 0
        # else:
        #     early_stop_counter += 1
        #     if early_stop_counter >= early_stop_patience:
        #         print(f"Early stopping triggered after {epoch+1} epochs.")
        #         break
            
    # Load best weights for evaluation
    if checkpoint_path and checkpoint_path.exists():
        print(f"Restoring model weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

    history_path = run_dir / f"history_{run_name}.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Saved training history to {history_path}")

    # Plot training and validation loss
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        
        plt.figure(figsize=(10, 6))
        epochs_range = range(1, len(history['loss']) + 1)
        plt.plot(epochs_range, history['loss'], label='Train Loss (pinball)', marker='o', linewidth=2)
        plt.plot(epochs_range, history['val_loss'], label='Val Loss (pinball)', marker='s', linewidth=2)
        plt.title('Training and Validation Loss', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plot_path = run_dir / f"loss_plot_{run_name}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved loss plot to {plot_path}")
    except ImportError:
        print("matplotlib or seaborn not installed, skipping plot generation.")

    # ---- CQR conformal calibration on the held-out cal split ----
    print("\nCalibrating CQR intervals on the cal split...")
    cal_loader = build_val_or_test_loader(cal_df)
    cal_preds, cal_true = predict_quantiles(model, cal_loader, device)
    q_hat = calibrate(cal_preds, cal_true)

    q_hat_path = run_dir / f"q_hat_{run_name}.json"
    with open(q_hat_path, "w") as f:
        json.dump({
            "max_age": float(max_age),
            "n_cal": int(len(cal_true)),
            # store both normalized and month-scale corrections
            "q_hat_norm": {str(c): q_hat[c] for c in CONFIDENCES},
            "q_hat_months": {str(c): q_hat[c] * float(max_age) for c in CONFIDENCES},
        }, f, indent=2)
    print(f"Saved CQR corrections to {q_hat_path}")

    # Evaluate on val (eval transforms, no shuffle) and test.
    val_eval_loader = build_val_or_test_loader(val_df)
    evaluate_and_save_metrics(model, val_eval_loader, val_df, max_age, q_hat,
                              run_name=run_name, split="val", device=device)

    print("\nRunning evaluation on test set...")
    test_loader = build_val_or_test_loader(test_df)
    evaluate_and_save_metrics(model, test_loader, test_df, max_age, q_hat,
                              run_name=run_name, split="test", device=device)

    print("Training process completed.")

if __name__ == "__main__":
    main()
