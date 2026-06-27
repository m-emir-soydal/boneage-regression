import argparse
import torch

from config import RANDOM_STATE
from data_loader import load_data, build_val_or_test_loader
from model import build_multi_input_model
from metrics import evaluate_and_save_metrics
from cqr_utils import predict_quantiles, calibrate

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CQR bone age model on the held-out TEST set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth model weights")
    parser.add_argument("--output-name", type=str, default="run", help="Prefix for output files")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                        help="Seed used for the per-seed split (must match training)")
    args = parser.parse_args()

    print("Loading data...")
    # Re-use the same per-seed split as training so max_age and the cal split match.
    _, _, _, cal_df, test_df, max_age = load_data(sample_frac=1.0, seed=args.seed)
    print(f"Cal set: {len(cal_df)} | Test set: {len(test_df)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model weights from {args.checkpoint}")
    model = build_multi_input_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    # Recompute the CQR correction on the cal split for this checkpoint.
    print("Calibrating CQR intervals on the cal split...")
    cal_loader = build_val_or_test_loader(cal_df)
    cal_preds, cal_true = predict_quantiles(model, cal_loader, device)
    q_hat = calibrate(cal_preds, cal_true)

    test_loader = build_val_or_test_loader(test_df)
    evaluate_and_save_metrics(
        model, test_loader, test_df, max_age, q_hat,
        run_name=args.output_name, device=device, split="test",
    )
    print("Test evaluation completed.")

if __name__ == "__main__":
    main()
