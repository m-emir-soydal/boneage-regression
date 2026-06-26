import argparse
import torch

from data_loader import load_data, build_val_or_test_loader
from model import build_multi_input_model, backbone_img_size
from metrics import evaluate_and_save_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained bone age model on the held-out TEST set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth model weights")
    parser.add_argument("--output-name", type=str, default="run", help="Prefix for output files")
    parser.add_argument("--backbone", type=str, default="efficientnet_b3",
                        choices=["efficientnet_b3", "vit_b_16"],
                        help="Image feature backbone (must match the checkpoint)")
    args = parser.parse_args()

    print("Loading data...")
    # Re-uses the same deterministic split (RANDOM_STATE) as training,
    # so max_age matches the normalization used at train time.
    _, _, _, test_df, max_age = load_data(sample_frac=1.0)
    print(f"Test set: {len(test_df)} samples")

    test_loader = build_val_or_test_loader(test_df, img_size=backbone_img_size(args.backbone))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model weights from {args.checkpoint} (backbone={args.backbone})")
    model = build_multi_input_model(backbone=args.backbone)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    evaluate_and_save_metrics(
        model, test_loader, test_df, max_age,
        run_name=args.output_name, device=device, split="test",
    )
    print("Test evaluation completed.")

if __name__ == "__main__":
    main()
