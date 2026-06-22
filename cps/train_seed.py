import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import cps_config as C
from model import build_multi_input_model
from cps.device import get_device
from cps.loaders import build_train_val_loaders
from cps.split_data import create_split, load_seed_split


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_one_seed(seed):
    set_all_seeds(seed)
    create_split(seed)
    dfs, _ = load_seed_split(seed)

    seed_dir = C.RUN_OUTPUT_DIR / f"seed_{seed:02d}"
    checkpoint_path = seed_dir / "model_best.pt"

    device = get_device()
    train_loader, val_loader = build_train_val_loaders(dfs["train"], dfs["val"], device)

    model = build_multi_input_model().to(device)
    criterion = nn.SmoothL1Loss()
    mae_metric = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_mae = float("inf")
    print(f"[seed {seed:02d}] Training for {C.EPOCHS} epochs on {device}")

    for epoch in range(C.EPOCHS):
        model.train()
        train_loss = 0.0

        progress = tqdm(train_loader, desc=f"seed {seed:02d} epoch {epoch + 1}/{C.EPOCHS}")
        for inputs, targets in progress:
            img = inputs["image_input"].to(device)
            sex = inputs["sex_input"].to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(img, sex)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * img.size(0)
            progress.set_postfix({"loss": loss.item()})

        train_loss /= max(1, len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                img = inputs["image_input"].to(device)
                sex = inputs["sex_input"].to(device)
                targets = targets.to(device)
                outputs = model(img, sex)
                val_loss += mae_metric(outputs, targets).item() * img.size(0)

        val_loss /= max(1, len(val_loader.dataset))
        scheduler.step(val_loss)
        print(
            f"[seed {seed:02d}] epoch {epoch + 1}/{C.EPOCHS} "
            f"train_loss={train_loss:.4f} val_mae={val_loss:.4f}"
        )

        if val_loss < best_val_mae:
            best_val_mae = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[seed {seed:02d}] saved {checkpoint_path}")

    return checkpoint_path


if __name__ == "__main__":
    for seed_value in C.SEEDS:
        train_one_seed(seed_value)
