import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from config import BATCH_SIZE, IMG_SIZE
from data_loader import BoneAgeDataset


def _loader_options(device):
    use_cuda = device.type == "cuda"
    return {
        "num_workers": 8 if use_cuda else 0,
        "pin_memory": use_cuda,
    }


def build_train_val_loaders(train_df, val_df, device, batch_size=BATCH_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    opts = _loader_options(device)
    train_loader = DataLoader(
        BoneAgeDataset(train_df, transform=train_transform),
        batch_size=batch_size,
        shuffle=True,
        **opts,
    )
    val_loader = DataLoader(
        BoneAgeDataset(val_df, transform=eval_transform),
        batch_size=batch_size,
        shuffle=False,
        **opts,
    )
    return train_loader, val_loader


def build_eval_loader(df, device, batch_size=BATCH_SIZE):
    eval_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return DataLoader(
        BoneAgeDataset(df, transform=eval_transform),
        batch_size=batch_size,
        shuffle=False,
        **_loader_options(device),
    )
