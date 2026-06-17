import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, RANDOM_STATE

class BoneAgeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
        self.filepaths = df['filepath'].values
        self.sex_norm = df['sex_norm'].values.astype('float32')
        self.boneage_norm = df['boneage_norm'].values.astype('float32')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        img = Image.open(path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        sex = torch.tensor([self.sex_norm[idx]], dtype=torch.float32)
        age = torch.tensor([self.boneage_norm[idx]], dtype=torch.float32)
        
        return {'image_input': img, 'sex_input': sex}, age


def _normalize_male(series):
    """Coerce True/False, 'True'/'FALSE', 1/0 -> bool."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )


def _read_source(csv_path, id_col, age_col, male_col, img_dir):
    """Read a source csv, unify schema to id,boneage,male,filepath; drop missing imgs."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    out = pd.DataFrame({
        "id": df[id_col].astype(int),
        "boneage": df[age_col].astype(int),
        "male": _normalize_male(df[male_col]),
    })
    out["filepath"] = out["id"].apply(
        lambda x: str(DATA_DIR / img_dir / f"{x}.png")
    )

    exists = out["filepath"].apply(lambda p: Path(p).exists())
    missing = int((~exists).sum())
    if missing:
        print(f"  warning: {missing} rows dropped (image not found) from {csv_path.name}")
    return out[exists].reset_index(drop=True)


def _add_norm_cols(df, max_age):
    df = df.copy()
    df["boneage_norm"] = df["boneage"] / max_age
    df["sex_norm"] = df["male"].astype("float32")
    return df


def load_data(sample_frac=1.0):
    """
    Loads the source train.csv + val.csv, unifies their schema, and splits
    the source training data 50/25/25 into train/val/calibration. The source
    validation set is used as the held-out TEST set.
    """
    train_full = _read_source(DATA_DIR / "train.csv", "id", "boneage", "male", "train")
    test_df = _read_source(
        DATA_DIR / "val.csv", "Image ID", "Bone Age (months)", "male", "val"
    )

    if sample_frac < 1.0:
        train_full = train_full.sample(frac=sample_frac, random_state=RANDOM_STATE)

    # 50 / 25 / 25 split, stratified by sex.
    train_df, temp_df = train_test_split(
        train_full,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=train_full["male"],
    )
    val_df, calib_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_df["male"],
    )

    # Normalize targets against the training split only.
    max_age = train_df["boneage"].max()
    train_df = _add_norm_cols(train_df, max_age)
    val_df = _add_norm_cols(val_df, max_age)
    calib_df = _add_norm_cols(calib_df, max_age)
    test_df = _add_norm_cols(test_df, max_age)

    return train_df, val_df, calib_df, test_df, max_age

def build_datasets(train_df, val_df, batch_size=BATCH_SIZE):
    """
    Converts pandas DataFrames into torch DataLoaders.
    """
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_ds = BoneAgeDataset(train_df, transform=train_transform)
    val_ds = BoneAgeDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=8, pin_memory=True)
                            
    return train_loader, val_loader
