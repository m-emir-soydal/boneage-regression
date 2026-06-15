import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from sklearn.model_selection import train_test_split
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, RANDOM_STATE

def parse_multi(filepath, sex, age):
    """
    Parses the image, normalizes, and packages features for the multi-input model.
    """
    # Load & preprocess image
    img = tf.io.read_file(filepath)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = effnet_preprocess(img)

    # Cast numeric inputs
    sex = tf.cast(sex, tf.float32)
    age = tf.cast(age, tf.float32)

    return (
        {'image_input': img, 'sex_input': tf.expand_dims(sex, -1)},
        tf.expand_dims(age, -1)
    )

def load_data(sample_frac=1.0):
    """
    Loads train.csv, handles paths and normalizations, and returns
    dataframes and max_age for denormalization later.
    
    Args:
        sample_frac: Fraction of data to use (e.g. 0.01 for quick test)
    """
    csv_path = DATA_DIR / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found at: {csv_path}")
        
    train_df = pd.read_csv(csv_path)
    
    if sample_frac < 1.0:
        train_df = train_df.sample(frac=sample_frac, random_state=RANDOM_STATE)
    
    # Add the image file path column
    train_df['filepath'] = train_df['id'].astype(str).apply(lambda x: str(DATA_DIR / f"{x}.png"))

    # Split off 20% for validation (stratified by sex)
    train_split, val_split = train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_df['male']
    )

    # Normalize target column
    max_age = train_split['boneage'].max()
    train_split = train_split.copy()
    val_split = val_split.copy()
    
    train_split['boneage_norm'] = train_split['boneage'] / max_age
    val_split['boneage_norm'] = val_split['boneage'] / max_age
    
    train_split['sex_norm'] = train_split['male'].astype('float32')
    val_split['sex_norm'] = val_split['male'].astype('float32')

    return train_split, val_split, max_age

def build_datasets(train_df, val_df, batch_size=BATCH_SIZE):
    """
    Converts pandas DataFrames into tf.data.Dataset objects for training.
    """
    ds_train = tf.data.Dataset.from_tensor_slices((
        train_df['filepath'].values,
        train_df['sex_norm'].values,
        train_df['boneage_norm'].values
    ))
    ds_train = (
        ds_train
          .map(parse_multi, num_parallel_calls=tf.data.AUTOTUNE)
          .cache()
          .shuffle(1024, seed=RANDOM_STATE)
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE)
    )

    ds_val = tf.data.Dataset.from_tensor_slices((
        val_df['filepath'].values,
        val_df['sex_norm'].values,
        val_df['boneage_norm'].values
    ))
    ds_val = (
        ds_val
          .map(parse_multi, num_parallel_calls=tf.data.AUTOTUNE)
          .cache()
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE)
    )
    
    return ds_train, ds_val
