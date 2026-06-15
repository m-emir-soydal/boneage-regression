import argparse
import pickle
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from config import OUTPUT_DIR
from data_loader import load_data, build_datasets
from model import build_multi_input_model
from metrics import evaluate_and_save_metrics

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Input Bone Age Model")
    parser.add_argument("--quick-test", action="store_true", help="Run a quick test with 1% of data and 2 epochs")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs to train")
    parser.add_argument("--output-name", type=str, default="run", help="Prefix for output files")
    args = parser.parse_args()

    # Determine execution parameters
    sample_frac = 0.01 if args.quick_test else 1.0
    epochs = 2 if args.quick_test else args.epochs
    run_name = args.output_name + ("_quicktest" if args.quick_test else "")

    print(f"Starting run: {run_name}")
    print(f"Using {sample_frac*100}% of data for {epochs} epochs.")

    # 1. Load and process data
    print("Loading data...")
    train_df, val_df, max_age = load_data(sample_frac=sample_frac)
    
    print("Building tf.data Datasets...")
    ds_train, ds_val = build_datasets(train_df, val_df)

    # 2. Build model
    print("Building model...")
    model = build_multi_input_model()
    
    # 3. Setup Callbacks
    checkpoint_path = OUTPUT_DIR / f"best_{run_name}_{{epoch:02d}}-{{val_mean_absolute_error:.3f}}.h5"
    checkpoint = ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_mean_absolute_error",
        save_best_only=True,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor="val_mean_absolute_error",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_mean_absolute_error',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # 4. Train Model
    print("Starting training...")
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=[early_stop, checkpoint, lr_scheduler],
        verbose=2
    )

    # Save training history
    history_path = OUTPUT_DIR / f"history_{run_name}.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Saved training history to {history_path}")

    # 5. Evaluate and save metrics
    evaluate_and_save_metrics(model, ds_val, val_df, max_age, run_name=run_name)
    print("Training process completed.")

if __name__ == "__main__":
    main()
