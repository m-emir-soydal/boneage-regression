import cps_config as C
from cps.collect_predictions import collect_predictions
from cps.predict_seed import predict_one_seed
from cps.train_seed import train_one_seed


def main():
    C.RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = C.save_active_config()
    print(f"Saved active config to {config_path}")

    for seed in C.SEEDS:
        print(f"\n=== Seed {seed:02d} ===")
        train_one_seed(seed)
        predict_one_seed(seed)

    collect_predictions()


if __name__ == "__main__":
    main()
