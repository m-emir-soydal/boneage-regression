"""Run the fair BNN / MC-Dropout pipeline across all configured seeds, then
build the comparison report against the conformal methods.

Usage
-----
Full run (10 seeds, 50 epochs — needs a GPU)::

    python -m UQ.pipeline.run_all

Quick end-to-end smoke test (override the config without editing it)::

    python -m UQ.pipeline.run_all --debug
    python -m UQ.pipeline.run_all --seeds 0 1 --methods mc_dropout --epochs 2 --data-fraction 0.01

Re-build only the tables/plots from existing per-seed metrics::

    python -m UQ.pipeline.run_all --report-only
"""
from __future__ import annotations

import argparse
import time

from . import pconfig as C


def _apply_overrides(args):
    if args.debug:
        C.SEEDS = C.DEBUG_SEEDS
        C.DATA_FRACTION = C.DEBUG_DATA_FRACTION
        C.EPOCHS = C.DEBUG_EPOCHS
    if args.seeds is not None:
        C.SEEDS = args.seeds
    if args.methods is not None:
        C.METHODS = args.methods
    if args.epochs is not None:
        C.EPOCHS = args.epochs
    if args.data_fraction is not None:
        C.DATA_FRACTION = args.data_fraction
    if args.final_samples is not None:
        C.FINAL_SAMPLES = args.final_samples
    if args.mc_eval_samples is not None:
        C.MC_EVAL_SAMPLES = args.mc_eval_samples
    if args.max_test is not None:
        C.MAX_TEST = args.max_test


def main():
    parser = argparse.ArgumentParser(description="Fair BNN/MC-Dropout pipeline vs conformal")
    parser.add_argument("--debug", action="store_true",
                        help="Use debug seeds/data-fraction/epochs from the config")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override the seed list")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        choices=["bnn", "mc_dropout"], help="Override the method list")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--data-fraction", type=float, default=None,
                        help="Override data fraction")
    parser.add_argument("--final-samples", type=int, default=None,
                        help="Override MC passes for the final calib fit + test eval")
    parser.add_argument("--mc-eval-samples", type=int, default=None,
                        help="Override MC passes for per-epoch calib scoring")
    parser.add_argument("--max-test", type=int, default=None,
                        help="Debug: cap TEST set size for a fast smoke test")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip training; rebuild tables/plots from existing metrics")
    args = parser.parse_args()
    _apply_overrides(args)

    # Import after overrides so train_eval reads the patched config.
    from .train_eval import run_method_seed
    from .report import build_report

    C.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cfg_path = C.save_active_config()
    print(f"Active config -> {cfg_path}")
    print(f"Mode: seeds={C.SEEDS} methods={C.METHODS} epochs={C.EPOCHS} "
          f"data_fraction={C.DATA_FRACTION}")
    print(f"Output root: {C.OUTPUT_ROOT}\n")

    if not args.report_only:
        t0 = time.time()
        for seed in C.SEEDS:
            for method in C.METHODS:
                print(f"=== {method} | seed {seed} ===")
                try:
                    run_method_seed(method, seed)
                except Exception as exc:  # noqa: BLE001 — keep going on a single failure
                    print(f"  ERROR: {method} seed {seed} failed: {exc}")
        print(f"\nAll training done in {(time.time() - t0)/60:.1f} min.")

    print("\nBuilding comparison report...")
    build_report()
    print("Done.")


if __name__ == "__main__":
    main()
