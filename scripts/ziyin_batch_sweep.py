#!/usr/bin/env python
"""
Suite 2C: Batch Size Sweep at K=20.

Varies batch size while keeping eta=1e-3 fixed.
Tests whether noise (proportional to 1/sqrt(BS)) stabilizes or destabilizes the plateau.

NOTE: The existing training code does NOT scale gradients by 1/BS.
Loss is computed by F.cross_entropy (mean over tokens in batch), so
the effective learning rate per example is eta/BS_implicit — but since
PyTorch's cross_entropy uses mean reduction by default, the gradient
magnitude is independent of BS. This means changing BS only changes
gradient noise, not the expected gradient, which is exactly what we want.
"""

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.experiment_helpers import (
    make_config, run_exists, run_single_experiment,
)

K = 20
BATCH_SIZES = [32, 64, 128, 256, 512]
SEED = 42
LR = 1e-3
MAX_STEPS = 200000
OUTPUT_DIR = "outputs"

# Existing runs to check
EXISTING_MAP = {
    64: "temp_lr1e3_bs64",
    128: "temp_lr1e3_bs128",
    256: "temp_lr1e3_bs256",
    512: "temp_lr1e3_bs512",
}


def canonical_name(bs):
    return f"ziyin_batch_K{K}_bs{bs}"


def find_existing(bs):
    name = canonical_name(bs)
    if run_exists(name, min_steps=MAX_STEPS // 4, output_dir=OUTPUT_DIR):
        return name
    alias = EXISTING_MAP.get(bs)
    if alias and run_exists(alias, min_steps=1, output_dir=OUTPUT_DIR):
        return alias
    return None


def run_suite2c():
    """Run batch size sweep."""
    print("=" * 60)
    print(f"SUITE 2C: BATCH SIZE SWEEP (K={K}, eta={LR})")
    print("=" * 60)

    for bs in BATCH_SIZES:
        existing = find_existing(bs)
        if existing:
            print(f"  [SKIP] BS={bs} (exists as {existing})")
            continue

        name = canonical_name(bs)
        print(f"  [RUN] {name} (BS={bs}, {MAX_STEPS//1000}K steps)")

        eval_every = max(MAX_STEPS // 1000, 50)
        checkpoint_every = max(MAX_STEPS // 20, 5000)

        cfg = make_config(
            experiment_name=name,
            task="bz_to_a",
            k=K,
            seed=SEED,
            lr=LR,
            bs=bs,
            max_steps=MAX_STEPS,
            eval_every=eval_every,
            checkpoint_every=checkpoint_every,
            early_stop_frac=0.05,
        )
        try:
            t0 = time.time()
            run_single_experiment(cfg, output_dir=OUTPUT_DIR)
            print(f"    Done in {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("SUITE 2C COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_suite2c()
