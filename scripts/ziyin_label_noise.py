#!/usr/bin/env python
"""
Suite 2A: Label Noise Sweep.

For each (K, noise_level), train bz_to_a with within-group label noise.
Noise replaces target A with a random wrong A from the same B-group.

Reuses existing K=20 noise runs where available.
"""

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.experiment_helpers import (
    make_config, run_exists, run_single_experiment, load_history,
)

K_VALUES = [10, 20, 36]
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
SEED = 42
LR = 1e-3
MAX_STEPS = 200000
OUTPUT_DIR = "outputs"

# Map of existing runs that can be reused (experiment_name -> noise_level)
EXISTING_ALIASES = {
    # K=20 clean run
    (20, 0.0): "landauer_dense_k20",
    (20, 0.05): "landauer_k20_noise0.05",
    (20, 0.10): "landauer_k20_noise0.10",
    (20, 0.20): "landauer_k20_noise0.20",
    # K=10 clean run
    (10, 0.0): "landauer_dense_k10",
    # K=36 clean run
    (36, 0.0): "landauer_dense_k36",
}


def canonical_name(k, noise):
    """Standard experiment name for a (K, noise) pair."""
    if noise == 0.0:
        return f"ziyin_noise_K{k}_p0.00"
    return f"ziyin_noise_K{k}_p{noise:.2f}"


def find_existing(k, noise):
    """Check if a run already exists (under canonical or alias name)."""
    # Check canonical name
    name = canonical_name(k, noise)
    if run_exists(name, min_steps=MAX_STEPS // 4, output_dir=OUTPUT_DIR):
        return name

    # Check alias
    alias = EXISTING_ALIASES.get((k, noise))
    if alias and run_exists(alias, min_steps=1, output_dir=OUTPUT_DIR):
        return alias

    return None


def run_suite2a():
    """Run label noise sweep."""
    print("=" * 60)
    print("SUITE 2A: LABEL NOISE SWEEP")
    print("=" * 60)

    for K in K_VALUES:
        print(f"\n--- K = {K} ---")
        for noise in NOISE_LEVELS:
            name = canonical_name(K, noise)
            existing = find_existing(K, noise)

            if existing:
                print(f"  [SKIP] K={K}, noise={noise:.2f} "
                      f"(exists as {existing})")
                continue

            print(f"  [RUN] {name} (noise={noise:.0%}, {MAX_STEPS//1000}K steps)")

            eval_every = max(MAX_STEPS // 1000, 50)
            checkpoint_every = max(MAX_STEPS // 20, 5000)

            cfg = make_config(
                experiment_name=name,
                task="bz_to_a",
                k=K,
                seed=SEED,
                lr=LR,
                max_steps=MAX_STEPS,
                eval_every=eval_every,
                checkpoint_every=checkpoint_every,
                label_noise_prob=noise,
                early_stop_frac=0.05 if noise < 0.10 else None,
            )
            try:
                t0 = time.time()
                run_single_experiment(cfg, output_dir=OUTPUT_DIR)
                print(f"    Done in {time.time()-t0:.0f}s", flush=True)
            except Exception as e:
                print(f"    FAILED: {e}", flush=True)
                traceback.print_exc()

    print(f"\n{'='*60}")
    print("SUITE 2A COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_suite2a()
