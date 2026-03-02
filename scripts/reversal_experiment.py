#!/usr/bin/env python
"""
Suite 1: The Reversal Curse in the Wind Tunnel.

Tests directional asymmetry by training on the same data in 4 configurations:
  Task F:  A -> B  (forward, trivial)
  Task Bz: (B,z) -> A  (backward with z, the standard disambiguation task)
  Task B:  B -> A  (backward without z, impossible below log K)
  Transfer: Train F for 50K, then switch to Bz for 150K

All tasks at each K use the SAME surjective function f.
"""

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.experiment_helpers import (
    make_config, run_exists, save_mapping, load_mapping,
    run_single_experiment, generate_mappings,
)

K_VALUES = [5, 10, 20, 36]
SEED = 42
LR = 1e-3
OUTPUT_DIR = "outputs"


def run_task(name, task, k, mapping, max_steps, early_stop_frac=None,
             model=None, checkpoint_every=None):
    """Run a single task, returning (model, history) or None on failure."""
    if checkpoint_every is None:
        checkpoint_every = max(max_steps // 20, 1000)

    # eval_every: aim for ~1000 eval points
    eval_every = max(max_steps // 1000, 50)

    cfg = make_config(
        experiment_name=name,
        task=task,
        k=k,
        seed=SEED,
        lr=LR,
        max_steps=max_steps,
        eval_every=eval_every,
        checkpoint_every=checkpoint_every,
        early_stop_frac=early_stop_frac,
    )
    try:
        t0 = time.time()
        result = run_single_experiment(
            cfg, mapping_data=mapping, output_dir=OUTPUT_DIR, model=model,
        )
        elapsed = time.time() - t0
        print(f"  Completed {name} in {elapsed:.0f}s", flush=True)
        return result[0], result[1]  # model, history
    except Exception as e:
        print(f"  FAILED {name}: {e}", flush=True)
        traceback.print_exc()
        return None, None


def run_suite1():
    """Run all Suite 1 experiments."""
    print("=" * 60)
    print("SUITE 1: THE REVERSAL CURSE IN THE WIND TUNNEL")
    print("=" * 60)

    for K in K_VALUES:
        print(f"\n{'='*60}")
        print(f"K = {K}")
        print(f"{'='*60}")

        # ---- Generate/load shared mapping ----
        mapping_path = Path(OUTPUT_DIR) / f"reversal_mapping_K{K}.json"
        if not mapping_path.exists():
            print(f"  Generating shared mapping for K={K}...")
            mapping = generate_mappings(
                n_unique_b=1000, k=K, b_length=6, a_length=4,
                z_length=2, vocab_chars="abcdefghijklmnopqrstuvwxyz0123456789",
                seed=SEED, task="bz_to_a",
            )
            save_mapping(mapping, str(mapping_path))
            print(f"  Saved mapping to {mapping_path}")
        else:
            print(f"  Loading existing mapping from {mapping_path}")
        mapping = load_mapping(str(mapping_path))

        # ---- Task F: A -> B with z distractor (50K steps) ----
        name_f = f"reversal_F_K{K}_s{SEED}"
        if run_exists(name_f, min_steps=40000, output_dir=OUTPUT_DIR):
            print(f"  [SKIP] {name_f} already exists")
        else:
            print(f"  [RUN] {name_f} (Task F: A->B, 50K steps)")
            run_task(name_f, "az_to_b", K, mapping, max_steps=50000)

        # ---- Task Bz: (B,z) -> A (200K steps with early stopping) ----
        name_bz = f"reversal_Bz_K{K}_s{SEED}"
        # Need longer budget for large K
        bz_steps = 200000 if K >= 20 else 100000
        min_check = bz_steps // 2
        if run_exists(name_bz, min_steps=min_check, output_dir=OUTPUT_DIR):
            print(f"  [SKIP] {name_bz} already exists")
        else:
            print(f"  [RUN] {name_bz} (Task Bz: (B,z)->A, {bz_steps//1000}K steps)")
            run_task(name_bz, "bz_to_a", K, mapping, max_steps=bz_steps,
                     early_stop_frac=0.05)

        # ---- Task B: B -> A without z (50K steps) ----
        name_b = f"reversal_B_K{K}_s{SEED}"
        if run_exists(name_b, min_steps=40000, output_dir=OUTPUT_DIR):
            print(f"  [SKIP] {name_b} already exists")
        else:
            print(f"  [RUN] {name_b} (Task B: B->A no z, 50K steps)")
            run_task(name_b, "b_to_a", K, mapping, max_steps=50000)

        # ---- Transfer: F for 50K -> Bz for 150K ----
        name_tr = f"reversal_Transfer_K{K}_s{SEED}"
        tr_steps = 150000 if K >= 20 else 100000
        if run_exists(name_tr, min_steps=tr_steps // 2, output_dir=OUTPUT_DIR):
            print(f"  [SKIP] {name_tr} already exists")
        else:
            print(f"  [RUN] Transfer task for K={K}")

            # Phase 1: Task F for 50K steps
            name_p1 = f"reversal_Transfer_P1_K{K}_s{SEED}"
            print(f"    Phase 1: {name_p1} (A->B, 50K steps)")
            model_p1, _ = run_task(name_p1, "az_to_b", K, mapping,
                                   max_steps=50000, checkpoint_every=50000)

            if model_p1 is not None:
                # Phase 2: Task Bz for 150K steps using model from Phase 1
                print(f"    Phase 2: {name_tr} (Bz->A, {tr_steps//1000}K steps)")
                run_task(name_tr, "bz_to_a", K, mapping, max_steps=tr_steps,
                         model=model_p1, early_stop_frac=0.05)
            else:
                print(f"    Phase 1 failed; skipping Phase 2")

    print(f"\n{'='*60}")
    print("SUITE 1 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_suite1()
