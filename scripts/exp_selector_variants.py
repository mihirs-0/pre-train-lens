#!/usr/bin/env python
"""
Experiment 7: Selector Variant Tests

Reviewer concern: z is always 2 characters at a fixed position.
Results might be an artifact of this specific format.

Design: Test with different z configurations:
- z_length=1: Shorter selector
- z_length=3: Longer selector (more redundancy)
- z_length=4: Even longer
- All at K=10, η=1e-3, n_unique_b=1000
- Compare with z_length=2 baseline (existing run)
- Total: 3 new runs

Expected output: τ for each z_length. If plateau height = log K and
transition is sharp regardless of z_length, the phenomenon is robust.
"""

import sys
import json
import math
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.experiment_helpers import (
    make_config, run_parallel, run_exists, load_history, detect_tau,
)
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
FIG_DIR = Path(OUTPUT_DIR) / "paper_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

K = 10
SEED = 42
LR = 1e-3
MAX_STEPS = 50000

Z_LENGTHS = [1, 2, 3, 4]  # 2 is baseline
BASELINE_NAMES = ["landauer_dense_k10", "k10_lr1e-3"]

# ── Build jobs ─────────────────────────────────────────────────────────────

print("=" * 70)
print("EXPERIMENT 7: SELECTOR VARIANT TESTS")
print("=" * 70)

run_names = {}
jobs = []

for z_len in Z_LENGTHS:
    if z_len == 2:
        # Try to use existing baseline
        found = False
        for name in BASELINE_NAMES:
            if run_exists(name, min_steps=MAX_STEPS * 0.5, output_dir=OUTPUT_DIR):
                run_names[z_len] = name
                print(f"  z_length={z_len}: EXISTING {name}")
                found = True
                break
        if found:
            continue

    name = f"selector_zlen{z_len}_K{K}"
    run_names[z_len] = name

    if run_exists(name, min_steps=MAX_STEPS * 0.5, output_dir=OUTPUT_DIR):
        print(f"  z_length={z_len}: EXISTING {name}")
        continue

    print(f"  z_length={z_len}: TRAIN {name}")
    cfg = make_config(
        experiment_name=name,
        k=K,
        z_length=z_len,
        lr=LR,
        seed=SEED,
        max_steps=MAX_STEPS,
        early_stop_frac=0.01,
    )
    jobs.append({
        "cfg_dict": OmegaConf.to_container(cfg, resolve=True),
        "mapping_path": None,
        "output_dir": OUTPUT_DIR,
        "name": name,
    })

# ── Run ────────────────────────────────────────────────────────────────────

if jobs:
    run_parallel(jobs, max_workers=4, label="selector-variants")
else:
    print("  All runs already exist!")

# ── Analyze ────────────────────────────────────────────────────────────────

print(f"\n{'z_length':<10} {'Run':<28} {'τ':>8} {'Plateau loss':>14} {'Final loss':>12}")
print("-" * 76)

log_k = math.log(K)
results = {}

for z_len in Z_LENGTHS:
    name = run_names.get(z_len)
    if name is None:
        continue
    h = load_history(name, OUTPUT_DIR)
    if h is None:
        print(f"  {z_len:<10} {name:<28} {'MISS':>8}")
        continue

    tau = detect_tau(h, log_k)
    # Plateau loss: average first_target_loss in steps 100-500
    early_steps = [(s, l) for s, l in zip(h["steps"], h["first_target_loss"])
                   if 100 <= s <= 500]
    plateau_loss = np.mean([l for _, l in early_steps]) if early_steps else None
    final_loss = h["first_target_loss"][-1] if h["first_target_loss"] else None

    results[z_len] = {
        "run_name": name,
        "tau": tau,
        "plateau_loss": float(plateau_loss) if plateau_loss is not None else None,
        "final_loss": float(final_loss) if final_loss is not None else None,
        "plateau_over_logk": float(plateau_loss / log_k) if plateau_loss else None,
    }

    tau_str = str(tau) if tau else "N/A"
    pl_str = f"{plateau_loss:.3f}" if plateau_loss else "N/A"
    fl_str = f"{final_loss:.4f}" if final_loss else "N/A"
    print(f"  {z_len:<10} {name:<28} {tau_str:>8} {pl_str:>14} {fl_str:>12}")

# Summary
if results:
    print(f"\n  log K = {log_k:.3f}")
    for z_len, r in sorted(results.items()):
        if r["plateau_over_logk"] is not None:
            print(f"  z_length={z_len}: plateau/log K = {r['plateau_over_logk']:.3f}")

# ── Figure ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Panel A: Loss curves
ax = axes[0]
colors = {1: "#E74C3C", 2: "#3498DB", 3: "#2ECC71", 4: "#9B59B6"}
for z_len in Z_LENGTHS:
    name = run_names.get(z_len)
    if name is None:
        continue
    h = load_history(name, OUTPUT_DIR)
    if h is None:
        continue
    ax.plot(h["steps"], h["first_target_loss"],
            color=colors.get(z_len, "gray"), linewidth=1.2,
            label=f"z_length={z_len}")

ax.axhline(log_k, color="gray", linestyle=":", alpha=0.5, label=f"log K = {log_k:.2f}")
ax.set_xlabel("Step", fontsize=10)
ax.set_ylabel("First target loss", fontsize=10)
ax.set_title(f"K={K}: Loss by z_length", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)

# Panel B: τ vs z_length
ax = axes[1]
z_lens = sorted([z for z, r in results.items() if r["tau"] is not None])
taus = [results[z]["tau"] for z in z_lens]
if z_lens:
    ax.bar(z_lens, taus, color=[colors.get(z, "gray") for z in z_lens], alpha=0.8)
    ax.set_xlabel("z_length", fontsize=10)
    ax.set_ylabel("τ (transition time)", fontsize=10)
    ax.set_title(f"K={K}: τ vs selector length", fontsize=11, fontweight="bold")
    ax.set_xticks(z_lens)
    ax.tick_params(labelsize=8)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(FIG_DIR / f"fig_selector_variants.{ext}", dpi=300,
                bbox_inches="tight")
plt.close()
print(f"\nFigure saved: {FIG_DIR}/fig_selector_variants")

# ── Save summary ───────────────────────────────────────────────────────────

summary = {
    "experiment": "selector_variants",
    "description": "Selector variant tests (Exp 7)",
    "k": K,
    "log_k": log_k,
    "results": {str(z): v for z, v in results.items()},
}
summary_path = Path(OUTPUT_DIR) / "selector_variants_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved: {summary_path}")
