#!/usr/bin/env python
"""
Experiment 1: Dataset-Size Confound Control

Reviewer concern: D = n_unique_b × K, so larger K means more training data.
The plateau and scaling laws might reflect dataset size, not ambiguity.

Design: Hold total dataset size D constant by varying n_unique_b inversely with K.
- Target D = 10,000 examples: K=5 (n=2000), K=10 (1000), K=20 (500), K=36 (~278)
- Target D = 20,000 examples: K=5 (4000), K=10 (2000), K=20 (1000), K=36 (~556)
- All at η=1e-3, other params = base.yaml defaults
- Total runs: 8

Expected output: Table of τ(K) at fixed D. If τ still scales with K,
the confound is ruled out.
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

TARGETS = {
    10000: {5: 2000, 10: 1000, 20: 500, 36: 278},
    20000: {5: 4000, 10: 2000, 20: 1000, 36: 556},
}

SEED = 42
LR = 1e-3
MAX_STEPS = 50000


def fit_power_law(k_arr, tau_arr, n_bootstrap=2000):
    """Fit τ = C * K^δ in log-log space with bootstrap CIs."""
    mask = np.isfinite(tau_arr) & (tau_arr > 0)
    k_arr, tau_arr = k_arr[mask], tau_arr[mask]
    if len(k_arr) < 2:
        return None, None, None
    log_k, log_tau = np.log(k_arr), np.log(tau_arr)
    A = np.vstack([log_k, np.ones(len(log_k))]).T
    delta, _ = np.linalg.lstsq(A, log_tau, rcond=None)[0]
    rng = np.random.RandomState(42)
    n = len(log_k)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            d_b, _ = np.linalg.lstsq(
                np.vstack([log_k[idx], np.ones(len(idx))]).T,
                log_tau[idx], rcond=None)[0]
            deltas.append(d_b)
        except Exception:
            continue
    deltas = np.array(deltas)
    return delta, np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)


def main():
    print("=" * 70)
    print("EXPERIMENT 1: DATASET-SIZE CONFOUND CONTROL")
    print("=" * 70)

    # ── Build jobs ─────────────────────────────────────────────────────────
    jobs = []
    for D, k_map in TARGETS.items():
        for k, n_b in k_map.items():
            name = f"confound_D{D}_K{k}_n{n_b}"
            if run_exists(name, min_steps=MAX_STEPS - 1000, output_dir=OUTPUT_DIR):
                print(f"  SKIP {name} (already exists)")
                continue

            cfg = make_config(
                experiment_name=name,
                k=k,
                n_unique_b=n_b,
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

    # ── Run ────────────────────────────────────────────────────────────────
    if jobs:
        run_parallel(jobs, max_workers=6, label="confound-control")
    else:
        print("  All runs already exist!")

    # ── Analyze ────────────────────────────────────────────────────────────
    print(f"\n{'D':<8} {'K':<6} {'n_b':<8} {'τ':>8} {'log K':>8}")
    print("-" * 42)

    results = {}
    for D, k_map in TARGETS.items():
        results[D] = {}
        for k, n_b in sorted(k_map.items()):
            name = f"confound_D{D}_K{k}_n{n_b}"
            h = load_history(name, OUTPUT_DIR)
            if h is None:
                print(f"  {D:<8} {k:<6} {n_b:<8} {'MISS':>8}")
                continue
            log_k = math.log(k)
            tau = detect_tau(h, log_k)
            results[D][k] = {"n_unique_b": n_b, "tau": tau}
            tau_str = str(tau) if tau else "N/A"
            print(f"  {D:<8} {k:<6} {n_b:<8} {tau_str:>8} {log_k:>8.3f}")

    # ── Power-law fits ─────────────────────────────────────────────────────
    print(f"\n{'D':<8} {'δ':>8} {'95% CI':>18} {'n':>4}")
    print("-" * 42)

    fit_results = {}
    for D in sorted(results.keys()):
        ks = sorted([k for k in results[D] if results[D][k]["tau"] is not None])
        if len(ks) < 2:
            continue
        k_arr = np.array(ks, dtype=float)
        tau_arr = np.array([results[D][k]["tau"] for k in ks], dtype=float)
        delta, ci_lo, ci_hi = fit_power_law(k_arr, tau_arr)
        fit_results[D] = {"delta": delta, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": len(ks)}
        if delta is not None:
            print(f"  {D:<8} {delta:>8.3f} [{ci_lo:.3f}, {ci_hi:.3f}] {len(ks):>4}")

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"10000": "#E74C3C", "20000": "#3498DB"}
    markers = {"10000": "o", "20000": "s"}

    for D in sorted(results.keys()):
        ks = sorted([k for k in results[D] if results[D][k]["tau"] is not None])
        if not ks:
            continue
        k_arr = np.array(ks, dtype=float)
        tau_arr = np.array([results[D][k]["tau"] for k in ks], dtype=float)
        label = f"D={D:,}"
        if D in fit_results and fit_results[D]["delta"] is not None:
            label += f" (δ={fit_results[D]['delta']:.2f})"
        ax.loglog(k_arr, tau_arr, f"{markers.get(str(D), 'o')}-",
                  color=colors.get(str(D), "gray"),
                  label=label, markersize=6, linewidth=1.5)

    ax.set_xlabel("K (ambiguity)", fontsize=10)
    ax.set_ylabel("τ (transition time, steps)", fontsize=10)
    ax.set_title("τ(K) at fixed dataset size D", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        plt.savefig(FIG_DIR / f"fig_dataset_confound.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {FIG_DIR}/fig_dataset_confound")

    # ── Save summary ───────────────────────────────────────────────────────
    summary = {
        "experiment": "dataset_confound",
        "description": "Dataset-size confound control (Exp 1)",
        "results": {str(D): {str(k): v for k, v in kv.items()}
                    for D, kv in results.items()},
        "power_law_fits": {str(D): v for D, v in fit_results.items()},
    }
    summary_path = Path(OUTPUT_DIR) / "dataset_confound_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
