#!/usr/bin/env python
"""
Plot two-layer linear network learning rate sweep results.

Compares K=10 forward runs at lr=1e-3 (200K), lr=1e-2 (100K), lr=1e-1 (100K)
to demonstrate that the "stuck at log(K)" behavior is a fundamental capacity
limitation, not a learning rate / optimization issue.

Key insight: MSE decreases (model learns group mean ā_g) but candidate_loss
stays at log(K) (model can't break symmetry between K candidates). A linear
model y = W_b @ b + W_z @ z can only produce an additive contribution from z
that is the SAME for all B groups, whereas the task requires z to select
DIFFERENT candidates per group — a fundamentally nonlinear interaction.

Usage:
    python scripts/plot_twolayer_lr_sweep.py
"""

import json
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    print("matplotlib required")
    raise


# ═══════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════

def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_lr_sweep_runs(output_dir):
    """Find all K=10 forward lr sweep files."""
    runs = {}
    base = Path(output_dir)
    patterns = [
        # Standard 30K runs from main sweep
        ("1e-3 (30K)", "twolayer_linear_forward_K10_H128_results.json"),
        # Extended / LR sweep runs
        ("1e-3 (200K)", "twolayer_linear_forward_K10_H128_lr1e-3_200k_results.json"),
        ("1e-2 (100K)", "twolayer_linear_forward_K10_H128_lr1e-2_100k_results.json"),
        ("1e-1 (100K)", "twolayer_linear_forward_K10_H128_lr1e-1_100k_results.json"),
    ]
    for label, fname in patterns:
        path = base / fname
        if path.exists():
            runs[label] = load_json(str(path))
    return runs


# ═══════════════════════════════════════════════════════════════════════
# Theoretical bounds
# ═══════════════════════════════════════════════════════════════════════

def compute_theoretical_mse(K, N_B, D):
    """
    Theoretical minimum MSE for the best linear predictor.

    A linear model y = W_b b + W_z z can at best predict:
      - W_b maps b_g to the group mean ā_g = (1/K) Σ_k a_{g,k}
      - W_z maps z_j to a correction ā_j = (1/N_B) Σ_g a_{g,j}

    The residual per-element MSE is approximately:
      MSE ≈ 1 - 1/K - 1/N_B   (for large D)

    This is the irreducible error of the linear function class.
    """
    return 1.0 - 1.0 / K - 1.0 / N_B


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

COLORS = {
    "1e-3 (30K)": "#1f77b4",
    "1e-3 (200K)": "#1f77b4",
    "1e-2 (100K)": "#ff7f0e",
    "1e-1 (100K)": "#2ca02c",
}
STYLES = {
    "1e-3 (30K)": "--",
    "1e-3 (200K)": "-",
    "1e-2 (100K)": "-",
    "1e-1 (100K)": "-",
}


def plot_lr_sweep(runs, output_path):
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.25)

    K = 10
    log_K = math.log(K)
    N_B = 1000
    D = 128

    # ── Panel A: Candidate Loss vs Step ──
    ax = fig.add_subplot(gs[0, 0])
    for label, data in sorted(runs.items()):
        c = COLORS.get(label, "gray")
        ls = STYLES.get(label, "-")
        ax.plot(data["steps"], data["candidate_loss"],
                ls, color=c, label=f"lr={label}", alpha=0.85, linewidth=1.5)
    ax.axhline(log_K, color="red", linestyle=":", alpha=0.5,
               label=f"log({K}) = {log_K:.3f}")
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Candidate Loss", fontsize=11)
    ax.set_title("(A) Candidate Loss — Stuck at log(K)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(log_K - 0.05, log_K + 0.15)

    # ── Panel B: MSE vs Step ──
    ax = fig.add_subplot(gs[0, 1])
    theoretical_mse = compute_theoretical_mse(K, N_B, D)
    for label, data in sorted(runs.items()):
        c = COLORS.get(label, "gray")
        ls = STYLES.get(label, "-")
        ax.plot(data["steps"], data["total_mse"],
                ls, color=c, label=f"lr={label}", alpha=0.85, linewidth=1.5)
    ax.axhline(theoretical_mse, color="red", linestyle=":",
               alpha=0.5, label=f"Theory: 1−1/K−1/N_B = {theoretical_mse:.3f}")
    ax.axhline(1.0 - 1.0 / K, color="orange", linestyle=":",
               alpha=0.4, label=f"Group-mean only: 1−1/K = {1-1/K:.2f}")
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("MSE (per element)", fontsize=11)
    ax.set_title("(B) MSE — Model Learns Group Mean", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)

    # ── Panel C: z-gap vs Step ──
    ax = fig.add_subplot(gs[1, 0])
    for label, data in sorted(runs.items()):
        c = COLORS.get(label, "gray")
        ls = STYLES.get(label, "-")
        ax.plot(data["steps"], data["z_gap"],
                ls, color=c, label=f"lr={label}", alpha=0.85, linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("z-gap", fontsize=11)
    ax.set_title("(C) z-gap — Never Learns to Use z", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # ── Panel D: Gradient Norm² vs Step ──
    ax = fig.add_subplot(gs[1, 1])
    for label, data in sorted(runs.items()):
        c = COLORS.get(label, "gray")
        ls = STYLES.get(label, "-")
        ax.plot(data["steps"], data["gradient_norm_sq"],
                ls, color=c, label=f"lr={label}", alpha=0.85, linewidth=1.5)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("||∇L||²", fontsize=11)
    ax.set_title("(D) Gradient Norm² — No Spike, Monotone Decay",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # ── Title ──
    fig.suptitle(
        "Two-Layer Linear Network: Learning Rate Sweep (K=10, D=128)\n"
        "The linear model learns to predict the group mean (MSE↓) but cannot "
        "break candidate symmetry (CandLoss=log K, z-gap≈0).\n"
        "This is a fundamental capacity limitation: y = W·[b;z] cannot represent "
        "the conditional selection a_{g,j(z)}.",
        fontsize=11, y=1.02, va="bottom"
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════

def print_summary(runs):
    K = 10
    log_K = math.log(K)

    print()
    print("Two-Layer Linear Network: Learning Rate Sweep (K=10)")
    print("══════════════════════════════════════════════════════════════")
    print(f"{'LR (steps)':>16}  {'final MSE':>10}  {'final CL':>10}  "
          f"{'CL-log(K)':>10}  {'z-gap':>8}  {'Acc':>8}")
    print("-" * 75)

    for label in sorted(runs.keys()):
        data = runs[label]
        final_mse = data["total_mse"][-1]
        final_cl = data["candidate_loss"][-1]
        final_zg = data["z_gap"][-1]
        final_acc = data["candidate_accuracy"][-1]
        cl_diff = final_cl - log_K

        print(f"{label:>16}  {final_mse:>10.6f}  {final_cl:>10.4f}  "
              f"{cl_diff:>+10.4f}  {final_zg:>8.4f}  {final_acc:>8.4f}")

    print("══════════════════════════════════════════════════════════════")
    print(f"log(K) = {log_K:.4f},  1/K = {1/K:.4f}")
    print(f"Theory MSE floor (linear) = {1 - 1/K - 1/1000:.4f}")
    print()
    print("INTERPRETATION:")
    print("  The two-layer linear network reduces MSE (learning the per-group")
    print(f"  mean ā_g) but candidate_loss stays at log({K}) and accuracy at 1/{K}")
    print("  across ALL learning rates (1e-3, 1e-2, 1e-1) and up to 200K steps.")
    print()
    print("  This is a representational capacity failure, NOT an optimization")
    print("  failure. A linear model y = W·[b;z] computes an additive function")
    print("  of b and z. The z-contribution is the SAME for all B groups, but")
    print("  the task requires z to select DIFFERENT candidates per group —")
    print("  a fundamentally nonlinear (b × z) interaction.")
    print()
    print("  PAPER FRAMING: 'The two-layer linear network converges to the")
    print("  group-mean predictor but cannot escape the log(K) candidate-loss")
    print("  plateau at any learning rate, confirming that the sharp transition")
    print("  requires the transformer's nonlinear conditional routing — the")
    print("  ability to form attention patterns that selectively gate")
    print("  information from z based on b.'")
    print("══════════════════════════════════════════════════════════════")
    print()


def main():
    output_dir = "outputs"
    runs = find_lr_sweep_runs(output_dir)

    if not runs:
        print("No learning rate sweep results found.")
        print("Expected files like: outputs/twolayer_linear_forward_K10_H128_lr1e-2_100k_results.json")
        return

    print(f"Found {len(runs)} runs:")
    for label in sorted(runs.keys()):
        n_steps = runs[label]["steps"][-1]
        print(f"  {label}: {n_steps} steps")

    print_summary(runs)

    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_lr_sweep(runs, str(fig_dir / "twolayer_linear_lr_sweep.png"))


if __name__ == "__main__":
    main()
