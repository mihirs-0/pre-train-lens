#!/usr/bin/env python
"""
Plot gradient norm overlaid with candidate loss and z_gap.

Tests Ziyin's effective free energy prediction:
  F = L + η·S,  where S = ||∇L||²

Generates:
  1. gradient_norm_overlay.png  — 3-panel: loss+grad, z_gap+grad, free energy
  2. gradient_norm_components.png — per-component gradient norms over training

Usage:
    python scripts/plot_gradient_norm_overlay.py --experiment temp_lr1e3_bs128_k20
"""

import sys
from pathlib import Path
import argparse
import json
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_json(path: Path):
    """Load a JSON file or return None if missing."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _plot_overlay(grad_data, cand_data, save_path: Path):
    """
    3-panel overlay figure.

    Panel A: candidate_loss + ||∇L||²
    Panel B: z_gap + ||∇L||²
    Panel C: L, η·S, F = L + η·S  (effective free energy components)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    grad_steps = grad_data["steps"]
    grad_norm_sq = grad_data["total_grad_norm_sq"]
    loss_at_ckpt = grad_data["loss_at_checkpoint"]
    lr = grad_data["config"]["learning_rate"]
    k = grad_data["config"]["k"]
    log_k = math.log(k)

    # Candidate eval data (may have different step grid)
    cand_steps = cand_data["steps"] if cand_data else None
    cand_loss = cand_data["candidate_loss"] if cand_data else None
    z_gap = cand_data["z_gap"] if cand_data else None
    binding_onset = cand_data.get("binding_onset_step") if cand_data else None

    # ── Panel A: Candidate Loss + Gradient Norm ──
    ax1 = axes[0]
    color_loss = "#d62728"
    color_grad = "#1f77b4"

    if cand_steps and cand_loss:
        ax1.plot(cand_steps, cand_loss, color=color_loss, linewidth=2, label="Candidate loss")
    ax1.axhline(y=log_k, color="gray", linestyle="--", alpha=0.6, label=f"log({k}) = {log_k:.2f}")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Candidate Loss", color=color_loss)
    ax1.tick_params(axis="y", labelcolor=color_loss)

    ax1_right = ax1.twinx()
    ax1_right.plot(grad_steps, grad_norm_sq, color=color_grad, linewidth=2, alpha=0.8, label="||∇L||²")
    ax1_right.set_ylabel("||∇L||² (gradient norm squared)", color=color_grad)
    ax1_right.tick_params(axis="y", labelcolor=color_grad)

    if binding_onset is not None:
        ax1.axvline(x=binding_onset, color="green", linestyle=":", alpha=0.7, label=f"Binding onset ({binding_onset})")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_right.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax1.set_title("A) Candidate Loss and Gradient Norm")
    ax1.grid(True, alpha=0.2)

    # ── Panel B: z-Usage Gap + Gradient Norm ──
    ax2 = axes[1]
    color_zgap = "#d62728"

    if cand_steps and z_gap:
        ax2.plot(cand_steps, z_gap, color=color_zgap, linewidth=2, label="z_gap")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("z_gap (loss_z_shuffled − loss_clean)", color=color_zgap)
    ax2.tick_params(axis="y", labelcolor=color_zgap)

    ax2_right = ax2.twinx()
    ax2_right.plot(grad_steps, grad_norm_sq, color=color_grad, linewidth=2, alpha=0.8, label="||∇L||²")
    ax2_right.set_ylabel("||∇L||² (gradient norm squared)", color=color_grad)
    ax2_right.tick_params(axis="y", labelcolor=color_grad)

    if binding_onset is not None:
        ax2.axvline(x=binding_onset, color="green", linestyle=":", alpha=0.7, label=f"Binding onset ({binding_onset})")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax2.set_title("B) z-Usage Gap and Gradient Norm")
    ax2.grid(True, alpha=0.2)

    # ── Panel C: Effective Free Energy Components ──
    ax3 = axes[2]
    eta_S = [lr * s for s in grad_norm_sq]
    F = [l + es for l, es in zip(loss_at_ckpt, eta_S)]

    ax3.plot(grad_steps, loss_at_ckpt, color="#2ca02c", linewidth=2, label="L (training loss)")
    ax3.plot(grad_steps, eta_S, color="#ff7f0e", linewidth=2, label=f"η·||∇L||²  (η={lr})")
    ax3.plot(grad_steps, F, color="#9467bd", linewidth=2.5, linestyle="--", label="F = L + η·||∇L||²")
    ax3.axhline(y=log_k, color="gray", linestyle="--", alpha=0.4, label=f"log({k})")

    if binding_onset is not None:
        ax3.axvline(x=binding_onset, color="green", linestyle=":", alpha=0.7, label=f"Binding onset ({binding_onset})")

    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Value")
    ax3.set_title("C) Effective Free Energy Components")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(True, alpha=0.2)

    exp_name = grad_data["config"]["experiment_name"]
    fig.suptitle(f"{exp_name}: Gradient Norm and Effective Free Energy", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def _plot_components(grad_data, cand_data, save_path: Path):
    """
    Per-component gradient norms over training.

    Log-scale y-axis to see relative magnitudes.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    grad_steps = grad_data["steps"]
    components = grad_data.get("component_grad_norms", {})
    binding_onset = cand_data.get("binding_onset_step") if cand_data else None

    colors = {
        "embedding": "#1f77b4",
        "attention": "#ff7f0e",
        "mlp": "#2ca02c",
        "unembedding": "#d62728",
        "layernorm": "#9467bd",
        "other": "#8c564b",
    }

    for bucket_name, norms in sorted(components.items()):
        color = colors.get(bucket_name, "#7f7f7f")
        ax.plot(grad_steps, norms, linewidth=2, color=color, label=bucket_name)

    # Also plot total for reference
    ax.plot(
        grad_steps,
        grad_data["total_grad_norm"],
        linewidth=2.5,
        color="black",
        linestyle="--",
        alpha=0.7,
        label="total ||∇L||",
    )

    if binding_onset is not None:
        ax.axvline(x=binding_onset, color="green", linestyle=":", alpha=0.7, label=f"Binding onset ({binding_onset})")

    ax.set_yscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient Norm  ||∇L||  (log scale)")
    exp_name = grad_data["config"]["experiment_name"]
    ax.set_title(f"{exp_name}: Per-Component Gradient Norms")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, which="both")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot gradient norm overlay with candidate loss and z_gap"
    )
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base output directory")
    args = parser.parse_args()

    experiment_dir = Path(args.output_dir) / args.experiment
    figures_dir = experiment_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load gradient norm results
    grad_path = experiment_dir / "gradient_norm_results.json"
    grad_data = load_json(grad_path)
    if grad_data is None:
        print(f"Error: Gradient norm results not found at {grad_path}")
        print("Run compute_gradient_norms.py first.")
        sys.exit(1)

    # Load candidate eval results (optional but expected)
    cand_path = experiment_dir / "candidate_eval_results.json"
    cand_data = load_json(cand_path)
    if cand_data is None:
        print(f"Warning: candidate_eval_results.json not found at {cand_path}")
        print("Panels A and B will be missing candidate loss / z_gap curves.")

    print(f"Gradient norm data: {len(grad_data['steps'])} checkpoints")
    if cand_data:
        print(f"Candidate eval data: {len(cand_data['steps'])} checkpoints")

    # Generate figures
    _plot_overlay(grad_data, cand_data, figures_dir / "gradient_norm_overlay.png")
    _plot_components(grad_data, cand_data, figures_dir / "gradient_norm_components.png")

    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
