#!/usr/bin/env python
"""
Plot Landauer dissipation test results.

Loads outputs/landauer_results.json and produces a 2×2 figure:
  A: Cumulative dissipation Q(t) vs training step
  B: Q_transition vs log(K) with linear fit
  C: Gradient norm profiles with transition windows
  D: Landauer ratio Q_transition/log(K) bar chart

Usage:
    python scripts/plot_landauer_test.py [--input outputs/landauer_results.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Consistent colors for each K
K_COLORS = {
    10: "#2196F3",   # blue
    20: "#FF9800",   # orange
    36: "#4CAF50",   # green
}

K_COLORS_LIGHT = {
    10: "#90CAF9",
    20: "#FFE0B2",
    36: "#C8E6C9",
}


def get_color(k):
    return K_COLORS.get(k, "#9E9E9E")


def get_color_light(k):
    return K_COLORS_LIGHT.get(k, "#E0E0E0")


def load_gradient_norms(experiment_name, output_dir):
    """Load gradient norm data for an experiment."""
    path = Path(output_dir) / experiment_name / "gradient_norm_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def panel_a_cumulative_dissipation(ax, experiments):
    """Panel A: Cumulative dissipation Q(t) for each K."""
    for exp in experiments:
        k = exp["K"]
        steps = exp["Q_trajectory_steps"]
        Q = exp["Q_trajectory"]
        color = get_color(k)

        ax.plot(steps, Q, color=color, linewidth=1.8, label=f"K={k}", zorder=3)

        # Mark transition window
        t_start = exp.get("transition_start")
        t_end = exp.get("transition_end")
        if t_start is not None:
            ax.axvline(t_start, color=color, linestyle="--", alpha=0.4, linewidth=1)
        if t_end is not None:
            ax.axvline(t_end, color=color, linestyle="--", alpha=0.4, linewidth=1)

        # Shade transition region
        if t_start is not None and t_end is not None:
            ax.axvspan(t_start, t_end, color=get_color_light(k), alpha=0.15, zorder=1)

    ax.set_xlabel("Training Step", fontsize=10)
    ax.set_ylabel(r"$Q(t) = \sum \eta(s) \cdot \|\nabla L(s)\|^2 \cdot \Delta s$", fontsize=9)
    ax.set_title("A. Cumulative Dissipation Q(t)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


def panel_b_scaling(ax, experiments):
    """Panel B: Q_transition vs log(K) with linear fit."""
    valid = [e for e in experiments if e["Q_transition"] is not None]
    if len(valid) < 2:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", fontsize=12)
        ax.set_title("B. Q_transition vs log(K)", fontsize=11, fontweight="bold")
        return

    log_ks = np.array([e["log_K"] for e in valid])
    q_trans = np.array([e["Q_transition"] for e in valid])

    # Plot data points
    for e in valid:
        ax.scatter(
            e["log_K"], e["Q_transition"],
            color=get_color(e["K"]), s=80, zorder=5, edgecolors="black", linewidths=0.8
        )
        ax.annotate(
            f"K={e['K']}", (e["log_K"], e["Q_transition"]),
            textcoords="offset points", xytext=(8, 5), fontsize=9
        )

    # Linear fit through data
    if len(valid) >= 2:
        coeffs = np.polyfit(log_ks, q_trans, 1)
        slope, intercept = coeffs

        # R² calculation
        predicted = np.polyval(coeffs, log_ks)
        ss_res = np.sum((q_trans - predicted) ** 2)
        ss_tot = np.sum((q_trans - np.mean(q_trans)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Plot fit line
        x_fit = np.linspace(min(log_ks) * 0.9, max(log_ks) * 1.1, 100)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, "k-", linewidth=1.2, alpha=0.7,
                label=f"Fit: slope={slope:.4f}, R²={r_sq:.3f}")

        # Line through origin for comparison
        slope_origin = np.sum(log_ks * q_trans) / np.sum(log_ks ** 2)
        y_origin = slope_origin * x_fit
        ax.plot(x_fit, y_origin, "k--", linewidth=1, alpha=0.4,
                label=f"Through origin: slope={slope_origin:.4f}")

    ax.set_xlabel("log(K)", fontsize=10)
    ax.set_ylabel("Q_transition", fontsize=10)
    ax.set_title("B. Transition Dissipation vs log(K)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)


def panel_c_gradient_profiles(ax, experiments, output_dir):
    """Panel C: Gradient norm profiles with transition windows shaded."""
    for exp in experiments:
        k = exp["K"]
        color = get_color(k)

        # Load gradient norm data
        grad_data = load_gradient_norms(exp["name"], output_dir)
        if grad_data is None:
            continue

        steps = grad_data["steps"]
        norm_sq = grad_data["total_grad_norm_sq"]

        ax.plot(steps, norm_sq, color=color, linewidth=1.5, label=f"K={k}", zorder=3)

        # Shade transition window
        t_start = exp.get("transition_start")
        t_end = exp.get("transition_end")
        if t_start is not None and t_end is not None:
            ax.axvspan(t_start, t_end, color=get_color_light(k), alpha=0.2, zorder=1)

    ax.set_xlabel("Training Step", fontsize=10)
    ax.set_ylabel(r"$\|\nabla L\|^2$", fontsize=10)
    ax.set_title(r"C. Gradient Norm Profiles $\|\nabla L\|^2$", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)


def panel_d_ratio_bars(ax, experiments):
    """Panel D: Landauer ratio Q_transition/log(K) bar chart."""
    valid = [e for e in experiments if e["Q_transition_over_logK"] is not None]
    if not valid:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=12)
        ax.set_title("D. Landauer Ratio", fontsize=11, fontweight="bold")
        return

    labels = [f"K={e['K']}" for e in valid]
    ratios = [e["Q_transition_over_logK"] for e in valid]
    colors = [get_color(e["K"]) for e in valid]

    x = np.arange(len(valid))
    bars = ax.bar(x, ratios, color=colors, edgecolor="black", linewidth=0.8, width=0.5)

    # Add value labels on bars
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
            f"{ratio:.5f}", ha="center", va="bottom", fontsize=8
        )

    # Mean line
    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    ax.axhline(mean_ratio, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"Mean: {mean_ratio:.5f}")

    # Std band
    ax.axhspan(mean_ratio - std_ratio, mean_ratio + std_ratio,
               color="red", alpha=0.08)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"$Q_{transition} / \log(K)$", fontsize=10)
    ax.set_title("D. Landauer Ratio: Q_trans / log(K)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    # Add CV annotation
    if len(ratios) >= 2 and mean_ratio > 0:
        cv = std_ratio / mean_ratio
        ax.text(
            0.95, 0.95,
            f"CV = {cv:.3f}\n(std/mean)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )


def main():
    parser = argparse.ArgumentParser(description="Plot Landauer dissipation test results")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/landauer_results.json",
        help="Path to landauer_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory (for loading gradient norm data)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="outputs/figures/landauer_test.png",
        help="Output figure path",
    )
    args = parser.parse_args()

    # Load results
    with open(args.input) as f:
        data = json.load(f)

    experiments = data["experiments"]
    print(f"Loaded {len(experiments)} experiments")
    for e in experiments:
        print(f"  K={e['K']}: Q_trans={e['Q_transition']}, Q_trans/logK={e['Q_transition_over_logK']}")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Landauer Dissipation Test — First Pass\n"
        "Does Q_transition scale with log(K)?",
        fontsize=13, fontweight="bold", y=0.98,
    )

    panel_a_cumulative_dissipation(axes[0, 0], experiments)
    panel_b_scaling(axes[0, 1], experiments)
    panel_c_gradient_profiles(axes[1, 0], experiments, args.output_dir)
    panel_d_ratio_bars(axes[1, 1], experiments)

    # Add note about experimental conditions
    batch_sizes = set(e["batch_size"] for e in experiments)
    max_steps_set = set(e["max_steps"] for e in experiments)
    notes = []
    if len(batch_sizes) > 1:
        notes.append(
            f"K=36 uses bs={max(batch_sizes)} (others bs={min(batch_sizes)}). "
            "Different T_eff is a confound."
        )
    if len(max_steps_set) > 1:
        notes.append(
            f"K=36 uses {max(max_steps_set)//1000}K cosine schedule (others {min(max_steps_set)//1000}K). "
            "Higher LR during K=36 transition inflates Q_trans."
        )
    if notes:
        fig.text(
            0.5, 0.01,
            "Note: " + " ".join(notes) + " Interpret K=36 ratio with caution.",
            ha="center", fontsize=9, style="italic", color="red",
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure to: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
