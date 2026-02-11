#!/usr/bin/env python
"""
Plot comprehensive Landauer analysis across 6 K values.

Produces a 2×3 figure:
  A: Q_total_conv vs log(K)  — Full Protocol
  B: Q_zgap vs log(K)        — Functional Window
  C: Q_original vs log(K)    — Original Window
  D: Phase Transition Profiles (candidate loss vs step)
  E: Gradient Norm Profiles
  F: Fit Quality Summary (grouped bar chart)

Usage:
    python scripts/plot_landauer_comprehensive.py \\
      --results outputs/landauer_comprehensive_results.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ═══════════════════════════════════════════════════════════════════════
# Color scheme
# ═══════════════════════════════════════════════════════════════════════

# Low-K regime (no plateau): warm tones
# Plateau regime (K≥5): cool/distinct tones
K_COLORS = {
    2:  "#E53935",  # red
    3:  "#F06292",  # pink
    5:  "#1E88E5",  # blue
    10: "#43A047",  # green
    20: "#FB8C00",  # orange
    36: "#00897B",  # teal / dark green
}

K_MARKERS = {
    2:  "v",
    3:  "v",
    5:  "o",
    10: "s",
    20: "^",
    36: "D",
}


def get_k_color(k):
    return K_COLORS.get(k, "#9E9E9E")


def get_k_marker(k):
    return K_MARKERS.get(k, "o")


def is_low_k(k):
    return k in {2, 3}


# ═══════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════

def load_candidate_eval(exp_name, output_dir):
    """Load candidate_eval_results.json for an experiment."""
    path = Path(output_dir) / exp_name / "candidate_eval_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_gradient_norms(exp_name, output_dir):
    """Load gradient_norm_results.json for an experiment."""
    path = Path(output_dir) / exp_name / "gradient_norm_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# Panel A: Q_total_conv vs log(K)
# ═══════════════════════════════════════════════════════════════════════

def panel_a_q_total_conv(ax, experiments, fits):
    """Q_total_conv vs log(K) with two fit lines."""
    q_name = "Q_total_conv"

    for exp in experiments:
        k = exp["K"]
        lk = exp["log_K"]
        q = exp[q_name]
        if q is None:
            continue

        ax.scatter(lk, q, color=get_k_color(k), s=120, zorder=5,
                   marker=get_k_marker(k), edgecolors="black", linewidths=0.8)
        ax.annotate(f"K={k}", (lk, q), textcoords="offset points",
                    xytext=(8, 6), fontsize=9, color=get_k_color(k),
                    fontweight="bold")

    # Fit lines
    x_line = np.linspace(0, 4.0, 200)

    # Fit on ALL 6 points
    fit_all = fits[q_name]["all"]
    if fit_all["slope"] is not None:
        y_all = fit_all["slope"] * x_line + fit_all["intercept"]
        ax.plot(x_line, y_all, color="gray", linewidth=2.0, linestyle="-",
                alpha=0.7, zorder=2,
                label=f"All 6 points: R²={fit_all['r_squared']:.3f}")

    # Fit on PLATEAU REGIME only
    fit_plat = fits[q_name]["plateau_regime"]
    if fit_plat["slope"] is not None:
        y_plat = fit_plat["slope"] * x_line + fit_plat["intercept"]
        ax.plot(x_line, y_plat, color="#1565C0", linewidth=2.0, linestyle="-",
                alpha=0.8, zorder=3,
                label=f"Plateau regime: R²={fit_plat['r_squared']:.3f}")

    ax.set_xlabel("log(K)", fontsize=12)
    ax.set_ylabel(r"$Q_{\mathrm{total\_conv}}$", fontsize=12)
    ax.set_title("A. Q_total_conv vs log(K) — Full Protocol",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, color="lightgray")
    ax.set_xlim(0, 4.0)
    ax.set_ylim(bottom=0)


# ═══════════════════════════════════════════════════════════════════════
# Panel B: Q_zgap vs log(K)
# ═══════════════════════════════════════════════════════════════════════

def panel_b_q_zgap(ax, experiments, fits):
    """Q_zgap vs log(K). Mark invalid points with ×."""
    q_name = "Q_zgap"

    for exp in experiments:
        k = exp["K"]
        lk = exp["log_K"]
        q = exp[q_name]

        if q is not None:
            ax.scatter(lk, q, color=get_k_color(k), s=120, zorder=5,
                       marker=get_k_marker(k), edgecolors="black", linewidths=0.8)
            ax.annotate(f"K={k}", (lk, q), textcoords="offset points",
                        xytext=(8, 6), fontsize=9, color=get_k_color(k),
                        fontweight="bold")
        else:
            # Invalid point: × marker at Q=0
            ax.scatter(lk, 0, color=get_k_color(k), s=120, zorder=5,
                       marker="x", linewidths=2.5)
            ax.annotate(f"K={k}\nz_gap < 0.5", (lk, 0),
                        textcoords="offset points",
                        xytext=(8, -15), fontsize=8, color=get_k_color(k),
                        style="italic")

    # Fit on valid points
    x_line = np.linspace(0, 4.0, 200)

    fit_all = fits[q_name]["all"]
    if fit_all["slope"] is not None:
        y_all = fit_all["slope"] * x_line + fit_all["intercept"]
        ax.plot(x_line, y_all, color="gray", linewidth=2.0, linestyle="-",
                alpha=0.7, zorder=2,
                label=f"Valid points: R²={fit_all['r_squared']:.3f}")

    fit_plat = fits[q_name]["plateau_regime"]
    if fit_plat["slope"] is not None:
        y_plat = fit_plat["slope"] * x_line + fit_plat["intercept"]
        ax.plot(x_line, y_plat, color="#C62828", linewidth=2.0, linestyle="-",
                alpha=0.8, zorder=3,
                label=f"Plateau regime: R²={fit_plat['r_squared']:.3f}")

    ax.set_xlabel("log(K)", fontsize=12)
    ax.set_ylabel(r"$Q_{\mathrm{zgap}}$", fontsize=12)
    ax.set_title("B. Q_zgap vs log(K) — Functional Window",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, color="lightgray")
    ax.set_xlim(0, 4.0)
    ax.set_ylim(bottom=min(0, -0.05))


# ═══════════════════════════════════════════════════════════════════════
# Panel C: Q_original vs log(K)
# ═══════════════════════════════════════════════════════════════════════

def panel_c_q_original(ax, experiments, fits):
    """Q_original vs log(K) with two fits."""
    q_name = "Q_original"

    for exp in experiments:
        k = exp["K"]
        lk = exp["log_K"]
        q = exp[q_name]
        if q is None:
            q = 0.0

        ax.scatter(lk, q, color=get_k_color(k), s=120, zorder=5,
                   marker=get_k_marker(k), edgecolors="black", linewidths=0.8)
        label = f"K={k}"
        if q == 0.0 and is_low_k(k):
            label += " (no plateau)"
        ax.annotate(label, (lk, q), textcoords="offset points",
                    xytext=(8, 6), fontsize=9, color=get_k_color(k),
                    fontweight="bold")

    x_line = np.linspace(0, 4.0, 200)

    fit_all = fits[q_name]["all"]
    if fit_all["slope"] is not None:
        y_all = fit_all["slope"] * x_line + fit_all["intercept"]
        ax.plot(x_line, y_all, color="gray", linewidth=2.0, linestyle="-",
                alpha=0.7, zorder=2,
                label=f"All 6 points: R²={fit_all['r_squared']:.3f}")

    fit_plat = fits[q_name]["plateau_regime"]
    if fit_plat["slope"] is not None:
        y_plat = fit_plat["slope"] * x_line + fit_plat["intercept"]
        ax.plot(x_line, y_plat, color="black", linewidth=2.0, linestyle="-",
                alpha=0.8, zorder=3,
                label=f"Plateau regime: R²={fit_plat['r_squared']:.3f}")

    ax.set_xlabel("log(K)", fontsize=12)
    ax.set_ylabel(r"$Q_{\mathrm{original}}$", fontsize=12)
    ax.set_title("C. Q_original vs log(K) — Original Window",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, color="lightgray")
    ax.set_xlim(0, 4.0)
    ax.set_ylim(bottom=min(0, -0.05))


# ═══════════════════════════════════════════════════════════════════════
# Panel D: Phase Transition Profiles
# ═══════════════════════════════════════════════════════════════════════

def panel_d_phase_profiles(ax, experiments, output_dir):
    """Candidate loss vs training step for all 6 K values overlaid."""
    for exp in experiments:
        k = exp["K"]
        cand = load_candidate_eval(exp["name"], output_dir)
        if cand is None:
            continue

        steps = cand["steps"]
        loss = cand["candidate_loss"]
        ax.plot(steps, loss, color=get_k_color(k), linewidth=1.5,
                label=f"K={k}", zorder=3)

    # Horizontal dashed lines at log(K) for each K
    for exp in experiments:
        k = exp["K"]
        log_k = math.log(k)
        ax.axhline(log_k, color=get_k_color(k), linestyle="--",
                    alpha=0.35, linewidth=0.8)
        # Small label on right side
        ax.annotate(f"log({k})", xy=(1.01, log_k),
                    xycoords=("axes fraction", "data"),
                    fontsize=7, color=get_k_color(k), va="center")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Candidate Loss", fontsize=12)
    ax.set_title("D. Phase Transition Profiles",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3, color="lightgray")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-0.1)


# ═══════════════════════════════════════════════════════════════════════
# Panel E: Gradient Norm Profiles
# ═══════════════════════════════════════════════════════════════════════

def panel_e_gradient_profiles(ax, experiments, output_dir):
    """||∇L||² vs training step for all 6 K values."""
    for exp in experiments:
        k = exp["K"]
        grad = load_gradient_norms(exp["name"], output_dir)
        if grad is None:
            continue

        steps = grad["steps"]
        norm_sq = grad["total_grad_norm_sq"]
        ax.plot(steps, norm_sq, color=get_k_color(k), linewidth=1.5,
                label=f"K={k}", zorder=3)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel(r"$\|\nabla L\|^2$", fontsize=12)
    ax.set_title(r"E. Gradient Norm Profiles $\|\nabla L\|^2$",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3, color="lightgray")
    ax.set_xlim(left=0)


# ═══════════════════════════════════════════════════════════════════════
# Panel F: Fit Quality Summary
# ═══════════════════════════════════════════════════════════════════════

def panel_f_fit_quality(ax, fits):
    """Grouped bar chart: R² for all vs plateau_regime, for each Q definition."""
    q_names = ["Q_original", "Q_total_conv", "Q_zgap", "Q_gradspike"]
    q_labels = [r"$Q_\mathrm{orig}$", r"$Q_\mathrm{total\_conv}$",
                r"$Q_\mathrm{zgap}$", r"$Q_\mathrm{gradspike}$"]

    r2_all = []
    r2_plat = []

    for q_name in q_names:
        fit_a = fits[q_name]["all"]
        fit_p = fits[q_name]["plateau_regime"]
        r2_all.append(fit_a["r_squared"] if fit_a["r_squared"] is not None else 0.0)
        r2_plat.append(fit_p["r_squared"] if fit_p["r_squared"] is not None else 0.0)

    x = np.arange(len(q_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, r2_all, width, color="#BBDEFB",
                   edgecolor="#1565C0", linewidth=0.8,
                   label="All 6 K values", alpha=0.85)
    bars2 = ax.bar(x + width / 2, r2_plat, width, color="#1565C0",
                   edgecolor="#0D47A1", linewidth=0.8,
                   label="Plateau regime", alpha=0.85)

    # Value labels
    for bar, val in zip(bars1, r2_all):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color="#1565C0")
    for bar, val in zip(bars2, r2_plat):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color="#0D47A1")

    ax.axhline(0.95, color="gray", linewidth=0.8, linestyle=":",
               alpha=0.6, label="R² = 0.95")
    ax.set_xticks(x)
    ax.set_xticklabels(q_labels, fontsize=10)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("F. Fit Quality Summary",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3, color="lightgray")
    ax.set_ylim(0, 1.12)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Plot comprehensive Landauer analysis"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="outputs/landauer_comprehensive_results.json",
        help="Path to comprehensive results JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory (for loading raw experiment data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/figures/landauer_comprehensive.png",
        help="Output figure path",
    )
    args = parser.parse_args()

    # Load results
    with open(args.results) as f:
        data = json.load(f)

    experiments = data["experiments"]
    fits = data["fits"]

    print(f"Loaded {len(experiments)} experiments")
    for e in experiments:
        print(f"  K={e['K']}: plateau={'YES' if e['has_plateau'] else 'NO'}, "
              f"Q_total_conv={e['Q_total_conv']:.4f}")

    # ── Create 2×3 figure ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        "Comprehensive Landauer Analysis — 6 K Values\n"
        "Phase Transition Dissipation Scales with log(K)",
        fontsize=16, fontweight="bold", y=0.99
    )

    # Panel A: Q_total_conv
    panel_a_q_total_conv(axes[0, 0], experiments, fits)

    # Panel B: Q_zgap
    panel_b_q_zgap(axes[0, 1], experiments, fits)

    # Panel C: Q_original
    panel_c_q_original(axes[0, 2], experiments, fits)

    # Panel D: Phase transition profiles
    panel_d_phase_profiles(axes[1, 0], experiments, args.output_dir)

    # Panel E: Gradient norm profiles
    panel_e_gradient_profiles(axes[1, 1], experiments, args.output_dir)

    # Panel F: Fit quality summary
    panel_f_fit_quality(axes[1, 2], fits)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
