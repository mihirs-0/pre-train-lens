#!/usr/bin/env python
"""
Plot tautology check and decomposition analyses.

2×2 figure:
  A: Tautology Test — Q_zgap vs ΔL_zgap
  B: Excess Dissipation — Q_excess vs log(K)
  C: Plateau Decomposition — Stacked Bars
  D: Regime-Aware Fit — Q = c · log(K/K*)

Usage:
    python scripts/plot_tautology_and_decomposition.py \\
      --tautology outputs/tautology_check_results.json \\
      --decomposition outputs/plateau_decomposition_results.json \\
      --regime outputs/regime_fit_results.json \\
      --comprehensive outputs/landauer_comprehensive_results.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ═══════════════════════════════════════════════════════════════════════
# Color scheme (consistent with plot_landauer_comprehensive.py)
# ═══════════════════════════════════════════════════════════════════════

K_COLORS = {
    2:  "#E53935",
    3:  "#F06292",
    5:  "#1E88E5",
    10: "#43A047",
    15: "#8E24AA",
    20: "#FB8C00",
    25: "#00ACC1",
    36: "#00897B",
}

K_MARKERS = {
    2:  "v",
    3:  "v",
    5:  "o",
    10: "s",
    15: "p",
    20: "^",
    25: "h",
    36: "D",
}


def get_k_color(k):
    return K_COLORS.get(k, "#9E9E9E")


def get_k_marker(k):
    return K_MARKERS.get(k, "o")


# ═══════════════════════════════════════════════════════════════════════
# Panel A: Tautology Test — Q_zgap vs ΔL_zgap
# ═══════════════════════════════════════════════════════════════════════

def panel_a_tautology(ax, tautology_data):
    """Q_zgap vs ΔL_zgap with Q = ΔL tautology line."""
    exps = tautology_data["experiments"]

    delta_L_vals = [e["delta_L_zgap"] for e in exps]
    Q_vals = [e["Q_zgap"] for e in exps]

    # Plot each point
    for e in exps:
        k = e["K"]
        ax.scatter(e["delta_L_zgap"], e["Q_zgap"],
                   color=get_k_color(k), s=140, zorder=5,
                   marker=get_k_marker(k), edgecolors="black", linewidths=0.8)
        ax.annotate(
            f"K={k}\nQ/ΔL={e['dissipation_ratio']:.0f}×",
            (e["delta_L_zgap"], e["Q_zgap"]),
            textcoords="offset points", xytext=(12, -5),
            fontsize=8.5, color=get_k_color(k), fontweight="bold",
        )

    # Tautology line: Q = ΔL
    max_dl = max(delta_L_vals) * 1.2
    ax.plot([0, max_dl], [0, max_dl], "k--", linewidth=1.5, alpha=0.5,
            label="Q = ΔL (tautology)")

    # Shade the region above the line
    ax.fill_between([0, max_dl], [0, max_dl], [max(Q_vals) * 1.3] * 2,
                    color="#E3F2FD", alpha=0.3, label="Excess dissipation")

    ax.set_xlabel("ΔL_zgap (loss decrease over z-gap window)", fontsize=11)
    ax.set_ylabel("Q_zgap (gradient dissipation)", fontsize=11)
    ax.set_title("A. Tautology Test: Q_zgap vs ΔL_zgap",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, color="lightgray")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


# ═══════════════════════════════════════════════════════════════════════
# Panel B: Excess Dissipation — Q_excess vs log(K)
# ═══════════════════════════════════════════════════════════════════════

def panel_b_excess(ax, tautology_data):
    """Q_excess = Q_zgap - ΔL_zgap vs log(K) with linear fit."""
    exps = tautology_data["experiments"]
    fits = tautology_data["fits"]

    log_k_vals = [e["log_K"] for e in exps]
    Q_excess_vals = [e["Q_excess"] for e in exps]

    # Plot points
    for e in exps:
        k = e["K"]
        ax.scatter(e["log_K"], e["Q_excess"],
                   color=get_k_color(k), s=140, zorder=5,
                   marker=get_k_marker(k), edgecolors="black", linewidths=0.8)
        ax.annotate(f"K={k}", (e["log_K"], e["Q_excess"]),
                    textcoords="offset points", xytext=(8, 6),
                    fontsize=9, color=get_k_color(k), fontweight="bold")

    # Linear fit
    fit = fits["Q_excess_vs_logK"]
    if fit["slope"] is not None:
        x_line = np.linspace(min(log_k_vals) - 0.2, max(log_k_vals) + 0.2, 200)
        y_line = fit["slope"] * x_line + fit["intercept"]
        ax.plot(x_line, y_line, color="#1565C0", linewidth=2.0,
                label=f"Q_excess = {fit['slope']:.1f}·log(K) "
                      f"+ ({fit['intercept']:.1f})\nR² = {fit['r_squared']:.3f}")

    ax.set_xlabel("log(K)", fontsize=11)
    ax.set_ylabel("Q_excess = Q_zgap − ΔL_zgap", fontsize=11)
    ax.set_title("B. Excess Dissipation vs log(K)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, color="lightgray")
    ax.set_ylim(bottom=0)


# ═══════════════════════════════════════════════════════════════════════
# Panel C: Plateau Decomposition — Stacked Bars
# ═══════════════════════════════════════════════════════════════════════

def panel_c_decomposition(ax, decomp_data):
    """Stacked bar chart showing Q_prior, Q_plateau, Q_transition."""
    exps = decomp_data["experiments"]
    fits = decomp_data["fits"]

    labels = [f"K={e['K']}" for e in exps]
    Q_prior = [e["Q_prior"] for e in exps]
    Q_plateau = [e["Q_plateau"] for e in exps]
    Q_transition = [e["Q_transition"] for e in exps]

    x = np.arange(len(labels))
    width = 0.55

    bars1 = ax.bar(x, Q_prior, width, label="Q_prior (learning prior)",
                   color="#BBDEFB", edgecolor="#1565C0", linewidth=0.8)
    bars2 = ax.bar(x, Q_plateau, width, bottom=Q_prior,
                   label="Q_plateau (memorization phase)",
                   color="#90CAF9", edgecolor="#1565C0", linewidth=0.8)
    bottom_trans = [p + pl for p, pl in zip(Q_prior, Q_plateau)]
    bars3 = ax.bar(x, Q_transition, width, bottom=bottom_trans,
                   label="Q_transition (Landauer signal)",
                   color="#1565C0", edgecolor="#0D47A1", linewidth=0.8)

    # Add R² annotations
    r2_prior = fits["Q_prior_vs_logK"]["r_squared"]
    r2_plateau = fits["Q_plateau_vs_logK"]["r_squared"]
    r2_trans = fits["Q_transition_vs_logK"]["r_squared"]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Dissipation Q", fontsize=11)
    ax.set_title("C. Plateau Decomposition",
                 fontsize=13, fontweight="bold")

    # Legend with R² info
    legend_labels = [
        f"Q_prior (R²={r2_prior:.3f})" if r2_prior is not None else "Q_prior",
        f"Q_plateau (R²={r2_plateau:.3f})" if r2_plateau is not None else "Q_plateau",
        f"Q_transition (R²={r2_trans:.3f})" if r2_trans is not None else "Q_transition",
    ]
    ax.legend(
        [bars1, bars2, bars3], legend_labels,
        fontsize=8.5, loc="upper left",
    )
    ax.grid(True, axis="y", alpha=0.3, color="lightgray")
    ax.set_ylim(bottom=0)


# ═══════════════════════════════════════════════════════════════════════
# Panel D: Regime-Aware Fit
# ═══════════════════════════════════════════════════════════════════════

def panel_d_regime_fit(ax, regime_data, comp_data):
    """Q = c · log(K/K*) with all data points."""
    # Use Q_zgap definition
    if "Q_zgap" not in regime_data:
        ax.text(0.5, 0.5, "No Q_zgap regime fit data",
                transform=ax.transAxes, ha="center", fontsize=12)
        return

    rd = regime_data["Q_zgap"]
    K_star = rd["best_K_star"]
    c = rd["best_c"]
    r2 = rd["best_r_squared"]

    # Plot all experiment points from comprehensive results
    experiments = [e for e in comp_data["experiments"] if e["K"] <= 36]
    experiments.sort(key=lambda x: x["K"])

    for e in experiments:
        k = e["K"]
        q = e["Q_zgap"]
        lk = math.log(k)
        if q is None:
            continue
        ax.scatter(lk, q, color=get_k_color(k), s=140, zorder=5,
                   marker=get_k_marker(k), edgecolors="black", linewidths=0.8)
        ax.annotate(f"K={k}", (lk, q), textcoords="offset points",
                    xytext=(8, 6), fontsize=9, color=get_k_color(k),
                    fontweight="bold")

    # Regime-aware fit curve
    x_line = np.linspace(0.3, 4.0, 500)
    K_line = np.exp(x_line)

    if K_star is not None and c is not None:
        y_regime = np.where(
            K_line > K_star,
            c * np.log(K_line / K_star),
            0.0,
        )
        ax.plot(x_line, y_regime, color="#1565C0", linewidth=2.5, zorder=3,
                label=f"Q = {c:.1f}·log(K/{K_star:.1f})\nR² = {r2:.3f}")

        # Vertical line at K*
        log_K_star = math.log(K_star)
        ax.axvline(log_K_star, color="#1565C0", linestyle=":", linewidth=1.5,
                   alpha=0.7, zorder=2)
        ax.annotate(f"K* = {K_star:.1f}", (log_K_star, 0),
                    textcoords="offset points", xytext=(6, 10),
                    fontsize=10, color="#1565C0", fontweight="bold")

    # Old affine fit for comparison
    af = rd.get("affine_fit", {})
    if af.get("slope") is not None:
        y_affine = af["slope"] * x_line + af["intercept"]
        ax.plot(x_line, y_affine, color="gray", linewidth=1.5,
                linestyle="--", alpha=0.6, zorder=2,
                label=f"Affine: {af['slope']:.1f}·log(K) "
                      f"+ ({af['intercept']:.0f})\n"
                      f"R² = {af['r_squared']:.3f}")

    ax.set_xlabel("log(K)", fontsize=11)
    ax.set_ylabel("Q_zgap", fontsize=11)
    ax.set_title("D. Regime-Aware Fit: Q = c · log(K/K*)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, color="lightgray")
    ax.set_xlim(0.3, 4.0)
    ax.set_ylim(bottom=-10)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Plot tautology check and decomposition analyses"
    )
    parser.add_argument(
        "--tautology",
        type=str,
        default="outputs/tautology_check_results.json",
    )
    parser.add_argument(
        "--decomposition",
        type=str,
        default="outputs/plateau_decomposition_results.json",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="outputs/regime_fit_results.json",
    )
    parser.add_argument(
        "--comprehensive",
        type=str,
        default="outputs/landauer_comprehensive_results.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/figures/tautology_and_decomposition.png",
    )
    args = parser.parse_args()

    # Load all results
    with open(args.tautology) as f:
        tautology_data = json.load(f)
    with open(args.decomposition) as f:
        decomp_data = json.load(f)
    with open(args.regime) as f:
        regime_data = json.load(f)
    with open(args.comprehensive) as f:
        comp_data = json.load(f)

    # Create 2×2 figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Tautology Check + Critical Analyses\n"
        "Excess Dissipation Is Non-Trivial and Scales with log(K)",
        fontsize=15, fontweight="bold", y=0.99,
    )

    # Panel A: Tautology test
    panel_a_tautology(axes[0, 0], tautology_data)

    # Panel B: Excess dissipation
    panel_b_excess(axes[0, 1], tautology_data)

    # Panel C: Plateau decomposition
    panel_c_decomposition(axes[1, 0], decomp_data)

    # Panel D: Regime-aware fit
    panel_d_regime_fit(axes[1, 1], regime_data, comp_data)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
