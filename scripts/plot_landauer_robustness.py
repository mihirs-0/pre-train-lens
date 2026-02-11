#!/usr/bin/env python
"""
Plot Landauer robustness analysis: compare four Q definitions.

Produces a 2×2 figure:
  A: Q_total vs log(K) — full training dissipation (no window)
  B: Q_zgap vs log(K)  — z-shuffle functional window
  C: Q_gradspike vs log(K) — gradient norm spike window
  D: Fit quality comparison (bar chart of intercept, R², R²_origin)

Usage:
    python scripts/plot_landauer_robustness.py \
      --results outputs/landauer_robustness_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Colors for K values (matching existing plots) ──
K_COLORS = {
    5:  "#9E9E9E",  # gray
    10: "#2196F3",  # blue
    20: "#FF9800",  # orange
    36: "#4CAF50",  # green
}

# ── Colors/styles for Q definitions ──
Q_STYLES = {
    "Q_original":  {"color": "black",   "linestyle": "-",  "marker": "o", "label": r"$Q_\mathrm{original}$"},
    "Q_total":     {"color": "#1565C0", "linestyle": "-",  "marker": "s", "label": r"$Q_\mathrm{total}$"},
    "Q_zgap":      {"color": "#C62828", "linestyle": "--", "marker": "^", "label": r"$Q_\mathrm{zgap}$"},
    "Q_gradspike": {"color": "#2E7D32", "linestyle": ":",  "marker": "D", "label": r"$Q_\mathrm{gradspike}$"},
}

# Bar chart colors for Q definitions
Q_BAR_COLORS = {
    "Q_original":  "#616161",
    "Q_total":     "#1565C0",
    "Q_zgap":      "#C62828",
    "Q_gradspike": "#2E7D32",
}


def get_k_color(k):
    return K_COLORS.get(k, "#9E9E9E")


def panel_q_vs_logk(ax, experiments, fits, q_name, title, subtitle):
    """
    Generic panel: Q vs log(K) with linear fit and through-origin fit.
    """
    fit = fits[q_name]
    style = Q_STYLES[q_name]

    # Extract data points
    log_k_vals = []
    q_vals = []
    k_vals = []
    for exp in experiments:
        if exp[q_name] is not None:
            log_k_vals.append(exp["log_K"])
            q_vals.append(exp[q_name])
            k_vals.append(exp["K"])

    log_k_arr = np.array(log_k_vals)
    q_arr = np.array(q_vals)

    # Plot data points colored by K
    for lk, q, k in zip(log_k_vals, q_vals, k_vals):
        ax.scatter(lk, q, color=get_k_color(k), s=100, zorder=5,
                   edgecolors="black", linewidths=0.8)
        ax.annotate(f"K={k}", (lk, q), textcoords="offset points",
                    xytext=(8, 5), fontsize=9, color=get_k_color(k),
                    fontweight="bold")

    # Linear fit
    if fit["slope"] is not None:
        x_line = np.linspace(0, max(log_k_arr) * 1.1, 100)
        y_line = fit["slope"] * x_line + fit["intercept"]
        ax.plot(x_line, y_line, color=style["color"], linewidth=1.8,
                linestyle="-", alpha=0.8, zorder=3,
                label=f"Linear: R²={fit['r_squared']:.3f}")

        # Annotate fit equation
        eq_str = f"Q = {fit['slope']:.2f}·log(K) {'+' if fit['intercept'] >= 0 else '−'} {abs(fit['intercept']):.2f}"
        ax.annotate(eq_str, xy=(0.05, 0.92), xycoords="axes fraction",
                    fontsize=10, fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="gray", alpha=0.9))

        # Through-origin fit
        if fit["slope_origin"] is not None:
            y_origin = fit["slope_origin"] * x_line
            ax.plot(x_line, y_origin, color=style["color"], linewidth=1.5,
                    linestyle="--", alpha=0.5, zorder=2,
                    label=f"Through origin: R²={fit['r_squared_origin']:.3f}")

    ax.set_xlabel("log(K)", fontsize=12)
    ax.set_ylabel(f"{style['label']}", fontsize=12)
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold",
                 linespacing=1.4)
    # Make subtitle part italic and smaller via two-line title
    # (matplotlib doesn't support mixed styles easily, so use annotation)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=18)
    ax.annotate(subtitle, xy=(0.5, 1.0), xycoords="axes fraction",
                fontsize=9, ha="center", va="bottom", style="italic",
                color="#555555", xytext=(0, 4), textcoords="offset points")

    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, color="lightgray")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=min(0, min(q_vals) * 1.1) if q_vals else 0)


def panel_d_comparison(ax, fits):
    """
    Panel D: Three side-by-side bar groups (Intercept, R², R²_origin)
    with separate visual treatment for the different scales.

    Uses the main ax for layout and creates three sub-regions via inset axes.
    """
    ax.set_axis_off()
    ax.set_title("D. Fit Quality Comparison", fontsize=14, fontweight="bold",
                 pad=12)

    q_names = ["Q_original", "Q_total", "Q_zgap", "Q_gradspike"]
    short_labels = [r"$Q_\mathrm{orig}$", r"$Q_\mathrm{total}$",
                    r"$Q_\mathrm{zgap}$", r"$Q_\mathrm{spike}$"]
    colors = [Q_BAR_COLORS[n] for n in q_names]

    # Collect metrics (skip None fits)
    intercepts, r2_vals, r2_origin_vals = [], [], []
    valid_labels, valid_colors = [], []
    n_points_list = []

    for name, label, color in zip(q_names, short_labels, colors):
        f = fits[name]
        if f["slope"] is not None:
            intercepts.append(f["intercept"])
            r2_vals.append(f["r_squared"])
            r2_origin_vals.append(f["r_squared_origin"])
            valid_labels.append(label)
            valid_colors.append(color)
            n_points_list.append(f["n_points"])

    n = len(valid_labels)
    if n == 0:
        ax.text(0.5, 0.5, "No fits available", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        return

    # Three inset axes side by side
    metrics = [
        ("Intercept", intercepts, "{:.1f}"),
        ("R²", r2_vals, "{:.3f}"),
        ("R² (origin)", r2_origin_vals, "{:.3f}"),
    ]

    gap = 0.06
    w = (1.0 - 4 * gap) / 3
    for mi, (metric_name, vals, fmt) in enumerate(metrics):
        left = gap + mi * (w + gap)
        inset = ax.inset_axes([left, 0.08, w, 0.78])

        x = np.arange(n)
        bars = inset.bar(x, vals, color=valid_colors, edgecolor="white",
                         linewidth=0.5, alpha=0.85, width=0.7)

        for bar, val in zip(bars, vals):
            y_pos = bar.get_height()
            va_pos = "bottom" if y_pos >= 0 else "top"
            offset_y = abs(max(vals) - min(vals)) * 0.03
            if y_pos < 0:
                offset_y = -offset_y
            inset.text(bar.get_x() + bar.get_width() / 2, y_pos + offset_y,
                       fmt.format(val), ha="center", va=va_pos, fontsize=8,
                       fontweight="bold")

        inset.set_xticks(x)
        inset.set_xticklabels(valid_labels, fontsize=8)
        inset.set_title(metric_name, fontsize=11, fontweight="bold")
        inset.grid(True, axis="y", alpha=0.3, color="lightgray")
        inset.axhline(0, color="black", linewidth=0.5)

        # R² reference line
        if "R²" in metric_name:
            inset.axhline(0.95, color="gray", linewidth=0.8,
                          linestyle=":", alpha=0.5)
            inset.set_ylim(0, 1.08)

        # Add n_points annotation for Q_zgap
        for i, np_ in enumerate(n_points_list):
            if np_ < 4 and "R²" not in metric_name:
                inset.annotate(f"n={np_}", (i, vals[i]),
                               textcoords="offset points",
                               xytext=(0, -12), fontsize=7,
                               ha="center", color="#888")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Landauer robustness analysis"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="outputs/landauer_robustness_results.json",
        help="Path to robustness results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/figures/landauer_robustness.png",
        help="Output figure path",
    )
    args = parser.parse_args()

    # Load results
    with open(args.results) as f:
        data = json.load(f)

    experiments = data["experiments"]
    fits = data["fits"]

    # ── Create 2×2 figure ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Landauer Robustness: Three Alternative Q Definitions",
        fontsize=16, fontweight="bold", y=0.98
    )

    # Panel A: Q_total
    panel_q_vs_logk(
        axes[0, 0], experiments, fits, "Q_total",
        title="A. Full Training Dissipation",
        subtitle="No window definition — eliminates kinematic confound",
    )

    # Panel B: Q_zgap
    panel_q_vs_logk(
        axes[0, 1], experiments, fits, "Q_zgap",
        title="B. z-Shuffle Diagnostic Window",
        subtitle="Window: z_gap ∈ [0.5, 0.9·max] — K-independent thresholds",
    )

    # Panel C: Q_gradspike
    panel_q_vs_logk(
        axes[1, 0], experiments, fits, "Q_gradspike",
        title="C. Gradient Norm Spike Window",
        subtitle="Window: ||∇L||² > 2× baseline — K-independent threshold",
    )

    # Panel D: Comparison bar chart
    panel_d_comparison(axes[1, 1], fits)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
