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
import matplotlib.gridspec as gridspec


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


def panel_d_fit_and_residuals(ax_fit, ax_resid, experiments):
    """Panel D: Q_trans vs log(K) with linear fit (top) and residuals (bottom)."""
    valid = [e for e in experiments if e["Q_transition"] is not None]
    if len(valid) < 2:
        ax_fit.text(0.5, 0.5, "Insufficient data", transform=ax_fit.transAxes,
                    ha="center", fontsize=12)
        ax_fit.set_title("D. Fit & Residuals", fontsize=11, fontweight="bold")
        return

    log_ks = np.array([e["log_K"] for e in valid])
    q_trans = np.array([e["Q_transition"] for e in valid])
    ks = np.array([e["K"] for e in valid])

    # --- Linear fit ---
    coeffs = np.polyfit(log_ks, q_trans, 1)
    slope, intercept = coeffs
    predicted = np.polyval(coeffs, log_ks)
    residuals = q_trans - predicted

    # R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((q_trans - np.mean(q_trans)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # RMSE and max residual
    rmse = np.sqrt(np.mean(residuals ** 2))
    max_abs_resid = np.max(np.abs(residuals))

    # --- Top panel: scatter + fit ---
    x_fit = np.linspace(min(log_ks) - 0.15, max(log_ks) + 0.15, 100)
    y_fit = np.polyval(coeffs, x_fit)

    # Confidence-like band (±RMSE around fit)
    ax_fit.fill_between(x_fit, y_fit - rmse, y_fit + rmse,
                        color="#BBDEFB", alpha=0.4, label=f"±RMSE = {rmse:.2f}")
    ax_fit.plot(x_fit, y_fit, "k-", linewidth=1.8, alpha=0.8,
                label=f"Fit: {slope:.2f}·log(K) − {abs(intercept):.2f}   R²={r_sq:.3f}")

    for e, pred_val in zip(valid, predicted):
        color = get_color(e["K"])
        ax_fit.scatter(e["log_K"], e["Q_transition"], color=color, s=90,
                       zorder=5, edgecolors="black", linewidths=0.8)
        ax_fit.annotate(f"K={e['K']}", (e["log_K"], e["Q_transition"]),
                        textcoords="offset points", xytext=(8, 6), fontsize=9)

    ax_fit.set_ylabel(r"$Q_{\mathrm{trans}}$", fontsize=10)
    ax_fit.set_title("D. Linear Fit & Residuals", fontsize=11, fontweight="bold")
    ax_fit.legend(fontsize=8, loc="upper left")
    ax_fit.grid(True, alpha=0.3)
    ax_fit.set_xlim(x_fit[0], x_fit[-1])
    ax_fit.set_ylim(bottom=0)
    plt.setp(ax_fit.get_xticklabels(), visible=False)

    # --- Bottom panel: residuals ---
    colors_resid = [get_color(k) for k in ks]
    markerline, stemlines, baseline = ax_resid.stem(
        log_ks, residuals, linefmt="-", markerfmt="o", basefmt="k-")
    baseline.set_linewidth(0.8)
    # Color each marker
    for i, (ml, sl) in enumerate(zip(
            [markerline] if len(valid) == 1 else [markerline],
            stemlines if hasattr(stemlines, '__iter__') else [stemlines])):
        pass  # stem doesn't easily support per-point colors; use scatter overlay

    # Overlay colored scatter on top of stem markers
    for e, res in zip(valid, residuals):
        color = get_color(e["K"])
        ax_resid.scatter(e["log_K"], res, color=color, s=70,
                         zorder=5, edgecolors="black", linewidths=0.7)
        pct = res / e["Q_transition"] * 100
        ax_resid.annotate(f"{res:+.2f}\n({pct:+.1f}%)",
                          (e["log_K"], res),
                          textcoords="offset points",
                          xytext=(12, -2 if res < 0 else 2),
                          fontsize=7.5, ha="left",
                          color="dimgray")

    ax_resid.axhline(0, color="black", linewidth=0.8)
    ax_resid.set_xlabel("log(K)", fontsize=10)
    ax_resid.set_ylabel("Residual", fontsize=9)
    ax_resid.grid(True, alpha=0.3)
    ax_resid.set_xlim(x_fit[0], x_fit[-1])

    # Symmetric y limits for residuals
    y_lim = max(abs(residuals.min()), abs(residuals.max())) * 1.6
    ax_resid.set_ylim(-y_lim, y_lim)

    # Summary annotation
    ax_resid.text(
        0.98, 0.95,
        f"RMSE = {rmse:.2f}\nmax |resid| = {max_abs_resid:.2f}",
        transform=ax_resid.transAxes, ha="right", va="top",
        fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
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

    # Create figure with GridSpec: Panel D gets two vertically stacked axes
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])

    # Panel D: split into fit (top 70%) and residuals (bottom 30%)
    gs_d = gs[1, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax_d_fit = fig.add_subplot(gs_d[0])
    ax_d_resid = fig.add_subplot(gs_d[1], sharex=ax_d_fit)

    fig.suptitle(
        "Landauer Dissipation Test — Constant LR\n"
        "Does Q_transition scale with log(K)?",
        fontsize=13, fontweight="bold", y=0.98,
    )

    panel_a_cumulative_dissipation(ax_a, experiments)
    panel_b_scaling(ax_b, experiments)
    panel_c_gradient_profiles(ax_c, experiments, args.output_dir)
    panel_d_fit_and_residuals(ax_d_fit, ax_d_resid, experiments)

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

    # Use subplots_adjust instead of tight_layout (incompatible with subgridspec)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.06, hspace=0.35, wspace=0.3)

    # Save
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure to: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
