"""
Phase 1 analysis: bootstrap CIs, go/no-go decision, figures.

Usage:
    python scripts/analyze_ppt.py
    python scripts/analyze_ppt.py --results outputs/ppt_phase1/all_results.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Bootstrap utilities ───


def bootstrap_ci(
    data: List[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for the mean.

    Returns (mean, lower, upper).
    """
    rng = np.random.RandomState(seed)
    arr = np.array(data, dtype=float)
    means = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = (1 - ci) / 2
    return float(arr.mean()), float(np.percentile(means, 100 * alpha)), float(
        np.percentile(means, 100 * (1 - alpha))
    )


def bootstrap_diff_ci(
    a: List[float],
    b: List[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for (mean(a) - mean(b)).

    Returns (mean_diff, lower, upper).
    """
    rng = np.random.RandomState(seed)
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    diffs = []
    for _ in range(n_bootstrap):
        a_sample = rng.choice(a_arr, size=len(a_arr), replace=True).mean()
        b_sample = rng.choice(b_arr, size=len(b_arr), replace=True).mean()
        diffs.append(a_sample - b_sample)
    diffs = np.array(diffs)
    alpha = (1 - ci) / 2
    mean_diff = float(a_arr.mean() - b_arr.mean())
    return mean_diff, float(np.percentile(diffs, 100 * alpha)), float(
        np.percentile(diffs, 100 * (1 - alpha))
    )


# ─── Grouping helpers ───


def group_by_condition(results: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for r in results:
        c = r["condition"]
        groups.setdefault(c, []).append(r)
    return groups


def extract_tau_z(results: List[dict], key: str = "tau_z_shuffle") -> List[float]:
    """Extract τ_z values, treating None (never detected) as max_steps."""
    vals = []
    for r in results:
        v = r.get(key)
        if v is None:
            # Use max_steps as a conservative upper bound
            max_steps = max(r.get("steps", [0])) if r.get("steps") else 20000
            vals.append(float(max_steps))
        else:
            vals.append(float(v))
    return vals


# ─── Summary + go/no-go ───


def print_summary(results: List[dict]) -> None:
    """Print summary table and go/no-go decision."""
    groups = group_by_condition(results)

    print("\n" + "=" * 70)
    print("  PPT Phase 1 Summary")
    print("=" * 70)

    tau_z_by_condition = {}

    for cond in ["C0", "C1", "C2"]:
        if cond not in groups:
            continue
        runs = groups[cond]
        tau_vals = extract_tau_z(runs, "tau_z_shuffle")
        tau_cand = extract_tau_z(runs, "tau_z_candidate")
        tau_z_by_condition[cond] = tau_vals

        mean, lo, hi = bootstrap_ci(tau_vals)
        mean_c, lo_c, hi_c = bootstrap_ci(tau_cand)

        converged = sum(1 for r in runs if r.get("ppt_converged", True))
        final_losses = [r["final_loss"] for r in runs if r["final_loss"] is not None]
        final_gaps = [r["final_z_gap"] for r in runs if r["final_z_gap"] is not None]

        print(f"\n  {cond} ({len(runs)} seeds)")
        print(f"    τ_z (shuffle):    {mean:8.0f}  [{lo:.0f}, {hi:.0f}]")
        print(f"    τ_z (candidate):  {mean_c:8.0f}  [{lo_c:.0f}, {hi_c:.0f}]")
        if final_losses:
            print(f"    Final loss:       {np.mean(final_losses):8.4f}")
        if final_gaps:
            print(f"    Final z_gap:      {np.mean(final_gaps):8.4f}")
        if cond != "C0":
            print(f"    PPT converged:    {converged}/{len(runs)}")

    # ── Go/no-go decision ──
    print("\n" + "-" * 70)
    print("  Go / No-Go Decision")
    print("-" * 70)

    if "C0" in tau_z_by_condition and "C2" in tau_z_by_condition:
        c0_tau = tau_z_by_condition["C0"]
        c2_tau = tau_z_by_condition["C2"]
        diff_mean, diff_lo, diff_hi = bootstrap_diff_ci(c0_tau, c2_tau)

        print(f"\n  C0 - C2 τ_z difference: {diff_mean:.0f}  [{diff_lo:.0f}, {diff_hi:.0f}]")

        if diff_lo > 0:
            print("  >> GO: C2 (Shuffle-Dyck) reliably accelerates z-learning")
            print("     (95% CI for τ_z reduction is entirely positive)")
        elif diff_mean > 0:
            print("  >> MARGINAL: C2 shows tendency to accelerate but CI crosses 0")
            print("     Consider more seeds or longer training")
        else:
            print("  >> NO-GO: C2 does not accelerate z-learning relative to C0")

    if "C0" in tau_z_by_condition and "C1" in tau_z_by_condition:
        c0_tau = tau_z_by_condition["C0"]
        c1_tau = tau_z_by_condition["C1"]
        diff_mean, diff_lo, diff_hi = bootstrap_diff_ci(c0_tau, c1_tau)

        print(f"\n  C0 - C1 τ_z difference: {diff_mean:.0f}  [{diff_lo:.0f}, {diff_hi:.0f}]")
        if diff_lo > 0:
            print("  >> C1 (Markov) also accelerates — structural specificity unclear")
        elif diff_mean > 0:
            print("  >> C1 shows weak acceleration")
        else:
            print("  >> C1 does not accelerate (expected for generic warm-start)")

    if (
        "C1" in tau_z_by_condition
        and "C2" in tau_z_by_condition
    ):
        c1_tau = tau_z_by_condition["C1"]
        c2_tau = tau_z_by_condition["C2"]
        diff_mean, diff_lo, diff_hi = bootstrap_diff_ci(c1_tau, c2_tau)

        print(f"\n  C1 - C2 τ_z difference: {diff_mean:.0f}  [{diff_lo:.0f}, {diff_hi:.0f}]")
        if diff_lo > 0:
            print("  >> C2 beats C1 — hierarchical structure matters beyond warm-start")
        else:
            print("  >> Cannot distinguish C2 from C1 — may be generic warm-start effect")

    print()


# ─── Plotting ───


def plot_z_gap_curves(results: List[dict], output_dir: Path) -> None:
    """Overlay z_gap traces for all conditions. PRIMARY FIGURE."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not installed, skipping plots")
        return

    groups = group_by_condition(results)
    colors = {"C0": "#1f77b4", "C1": "#ff7f0e", "C2": "#2ca02c"}

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for cond in ["C0", "C1", "C2"]:
        if cond not in groups:
            continue
        for r in groups[cond]:
            steps = r["steps"]
            z_gaps = r["z_gap_trace"]
            ax.plot(
                steps,
                z_gaps,
                color=colors.get(cond, "gray"),
                alpha=0.3,
                linewidth=0.8,
            )
        # Plot mean
        all_steps = groups[cond][0]["steps"]
        all_gaps = np.array([r["z_gap_trace"] for r in groups[cond]])
        if len(all_gaps) > 0 and all(len(g) == len(all_steps) for g in all_gaps):
            mean_gaps = all_gaps.mean(axis=0)
            ax.plot(
                all_steps,
                mean_gaps,
                color=colors.get(cond, "gray"),
                linewidth=2.5,
                label=f"{cond} (mean)",
            )

    ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="threshold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("z_gap (loss_shuffled - loss_clean)")
    ax.set_title("Z-Gap Traces: When Does the Model Start Using z?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "z_gap_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved z_gap_curves.png")


def plot_candidate_loss_curves(results: List[dict], output_dir: Path) -> None:
    """Overlay candidate loss with log(K) line."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    groups = group_by_condition(results)
    colors = {"C0": "#1f77b4", "C1": "#ff7f0e", "C2": "#2ca02c"}
    k = results[0].get("K", 10)
    log_k = math.log(k)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for cond in ["C0", "C1", "C2"]:
        if cond not in groups:
            continue
        for r in groups[cond]:
            if r["candidate_steps"] and r["candidate_loss_trace"]:
                ax.plot(
                    r["candidate_steps"],
                    r["candidate_loss_trace"],
                    color=colors.get(cond, "gray"),
                    alpha=0.3,
                    linewidth=0.8,
                )

    ax.axhline(y=log_k, color="red", linestyle="--", alpha=0.5, label=f"log(K)={log_k:.2f}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Candidate Loss (nats)")
    ax.set_title("Candidate-Normalized Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "candidate_loss_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved candidate_loss_curves.png")


def plot_tau_z_distributions(results: List[dict], output_dir: Path) -> None:
    """Box plot of τ_z by condition."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    groups = group_by_condition(results)
    conditions = ["C0", "C1", "C2"]
    data = []
    labels = []
    for cond in conditions:
        if cond in groups:
            tau_vals = extract_tau_z(groups[cond], "tau_z_shuffle")
            data.append(tau_vals)
            labels.append(cond)

    if not data:
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("τ_z (training step)")
    ax.set_title("Distribution of τ_z by Condition")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "tau_z_distributions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved tau_z_distributions.png")


def plot_ppt_loss_curves(results: List[dict], output_dir: Path) -> None:
    """PPT convergence verification for C1 and C2."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    groups = group_by_condition(results)
    colors = {"C1": "#ff7f0e", "C2": "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, cond in enumerate(["C1", "C2"]):
        if cond not in groups:
            continue
        ax = axes[idx]
        for r in groups[cond]:
            if r["ppt_loss_curve"]:
                ax.plot(r["ppt_loss_curve"], alpha=0.4, color=colors[cond], linewidth=0.5)
        ax.set_xlabel("PPT Step")
        ax.set_ylabel("PPT Loss")
        ax.set_title(f"{cond} PPT Convergence")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "ppt_loss_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved ppt_loss_curves.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="outputs/ppt_phase1/all_results.json",
        help="Path to combined results JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ppt_phase1",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        # Try loading individual results
        print(f"  Combined results not found at {results_path}")
        print("  Scanning for individual result.json files...")
        results = []
        for p in output_dir.glob("ppt_*/result.json"):
            with open(p) as f:
                results.append(json.load(f))
        if not results:
            print("  No results found. Run the experiment first.")
            return
    else:
        with open(results_path) as f:
            results = json.load(f)

    print_summary(results)

    # Generate figures
    print("\nGenerating figures...")
    plot_z_gap_curves(results, output_dir)
    plot_candidate_loss_curves(results, output_dir)
    plot_tau_z_distributions(results, output_dir)
    plot_ppt_loss_curves(results, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
