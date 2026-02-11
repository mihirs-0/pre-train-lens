#!/usr/bin/env python
"""
Compute dissipation Q under three alternative window definitions for the
Landauer robustness analysis.

Addresses three reviewer objections to the original Q_transition metric:
  1. Kinematic confound: window width ∝ log(K) could trivially produce scaling
  2. Negative intercept: fit extrapolates to Q < 0 at small K
  3. Plateau exclusion: Landauer's principle covers the full protocol

Three new Q definitions (none use log(K)-dependent bounds):
  Q_total:     Full training dissipation (step 0 → last checkpoint)
  Q_zgap:      Functional transition window via z-shuffle diagnostic
               (z_gap > 0.5 to z_gap > 0.9·max — absolute thresholds)
  Q_gradspike: Gradient norm spike window
               (smoothed ||∇L||² > 2× baseline — absolute threshold)

Usage:
    python scripts/compute_landauer_robustness.py \
      --experiments landauer_k5 landauer_k10 landauer_k20 landauer_k36 \
      --output outputs/landauer_robustness_results.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


def load_experiment_data(exp_name: str, output_dir: str) -> dict:
    """Load all data files for a single experiment."""
    exp_dir = Path(output_dir) / exp_name

    # Config
    with open(exp_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Gradient norms
    with open(exp_dir / "gradient_norm_results.json") as f:
        grad_data = json.load(f)

    # Candidate eval
    with open(exp_dir / "candidate_eval_results.json") as f:
        cand_data = json.load(f)

    # Training history
    with open(exp_dir / "training_history.json") as f:
        hist_data = json.load(f)

    return {
        "config": cfg,
        "grad": grad_data,
        "cand": cand_data,
        "hist": hist_data,
    }


def compute_cumulative_dissipation(steps, grad_norm_sq, lr):
    """
    Compute cumulative dissipation Q(t) = Σ η × ||∇L(s)||² × Δs.

    Parameters
    ----------
    steps : array-like
        Checkpoint steps.
    grad_norm_sq : array-like
        ||∇L||² at each checkpoint.
    lr : float
        Constant learning rate.

    Returns
    -------
    steps : np.ndarray
    Q_cumulative : np.ndarray
    dissipation_per_interval : np.ndarray
    """
    steps = np.array(steps, dtype=float)
    norms_sq = np.array(grad_norm_sq, dtype=float)
    delta_s = np.diff(steps, prepend=0)
    dissipation = lr * norms_sq * delta_s
    Q_cumulative = np.cumsum(dissipation)
    return steps, Q_cumulative, dissipation


def compute_Q_in_window(steps, Q_cumulative, window_start, window_end):
    """
    Compute dissipation accumulated within [window_start, window_end]
    by interpolating the cumulative dissipation curve.
    """
    Q_at_start = float(np.interp(window_start, steps, Q_cumulative))
    Q_at_end = float(np.interp(window_end, steps, Q_cumulative))
    return Q_at_end - Q_at_start


def find_zgap_window(cand_steps, z_gap):
    """
    Find the functional transition window using z-shuffle diagnostic.

    Window:
      zgap_start: FIRST step where z_gap > 0.5
      zgap_end:   FIRST step where z_gap > 0.9 × max(z_gap)

    Both thresholds are absolute (K-independent).
    """
    cand_steps = np.array(cand_steps, dtype=float)
    z_gap = np.array(z_gap, dtype=float)

    max_zgap = np.max(z_gap)

    # FIRST step where z_gap > 0.5
    zgap_start = None
    mask_start = z_gap > 0.5
    if np.any(mask_start):
        zgap_start = float(cand_steps[np.argmax(mask_start)])

    # FIRST step where z_gap > 0.9 × max(z_gap)
    zgap_end = None
    threshold_end = 0.9 * max_zgap
    mask_end = z_gap > threshold_end
    if np.any(mask_end):
        zgap_end = float(cand_steps[np.argmax(mask_end)])

    # Handle degenerate case
    if zgap_start is not None and zgap_end is not None:
        if zgap_start >= zgap_end:
            # Set zgap_end to step after zgap_start where z_gap is maximized
            after_start = cand_steps >= zgap_start
            if np.any(after_start):
                idx_after = np.where(after_start)[0]
                best_idx = idx_after[np.argmax(z_gap[idx_after])]
                zgap_end = float(cand_steps[best_idx])
                # Ensure end > start
                if zgap_end <= zgap_start and best_idx + 1 < len(cand_steps):
                    zgap_end = float(cand_steps[best_idx + 1])

    return zgap_start, zgap_end, max_zgap


def compute_global_spike_threshold(all_grad_data: list) -> tuple:
    """
    Compute a GLOBAL fixed threshold for the gradient spike window.

    Uses the mean ||∇L||² at the first checkpoint (step 100) across all
    experiments. At initialization, the gradient norm depends only on
    architecture and learning rate — not K — so this is K-independent.

    Threshold = 10 × mean_initial_norm (≈ 2.0 for this architecture).
    The multiplier 10× captures the regime where gradient norms are
    substantially elevated above the pre-transition baseline.

    Returns (global_baseline, global_threshold).
    """
    initial_norms = []
    for grad in all_grad_data:
        norms_sq = np.array(grad["total_grad_norm_sq"], dtype=float)
        # First checkpoint = pre-transition gradient norm
        initial_norms.append(float(norms_sq[0]))

    global_baseline = float(np.mean(initial_norms))
    global_threshold = 10.0 * global_baseline

    return global_baseline, global_threshold


def find_gradspike_window(steps, grad_norm_sq, global_threshold):
    """
    Find the gradient norm spike window using a GLOBAL absolute threshold.

    The threshold is the same for all experiments (K-independent), computed
    from the mean initial gradient norm across all experiments.

    Smoothing: simple moving average (window=5) for boundary detection.
    The actual integral uses UNSMOOTHED gradient norms.

    Window:
      spike_start: FIRST step where smoothed ||∇L||² > global_threshold
      spike_end:   LAST step where smoothed ||∇L||² > global_threshold
    """
    steps = np.array(steps, dtype=float)
    norms_sq = np.array(grad_norm_sq, dtype=float)

    # Per-experiment baseline for reporting (p25)
    per_exp_baseline = float(np.percentile(norms_sq, 25))

    # Smooth with moving average (window=5)
    window = 5
    if len(norms_sq) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(norms_sq, kernel, mode="same")
    else:
        smoothed = norms_sq.copy()

    # Find spike boundaries on smoothed curve using GLOBAL threshold
    above = smoothed > global_threshold
    spike_start = None
    spike_end = None

    if np.any(above):
        indices = np.where(above)[0]
        spike_start = float(steps[indices[0]])
        spike_end = float(steps[indices[-1]])

    return spike_start, spike_end, per_exp_baseline, global_threshold


def linear_fit(x, y):
    """Compute linear fit y = slope*x + intercept, return (slope, intercept, R²)."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r_squared)


def through_origin_fit(x, y):
    """Compute proportional fit y = slope*x (through origin)."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    slope = float(np.sum(x * y) / np.sum(x ** 2))
    y_pred = slope * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(r_squared)


def coefficient_of_variation(log_k_vals, q_vals):
    """Compute CV of Q/log(K) ratios."""
    ratios = np.array(q_vals) / np.array(log_k_vals)
    return float(np.std(ratios) / np.mean(ratios)) if np.mean(ratios) != 0 else float("inf")


def process_experiment(exp_name: str, output_dir: str, global_spike_threshold: float) -> dict:
    """Process a single experiment: compute Q under all three definitions."""
    data = load_experiment_data(exp_name, output_dir)
    cfg = data["config"]
    grad = data["grad"]
    cand = data["cand"]

    k = int(cfg["data"]["k"])
    log_k = math.log(k)
    lr = float(cfg["training"]["learning_rate"])
    batch_size = int(cfg["training"]["batch_size"])

    # ── Compute cumulative dissipation over all steps ──
    grad_steps = grad["steps"]
    grad_norm_sq = grad["total_grad_norm_sq"]
    steps_arr, Q_cum, dissipation = compute_cumulative_dissipation(
        grad_steps, grad_norm_sq, lr
    )

    # ── Definition 1: Q_total (full training) ──
    Q_total = float(Q_cum[-1])

    # ── Definition 2: Q_zgap (functional transition window) ──
    cand_steps = cand["steps"]
    z_gap = cand["z_gap"]

    zgap_start, zgap_end, max_zgap = find_zgap_window(cand_steps, z_gap)

    Q_zgap = None
    if zgap_start is not None and zgap_end is not None:
        Q_zgap = compute_Q_in_window(steps_arr, Q_cum, zgap_start, zgap_end)

    # ── Definition 3: Q_gradspike (gradient norm spike window) ──
    spike_start, spike_end, per_exp_baseline, threshold = find_gradspike_window(
        grad_steps, grad_norm_sq, global_spike_threshold
    )

    Q_gradspike = None
    if spike_start is not None and spike_end is not None:
        Q_gradspike = compute_Q_in_window(steps_arr, Q_cum, spike_start, spike_end)

    result = {
        "name": exp_name,
        "K": k,
        "log_K": round(log_k, 4),
        "learning_rate": lr,
        "batch_size": batch_size,
        # Q values
        "Q_total": Q_total,
        "Q_zgap": Q_zgap,
        "Q_gradspike": Q_gradspike,
        # z-gap window details
        "zgap_start": zgap_start,
        "zgap_end": zgap_end,
        "max_zgap": float(max_zgap),
        # gradspike window details
        "spike_start": spike_start,
        "spike_end": spike_end,
        "plateau_baseline": per_exp_baseline,
        "spike_threshold": threshold,
    }

    return result


def print_summary(experiments, fits):
    """Print the comparison table."""
    print()
    print("=" * 85)
    print("Landauer Robustness Analysis")
    print("=" * 85)
    print()
    print(f"{'Q Definition':<18} | {'slope':>8}  {'intercept':>10}  {'R²':>7}  {'R²(origin)':>11}  {'CV(Q/logK)':>11}")
    print("-" * 18 + "-+-" + "-" * 60)
    for name in ["Q_original", "Q_total", "Q_zgap", "Q_gradspike"]:
        f = fits[name]
        if f["slope"] is not None:
            print(
                f"{name:<18} | {f['slope']:>8.2f}  {f['intercept']:>10.2f}  "
                f"{f['r_squared']:>7.3f}  {f['r_squared_origin']:>11.3f}  "
                f"{f['cv']:>11.3f}"
            )
        else:
            print(f"{name:<18} |   (insufficient data)")
    print("=" * 85)
    print()

    # Per-experiment values
    print(f"{'K':>4}  {'log(K)':>7}  {'Q_orig':>10}  {'Q_total':>10}  {'Q_zgap':>10}  {'Q_gradspike':>12}")
    print("-" * 67)
    for exp in experiments:
        def _fmt(val, width=10):
            return f"{val:>{width}.3f}" if val is not None else f"{'N/A':>{width}}"

        print(
            f"{exp['K']:>4}  {exp['log_K']:>7.3f}  "
            f"{_fmt(exp.get('Q_original'))}  {_fmt(exp['Q_total'])}  "
            f"{_fmt(exp['Q_zgap'])}  {_fmt(exp['Q_gradspike'], 12)}"
        )

    # Window details
    print()
    print("Window details:")
    print(f"{'K':>4}  {'zgap_start':>11}  {'zgap_end':>9}  {'spike_start':>12}  {'spike_end':>10}  {'baseline':>10}  {'threshold':>10}")
    print("-" * 80)
    for exp in experiments:
        def fmt(v):
            return f"{v:>10.0f}" if v is not None else f"{'N/A':>10}"

        print(
            f"{exp['K']:>4}  "
            f"{fmt(exp['zgap_start']):>11}  {fmt(exp['zgap_end']):>9}  "
            f"{fmt(exp['spike_start']):>12}  {fmt(exp['spike_end']):>10}  "
            f"{exp['plateau_baseline']:>10.6f}  {exp['spike_threshold']:>10.6f}"
        )

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compute dissipation Q under three alternative definitions"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["landauer_k5", "landauer_k10", "landauer_k20", "landauer_k36"],
        help="Experiment names to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/landauer_robustness_results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    # ── Load original landauer_results.json for Q_original ──
    original_path = Path(args.output_dir) / "landauer_results.json"
    original_Q = {}
    if original_path.exists():
        with open(original_path) as f:
            orig_data = json.load(f)
        for exp in orig_data["experiments"]:
            original_Q[exp["name"]] = exp.get("Q_transition")
        print(f"Loaded original Q_transition values from {original_path}")
    else:
        print(f"WARNING: {original_path} not found — Q_original will be None")

    # ── Pre-load gradient data for global spike threshold ──
    all_grad_data = []
    for exp_name in args.experiments:
        exp_dir = Path(args.output_dir) / exp_name
        with open(exp_dir / "gradient_norm_results.json") as f:
            all_grad_data.append(json.load(f))

    global_baseline, global_spike_threshold = compute_global_spike_threshold(all_grad_data)
    print(f"Global spike threshold: {global_spike_threshold:.4f} "
          f"(10 × mean initial ||∇L||² = 10 × {global_baseline:.4f})")

    # ── Process each experiment ──
    experiments = []
    for exp_name in args.experiments:
        print(f"\nProcessing: {exp_name}")
        result = process_experiment(exp_name, args.output_dir, global_spike_threshold)

        # Attach Q_original from the original analysis
        result["Q_original"] = original_Q.get(exp_name)
        if result["Q_original"] is not None:
            print(f"  Q_original  = {result['Q_original']:.4f}")
        print(f"  Q_total     = {result['Q_total']:.4f}")
        if result["Q_zgap"] is not None:
            print(f"  Q_zgap      = {result['Q_zgap']:.4f}  (window: {result['zgap_start']:.0f} → {result['zgap_end']:.0f})")
        else:
            print(f"  Q_zgap      = N/A (z-gap window not found)")
        if result["Q_gradspike"] is not None:
            print(f"  Q_gradspike = {result['Q_gradspike']:.4f}  (window: {result['spike_start']:.0f} → {result['spike_end']:.0f})")
        else:
            print(f"  Q_gradspike = N/A (spike window not found)")

        experiments.append(result)

    # Sort by K
    experiments.sort(key=lambda x: x["K"])

    # ── Compute linear fits for each Q definition ──
    log_k_vals = [exp["log_K"] for exp in experiments]

    fits = {}
    for q_name in ["Q_original", "Q_total", "Q_zgap", "Q_gradspike"]:
        # Collect non-None values
        valid = [(lk, exp[q_name]) for lk, exp in zip(log_k_vals, experiments)
                 if exp[q_name] is not None]

        if len(valid) >= 2:
            lk_arr = [v[0] for v in valid]
            q_arr = [v[1] for v in valid]
            slope, intercept, r2 = linear_fit(lk_arr, q_arr)
            slope_o, r2_o = through_origin_fit(lk_arr, q_arr)
            cv = coefficient_of_variation(lk_arr, q_arr)
            fits[q_name] = {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r2,
                "slope_origin": slope_o,
                "r_squared_origin": r2_o,
                "cv": cv,
                "n_points": len(valid),
            }
        else:
            fits[q_name] = {
                "slope": None,
                "intercept": None,
                "r_squared": None,
                "slope_origin": None,
                "r_squared_origin": None,
                "cv": None,
                "n_points": len(valid),
            }

    # ── Print summary ──
    print_summary(experiments, fits)

    # ── Save results ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "experiments": experiments,
        "fits": fits,
        "global_spike_baseline": global_baseline,
        "global_spike_threshold": global_spike_threshold,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
