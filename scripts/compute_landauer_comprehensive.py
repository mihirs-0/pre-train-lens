#!/usr/bin/env python
"""
Comprehensive Landauer analysis across 6 K values.

Computes four Q definitions for all experiments and performs scaling
analysis on multiple subsets (all, plateau_regime, high_K).

Q definitions:
  Q_original:   Transition window from candidate_loss thresholds
                 (0.9·log(K) → 0.1·log(K))
  Q_total_conv: Full protocol from step 0 to convergence point
                 (candidate_loss < 0.5 for 10 consecutive checkpoints)
  Q_zgap:       Functional transition via z-shuffle diagnostic
                 (z_gap > 0.5 → z_gap > 0.9·max)
  Q_gradspike:  Gradient norm spike window
                 (smoothed ||∇L||² > 2 × baseline, baseline = p25)

Usage:
    python scripts/compute_landauer_comprehensive.py \\
      --experiments landauer_k2 landauer_k3 landauer_k5 \\
                    landauer_k10 landauer_k20 landauer_k36_60k
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_experiment_data(exp_name: str, output_dir: str) -> dict:
    """Load all data files for a single experiment."""
    exp_dir = Path(output_dir) / exp_name

    with open(exp_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    with open(exp_dir / "gradient_norm_results.json") as f:
        grad_data = json.load(f)

    with open(exp_dir / "candidate_eval_results.json") as f:
        cand_data = json.load(f)

    hist_path = exp_dir / "training_history.json"
    hist_data = None
    if hist_path.exists():
        with open(hist_path) as f:
            hist_data = json.load(f)

    return {
        "config": cfg,
        "grad": grad_data,
        "cand": cand_data,
        "hist": hist_data,
    }


# ═══════════════════════════════════════════════════════════════════════
# Dissipation computation
# ═══════════════════════════════════════════════════════════════════════

def compute_cumulative_dissipation(steps, grad_norm_sq, lr):
    """
    Compute cumulative dissipation Q(t) = Σ η × ||∇L(s)||² × Δs.

    For constant LR, η is the same at every step.
    """
    steps = np.array(steps, dtype=float)
    norms_sq = np.array(grad_norm_sq, dtype=float)
    delta_s = np.diff(steps, prepend=0)
    dissipation = lr * norms_sq * delta_s
    Q_cumulative = np.cumsum(dissipation)
    return steps, Q_cumulative, dissipation


def compute_Q_in_window(steps, Q_cumulative, window_start, window_end):
    """Compute dissipation within [window_start, window_end] via interpolation."""
    Q_at_start = float(np.interp(window_start, steps, Q_cumulative))
    Q_at_end = float(np.interp(window_end, steps, Q_cumulative))
    return Q_at_end - Q_at_start


# ═══════════════════════════════════════════════════════════════════════
# Q_original: transition window from candidate_loss
# ═══════════════════════════════════════════════════════════════════════

def find_original_transition_window(cand_steps, candidate_loss, log_k):
    """
    transition_start: last step where candidate_loss > 0.9 × log(K)
    transition_end:   first step where candidate_loss < 0.1 × log(K)

    Returns (start, end) or (None, None).
    """
    threshold_high = 0.9 * log_k
    threshold_low = 0.1 * log_k

    transition_start = None
    transition_end = None

    for i, cl in enumerate(candidate_loss):
        if cl > threshold_high:
            transition_start = cand_steps[i]

    for i, cl in enumerate(candidate_loss):
        if cl < threshold_low:
            transition_end = cand_steps[i]
            break

    return transition_start, transition_end


# ═══════════════════════════════════════════════════════════════════════
# Q_total_conv: convergence-based cutoff
# ═══════════════════════════════════════════════════════════════════════

def find_convergence_step(cand_steps, candidate_loss, threshold=0.5,
                          n_consecutive=10):
    """
    Find the first checkpoint where candidate_loss < threshold for
    n_consecutive consecutive checkpoints.

    Returns the step of the FIRST such checkpoint (start of the streak).
    If never converges, returns the last checkpoint step.
    """
    count = 0
    first_below = None

    for i, cl in enumerate(candidate_loss):
        if cl < threshold:
            if count == 0:
                first_below = cand_steps[i]
            count += 1
            if count >= n_consecutive:
                return first_below
        else:
            count = 0
            first_below = None

    # Never converged → use last checkpoint
    return cand_steps[-1]


# ═══════════════════════════════════════════════════════════════════════
# Q_zgap: z-shuffle functional window
# ═══════════════════════════════════════════════════════════════════════

def find_zgap_window(cand_steps, z_gap):
    """
    zgap_start: FIRST step where z_gap > 0.5
    zgap_end:   FIRST step where z_gap > 0.9 × max(z_gap)

    Returns (start, end, max_zgap) — start/end may be None.
    """
    cand_steps = np.array(cand_steps, dtype=float)
    z_gap = np.array(z_gap, dtype=float)
    max_zgap = float(np.max(z_gap))

    zgap_start = None
    mask_start = z_gap > 0.5
    if np.any(mask_start):
        zgap_start = float(cand_steps[np.argmax(mask_start)])

    zgap_end = None
    threshold_end = 0.9 * max_zgap
    mask_end = z_gap > threshold_end
    if np.any(mask_end):
        zgap_end = float(cand_steps[np.argmax(mask_end)])

    # Handle degenerate case: end <= start
    if zgap_start is not None and zgap_end is not None:
        if zgap_start >= zgap_end:
            after_start = cand_steps >= zgap_start
            if np.any(after_start):
                idx_after = np.where(after_start)[0]
                best_idx = idx_after[np.argmax(z_gap[idx_after])]
                zgap_end = float(cand_steps[best_idx])
                if zgap_end <= zgap_start and best_idx + 1 < len(cand_steps):
                    zgap_end = float(cand_steps[best_idx + 1])

    return zgap_start, zgap_end, max_zgap


# ═══════════════════════════════════════════════════════════════════════
# Q_gradspike: gradient norm spike window
# ═══════════════════════════════════════════════════════════════════════

def find_gradspike_window(steps, grad_norm_sq):
    """
    Baseline = 25th percentile of ||∇L||² across all checkpoints.
    Threshold = 2 × baseline.
    Smoothing: moving average (window=5) for boundary detection.

    spike_start: FIRST step where smoothed > threshold
    spike_end:   LAST step where smoothed > threshold

    Returns (start, end, baseline, threshold).
    """
    steps = np.array(steps, dtype=float)
    norms_sq = np.array(grad_norm_sq, dtype=float)

    baseline = float(np.percentile(norms_sq, 25))
    threshold = 2.0 * baseline

    # Guard against degenerate baseline (fast-converging models have
    # near-zero gradient norms for most of training, giving baseline ≈ 0)
    if baseline < 1e-4:
        return None, None, baseline, threshold

    # Smooth with moving average
    window = 5
    if len(norms_sq) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(norms_sq, kernel, mode="same")
    else:
        smoothed = norms_sq.copy()

    above = smoothed > threshold
    spike_start = None
    spike_end = None

    if np.any(above):
        indices = np.where(above)[0]
        spike_start = float(steps[indices[0]])
        spike_end = float(steps[indices[-1]])

    return spike_start, spike_end, baseline, threshold


# ═══════════════════════════════════════════════════════════════════════
# Plateau detection
# ═══════════════════════════════════════════════════════════════════════

def detect_plateau(cand_steps, candidate_loss, log_k, min_duration=500):
    """
    Detect whether the experiment shows a clear plateau phase.

    A plateau exists if candidate_loss stays within 10% of log(K)
    (i.e., between 0.9·log(K) and 1.1·log(K)) for a contiguous stretch
    of at least min_duration training steps.

    Returns (has_plateau, max_plateau_duration).
    """
    cand_steps = np.array(cand_steps, dtype=float)
    candidate_loss = np.array(candidate_loss, dtype=float)

    low = 0.9 * log_k
    high = 1.1 * log_k

    in_plateau = (candidate_loss >= low) & (candidate_loss <= high)

    max_duration = 0
    current_start = None

    for i in range(len(in_plateau)):
        if in_plateau[i]:
            if current_start is None:
                current_start = cand_steps[i]
        else:
            if current_start is not None:
                duration = cand_steps[i] - current_start
                max_duration = max(max_duration, duration)
                current_start = None

    # Check trailing plateau
    if current_start is not None:
        duration = cand_steps[-1] - current_start
        max_duration = max(max_duration, duration)

    return bool(max_duration >= min_duration), float(max_duration)


# ═══════════════════════════════════════════════════════════════════════
# Linear fits
# ═══════════════════════════════════════════════════════════════════════

def linear_fit(x, y):
    """Compute linear fit y = slope*x + intercept, return (slope, intercept, R²)."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(x) < 2:
        return None, None, None
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
    if len(x) < 1:
        return None, None
    slope = float(np.sum(x * y) / np.sum(x ** 2))
    y_pred = slope * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(r_squared)


def compute_fits_for_subset(log_k_vals, q_vals):
    """Compute linear fit and through-origin fit for a subset."""
    valid = [(lk, q) for lk, q in zip(log_k_vals, q_vals)
             if q is not None and not (isinstance(q, float) and np.isnan(q))]

    if len(valid) < 2:
        return {
            "slope": None, "intercept": None, "r_squared": None,
            "slope_origin": None, "r_squared_origin": None,
            "n_points": len(valid),
        }

    lk_arr = [v[0] for v in valid]
    q_arr = [v[1] for v in valid]
    slope, intercept, r2 = linear_fit(lk_arr, q_arr)
    slope_o, r2_o = through_origin_fit(lk_arr, q_arr)

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r2,
        "slope_origin": slope_o,
        "r_squared_origin": r2_o,
        "n_points": len(valid),
    }


# ═══════════════════════════════════════════════════════════════════════
# Process a single experiment
# ═══════════════════════════════════════════════════════════════════════

def process_experiment(exp_name: str, output_dir: str) -> dict:
    """Process one experiment: compute all four Q definitions."""
    print(f"\nProcessing: {exp_name}")
    data = load_experiment_data(exp_name, output_dir)
    cfg = data["config"]
    grad = data["grad"]
    cand = data["cand"]

    k = int(cfg["data"]["k"])
    log_k = math.log(k)
    lr = float(cfg["training"]["learning_rate"])

    grad_steps = grad["steps"]
    grad_norm_sq = grad["total_grad_norm_sq"]
    cand_steps = cand["steps"]
    candidate_loss = cand["candidate_loss"]
    z_gap = cand["z_gap"]

    # Get final accuracy from training history or candidate eval
    final_accuracy = None
    if data["hist"] is not None:
        acc_list = data["hist"].get("train_accuracy", [])
        if acc_list:
            final_accuracy = float(acc_list[-1])
    if final_accuracy is None:
        final_accuracy = float(cand["candidate_accuracy"][-1])

    # ── Cumulative dissipation ──
    steps_arr, Q_cum, dissipation = compute_cumulative_dissipation(
        grad_steps, grad_norm_sq, lr
    )

    # ── 1. Q_original ──
    trans_start, trans_end = find_original_transition_window(
        cand_steps, candidate_loss, log_k
    )
    Q_original = None
    if trans_start is not None and trans_end is not None:
        window_width = trans_end - trans_start
        if window_width < 100:
            Q_original = 0.0
            print(f"  Q_original: degenerate window ({window_width:.0f} steps < 100) → set to 0")
        else:
            Q_original = compute_Q_in_window(steps_arr, Q_cum, trans_start, trans_end)
            print(f"  Q_original = {Q_original:.4f}  (window: {trans_start:.0f} → {trans_end:.0f})")
    else:
        Q_original = 0.0
        print(f"  Q_original: no transition window found → set to 0")

    # ── 2. Q_total_conv ──
    converged_step = find_convergence_step(cand_steps, candidate_loss)
    Q_total_conv = compute_Q_in_window(steps_arr, Q_cum, 0, converged_step)
    print(f"  Q_total_conv = {Q_total_conv:.4f}  (converged at step {converged_step:.0f})")

    # ── 3. Q_zgap ──
    zgap_start, zgap_end, max_zgap = find_zgap_window(cand_steps, z_gap)
    Q_zgap = None
    if max_zgap < 0.5:
        print(f"  Q_zgap: max z_gap = {max_zgap:.3f} < 0.5 → NaN (no functional transition)")
    elif zgap_start is not None and zgap_end is not None:
        Q_zgap = compute_Q_in_window(steps_arr, Q_cum, zgap_start, zgap_end)
        print(f"  Q_zgap = {Q_zgap:.4f}  (window: {zgap_start:.0f} → {zgap_end:.0f})")
    else:
        print(f"  Q_zgap: window not found → NaN")

    # ── 4. Q_gradspike ──
    spike_start, spike_end, baseline, threshold = find_gradspike_window(
        grad_steps, grad_norm_sq
    )
    Q_gradspike = None
    if spike_start is not None and spike_end is not None:
        Q_gradspike = compute_Q_in_window(steps_arr, Q_cum, spike_start, spike_end)
        print(f"  Q_gradspike = {Q_gradspike:.4f}  (window: {spike_start:.0f} → {spike_end:.0f}, "
              f"baseline={baseline:.4f}, threshold={threshold:.4f})")
    else:
        print(f"  Q_gradspike: no spike detected → None")

    # ── Plateau detection ──
    has_plateau, plateau_duration = detect_plateau(
        cand_steps, candidate_loss, log_k
    )
    print(f"  Plateau: {'YES' if has_plateau else 'NO'} "
          f"(max contiguous duration = {plateau_duration:.0f} steps)")

    return {
        "name": exp_name,
        "K": k,
        "log_K": round(log_k, 4),
        "has_plateau": has_plateau,
        "plateau_duration": plateau_duration,
        "converged_step": converged_step,
        "final_accuracy": final_accuracy,
        "Q_original": Q_original,
        "Q_total_conv": Q_total_conv,
        "Q_zgap": Q_zgap,
        "Q_gradspike": Q_gradspike,
        "transition_start_original": trans_start,
        "transition_end_original": trans_end,
        "zgap_start": zgap_start,
        "zgap_end": zgap_end,
        "max_zgap": max_zgap,
        "spike_start": spike_start,
        "spike_end": spike_end,
        "spike_baseline": baseline,
        "spike_threshold": threshold,
    }


# ═══════════════════════════════════════════════════════════════════════
# Summary printing
# ═══════════════════════════════════════════════════════════════════════

def print_summary(experiments, fits):
    """Print comprehensive summary tables."""
    print()
    print("══════════════════════════════════════════════════════════════════")
    print("Comprehensive Landauer Analysis — 6 K Values")
    print("══════════════════════════════════════════════════════════════════")
    print()

    # ── Experiment summary ──
    print("Experiment Summary:")
    header = (f"{'K':>4}  {'plateau?':>8}  {'conv_step':>10}  {'final_acc':>10}  "
              f"{'Q_orig':>10}  {'Q_total_cv':>10}  {'Q_zgap':>10}  {'Q_gradspk':>10}")
    print(header)
    print("-" * len(header))

    def fmt(val, width=10):
        if val is None:
            return f"{'NaN':>{width}}"
        return f"{val:>{width}.4f}"

    for exp in experiments:
        plateau_str = "YES" if exp["has_plateau"] else "NO"
        acc_str = f"{exp['final_accuracy']:.2%}" if exp["final_accuracy"] is not None else "N/A"
        print(
            f"{exp['K']:>4}  {plateau_str:>8}  {exp['converged_step']:>10.0f}  "
            f"{acc_str:>10}  "
            f"{fmt(exp['Q_original'])}  {fmt(exp['Q_total_conv'])}  "
            f"{fmt(exp['Q_zgap'])}  {fmt(exp['Q_gradspike'])}"
        )

    print()

    # ── Scaling fits ──
    print("Scaling Fits (Q_definition × subset):")
    fit_header = (f"{'Q_definition':<15}  {'subset':<16}  {'N':>3}  "
                  f"{'slope':>8}  {'intercept':>10}  {'R²':>7}  {'R²_origin':>10}")
    print(fit_header)
    print("-" * len(fit_header))

    for q_name in ["Q_original", "Q_total_conv", "Q_zgap", "Q_gradspike"]:
        for subset_name in ["all", "plateau_regime", "high_K"]:
            f = fits[q_name][subset_name]
            if f["slope"] is not None:
                print(
                    f"{q_name:<15}  {subset_name:<16}  {f['n_points']:>3}  "
                    f"{f['slope']:>8.2f}  {f['intercept']:>10.2f}  "
                    f"{f['r_squared']:>7.3f}  {f['r_squared_origin']:>10.3f}"
                )
            else:
                print(
                    f"{q_name:<15}  {subset_name:<16}  {f['n_points']:>3}  "
                    f"{'N/A':>8}  {'N/A':>10}  {'N/A':>7}  {'N/A':>10}"
                )

    print()
    print("══════════════════════════════════════════════════════════════════")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Landauer analysis across 6 K values"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=[
            "landauer_k2", "landauer_k3", "landauer_k5",
            "landauer_k10", "landauer_k20", "landauer_k36_60k",
        ],
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
        default="outputs/landauer_comprehensive_results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    # ── Process each experiment ──
    experiments = []
    for exp_name in args.experiments:
        result = process_experiment(exp_name, args.output_dir)
        experiments.append(result)

    # Sort by K
    experiments.sort(key=lambda x: x["K"])

    # ── Build subsets ──
    # Indices for each subset
    all_indices = list(range(len(experiments)))
    plateau_indices = [i for i, e in enumerate(experiments) if e["has_plateau"]]
    high_k_indices = [i for i, e in enumerate(experiments) if e["K"] in {10, 20, 36}]

    subsets = {
        "all": all_indices,
        "plateau_regime": plateau_indices,
        "high_K": high_k_indices,
    }

    # ── Compute fits for each (Q_definition, subset) ──
    q_names = ["Q_original", "Q_total_conv", "Q_zgap", "Q_gradspike"]
    fits = {}

    for q_name in q_names:
        fits[q_name] = {}
        for subset_name, indices in subsets.items():
            log_k_vals = [experiments[i]["log_K"] for i in indices]
            q_vals = [experiments[i][q_name] for i in indices]
            fits[q_name][subset_name] = compute_fits_for_subset(log_k_vals, q_vals)

    # ── Print summary ──
    print_summary(experiments, fits)

    # ── Save results ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "experiments": experiments,
        "fits": fits,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
