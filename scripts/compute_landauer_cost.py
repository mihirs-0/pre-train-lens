#!/usr/bin/env python
"""
Compute cumulative dissipation integral for the Landauer test.

For each experiment, computes:
  Q(t) = Σ_{s} η(s) × ||∇L(s)||² × Δs

where η(s) is the ACTUAL learning rate at step s (warmup + cosine decay),
||∇L(s)||² is the measured gradient norm squared, and Δs is the step spacing.

Defines the transition window from candidate_eval_results.json:
  - transition_start: last step where candidate_loss > 0.9 × log(K)
  - transition_end:   first step where candidate_loss < 0.1 × log(K)

Outputs Q_transition, Q_total, Q_plateau, Q_post for each experiment,
and a summary table testing whether Q_transition scales with log(K).

Usage:
    python scripts/compute_landauer_cost.py \
      --experiments temp_lr1e3_bs128 temp_lr1e3_bs128_k20 temp_lr1e3_bs512_k36 \
      --output-dir outputs
"""

import sys
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


def reconstruct_lr_schedule(peak_lr: float, warmup_steps: int, max_steps: int) -> np.ndarray:
    """
    Reconstruct the learning rate at each training step.

    Matches src/training/trainer.py::get_lr_scheduler exactly:
      - Linear warmup from 0 to peak_lr over warmup_steps
      - Cosine decay from peak_lr to 0 over remaining steps
    """
    lrs = np.zeros(max_steps)
    for step in range(max_steps):
        if step < warmup_steps:
            lrs[step] = peak_lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            lrs[step] = peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lrs


def find_transition_window(steps, candidate_loss, log_k):
    """
    Define the phase transition window from candidate_loss trajectory.

    transition_start: last step where candidate_loss > 0.9 × log(K)
    transition_end:   first step where candidate_loss < 0.1 × log(K)

    Returns (transition_start, transition_end) or (None, None) if not found.
    """
    threshold_high = 0.9 * log_k
    threshold_low = 0.1 * log_k

    transition_start = None
    transition_end = None

    # Last step where candidate_loss > 0.9 * log(K)
    for i, cl in enumerate(candidate_loss):
        if cl > threshold_high:
            transition_start = steps[i]

    # First step where candidate_loss < 0.1 * log(K)
    for i, cl in enumerate(candidate_loss):
        if cl < threshold_low:
            transition_end = steps[i]
            break

    return transition_start, transition_end


def compute_dissipation(
    grad_steps: list,
    grad_norm_sq: list,
    lr_schedule: np.ndarray,
) -> tuple:
    """
    Compute cumulative dissipation Q(t) = Σ η(s) × ||∇L(s)||² × Δs.

    Uses the actual LR at each checkpoint step. Δs is the spacing between
    consecutive checkpoint steps.

    Returns:
        steps: array of checkpoint steps
        Q_cumulative: array of cumulative dissipation at each checkpoint
        dissipation_per_step: array of per-interval dissipation contribution
    """
    steps = np.array(grad_steps)
    norms_sq = np.array(grad_norm_sq)

    # Get actual LR at each checkpoint step
    # Clip steps to valid range for the schedule
    max_step = len(lr_schedule) - 1
    safe_steps = np.clip(steps, 0, max_step).astype(int)
    eta_at_checkpoints = lr_schedule[safe_steps]

    # Compute step spacings (Δs)
    # First interval: from 0 to first checkpoint
    delta_s = np.diff(steps, prepend=0)

    # Per-interval dissipation: η(s) × ||∇L(s)||² × Δs
    dissipation = eta_at_checkpoints * norms_sq * delta_s

    # Cumulative sum
    Q_cumulative = np.cumsum(dissipation)

    return steps, Q_cumulative, dissipation


def process_experiment(experiment_name: str, output_dir: str) -> dict:
    """Process a single experiment and return its Landauer metrics."""
    exp_dir = Path(output_dir) / experiment_name

    # Load config
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        print(f"  ERROR: Config not found at {config_path}")
        return None

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    k = int(cfg["data"]["k"])
    log_k = math.log(k)
    log2_k = math.log2(k)
    peak_lr = float(cfg["training"]["learning_rate"])
    batch_size = int(cfg["training"]["batch_size"])
    warmup_steps = int(cfg["training"].get("warmup_steps", 500))
    max_steps = int(cfg["training"].get("max_steps", 10000))
    T_eff_peak = peak_lr / batch_size

    # Load gradient norms
    grad_path = exp_dir / "gradient_norm_results.json"
    if not grad_path.exists():
        print(f"  ERROR: Gradient norms not found at {grad_path}")
        return None

    with open(grad_path, "r") as f:
        grad_data = json.load(f)

    grad_steps = grad_data["steps"]
    grad_norm_sq = grad_data["total_grad_norm_sq"]

    # Load candidate eval (optional but preferred)
    cand_path = exp_dir / "candidate_eval_results.json"
    has_candidate = cand_path.exists()
    transition_start = None
    transition_end = None

    if has_candidate:
        with open(cand_path, "r") as f:
            cand_data = json.load(f)

        cand_steps = cand_data["steps"]
        cand_loss = cand_data["candidate_loss"]

        transition_start, transition_end = find_transition_window(
            cand_steps, cand_loss, log_k
        )
        print(f"  Transition window (from candidate_loss): {transition_start} -> {transition_end}")
    else:
        # Fallback: use binding_onset_step from gradient_norm_results
        onset = grad_data.get("binding_onset_step")
        if onset is not None:
            transition_start = onset
            # Rough estimate: transition ends ~1000 steps after onset
            transition_end = min(onset + 1000, grad_steps[-1])
            print(f"  Transition window (fallback from binding_onset): {transition_start} -> {transition_end}")
        else:
            print("  WARNING: No transition window found!")

    # Reconstruct LR schedule
    lr_schedule = reconstruct_lr_schedule(peak_lr, warmup_steps, max_steps)

    # Compute dissipation
    steps, Q_cum, dissipation = compute_dissipation(grad_steps, grad_norm_sq, lr_schedule)

    # Extract dissipation values
    Q_total = float(Q_cum[-1])

    # Find Q at transition boundaries by interpolation
    Q_transition = None
    Q_plateau = None
    Q_post = None
    transition_duration = None

    if transition_start is not None and transition_end is not None:
        # Interpolate Q at transition_start and transition_end
        Q_at_start = float(np.interp(transition_start, steps, Q_cum))
        Q_at_end = float(np.interp(transition_end, steps, Q_cum))

        Q_transition = Q_at_end - Q_at_start
        Q_plateau = Q_at_start
        Q_post = Q_total - Q_at_end
        transition_duration = transition_end - transition_start

    # Compute S statistics (S = ||∇L||²)
    # Plateau S: mean grad norm squared before transition
    if transition_start is not None:
        plateau_mask = np.array(grad_steps) < transition_start
        if np.any(plateau_mask):
            plateau_S_mean = float(np.mean(np.array(grad_norm_sq)[plateau_mask]))
        else:
            plateau_S_mean = float(grad_norm_sq[0])
    else:
        plateau_S_mean = float(np.mean(grad_norm_sq))

    peak_S = float(max(grad_norm_sq))
    peak_S_step = int(grad_steps[grad_norm_sq.index(peak_S)] if isinstance(grad_norm_sq, list) else grad_steps[np.argmax(grad_norm_sq)])

    # Build note about experiment conditions
    notes = []
    if batch_size != 128:
        notes.append(f"batch_size={batch_size} (differs from K=10,K=20 which use bs=128)")
    if not has_candidate:
        notes.append("transition window from binding_onset fallback (no candidate_eval)")
    note = "; ".join(notes) if notes else ""

    result = {
        "name": experiment_name,
        "K": k,
        "log_K": round(log_k, 4),
        "log2_K": round(log2_k, 4),
        "peak_lr": peak_lr,
        "batch_size": batch_size,
        "T_eff_peak": T_eff_peak,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "transition_start": transition_start,
        "transition_end": transition_end,
        "transition_duration": transition_duration,
        "Q_transition": Q_transition,
        "Q_total": Q_total,
        "Q_plateau": Q_plateau,
        "Q_post": Q_post,
        "plateau_S_mean": plateau_S_mean,
        "peak_S": peak_S,
        "peak_S_step": peak_S_step,
        "Q_transition_over_logK": round(Q_transition / log_k, 6) if Q_transition is not None else None,
        "Q_transition_over_log2K": round(Q_transition / log2_k, 6) if Q_transition is not None else None,
        "note": note,
        # Full trajectories for plotting
        "Q_trajectory_steps": steps.tolist(),
        "Q_trajectory": Q_cum.tolist(),
        "dissipation_per_interval": dissipation.tolist(),
    }

    return result


def print_summary_table(experiments: list):
    """Print a nicely formatted summary table."""
    print()
    print("=" * 80)
    print("Landauer Dissipation Test — First Pass (3 data points, single seed)")
    print("=" * 80)
    print()

    # Header
    header = f"{'K':>4}  {'log(K)':>7}  {'BS':>4}  {'T_eff':>10}  {'Q_trans':>12}  {'Q_trans/logK':>14}  {'plat_S':>10}  {'peak_S':>10}"
    print(header)
    print("-" * len(header))

    ratios = []
    for exp in experiments:
        q_trans = exp["Q_transition"]
        ratio = exp["Q_transition_over_logK"]
        if q_trans is not None:
            ratios.append(ratio)
            print(
                f"{exp['K']:>4}  {exp['log_K']:>7.3f}  {exp['batch_size']:>4}  "
                f"{exp['T_eff_peak']:>10.2e}  {q_trans:>12.6f}  {ratio:>14.6f}  "
                f"{exp['plateau_S_mean']:>10.4f}  {exp['peak_S']:>10.4f}"
            )
        else:
            print(
                f"{exp['K']:>4}  {exp['log_K']:>7.3f}  {exp['batch_size']:>4}  "
                f"{exp['T_eff_peak']:>10.2e}  {'N/A':>12}  {'N/A':>14}  "
                f"{exp['plateau_S_mean']:>10.4f}  {exp['peak_S']:>10.4f}"
            )

    print("-" * len(header))

    if len(ratios) >= 2:
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        cv = std_ratio / mean_ratio if mean_ratio != 0 else float("inf")
        print(f"\n  Q_trans/log(K): mean = {mean_ratio:.6f}, std = {std_ratio:.6f}, CV = {cv:.3f}")
        print()
        if cv < 0.3:
            print("  → CONSISTENT with Landauer scaling: Q_trans/log(K) ≈ constant")
            print(f"    (CV = {cv:.3f} < 0.3, but only {len(ratios)} data points)")
        elif cv < 0.5:
            print(f"  → WEAKLY consistent with Landauer scaling (CV = {cv:.3f})")
            print(f"    (marginal; {len(ratios)} points, single seed)")
        else:
            print(f"  → NOT consistent with Landauer scaling (CV = {cv:.3f})")
            print(f"    Q_trans/log(K) varies too much across K values")
    else:
        print("\n  Insufficient data points for scaling test")

    # Note about different batch sizes
    batch_sizes = set(exp["batch_size"] for exp in experiments)
    if len(batch_sizes) > 1:
        print(f"\n  ⚠ CAVEAT: Experiments use different batch sizes {sorted(batch_sizes)}.")
        print("    T_eff = lr/bs differs, so Q values are not on identical footing.")
        print("    This is a confound. A clean test would use matched T_eff.")

    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compute cumulative dissipation integral for Landauer test"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        required=True,
        help="Experiment names to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory",
    )
    args = parser.parse_args()

    results = []
    for exp_name in args.experiments:
        print(f"\nProcessing: {exp_name}")
        result = process_experiment(exp_name, args.output_dir)
        if result is not None:
            results.append(result)
        else:
            print(f"  SKIPPED (missing data)")

    # Sort by K
    results.sort(key=lambda x: x["K"])

    # Save full results
    output_path = Path(args.output_dir) / "landauer_results.json"
    output = {"experiments": results}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to: {output_path}")

    # Print summary
    print_summary_table(results)


if __name__ == "__main__":
    main()
