#!/usr/bin/env python
"""
Plateau Decomposition: Decompose total dissipation into three phases.

  Q_prior:      step 0 → prior_end      (model learns K-candidate prior)
  Q_plateau:    prior_end → plateau_end  (plateau / memorization phase)
  Q_transition: plateau_end → convergence (symmetry breaking / Landauer signal)

This shows that Q_plateau is weakly dependent on K while Q_transition
carries the log(K) scaling.

Usage:
    python scripts/compute_plateau_decomposition.py \\
      --experiments landauer_k10 landauer_k15 landauer_k20 \\
                    landauer_k25 landauer_k36_60k
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_comprehensive_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_candidate_eval(exp_name: str, output_dir: str) -> dict:
    path = Path(output_dir) / exp_name / "candidate_eval_results.json"
    with open(path) as f:
        return json.load(f)


def load_gradient_norms(exp_name: str, output_dir: str) -> dict:
    path = Path(output_dir) / exp_name / "gradient_norm_results.json"
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# Dissipation computation (mirrors compute_landauer_comprehensive.py)
# ═══════════════════════════════════════════════════════════════════════

def compute_cumulative_dissipation(steps, grad_norm_sq, lr):
    """
    Compute cumulative dissipation Q(t) = Σ η × ||∇L(s)||² × Δs.
    """
    steps = np.array(steps, dtype=float)
    norms_sq = np.array(grad_norm_sq, dtype=float)
    delta_s = np.diff(steps, prepend=0)
    dissipation = lr * norms_sq * delta_s
    Q_cumulative = np.cumsum(dissipation)
    return steps, Q_cumulative


def compute_Q_in_window(steps, Q_cumulative, window_start, window_end):
    """Compute dissipation within [window_start, window_end] via interpolation."""
    Q_at_start = float(np.interp(window_start, steps, Q_cumulative))
    Q_at_end = float(np.interp(window_end, steps, Q_cumulative))
    return Q_at_end - Q_at_start


# ═══════════════════════════════════════════════════════════════════════
# Phase boundary detection
# ═══════════════════════════════════════════════════════════════════════

def find_prior_end(cand_steps, candidate_loss, log_k):
    """
    prior_end = first step where candidate_loss < 1.1 × log(K).
    This is where the model has learned the K-candidate prior
    (loss has dropped from initial to near log(K)).
    """
    threshold = 1.1 * log_k
    cand_steps = np.array(cand_steps, dtype=float)
    candidate_loss = np.array(candidate_loss, dtype=float)

    for i, cl in enumerate(candidate_loss):
        if cl < threshold:
            return float(cand_steps[i])

    # If never reaches threshold, use first step
    return float(cand_steps[0])


# ═══════════════════════════════════════════════════════════════════════
# Linear fits
# ═══════════════════════════════════════════════════════════════════════

def linear_fit(x, y):
    """Compute linear fit y = slope*x + intercept, return dict."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(x) < 2:
        return {"slope": None, "intercept": None, "r_squared": None}
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
    }


# ═══════════════════════════════════════════════════════════════════════
# Process one experiment
# ═══════════════════════════════════════════════════════════════════════

def process_experiment(exp_info: dict, output_dir: str) -> dict:
    """Decompose dissipation for one experiment."""
    exp_name = exp_info["name"]
    K = exp_info["K"]
    log_K = math.log(K)
    zgap_start = exp_info["zgap_start"]      # plateau_end = transition start
    converged_step = exp_info["converged_step"]

    # Load raw data
    cand = load_candidate_eval(exp_name, output_dir)
    grad = load_gradient_norms(exp_name, output_dir)
    lr = float(grad["config"]["learning_rate"])

    cand_steps = cand["steps"]
    candidate_loss = cand["candidate_loss"]
    grad_steps = grad["steps"]
    grad_norm_sq = grad["total_grad_norm_sq"]

    # Compute cumulative dissipation
    steps_arr, Q_cum = compute_cumulative_dissipation(grad_steps, grad_norm_sq, lr)

    # Phase boundaries
    prior_end = find_prior_end(cand_steps, candidate_loss, log_K)
    plateau_end = zgap_start  # transition starts where z-gap begins

    # For convergence: use the larger of converged_step and zgap_end
    # to capture the full transition
    zgap_end = exp_info["zgap_end"]
    convergence = max(converged_step, zgap_end)

    # Ensure ordering: prior_end <= plateau_end <= convergence
    prior_end = min(prior_end, plateau_end)

    # Compute Q for each phase
    Q_prior = compute_Q_in_window(steps_arr, Q_cum, 0, prior_end)
    Q_plateau = compute_Q_in_window(steps_arr, Q_cum, prior_end, plateau_end)
    Q_transition = compute_Q_in_window(steps_arr, Q_cum, plateau_end, convergence)
    Q_total = compute_Q_in_window(steps_arr, Q_cum, 0, convergence)

    # Fractions
    if Q_total > 0:
        f_prior = Q_prior / Q_total
        f_plateau = Q_plateau / Q_total
        f_transition = Q_transition / Q_total
    else:
        f_prior = f_plateau = f_transition = 0.0

    return {
        "name": exp_name,
        "K": K,
        "log_K": round(log_K, 4),
        "Q_prior": round(Q_prior, 6),
        "Q_plateau": round(Q_plateau, 6),
        "Q_transition": round(Q_transition, 6),
        "Q_total": round(Q_total, 6),
        "f_prior": round(f_prior, 4),
        "f_plateau": round(f_plateau, 4),
        "f_transition": round(f_transition, 4),
        "prior_end_step": prior_end,
        "plateau_end_step": plateau_end,
        "convergence_step": convergence,
    }


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════

def print_summary(results, fits):
    print()
    print("Plateau Decomposition")
    print("══════════════════════════════════════════════════════════════")
    header = (f"{'K':>4}  {'Q_prior':>10}  {'Q_plateau':>10}  "
              f"{'Q_transition':>12}  {'Q_total':>10}  "
              f"{'f_prior':>8}  {'f_plat':>8}  {'f_trans':>8}")
    print(header)
    print("─" * len(header))

    for r in results:
        print(
            f"{r['K']:>4}  {r['Q_prior']:>10.3f}  {r['Q_plateau']:>10.3f}  "
            f"{r['Q_transition']:>12.3f}  {r['Q_total']:>10.3f}  "
            f"{r['f_prior']:>8.3f}  {r['f_plateau']:>8.3f}  "
            f"{r['f_transition']:>8.3f}"
        )

    print("══════════════════════════════════════════════════════════════")
    print()
    print("Phase boundaries:")
    for r in results:
        print(f"  K={r['K']:>2}: prior→{r['prior_end_step']:.0f}, "
              f"plateau→{r['plateau_end_step']:.0f}, "
              f"conv→{r['convergence_step']:.0f}")

    print()
    print("R² of each phase vs log(K):")
    for phase, label in [("Q_prior_vs_logK", "Q_prior"),
                         ("Q_plateau_vs_logK", "Q_plateau"),
                         ("Q_transition_vs_logK", "Q_transition")]:
        f = fits[phase]
        if f["r_squared"] is not None:
            print(f"  {label:<14}: slope={f['slope']:>8.2f}, R² = {f['r_squared']:.3f}")
        else:
            print(f"  {label:<14}: insufficient data")

    print()
    print("══════════════════════════════════════════════════════════════")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Plateau decomposition of dissipation"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="outputs/landauer_comprehensive_results.json",
        help="Path to comprehensive results JSON",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=[
            "landauer_k10", "landauer_k15", "landauer_k20",
            "landauer_k25", "landauer_k36_60k",
        ],
        help="Experiment names (plateau regime)",
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
        default="outputs/plateau_decomposition_results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    # Load comprehensive results
    comp = load_comprehensive_results(args.results)
    exp_lookup = {e["name"]: e for e in comp["experiments"]}

    # Process each experiment
    results = []
    for exp_name in args.experiments:
        if exp_name not in exp_lookup:
            print(f"WARNING: {exp_name} not found, skipping")
            continue
        exp_info = exp_lookup[exp_name]
        print(f"Processing: {exp_name} (K={exp_info['K']})")
        result = process_experiment(exp_info, args.output_dir)
        results.append(result)
        print(f"  Q_prior={result['Q_prior']:.3f}, "
              f"Q_plateau={result['Q_plateau']:.3f}, "
              f"Q_transition={result['Q_transition']:.3f}")

    # Sort by K
    results.sort(key=lambda x: x["K"])

    # Compute fits for each phase vs log(K)
    log_k_vals = [r["log_K"] for r in results]
    fits = {
        "Q_prior_vs_logK": linear_fit(
            log_k_vals, [r["Q_prior"] for r in results]),
        "Q_plateau_vs_logK": linear_fit(
            log_k_vals, [r["Q_plateau"] for r in results]),
        "Q_transition_vs_logK": linear_fit(
            log_k_vals, [r["Q_transition"] for r in results]),
        "Q_total_vs_logK": linear_fit(
            log_k_vals, [r["Q_total"] for r in results]),
    }

    # Print summary
    print_summary(results, fits)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "experiments": results,
        "fits": fits,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
