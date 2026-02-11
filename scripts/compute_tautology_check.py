#!/usr/bin/env python
"""
Tautology Check: Is Q_zgap just tracking ΔL?

In continuous-time gradient flow: dL/dt = -||∇L||²
So ∫ ||∇L||² dt = ΔL (loss decrease).

Our Q = η × Σ ||∇L||² × Δs.  If the gradient flow identity held
exactly in discrete SGD, we'd have Q_zgap ≈ ΔL_zgap (the tautology).

This script computes the dissipation ratio R = Q_zgap / ΔL_zgap.
If R ≈ 1 for all K, Q is trivially tracking ΔL.
If R >> 1 and Q_excess = Q_zgap - ΔL_zgap still scales with log(K),
the Landauer result is non-trivial.

Usage:
    python scripts/compute_tautology_check.py \\
      --results outputs/landauer_comprehensive_results.json \\
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
# Tautology computation
# ═══════════════════════════════════════════════════════════════════════

def interpolate_candidate_loss(cand_steps, candidate_loss, target_step):
    """Interpolate candidate_loss at target_step."""
    cand_steps = np.array(cand_steps, dtype=float)
    candidate_loss = np.array(candidate_loss, dtype=float)
    return float(np.interp(target_step, cand_steps, candidate_loss))


def process_experiment(exp_info: dict, output_dir: str) -> dict:
    """Compute tautology metrics for one experiment."""
    exp_name = exp_info["name"]
    K = exp_info["K"]
    log_K = math.log(K)
    Q_zgap = exp_info["Q_zgap"]
    zgap_start = exp_info["zgap_start"]
    zgap_end = exp_info["zgap_end"]

    # Load candidate eval data
    cand = load_candidate_eval(exp_name, output_dir)
    cand_steps = cand["steps"]
    candidate_loss = cand["candidate_loss"]

    # Interpolate candidate_loss at zgap window boundaries
    loss_at_start = interpolate_candidate_loss(cand_steps, candidate_loss, zgap_start)
    loss_at_end = interpolate_candidate_loss(cand_steps, candidate_loss, zgap_end)

    # ΔL_zgap = loss decrease over z-gap window (positive value)
    delta_L_zgap = loss_at_start - loss_at_end

    # Guard against zero/negative ΔL
    if delta_L_zgap <= 0:
        print(f"  WARNING: ΔL_zgap <= 0 for {exp_name} "
              f"(loss_start={loss_at_start:.4f}, loss_end={loss_at_end:.4f})")
        dissipation_ratio = float("inf")
    else:
        dissipation_ratio = Q_zgap / delta_L_zgap

    Q_excess = Q_zgap - delta_L_zgap

    return {
        "name": exp_name,
        "K": K,
        "log_K": round(log_K, 4),
        "Q_zgap": Q_zgap,
        "delta_L_zgap": round(delta_L_zgap, 6),
        "delta_L_over_logK": round(delta_L_zgap / log_K, 4) if log_K > 0 else None,
        "dissipation_ratio": round(dissipation_ratio, 4),
        "Q_excess": round(Q_excess, 6),
        "Q_excess_over_logK": round(Q_excess / log_K, 4) if log_K > 0 else None,
        "zgap_start": zgap_start,
        "zgap_end": zgap_end,
        "candidate_loss_at_start": round(loss_at_start, 6),
        "candidate_loss_at_end": round(loss_at_end, 6),
    }


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════

def print_summary(results, fits, tautology_test):
    print()
    print("Tautology Check: Is Q_zgap Just Tracking ΔL?")
    print("══════════════════════════════════════════════════════════════")
    header = (f"{'K':>4}  {'log(K)':>7}  {'Q_zgap':>10}  {'ΔL_zgap':>10}  "
              f"{'Q/ΔL':>8}  {'Q_excess':>10}  {'ΔL/logK':>8}")
    print(header)
    print("─" * len(header))

    for r in results:
        print(
            f"{r['K']:>4}  {r['log_K']:>7.3f}  {r['Q_zgap']:>10.2f}  "
            f"{r['delta_L_zgap']:>10.4f}  {r['dissipation_ratio']:>8.1f}  "
            f"{r['Q_excess']:>10.2f}  {r['delta_L_over_logK']:>8.3f}"
        )

    print("══════════════════════════════════════════════════════════════")
    print()

    ratios = [r["dissipation_ratio"] for r in results]
    print(f"If tautology held: Q/ΔL ≈ 1 for all K.")
    print(f"Actual Q/ΔL range: [{min(ratios):.1f}, {max(ratios):.1f}]")
    print(f"Ratio CV (coefficient of variation): {tautology_test['ratio_CV']:.3f}")
    print()

    fit_excess = fits["Q_excess_vs_logK"]
    print(f"Q_excess vs log(K): slope={fit_excess['slope']:.2f}, "
          f"intercept={fit_excess['intercept']:.2f}, "
          f"R²={fit_excess['r_squared']:.3f}")

    fit_ratio = fits["ratio_vs_logK"]
    print(f"Q/ΔL vs log(K):    slope={fit_ratio['slope']:.2f}, "
          f"intercept={fit_ratio['intercept']:.2f}, "
          f"R²={fit_ratio['r_squared']:.3f}")
    print()
    print(f"VERDICT: {tautology_test['verdict']}")
    print("══════════════════════════════════════════════════════════════")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Tautology check: is Q_zgap just tracking ΔL?"
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
        help="Experiment names (plateau regime only)",
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
        default="outputs/tautology_check_results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    # Load comprehensive results to get Q_zgap and window boundaries
    comp = load_comprehensive_results(args.results)
    exp_lookup = {e["name"]: e for e in comp["experiments"]}

    # Process each experiment
    results = []
    for exp_name in args.experiments:
        if exp_name not in exp_lookup:
            print(f"WARNING: {exp_name} not found in comprehensive results, skipping")
            continue
        exp_info = exp_lookup[exp_name]
        if exp_info["Q_zgap"] is None:
            print(f"WARNING: {exp_name} has no Q_zgap, skipping")
            continue
        print(f"Processing: {exp_name} (K={exp_info['K']})")
        result = process_experiment(exp_info, args.output_dir)
        results.append(result)
        print(f"  ΔL_zgap = {result['delta_L_zgap']:.4f}, "
              f"Q/ΔL = {result['dissipation_ratio']:.1f}, "
              f"Q_excess = {result['Q_excess']:.2f}")

    # Sort by K
    results.sort(key=lambda x: x["K"])

    # Compute fits
    log_k_vals = [r["log_K"] for r in results]
    q_excess_vals = [r["Q_excess"] for r in results]
    ratio_vals = [r["dissipation_ratio"] for r in results]

    fits = {
        "Q_excess_vs_logK": linear_fit(log_k_vals, q_excess_vals),
        "ratio_vs_logK": linear_fit(log_k_vals, ratio_vals),
        "Q_zgap_vs_logK": linear_fit(log_k_vals, [r["Q_zgap"] for r in results]),
        "delta_L_vs_logK": linear_fit(log_k_vals, [r["delta_L_zgap"] for r in results]),
    }

    # Tautology test: is the ratio constant?
    ratios = np.array(ratio_vals)
    ratio_mean = float(np.mean(ratios))
    ratio_std = float(np.std(ratios))
    ratio_cv = ratio_std / ratio_mean if ratio_mean > 0 else float("inf")

    # If CV < 0.15, the ratio is approximately constant → tautology concern
    # If CV > 0.15 and Q_excess scales with log(K), tautology is killed
    is_ratio_constant = ratio_cv < 0.15

    if is_ratio_constant:
        verdict = ("TAUTOLOGY CONCERN REMAINS: Q/ΔL ratio is approximately "
                   f"constant (CV={ratio_cv:.3f}), suggesting Q may track ΔL "
                   "with a fixed multiplier. However, this multiplier >> 1 "
                   "still indicates excess dissipation beyond loss descent.")
    else:
        r2_excess = fits["Q_excess_vs_logK"]["r_squared"]
        if r2_excess is not None and r2_excess > 0.8:
            verdict = (f"TAUTOLOGY KILLED: Q/ΔL ratio varies across K "
                       f"(CV={ratio_cv:.3f}), and Q_excess = Q_zgap - ΔL_zgap "
                       f"scales with log(K) (R²={r2_excess:.3f}). "
                       "The Landauer signal is NOT an artifact of loss descent.")
        else:
            verdict = (f"TAUTOLOGY WEAKENED: Q/ΔL ratio varies across K "
                       f"(CV={ratio_cv:.3f}), but Q_excess vs log(K) fit "
                       f"is moderate (R²={r2_excess:.3f}). Excess dissipation "
                       "exists but scaling is noisy.")

    tautology_test = {
        "is_ratio_constant": bool(is_ratio_constant),
        "ratio_CV": round(ratio_cv, 4),
        "ratio_mean": round(ratio_mean, 4),
        "ratio_std": round(ratio_std, 4),
        "ratio_range": [round(float(np.min(ratios)), 4),
                        round(float(np.max(ratios)), 4)],
        "verdict": verdict,
    }

    # Print summary
    print_summary(results, fits, tautology_test)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "experiments": results,
        "fits": fits,
        "tautology_test": tautology_test,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
