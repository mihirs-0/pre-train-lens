#!/usr/bin/env python
"""
Regime-Aware Fit: Q = c · log(K / K*)  for K ≥ K*.

Instead of Q = a·log(K) + b with a negative intercept, fit a
parameterization that passes through Q=0 at the phase boundary K*.

K* is the minimum K where the plateau regime begins.
c is the Landauer constant (dissipation per nat of ambiguity erased).

Fitting procedure:
  1. Grid search over K* from 1 to 10 (step 0.1)
  2. For each K*, fit Q = c · log(K/K*) on experiments with K > K*
  3. Select K* that maximizes R² on included points

Usage:
    python scripts/compute_regime_fit.py \\
      --results outputs/landauer_comprehensive_results.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Linear fits
# ═══════════════════════════════════════════════════════════════════════

def fit_proportional(x, y):
    """
    Fit y = c * x through the origin using OLS.
    Returns (c, R², residuals).
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(x) < 1 or np.sum(x ** 2) == 0:
        return None, None
    c = float(np.sum(x * y) / np.sum(x ** 2))
    y_pred = c * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return c, float(r_squared)


def linear_fit(x, y):
    """Affine fit y = slope*x + intercept."""
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
# Regime-aware fit
# ═══════════════════════════════════════════════════════════════════════

def regime_fit_scan(K_vals, Q_vals, q_name="Q_zgap"):
    """
    Scan K* values and fit Q = c · log(K/K*) for K > K*.

    Returns scan results and best fit.
    """
    K_vals = np.array(K_vals, dtype=float)
    Q_vals = np.array(Q_vals, dtype=float)

    # Grid of K* values
    K_star_grid = np.arange(1.0, 10.1, 0.1)

    scan_results = []
    best_score = -np.inf
    best_K_star = None
    best_c = None
    best_r2 = None

    for K_star in K_star_grid:
        # Select experiments with K > K*
        mask = K_vals > K_star
        n_points = int(np.sum(mask))

        if n_points < 2:
            scan_results.append({
                "K_star": round(float(K_star), 1),
                "c": None,
                "r_squared": None,
                "n_points": n_points,
            })
            continue

        # Compute log(K/K*) for included points
        log_ratio = np.log(K_vals[mask] / K_star)
        Q_included = Q_vals[mask]

        # Fit Q = c · log(K/K*) through origin
        c, r2 = fit_proportional(log_ratio, Q_included)

        scan_results.append({
            "K_star": round(float(K_star), 1),
            "c": round(c, 4) if c is not None else None,
            "r_squared": round(r2, 6) if r2 is not None else None,
            "n_points": n_points,
        })

        # Score: prioritize R² but penalize dropping too many points
        if r2 is not None and n_points >= 3:
            score = r2
            if score > best_score:
                best_score = score
                best_K_star = float(K_star)
                best_c = c
                best_r2 = r2

    return scan_results, best_K_star, best_c, best_r2


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Regime-aware fit: Q = c · log(K/K*)"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="outputs/landauer_comprehensive_results.json",
        help="Path to comprehensive results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/regime_fit_results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    # Load comprehensive results
    with open(args.results) as f:
        comp = json.load(f)

    # Extract K and Q values for all experiments (excluding K=50 outlier)
    experiments = [e for e in comp["experiments"] if e["K"] <= 36]
    experiments.sort(key=lambda x: x["K"])

    K_vals = [e["K"] for e in experiments]
    log_K_vals = [math.log(e["K"]) for e in experiments]

    # We'll fit multiple Q definitions
    q_definitions = {}

    for q_name in ["Q_zgap", "Q_total_conv", "Q_original"]:
        Q_vals = []
        valid_K = []
        valid_logK = []
        for e in experiments:
            q = e.get(q_name)
            if q is not None:
                Q_vals.append(q)
                valid_K.append(e["K"])
                valid_logK.append(math.log(e["K"]))

        if len(Q_vals) < 3:
            continue

        # Regime-aware scan
        scan, best_K_star, best_c, best_r2 = regime_fit_scan(
            valid_K, Q_vals, q_name
        )

        # Simple proportional fit Q = c·log(K) on plateau-regime points
        plateau_mask = [K >= 10 for K in valid_K]
        plat_logK = [lk for lk, m in zip(valid_logK, plateau_mask) if m]
        plat_Q = [q for q, m in zip(Q_vals, plateau_mask) if m]
        prop_c, prop_r2 = fit_proportional(plat_logK, plat_Q)

        # Affine fit on plateau-regime points
        affine = linear_fit(plat_logK, plat_Q)

        q_definitions[q_name] = {
            "K_values": valid_K,
            "Q_values": [round(q, 4) for q in Q_vals],
            "best_K_star": round(best_K_star, 1) if best_K_star else None,
            "best_c": round(best_c, 4) if best_c else None,
            "best_r_squared": round(best_r2, 6) if best_r2 else None,
            "K_star_scan": scan,
            "proportional_fit": {
                "c": round(prop_c, 4) if prop_c else None,
                "r_squared": round(prop_r2, 6) if prop_r2 else None,
            },
            "affine_fit": {
                "slope": round(affine["slope"], 4) if affine["slope"] else None,
                "intercept": round(affine["intercept"], 4) if affine["intercept"] else None,
                "r_squared": round(affine["r_squared"], 6) if affine["r_squared"] else None,
            },
        }

    # Print summary
    print()
    print("Regime-Aware Fit: Q = c · log(K/K*)")
    print("══════════════════════════════════════════════════════════════")
    print()

    for q_name, qd in q_definitions.items():
        print(f"  {q_name}:")
        print(f"    Best K*         = {qd['best_K_star']}")
        print(f"    Landauer c      = {qd['best_c']}")
        print(f"    R² (regime)     = {qd['best_r_squared']}")
        print(f"    Proportional c  = {qd['proportional_fit']['c']}  "
              f"R² = {qd['proportional_fit']['r_squared']}")
        af = qd["affine_fit"]
        print(f"    Affine fit      = {af['slope']}·log(K) + ({af['intercept']}), "
              f"R² = {af['r_squared']}")
        print()

    # Detailed scan for primary Q (Q_zgap)
    if "Q_zgap" in q_definitions:
        qd = q_definitions["Q_zgap"]
        print("K* scan for Q_zgap (top 10 by R²):")
        print(f"  {'K*':>5}  {'c':>8}  {'R²':>8}  {'N':>3}")
        print("  " + "─" * 28)
        top_scans = sorted(
            [s for s in qd["K_star_scan"] if s["r_squared"] is not None],
            key=lambda s: s["r_squared"],
            reverse=True,
        )[:10]
        for s in top_scans:
            print(f"  {s['K_star']:>5.1f}  {s['c']:>8.2f}  "
                  f"{s['r_squared']:>8.4f}  {s['n_points']:>3}")

    print()
    print("══════════════════════════════════════════════════════════════")
    print()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(q_definitions, f, indent=2)

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
