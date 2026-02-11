#!/usr/bin/env python
"""
Analysis of two-layer linear network ablation results.

Checks for:
  1. Plateau → cliff dynamics in forward direction
  2. Q ∝ log(K) Landauer scaling
  3. Directional asymmetry (forward vs reverse)
  4. Comparison with transformer results

Produces:
  - outputs/twolayer_linear_analysis.json (structured results)
  - outputs/figures/twolayer_linear_analysis.png (5-panel figure)

Usage:
    python scripts/analyze_twolayer_linear.py
    python scripts/analyze_twolayer_linear.py --H 64  # bottleneck runs
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_run(path: str) -> dict:
    """Load a single run's JSON results."""
    with open(path) as f:
        return json.load(f)


def find_runs(output_dir: str, directions, K_values, H_value):
    """Discover available result files."""
    runs = {}
    for direction in directions:
        for K in K_values:
            fname = f"twolayer_linear_{direction}_K{K}_H{H_value}_results.json"
            path = Path(output_dir) / fname
            if path.exists():
                key = (direction, K, H_value)
                runs[key] = load_run(str(path))
    return runs


def load_transformer_baseline(output_dir: str):
    """Load transformer comprehensive results for comparison."""
    path = Path(output_dir) / "landauer_comprehensive_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# Plateau detection
# ═══════════════════════════════════════════════════════════════════════

def detect_plateau(steps, candidate_loss, log_K, min_duration=500):
    """
    Detect whether candidate_loss plateaus near log(K).

    A plateau exists if candidate_loss stays within [0.8·log(K), 1.2·log(K)]
    for a contiguous stretch of at least min_duration training steps.
    """
    steps = np.array(steps, dtype=float)
    cl = np.array(candidate_loss, dtype=float)

    low = 0.8 * log_K
    high = 1.2 * log_K

    in_plateau = (cl >= low) & (cl <= high)

    max_duration = 0
    current_start = None

    for i in range(len(in_plateau)):
        if in_plateau[i]:
            if current_start is None:
                current_start = steps[i]
        else:
            if current_start is not None:
                duration = steps[i] - current_start
                max_duration = max(max_duration, duration)
                current_start = None

    if current_start is not None:
        duration = steps[-1] - current_start
        max_duration = max(max_duration, duration)

    return bool(max_duration >= min_duration), float(max_duration)


def compute_transition_sharpness(steps, candidate_loss):
    """Max absolute rate of change of candidate_loss (smoothed)."""
    steps = np.array(steps, dtype=float)
    cl = np.array(candidate_loss, dtype=float)

    if len(cl) < 5:
        return 0.0

    # Smooth with window of 5
    kernel = np.ones(5) / 5
    cl_smooth = np.convolve(cl, kernel, mode="valid")
    steps_smooth = np.convolve(steps, kernel, mode="valid")

    ds = np.diff(steps_smooth)
    ds[ds == 0] = 1.0
    dcl = np.diff(cl_smooth)
    rate = np.abs(dcl / ds)

    return float(np.max(rate)) if len(rate) > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════
# Q_zgap computation (mirrors compute_landauer_comprehensive.py)
# ═══════════════════════════════════════════════════════════════════════

def find_zgap_window(steps, z_gap):
    """
    zgap_start: first step where z_gap > 0.5
    zgap_end:   first step where z_gap > 0.9 × max(z_gap)
    """
    steps = np.array(steps, dtype=float)
    z_gap = np.array(z_gap, dtype=float)
    max_zgap = float(np.max(z_gap))

    zgap_start = None
    mask_start = z_gap > 0.5
    if np.any(mask_start):
        zgap_start = float(steps[np.argmax(mask_start)])

    zgap_end = None
    threshold_end = 0.9 * max_zgap
    mask_end = z_gap > threshold_end
    if np.any(mask_end):
        zgap_end = float(steps[np.argmax(mask_end)])

    if zgap_start is not None and zgap_end is not None:
        if zgap_start >= zgap_end:
            after = steps >= zgap_start
            if np.any(after):
                idx_after = np.where(after)[0]
                best = idx_after[np.argmax(z_gap[idx_after])]
                zgap_end = float(steps[best])
                if zgap_end <= zgap_start and best + 1 < len(steps):
                    zgap_end = float(steps[best + 1])

    return zgap_start, zgap_end, max_zgap


def compute_cumulative_dissipation(steps, grad_norm_sq, lr):
    """Q(t) = Σ η × ||∇L(s)||² × Δs."""
    steps = np.array(steps, dtype=float)
    norms = np.array(grad_norm_sq, dtype=float)
    delta_s = np.diff(steps, prepend=0)
    Q_cum = np.cumsum(lr * norms * delta_s)
    return steps, Q_cum


def compute_Q_in_window(steps, Q_cum, start, end):
    Q_start = float(np.interp(start, steps, Q_cum))
    Q_end = float(np.interp(end, steps, Q_cum))
    return Q_end - Q_start


def interpolate(steps, values, target_step):
    return float(np.interp(target_step, steps, values))


# ═══════════════════════════════════════════════════════════════════════
# Linear fit
# ═══════════════════════════════════════════════════════════════════════

def linear_fit(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(x) < 2:
        return None, None, None
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)


# ═══════════════════════════════════════════════════════════════════════
# Process one run
# ═══════════════════════════════════════════════════════════════════════

def process_run(data: dict) -> dict:
    """Analyze a single experiment run."""
    cfg = data["config"]
    K = cfg["K"]
    log_K = math.log(K)
    lr = cfg["lr"]
    direction = cfg["direction"]

    steps = data["steps"]
    cl = data["candidate_loss"]
    zg = data["z_gap"]
    gnsq = data["gradient_norm_sq"]

    # Plateau detection
    has_plateau, plateau_duration = detect_plateau(steps, cl, log_K)
    transition_sharpness = compute_transition_sharpness(steps, cl)

    # Q_zgap
    Q_zgap = None
    delta_L_zgap = None
    Q_over_deltaL = None
    zgap_start = None
    zgap_end = None

    zg_start, zg_end, max_zgap = find_zgap_window(steps, zg)
    zgap_start = zg_start
    zgap_end = zg_end

    if zg_start is not None and zg_end is not None and max_zgap >= 0.5:
        steps_arr, Q_cum = compute_cumulative_dissipation(steps, gnsq, lr)
        Q_zgap = compute_Q_in_window(steps_arr, Q_cum, zg_start, zg_end)

        cl_at_start = interpolate(steps, cl, zg_start)
        cl_at_end = interpolate(steps, cl, zg_end)
        delta_L_zgap = cl_at_start - cl_at_end

        if delta_L_zgap > 0:
            Q_over_deltaL = Q_zgap / delta_L_zgap

    return {
        "K": K,
        "H": cfg["H"],
        "direction": direction,
        "log_K": round(log_K, 4),
        "plateau_detected": has_plateau,
        "plateau_duration": plateau_duration,
        "transition_sharpness": round(transition_sharpness, 6),
        "Q_zgap": round(Q_zgap, 4) if Q_zgap is not None else None,
        "delta_L_zgap": round(delta_L_zgap, 4) if delta_L_zgap is not None else None,
        "Q_over_deltaL": round(Q_over_deltaL, 4) if Q_over_deltaL is not None else None,
        "zgap_start": zgap_start,
        "zgap_end": zgap_end,
        "max_zgap": round(max_zgap, 4) if max_zgap is not None else None,
        "final_candidate_loss": round(cl[-1], 4),
        "final_accuracy": round(data["candidate_accuracy"][-1], 4),
        "final_zgap": round(zg[-1], 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

COLORS = {10: "#1f77b4", 15: "#ff7f0e", 20: "#2ca02c", 25: "#d62728", 36: "#9467bd"}


def plot_results(runs, forward_results, reverse_results, scaling_fit,
                 transformer_baseline, output_path):
    """Generate the 5-panel analysis figure."""
    if not HAS_MPL:
        print("matplotlib not available — skipping plots")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Two-Layer Linear Network: Ziyin Ablation", fontsize=14, y=0.98)

    # ── Panel 1: candidate_loss vs step (forward) ──
    ax = axes[0, 0]
    for key, data in sorted(runs.items()):
        direction, K, H = key
        if direction != "forward":
            continue
        c = COLORS.get(K, "gray")
        ax.plot(data["steps"], data["candidate_loss"],
                label=f"K={K}", color=c, alpha=0.8)
        ax.axhline(math.log(K), color=c, linestyle=":", alpha=0.3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Candidate Loss")
    ax.set_title("Forward (Bz→A): Candidate Loss")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=-0.1)

    # ── Panel 2: candidate_loss vs step (reverse) ──
    ax = axes[0, 1]
    for key, data in sorted(runs.items()):
        direction, K, H = key
        if direction != "reverse":
            continue
        c = COLORS.get(K, "gray")
        ax.plot(data["steps"], data["candidate_loss"],
                label=f"K={K}", color=c, alpha=0.8)
        ax.axhline(math.log(K), color=c, linestyle=":", alpha=0.3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Candidate Loss")
    ax.set_title("Reverse (Az→B): Candidate Loss")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=-0.1)

    # ── Panel 3: z_gap vs step (forward) ──
    ax = axes[0, 2]
    for key, data in sorted(runs.items()):
        direction, K, H = key
        if direction != "forward":
            continue
        c = COLORS.get(K, "gray")
        ax.plot(data["steps"], data["z_gap"],
                label=f"K={K}", color=c, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("z-gap")
    ax.set_title("Forward (Bz→A): z-gap")
    ax.legend(fontsize=8)

    # ── Panel 4: Q_zgap vs log(K) (if available) ──
    ax = axes[1, 0]
    # Two-layer linear data
    fwd_with_Q = [r for r in forward_results if r["Q_zgap"] is not None]
    if fwd_with_Q:
        lk = [r["log_K"] for r in fwd_with_Q]
        qq = [r["Q_zgap"] for r in fwd_with_Q]
        ax.scatter(lk, qq, marker="o", s=80, zorder=5,
                   label="Two-layer linear", color="#1f77b4")
        if scaling_fit and scaling_fit.get("slope") is not None:
            x_fit = np.linspace(min(lk) - 0.1, max(lk) + 0.1, 50)
            y_fit = scaling_fit["slope"] * x_fit + scaling_fit["intercept"]
            ax.plot(x_fit, y_fit, "--", color="#1f77b4", alpha=0.7,
                    label=f"Linear: slope={scaling_fit['slope']:.1f}, "
                          f"R²={scaling_fit['r_squared']:.3f}")
    # Transformer comparison
    if transformer_baseline is not None:
        t_exps = transformer_baseline.get("experiments", [])
        t_plateau = [e for e in t_exps if e.get("has_plateau") and e.get("Q_zgap")]
        if t_plateau:
            t_lk = [e["log_K"] for e in t_plateau]
            t_qq = [e["Q_zgap"] for e in t_plateau]
            ax.scatter(t_lk, t_qq, marker="^", s=80, zorder=5,
                       label="Transformer", color="#d62728")
            # Transformer fit line
            ts, ti, tr = linear_fit(t_lk, t_qq)
            if ts is not None:
                x_t = np.linspace(min(t_lk) - 0.1, max(t_lk) + 0.1, 50)
                ax.plot(x_t, ts * x_t + ti, "--", color="#d62728", alpha=0.5,
                        label=f"Transformer: slope={ts:.1f}, R²={tr:.3f}")
    ax.set_xlabel("log(K)")
    ax.set_ylabel("Q_zgap")
    ax.set_title("Landauer Scaling: Q_zgap vs log(K)")
    ax.legend(fontsize=7)

    # ── Panel 5: Forward vs Reverse for K=20 ──
    ax = axes[1, 1]
    for key, data in runs.items():
        direction, K, H = key
        if K == 20:
            style = "-" if direction == "forward" else "--"
            label = f"{direction.capitalize()} (K=20)"
            ax.plot(data["steps"], data["candidate_loss"],
                    style, label=label, alpha=0.8)
    # Fallback to K=10 if K=20 not available
    if not any(K == 20 for (_, K, _) in runs):
        for key, data in runs.items():
            direction, K, H = key
            if K == 10:
                style = "-" if direction == "forward" else "--"
                ax.plot(data["steps"], data["candidate_loss"],
                        style, label=f"{direction.capitalize()} (K={K})", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Candidate Loss")
    ax.set_title("Directional Asymmetry")
    ax.legend(fontsize=8)

    # ── Panel 6: gradient_norm_sq vs step (forward) ──
    ax = axes[1, 2]
    for key, data in sorted(runs.items()):
        direction, K, H = key
        if direction != "forward":
            continue
        c = COLORS.get(K, "gray")
        ax.plot(data["steps"], data["gradient_norm_sq"],
                label=f"K={K}", color=c, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("||∇L||²")
    ax.set_title("Forward (Bz→A): Gradient Norm²")
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Verdict
# ═══════════════════════════════════════════════════════════════════════

def determine_verdict(forward_results, reverse_results, scaling_fit):
    """Determine the scientific verdict from the results."""
    fwd_plateaus = [r["plateau_detected"] for r in forward_results]
    rev_plateaus = [r["plateau_detected"] for r in reverse_results]

    any_fwd_plateau = any(fwd_plateaus)
    any_rev_plateau = any(rev_plateaus)

    # Check if ANY run actually converged (candidate_loss << log(K))
    fwd_converged = [r for r in forward_results
                     if r["final_candidate_loss"] < 0.5 * r["log_K"]]
    rev_converged = [r for r in reverse_results
                     if r["final_candidate_loss"] < 0.5 * r["log_K"]]

    has_scaling = (scaling_fit is not None
                   and scaling_fit.get("r_squared") is not None
                   and scaling_fit["r_squared"] > 0.8)

    # ── Primary classification: did the model solve the task at all? ──
    if not fwd_converged and not rev_converged:
        # Neither direction converged — capacity failure
        verdict = (
            "CAPACITY FAILURE: The two-layer linear network cannot solve "
            "the disambiguation task in either direction. Candidate loss "
            "stays at log(K) (uniform over candidates) and accuracy stays "
            "at 1/K (random guessing) indefinitely. This is a fundamental "
            "representational limitation: a linear model y = W·[b;z] "
            "computes an additive function of b and z, but the task "
            "requires a nonlinear (b × z) interaction — z must select "
            "DIFFERENT candidates per B group. The model converges to the "
            "group-mean predictor (MSE decreases) but cannot break the "
            "per-group candidate symmetry. The sharp transition in the "
            "transformer requires nonlinear conditional routing (attention "
            "patterns that gate information from z based on b), which is "
            "outside the linear function class."
        )
    elif fwd_converged and has_scaling:
        verdict = (
            "Two-layer linear network shows plateau dynamics WITH log(K) "
            "dissipation scaling → Landauer scaling is universal across "
            "architectures, driven by information structure not architecture."
        )
    elif fwd_converged and not has_scaling:
        verdict = (
            "Two-layer linear network solves the task but shows NO log(K) "
            "dissipation scaling → Landauer scaling requires nonlinear "
            "representational dynamics beyond eigenvalue plateau effects."
        )
    elif not fwd_converged and rev_converged:
        verdict = (
            "Two-layer linear network cannot solve forward direction but "
            "solves reverse → directional asymmetry exists but with "
            "opposite polarity to the transformer (unexpected)."
        )
    else:
        verdict = "Inconclusive — insufficient data for clear determination."

    # ── Directional asymmetry assessment ──
    if not fwd_converged and not rev_converged:
        verdict += (
            " Directional asymmetry is UNINFORMATIVE: both directions "
            "fail, unlike the transformer where reverse converges "
            "immediately. The linear network's failure is symmetric."
        )
    elif fwd_converged and not rev_converged:
        verdict += (
            " Directional asymmetry (forward solves, reverse stuck) — "
            "unexpected; differs from transformer pattern."
        )
    elif fwd_converged and rev_converged:
        fwd_with_plateau = [r for r in forward_results if r["plateau_detected"]]
        rev_with_plateau = [r for r in reverse_results if r["plateau_detected"]]
        if fwd_with_plateau and not rev_with_plateau:
            verdict += (
                " Directional asymmetry (forward plateau, reverse smooth) "
                "matches transformer → asymmetry is a property of "
                "information structure, not transformer-specific computation."
            )

    return verdict


# ═══════════════════════════════════════════════════════════════════════
# Summary printing
# ═══════════════════════════════════════════════════════════════════════

def print_summary(forward_results, reverse_results, scaling_fit, verdict):
    print()
    print("Two-Layer Linear Network: Ziyin Ablation")
    print("══════════════════════════════════════════════════════════════")

    def fmt(v, w=10):
        if v is None:
            return f"{'—':>{w}}"
        return f"{v:>{w}.4f}"

    if forward_results:
        print("FORWARD (Bz → A):")
        header = (f"{'K':>4}  {'plateau?':>8}  {'duration':>8}  "
                  f"{'Q_zgap':>10}  {'Q/ΔL':>10}  {'sharpness':>10}")
        print(header)
        print("-" * len(header))
        for r in forward_results:
            plateau_str = "YES" if r["plateau_detected"] else "NO"
            print(
                f"{r['K']:>4}  {plateau_str:>8}  {r['plateau_duration']:>8.0f}  "
                f"{fmt(r['Q_zgap'])}  {fmt(r['Q_over_deltaL'])}  "
                f"{r['transition_sharpness']:>10.6f}"
            )
        print()

    if reverse_results:
        print("REVERSE (Az → B):")
        header = (f"{'K':>4}  {'plateau?':>8}  {'duration':>8}  "
                  f"{'sharpness':>10}")
        print(header)
        print("-" * len(header))
        for r in reverse_results:
            plateau_str = "YES" if r["plateau_detected"] else "NO"
            print(
                f"{r['K']:>4}  {plateau_str:>8}  {r['plateau_duration']:>8.0f}  "
                f"{r['transition_sharpness']:>10.6f}"
            )
        print()

    if scaling_fit and scaling_fit.get("slope") is not None:
        print("SCALING FIT:")
        print(f"  Q_zgap = {scaling_fit['slope']:.2f} × log(K) + "
              f"{scaling_fit['intercept']:.2f}, R² = {scaling_fit['r_squared']:.3f}")
        tc = scaling_fit.get("transformer_comparison", {})
        if tc.get("slope") is not None:
            print(f"  (Transformer: Q_zgap = {tc['slope']:.1f} × log(K) "
                  f"+ {tc['intercept']:.1f}, R² = {tc['r_squared']:.3f})")
        print()
    else:
        print("SCALING FIT: insufficient data (< 3 forward runs with Q_zgap)\n")

    print(f"VERDICT: {verdict}")
    print("══════════════════════════════════════════════════════════════")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyze two-layer linear network ablation results"
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--K-values", type=int, nargs="+",
                        default=[10, 15, 20, 25, 36])
    parser.add_argument("--H", type=int, default=128,
                        help="Hidden dimension to analyze")
    parser.add_argument("--output", type=str,
                        default="outputs/twolayer_linear_analysis.json")
    parser.add_argument("--figure", type=str,
                        default="outputs/figures/twolayer_linear_analysis.png")
    args = parser.parse_args()

    # ── Discover runs ──
    runs = find_runs(args.output_dir, ["forward", "reverse"],
                     args.K_values, args.H)

    if not runs:
        print(f"No results found in {args.output_dir}/ for H={args.H}")
        print("Expected files like: twolayer_linear_forward_K10_H128_results.json")
        return

    print(f"Found {len(runs)} runs:")
    for key in sorted(runs.keys()):
        d, K, H = key
        print(f"  {d} K={K} H={H}")

    # ── Process each run ──
    forward_results = []
    reverse_results = []

    for key, data in sorted(runs.items()):
        direction, K, H = key
        result = process_run(data)
        if direction == "forward":
            forward_results.append(result)
        else:
            reverse_results.append(result)

    forward_results.sort(key=lambda r: r["K"])
    reverse_results.sort(key=lambda r: r["K"])

    # ── Scaling fit (forward only, needs >= 3 runs with Q_zgap) ──
    fwd_with_Q = [r for r in forward_results if r["Q_zgap"] is not None]
    scaling_fit = None

    if len(fwd_with_Q) >= 3:
        lk = [r["log_K"] for r in fwd_with_Q]
        qq = [r["Q_zgap"] for r in fwd_with_Q]
        slope, intercept, r2 = linear_fit(lk, qq)
        scaling_fit = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r2,
            "n_points": len(fwd_with_Q),
            "transformer_comparison": {
                "slope": 120.3, "intercept": -235.5, "r_squared": 0.919,
            },
        }
    elif len(fwd_with_Q) >= 2:
        lk = [r["log_K"] for r in fwd_with_Q]
        qq = [r["Q_zgap"] for r in fwd_with_Q]
        slope, intercept, r2 = linear_fit(lk, qq)
        scaling_fit = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r2,
            "n_points": len(fwd_with_Q),
            "transformer_comparison": {
                "slope": 120.3, "intercept": -235.5, "r_squared": 0.919,
            },
            "warning": "only 2 points — fit is underdetermined",
        }

    # ── Directional asymmetry ──
    fwd_K_set = {r["K"] for r in forward_results}
    rev_K_set = {r["K"] for r in reverse_results}
    common_K = sorted(fwd_K_set & rev_K_set)

    fwd_plateau_by_K = {r["K"]: r["plateau_detected"] for r in forward_results}
    rev_plateau_by_K = {r["K"]: r["plateau_detected"] for r in reverse_results}

    directional_asymmetry = {
        "K_values_compared": common_K,
        "forward_has_plateau": [fwd_plateau_by_K.get(K, None) for K in common_K],
        "reverse_has_plateau": [rev_plateau_by_K.get(K, None) for K in common_K],
        "asymmetry_detected": (
            any(fwd_plateau_by_K.get(K, False) for K in common_K)
            and not any(rev_plateau_by_K.get(K, False) for K in common_K)
        ),
    }

    # ── Verdict ──
    verdict = determine_verdict(forward_results, reverse_results, scaling_fit)

    # ── Print summary ──
    print_summary(forward_results, reverse_results, scaling_fit, verdict)

    # ── Load transformer baseline for plots ──
    transformer_baseline = load_transformer_baseline(args.output_dir)

    # ── Plot ──
    fig_path = Path(args.figure)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_results(runs, forward_results, reverse_results, scaling_fit,
                 transformer_baseline, str(fig_path))

    # ── Save JSON ──
    output = {
        "forward_results": forward_results,
        "reverse_results": reverse_results,
        "scaling_fit": scaling_fit,
        "directional_asymmetry": directional_asymmetry,
        "verdict": verdict,
        "H_analyzed": args.H,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved results: {output_path}")


if __name__ == "__main__":
    main()
