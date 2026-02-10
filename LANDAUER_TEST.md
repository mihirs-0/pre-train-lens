# Landauer Dissipation Test — First Pass

**Question:** Does the cumulative gradient-norm cost during the phase transition scale with log(K)?

**Status:** Consistent with Landauer scaling for K=5, 10, 20 (matched hyperparameters). K=36 confounded by different LR schedule.

## Background

Ziyin's effective free energy framework predicts that escaping the symmetry-preserving plateau requires overcoming an entropic barrier. If the phase transition is analogous to an irreversible bit-erasure operation (Landauer's principle), the minimum dissipation cost should scale with the information content being resolved — which is log(K) bits, since the model must learn to distinguish K candidates per base string.

We define the **dissipation integral**:

    Q(t) = Σ_{s=0}^{t} η(s) · ||∇L(s)||² · Δs

where η(s) is the actual learning rate at step s (reconstructed from the warmup + cosine decay schedule), ||∇L(s)||² is the measured gradient norm squared, and Δs is the step spacing between checkpoints.

The **transition dissipation** Q_transition = Q(transition_end) - Q(transition_start) captures the cumulative dissipation during the phase transition only.

If Landauer scaling holds: **Q_transition / log(K) ≈ constant** across K values.

## Experiments

All experiments use: 4-layer transformer, d_model=128, lr=1e-3, weight_decay=0.01, warmup=500 steps, seed=42.

| Experiment | K | BS | T_eff | max_steps | Schedule | Transitioned? |
|---|---|---|---|---|---|---|
| `temp_lr1e3_bs128_k5` | 5 | 128 | 7.81e-6 | 10,000 | 10K cosine | Yes (step ~900) |
| `temp_lr1e3_bs128` | 10 | 128 | 7.81e-6 | 10,000 | 10K cosine | Yes (step ~2,100) |
| `temp_lr1e3_bs128_k20` | 20 | 128 | 7.81e-6 | 10,000 | 10K cosine | Yes (step ~4,700) |
| `temp_lr1e3_bs128_k36_30k` | 36 | 128 | 7.81e-6 | 30,000 | 30K cosine | Yes (step ~22,600) |

**Note on K=36:** The original K=36 run at 10K steps (`temp_lr1e3_bs128_k36`) never escaped the plateau. The barrier is real but not permanent — it required 3x the training budget. A separate run with bs=512 (`temp_lr1e3_bs512_k36`) completed the transition at 10K steps, but that uses a different T_eff.

## Results

### Matched subset: K=5, 10, 20 (identical hyperparameters + schedule)

```
   K   log(K)   Transition    Q_trans    Q_trans/logK   plateau_S   peak_S
   5    1.609    500→ 900       3.859        2.398         0.647      9.785
  10    2.303    900→2100       7.663        3.328         0.367     10.646
  20    2.996   2700→4700       8.440        2.817         0.379      7.437
  ──────────────────────────────────────────────────────────────────────────
  Q_trans/log(K):  mean = 2.848    std = 0.381    CV = 0.134
```

**Q_trans/log(K) is approximately constant (CV = 0.134).** This is consistent with Landauer scaling across a factor of 4 in K, at matched effective temperature.

### Full set including K=36 (30K cosine — different schedule)

```
  36    3.583   6100→10900     25.642        7.156         0.188     10.577
  ──────────────────────────────────────────────────────────────────────────
  All 4:  mean = 3.925    std = 1.894    CV = 0.483
```

K=36's ratio (7.16) is ~2.5x higher than the matched trio's mean (2.85). This is **confounded** by the different LR schedule: with max_steps=30K, the cosine decay is much slower, so the LR during K=36's transition window (steps 6100–10900) is ~72–90% of peak. For the same step range in a 10K schedule, the LR would already be near zero. The different schedule inflates K=36's dissipation integral.

## Transition window definition

The transition window is defined from `candidate_eval_results.json`:
- **transition_start**: last step where candidate_loss > 0.9 × log(K)
- **transition_end**: first step where candidate_loss < 0.1 × log(K)

This captures the interval over which the model goes from random-guessing among K candidates to reliably selecting the correct one.

## Key observations

1. **Plateau duration scales with K.** K=5 transitions by step ~900, K=10 by ~2,100, K=20 by ~4,700, K=36 by ~22,600. The barrier height increases with K.

2. **K=36 at bs=128 required 30K steps.** The original 10K run never escaped the plateau. This confirms the barrier is physically real and grows substantially with K — at some point it becomes impassable for a given T_eff and training budget.

3. **Gradient norm spikes during transition** for all K values (peak_S ≈ 7–11), consistent with symmetry breaking driving coherent gradients.

4. **Plateau gradient norms decrease with K** (plateau_S: 0.65 for K=5, 0.37 for K=10/20, 0.19 for K=36). More candidates → more gradient cancellation during the symmetric phase.

## Confounds and limitations

1. **K=36 uses a different LR schedule** (30K cosine vs 10K). The comparison is not clean. A proper test would use matched schedules for all K values.

2. **Single seed.** No error bars. The CV of 0.134 looks good but with N=3 the uncertainty on CV itself is large.

3. **Only 3 fully matched data points.** K=5, 10, 20 span a factor of 4 in K (log(K) from 1.6 to 3.0). Narrow range.

4. **Incomplete convergence for K=36.** Final first_target_loss = 0.77 (not zero) because the LR had decayed to ~3% by the time of transition. The transition_end estimate may be imprecise.

## Recommendations for Phase 2

1. **Match schedules.** Retrain K=5, 10, 20 with 30K max_steps so all four share the same cosine profile. Or use constant LR for the Landauer comparison.

2. **Multiple seeds.** At least 3 seeds per K for error bars.

3. **Wider K range.** Add K=50 or K=100 if computationally feasible (may require larger batch size or longer training).

4. **Constant-LR control.** Run with constant η (no cosine decay) to isolate the effect of the schedule on Q_transition. This gives a cleaner integral.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/compute_landauer_cost.py` | Computes dissipation integral Q(t) with actual LR schedule |
| `scripts/plot_landauer_test.py` | 2×2 diagnostic figure |
| `scripts/compute_gradient_norms.py` | Measures ||∇L||² at each checkpoint |
| `scripts/run_candidate_eval.py` | Measures candidate_loss for transition window |

## Reproducing

```bash
# K=5 (new experiment)
python scripts/train.py --config-name=temp_k5_bs128

# K=36 at 30K steps (new experiment)
python scripts/train.py --config-name=temp_k36_bs128_30k

# Gradient norms and candidate eval (for each experiment)
python scripts/compute_gradient_norms.py --experiment <NAME> --every-n 2
python scripts/run_candidate_eval.py --experiment <NAME> --every-n 2

# Compute Landauer cost
python scripts/compute_landauer_cost.py \
  --experiments temp_lr1e3_bs128_k5 temp_lr1e3_bs128 temp_lr1e3_bs128_k20 temp_lr1e3_bs128_k36_30k

# Plot
python scripts/plot_landauer_test.py
```

## Figure

See `outputs/figures/landauer_test.png` for the 2×2 diagnostic figure:
- **Panel A:** Cumulative dissipation Q(t) — most accumulates during the transition
- **Panel B:** Q_transition vs log(K) — linear fit with R²
- **Panel C:** Gradient norm profiles with transition windows shaded
- **Panel D:** Landauer ratio bar chart — approximate constancy for K=5, 10, 20
