# Pre-Pre-Training Phase 1 Results

**Date:** 2026-04-05  
**Result:** NO-GO — Shuffle-Dyck PPT does not accelerate disambiguation learning at K=10.

## Hypothesis

Pre-pre-training a transformer on k-Shuffle Dyck sequences (a hierarchical formal language) transfers internal representations that accelerate learning of a downstream disambiguation task, measured by earlier onset of z-usage (τ_z).

This is motivated by Hu et al. (ACL 2025), who found that pre-training on Shuffle-Dyck transfers effectively to natural language. We test whether this extends to a controlled disambiguation setting.

## Experimental Setup

**Conditions:**
- **C0 (baseline):** Random initialization, no pre-pre-training
- **C2 (Shuffle-Dyck):** 5,000 steps of next-token prediction on 5-Shuffle Dyck sequences (vocab=10, max depth=8), then transfer attention + MLP weights to the target model

**Target task:** `bz_to_a` disambiguation with K=10 candidate mappings, 1000 unique base strings, trained for 20,000 steps.

**Model:** 4-layer HookedTransformer (d_model=128, d_head=32, d_mlp=512, 4 heads). Architecture identical between PPT and target models; only vocab size differs (PPT: 10, target: 40).

**Weight transfer:** All attention, MLP, and LayerNorm weights transferred. Token embeddings, positional embeddings, and unembedding re-initialized (different vocab size / context length).

**Seeds:** 3 per condition (seeds 0, 1, 2). Quick validation subset of the planned 6-seed Phase 1.

**τ_z detection:**
- *Z-shuffle*: First step where z_gap (loss_z_shuffled − loss_clean) exceeds 0.1 nats for 3 consecutive evaluations (eval every 50 steps)
- *Candidate loss*: First step where candidate-normalized loss drops below 0.8 × log(K)

**Device:** Apple M4 (MPS), 4 parallel workers.

## Results

| Condition | Seed | τ_z (shuffle) | τ_z (candidate) | PPT converged |
|-----------|------|---------------|-----------------|---------------|
| C0        | 0    | 1200          | 1500            | n/a           |
| C0        | 1    | 1200          | 1500            | n/a           |
| C0        | 2    | 1150          | 1500            | n/a           |
| C2        | 0    | 1200          | 1500            | No            |
| C2        | 1    | 1150          | 1500            | No            |
| C2        | 2    | 1300          | 1500            | No            |

**Bootstrap 95% CIs:**
- C0 mean τ_z (shuffle): **1183** [1150, 1200]
- C2 mean τ_z (shuffle): **1217** [1150, 1300]
- C0 − C2 difference: **−33** [−117, +33]
- Candidate-loss τ_z: **1500** for all 6 runs (no variance)

## Interpretation

**The hypothesis is not supported.** Shuffle-Dyck PPT provides no measurable acceleration of z-learning. The 95% CI for the τ_z difference spans zero and trends slightly negative (C2 marginally slower). Z-gap trajectories are nearly identical across conditions at every evaluation checkpoint.

### Confounds and caveats

1. **PPT did not converge.** All 3 C2 runs flagged `converged=False` — the PPT loss did not drop below 50% of its initial value in 5,000 steps. The model may not have learned meaningful Shuffle-Dyck structure, making the transferred weights effectively a different random initialization rather than a structured one. This means the hypothesis was not fully tested.

2. **τ_z detection granularity.** Evaluations occur every 50 steps, and τ_z requires 3 consecutive crossings. Any real effect smaller than ~150 steps would be invisible.

3. **Small seed count.** 3 seeds per condition limits statistical power. However, the complete absence of directional signal (and zero variance in candidate-loss τ_z) suggests more seeds would not change the conclusion.

4. **No C1 (Markov) control.** We skipped the generic warm-start control in this quick validation. Without it, we cannot distinguish "PPT structure doesn't help" from "PPT of any kind doesn't help."

## What would change the conclusion

- **Longer PPT** (20K+ steps) ensuring actual convergence on Shuffle-Dyck, then re-running
- **Larger K** (e.g., K=25 or K=36) where the disambiguation task is harder and τ_z is later, giving more room for acceleration
- **Ablation on transfer mode** (attn_only vs mlp_only vs full) to test whether specific components carry the structural prior

## Files

- Results: `outputs/ppt_phase1/ppt_{C0,C2}_seed{0,1,2}_k10/result.json`
- Figures: `outputs/ppt_phase1/{z_gap_curves,candidate_loss_curves,tau_z_distributions,ppt_loss_curves}.png`
- Analysis script: `scripts/analyze_ppt.py`
- Runner: `scripts/run_ppt_experiment.py`
