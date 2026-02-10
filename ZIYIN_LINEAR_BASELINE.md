## Linear Baseline vs Transformer Plateau (Ziyin Contrast)

This document records the linear baselines and contrast figure added to
address the reviewer hypothesis that the transformer plateau is just an
ill-conditioning artifact.

### Goal
- Show ill-conditioned linear regression yields smooth, exponential-like decay
  from step 0 (no plateau).
- Show the transformer exhibits a flat plateau near `log(K)` followed by a sharp
  transition, indicating a qualitatively different learning dynamic.

### Added Scripts
- `scripts/linear_baseline_illconditioned.py`
  - Ill-conditioned inverse regression task (learn `M^{-1}` from `y = M x`).
  - Logs full MSE every step and per-component error every 100 steps.
  - Per-component error is computed in the eigenbasis of `E[yy^T]` using
    row norms of `V^T (W - M^{-1})`.
- `scripts/linear_baseline_planted.py`
  - Linear "selector usage" analogue to candidate disambiguation.
  - Logs `selector_gap = mse_shuffled - mse_clean` every 100 steps.
- `scripts/plot_ziyin_contrast.py`
  - 1x3 figure comparing loss shape, linear component dynamics, and selector use.
- `scripts/plot_probe_candidate_overlay.py`
  - Overlays candidate loss with probe metrics (attention-to-z + logit lens).
- `scripts/analyze.py`
  - Now supports `--every-n` to analyze every N-th checkpoint for probes.

### Key Design Choices
- Ill-conditioned baseline uses `D=100`, `batch_size=128`, `steps=20000`.
- Condition number set to 100 via `s_min=0.01` (default `s_max=1.0`).
- For Panel B visibility, per-component curves are normalized to their
  initial values and plotted on log scale in `[1e-3, 1.0]`.
- Panel C uses raw gaps with dual y-axes to avoid normalization artifacts.
- Probe overlay uses mean attention-to-z (averaged over layers/heads) and
  logit-lens correct probability at the final layer.

### Execution (recommended)
```
python scripts/linear_baseline_illconditioned.py --s-min 0.01 --lr 20
python scripts/linear_baseline_planted.py
python scripts/linear_baseline_planted.py --batch-size 32 --output-path outputs/linear_baseline/planted_bs32_results.json
python scripts/linear_baseline_planted.py --batch-size 64 --output-path outputs/linear_baseline/planted_bs64_results.json
python scripts/plot_ziyin_contrast.py --transformer-run temp_lr1e3_bs128_k20 \
  --planted-results outputs/linear_baseline/planted_bs32_results.json \
  outputs/linear_baseline/planted_bs64_results.json
python scripts/analyze.py --experiment temp_lr1e3_bs128_k20 \
  --probes attention_to_z logit_lens --batch-size 64 --every-n 10
python scripts/plot_probe_candidate_overlay.py --experiment temp_lr1e3_bs128_k20
```

Notes on `--lr 20` for the ill-conditioned baseline:
- PyTorch `MSELoss` uses mean reduction over `batch_size * D`.
- This makes the effective per-component step size smaller by ~`1/D`.
- `lr=20` yields an effective step size of ~`0.2` when `D=100`, so the slow
  component decays within 20k steps. If you prefer smaller `lr`, increase
  `steps` accordingly.

### Outputs
- `outputs/linear_baseline/illconditioned_results.json`
- `outputs/linear_baseline/planted_results.json`
- `outputs/linear_baseline/planted_bs32_results.json`
- `outputs/linear_baseline/planted_bs64_results.json`
- `outputs/linear_baseline/figures/ziyin_contrast.png`
- `outputs/temp_lr1e3_bs128_k20/probe_results/all_probes.json`
- `outputs/temp_lr1e3_bs128_k20/figures/probe_candidate_overlay.png`

### Plateau Magnitude (avoid "zero" claims)
- In `temp_lr1e3_bs128_k20`, `z_gap` stays below **0.025 nats** through step 2100.
- Final `z_gap` is **12.336 nats** → **~494×** larger than the plateau maximum.
- Early drift exists (e.g., ~0.006 at step 100 to ~0.025 by step 2100), but the
  scale separation is the argument, not exact zero.

### Interpretation Guide
- **Panel A (Loss Shape Comparison)**: Linear MSE decays smoothly from step 0,
  while transformer candidate loss shows a flat plateau then a cliff.
- **Panel B (Per-Component Dynamics)**: All linear components decay from step 0
  at different rates (fast, mid, slow). No component is truly flat.
- **Panel C (Selector Usage)**: Linear selector gap is nonzero immediately and
  decays as the model converges; transformer `z_gap` stays near 0 for thousands
  of steps, then rises sharply.
- **Probe overlay**: attention-to-z and logit-lens signals are flat/low during
  the plateau and spike near the same transition, providing independent support
  against the “invisible gradual change” objection.
