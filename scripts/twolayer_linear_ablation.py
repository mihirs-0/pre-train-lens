#!/usr/bin/env python
"""
Two-layer linear network ablation for Ziyin's follow-up experiment.

A two-layer linear network y = W2 @ W1 @ x computes a linear function
but has a non-convex loss landscape with saddle points. Ziyin predicts
this non-convexity alone can produce plateau → cliff dynamics via
eigenvalue dynamics, without any nonlinear feature extraction.

Tests whether the two-layer linear network, on our exact disambiguation
task, shows:
  (a) plateau → cliff dynamics (candidate_loss stuck at log(K) then drops)
  (b) Q ∝ log(K) Landauer dissipation scaling
  (c) directional asymmetry (forward Bz→A vs reverse Az→B)

Encoding:
  B_embs ∈ R^{N_B × D}     — one random embedding per B group
  Z_embs ∈ R^{K × D}       — one random embedding per z selector value
  A_embs ∈ R^{N_B × K × D} — one random embedding per candidate per group

  Forward (Bz→A): input = [b_g; z_j] ∈ R^{2D}, target = a_{g,j} ∈ R^D
  Reverse (Az→B): input = [a_{g,j}; z_j] ∈ R^{2D}, target = b_g ∈ R^D

Priority sweep:
  for K in 10 20 36; do
    for dir in forward reverse; do
      python scripts/twolayer_linear_ablation.py \\
        --K $K --D 128 --direction $dir \\
        --output outputs/twolayer_linear_${dir}_K${K}_H128_results.json
    done
  done

Bottleneck sweep (optional):
  for K in 10 20 36; do
    python scripts/twolayer_linear_ablation.py \\
      --K $K --D 128 --H 64 --direction forward \\
      --output outputs/twolayer_linear_forward_K${K}_H64_results.json
  done
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# Data Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_embeddings(N_B, K, D, seed=42):
    """Generate fixed random embeddings for B groups, z selectors, A candidates."""
    gen = torch.Generator().manual_seed(seed)
    B_embs = torch.randn(N_B, D, generator=gen)
    Z_embs = torch.randn(K, D, generator=gen)
    A_embs = torch.randn(N_B, K, D, generator=gen)
    return B_embs, Z_embs, A_embs


# ═══════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════

class TwoLayerLinear(nn.Module):
    """Two-layer linear network: y = W2(W1(x)), no bias, no nonlinearity."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, output_dim, bias=False)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, x):
        return self.W2(self.W1(x))


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, B_embs, Z_embs, A_embs, direction,
             eval_size=4096, rng_seed=54321):
    """
    Compute evaluation metrics at the current parameters.

    Returns dict with:
      total_mse          — average MSE over eval batch
      candidate_loss     — -log(p_correct) via softmax over distances to K candidates
      candidate_accuracy — fraction where nearest candidate is correct
      z_gap              — candidate_loss(shuffled_z) - candidate_loss(original_z)
    """
    model.eval()
    device = next(model.parameters()).device
    N_B, K, D = A_embs.shape

    rng = torch.Generator().manual_seed(rng_seed)

    # ── Sample eval indices (CPU generator, then move) ──
    g_idx = torch.randint(0, N_B, (eval_size,), generator=rng)
    j_idx = torch.randint(0, K, (eval_size,), generator=rng)

    b_batch = B_embs[g_idx]                  # (E, D)
    z_batch = Z_embs[j_idx]                  # (E, D)
    a_batch = A_embs[g_idx, j_idx]           # (E, D)

    # ── Forward pass (original z) ──
    if direction == "forward":
        x = torch.cat([b_batch, z_batch], dim=1)
        y = a_batch
    else:
        x = torch.cat([a_batch, z_batch], dim=1)
        y = b_batch

    y_hat = model(x)
    total_mse = F.mse_loss(y_hat, y).item()

    # ── Build candidate set ──
    if direction == "forward":
        # K candidates = the K A-embeddings for each group g
        candidates = A_embs[g_idx]            # (E, K, D)
        correct_k = j_idx.to(device)
    else:
        # 1 correct B + (K-1) random distractor B's
        distractors = torch.randint(0, N_B - 1, (eval_size, K - 1), generator=rng)
        distractors[distractors >= g_idx.unsqueeze(1)] += 1
        all_b_idx = torch.cat([g_idx.unsqueeze(1), distractors], dim=1)  # (E, K)
        candidates = B_embs[all_b_idx]       # (E, K, D)
        correct_k = torch.zeros(eval_size, dtype=torch.long, device=device)

    # ── Candidate loss (original z) ──
    # Scale distances by 1/D so the softmax operates on per-element scale.
    # Without scaling, ||y_hat - a_k||² ~ O(D) and the softmax is extremely
    # peaked even for random predictions. With 1/D scaling, the initial
    # candidate_loss ≈ log(K) (approximately uniform over candidates),
    # matching the transformer's token-level candidate_loss semantics.
    diff = y_hat.unsqueeze(1) - candidates   # (E, K, D)
    dist_sq = (diff ** 2).sum(dim=2) / D     # (E, K), scaled by 1/D
    log_probs = F.log_softmax(-dist_sq, dim=1)
    arange = torch.arange(eval_size, device=device)
    candidate_loss = -log_probs[arange, correct_k].mean().item()
    candidate_acc = (log_probs.argmax(dim=1) == correct_k).float().mean().item()

    # ── Shuffled z (z-gap diagnostic) ──
    j_shuffled = torch.randint(0, K - 1, (eval_size,), generator=rng)
    j_shuffled[j_shuffled >= j_idx] += 1     # guarantees j' ≠ j for each example

    z_shuffled = Z_embs[j_shuffled]

    if direction == "forward":
        x_shuf = torch.cat([b_batch, z_shuffled], dim=1)   # B stays, z changes
    else:
        x_shuf = torch.cat([a_batch, z_shuffled], dim=1)   # A stays, z changes

    y_hat_shuf = model(x_shuf)

    diff_s = y_hat_shuf.unsqueeze(1) - candidates
    dist_sq_s = (diff_s ** 2).sum(dim=2) / D  # same 1/D scaling
    log_probs_s = F.log_softmax(-dist_sq_s, dim=1)
    candidate_loss_shuf = -log_probs_s[arange, correct_k].mean().item()

    z_gap = candidate_loss_shuf - candidate_loss

    model.train()
    return {
        "total_mse": total_mse,
        "candidate_loss": candidate_loss,
        "candidate_accuracy": candidate_acc,
        "z_gap": z_gap,
    }


def compute_grad_norm_sq(model, x_fixed, y_fixed):
    """Compute ||∇L||² at current parameters on a fixed batch."""
    model.zero_grad()
    y_hat = model(x_fixed)
    loss = F.mse_loss(y_hat, y_fixed)
    loss.backward()
    grad_norm_sq = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.parameters() if p.grad is not None
    )
    model.zero_grad()
    return grad_norm_sq


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Two-layer linear network ablation (Ziyin follow-up)"
    )
    parser.add_argument("--K", type=int, required=True,
                        help="Number of candidates per B group")
    parser.add_argument("--D", type=int, default=128,
                        help="Embedding dimension")
    parser.add_argument("--H", type=int, default=None,
                        help="Hidden (bottleneck) dim (default: same as D)")
    parser.add_argument("--N_B", type=int, default=1000,
                        help="Number of unique B groups")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (constant)")
    parser.add_argument("--bs", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--steps", type=int, default=30000,
                        help="Training steps")
    parser.add_argument("--direction", type=str, default="forward",
                        choices=["forward", "reverse"],
                        help="forward = Bz→A, reverse = Az→B")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--eval-every", type=int, default=50,
                        help="Evaluate every N steps")
    parser.add_argument("--eval-size", type=int, default=4096,
                        help="Evaluation batch size")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path")
    args = parser.parse_args()

    if args.H is None:
        args.H = args.D

    log_K = math.log(args.K)

    print("=" * 60)
    print("Two-Layer Linear Network: Ziyin Ablation")
    print(f"  K={args.K} (log K={log_K:.4f}), D={args.D}, H={args.H}, "
          f"N_B={args.N_B}")
    print(f"  Direction: {args.direction}")
    print(f"  Optimizer: SGD (lr={args.lr}, no momentum, no weight decay)")
    print(f"  BS={args.bs}, Steps={args.steps}, Eval every={args.eval_every}")
    print(f"  Seed={args.seed}")
    print("=" * 60)

    # ── Seed ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Device ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Generate embeddings (same seed → same data for all runs) ──
    B_embs, Z_embs, A_embs = generate_embeddings(
        args.N_B, args.K, args.D, seed=args.seed
    )
    B_embs = B_embs.to(device)
    Z_embs = Z_embs.to(device)
    A_embs = A_embs.to(device)
    print(f"Embeddings: B{tuple(B_embs.shape)}, Z{tuple(Z_embs.shape)}, "
          f"A{tuple(A_embs.shape)}")
    print(f"Expected initial candidate_loss ≈ log({args.K}) = {log_K:.4f}")

    # ── Model ──
    model = TwoLayerLinear(2 * args.D, args.H, args.D).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  "
          f"(W1: {args.H}×{2*args.D} = {args.H*2*args.D:,}, "
          f"W2: {args.D}×{args.H} = {args.D*args.H:,})")

    # ── Pure SGD (no momentum, no weight decay) ──
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # ── Pre-compute fixed batch for gradient norm evaluation ──
    grad_rng = torch.Generator().manual_seed(args.seed + 777)
    g_grad = torch.randint(0, args.N_B, (args.bs,), generator=grad_rng)
    j_grad = torch.randint(0, args.K, (args.bs,), generator=grad_rng)
    b_grad = B_embs[g_grad]
    z_grad = Z_embs[j_grad]
    a_grad = A_embs[g_grad, j_grad]
    if args.direction == "forward":
        x_grad_fixed = torch.cat([b_grad, z_grad], dim=1)
        y_grad_fixed = a_grad
    else:
        x_grad_fixed = torch.cat([a_grad, z_grad], dim=1)
        y_grad_fixed = b_grad

    # ── Training RNG ──
    train_rng = torch.Generator().manual_seed(args.seed + 1)

    # ── Results storage ──
    results = {
        "config": {
            "K": args.K, "D": args.D, "H": args.H, "N_B": args.N_B,
            "lr": args.lr, "bs": args.bs, "steps": args.steps,
            "direction": args.direction, "seed": args.seed,
            "n_params": n_params, "log_K": round(log_K, 6),
        },
        "steps": [],
        "total_mse": [],
        "candidate_loss": [],
        "candidate_accuracy": [],
        "z_gap": [],
        "gradient_norm_sq": [],
    }

    # ── Helpers ──
    eval_rng_seed = args.seed + 10000  # fixed eval seed for smooth curves

    def record_eval(step):
        em = evaluate(model, B_embs, Z_embs, A_embs,
                      args.direction, args.eval_size, eval_rng_seed)
        gnsq = compute_grad_norm_sq(model, x_grad_fixed, y_grad_fixed)
        results["steps"].append(step)
        results["total_mse"].append(em["total_mse"])
        results["candidate_loss"].append(em["candidate_loss"])
        results["candidate_accuracy"].append(em["candidate_accuracy"])
        results["z_gap"].append(em["z_gap"])
        results["gradient_norm_sq"].append(gnsq)
        return em, gnsq

    # ── Initial evaluation (step 0) ──
    em0, gn0 = record_eval(0)
    print(f"Step     0 | MSE={em0['total_mse']:.6f} | "
          f"CandLoss={em0['candidate_loss']:.4f} | "
          f"Acc={em0['candidate_accuracy']:.4f} | "
          f"z_gap={em0['z_gap']:.4f} | ||∇L||²={gn0:.6f}")

    # ── Training loop ──
    start_time = time.time()
    model.train()

    for step in range(1, args.steps + 1):
        # Sample training batch
        g = torch.randint(0, args.N_B, (args.bs,), generator=train_rng)
        j = torch.randint(0, args.K, (args.bs,), generator=train_rng)

        b = B_embs[g]
        z = Z_embs[j]
        a = A_embs[g, j]

        if args.direction == "forward":
            x = torch.cat([b, z], dim=1)
            y = a
        else:
            x = torch.cat([a, z], dim=1)
            y = b

        # Forward + backward + step
        optimizer.zero_grad()
        y_hat = model(x)
        loss = F.mse_loss(y_hat, y)
        loss.backward()
        optimizer.step()

        # Evaluate periodically
        if step % args.eval_every == 0:
            em, gnsq = record_eval(step)

            if step % 2000 == 0 or step == args.eval_every:
                elapsed = time.time() - start_time
                print(
                    f"Step {step:>5d} | MSE={em['total_mse']:.6f} | "
                    f"CandLoss={em['candidate_loss']:.4f} | "
                    f"Acc={em['candidate_accuracy']:.4f} | "
                    f"z_gap={em['z_gap']:.4f} | "
                    f"||∇L||²={gnsq:.6f} | {elapsed:.1f}s"
                )

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")

    # ── Final summary ──
    final_cl = results["candidate_loss"][-1]
    final_acc = results["candidate_accuracy"][-1]
    final_zgap = results["z_gap"][-1]
    print(f"Final: candidate_loss={final_cl:.4f}, "
          f"accuracy={final_acc:.4f}, z_gap={final_zgap:.4f}")

    if final_cl < 0.5 * log_K:
        print("  → Model learned the mapping (candidate_loss < 0.5 × log K)")
    else:
        print(f"  → Model did NOT converge (candidate_loss = {final_cl:.4f} "
              f"vs log K = {log_K:.4f})")

    # ── Save ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
