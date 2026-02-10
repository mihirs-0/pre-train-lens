#!/usr/bin/env python
"""
Compute gradient norms at each checkpoint for entropic barrier analysis.

For each checkpoint, loads the model, computes the training loss on a fixed
set of batches, backpropagates, and records ||∇L|| and ||∇L||².

These measurements test Ziyin's effective free energy prediction:
  F = L + η·S,  where S = ||∇L||²

If the plateau is stabilised by the entropic term, we expect:
  Phase 1 (Plateau):    ||∇L||² LOW  (competing gradients cancel)
  Phase 2 (Transition): ||∇L||² HIGH (symmetry broken, coherent gradients)
  Phase 3 (Converged):  ||∇L||² LOW  (at loss minimum)

Usage:
    python scripts/compute_gradient_norms.py --experiment temp_lr1e3_bs128_k20 --every-n 2
"""

import sys
from pathlib import Path
import argparse
import json
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config, create_datasets_from_config, collate_fn
from src.model import create_model_from_config
from src.training.trainer import compute_loss
from src.training.checkpoint import list_checkpoints, load_checkpoint
import yaml


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def _to_namespace(obj):
    """Recursively convert dicts to SimpleNamespace (matches run_candidate_eval.py)."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _select_steps(all_steps, every_n: int):
    """Subsample checkpoints, always keeping first and last."""
    if every_n <= 1:
        return list(all_steps)
    selected = {all_steps[0], all_steps[-1]}
    for idx, step in enumerate(all_steps):
        if idx % every_n == 0:
            selected.add(step)
    return [step for step in all_steps if step in selected]


def _bucket_param(name: str) -> str:
    """Bucket a parameter name into a component category."""
    name_lower = name.lower()
    if "unembed" in name_lower:
        return "unembedding"
    if "embed" in name_lower or "pos_embed" in name_lower:
        return "embedding"
    if "attn" in name_lower:
        return "attention"
    if "mlp" in name_lower:
        return "mlp"
    if "ln" in name_lower or "norm" in name_lower:
        return "layernorm"
    return "other"


def compute_gradient_norms_at_checkpoint(
    model,
    data_loader,
    n_batches: int,
    device: str,
):
    """
    Compute gradient norms averaged over n_batches.

    Returns dict with total_grad_norm, total_grad_norm_sq, component norms,
    and the loss value.
    """
    model.train()

    total_norms = []
    total_norms_sq = []
    component_norms_accum = {}
    losses = []
    first_target_losses = []

    batch_iter = iter(data_loader)
    for _ in range(n_batches):
        try:
            batch = next(batch_iter)
        except StopIteration:
            # Wrap around if dataset is smaller than n_batches
            batch_iter = iter(data_loader)
            batch = next(batch_iter)

        # Move batch to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        model.zero_grad()
        loss, accuracy, first_target_loss = compute_loss(model, batch)
        loss.backward()

        # Compute total gradient norm
        total_norm_sq = 0.0
        component_norms_batch = {}

        for name, p in model.named_parameters():
            if p.grad is not None:
                param_norm_sq = p.grad.data.norm(2).item() ** 2
                total_norm_sq += param_norm_sq

                bucket = _bucket_param(name)
                if bucket not in component_norms_batch:
                    component_norms_batch[bucket] = 0.0
                component_norms_batch[bucket] += param_norm_sq

        total_norm = total_norm_sq ** 0.5

        total_norms.append(total_norm)
        total_norms_sq.append(total_norm_sq)
        losses.append(loss.item())
        first_target_losses.append(first_target_loss)

        # Accumulate component norms
        for bucket, norm_sq in component_norms_batch.items():
            if bucket not in component_norms_accum:
                component_norms_accum[bucket] = []
            component_norms_accum[bucket].append(norm_sq ** 0.5)

    # Average across batches
    avg_total_norm = sum(total_norms) / len(total_norms)
    avg_total_norm_sq = sum(total_norms_sq) / len(total_norms_sq)
    avg_loss = sum(losses) / len(losses)
    avg_first_target_loss = sum(first_target_losses) / len(first_target_losses)

    avg_component_norms = {}
    for bucket, norms in component_norms_accum.items():
        avg_component_norms[bucket] = sum(norms) / len(norms)

    return {
        "total_grad_norm": avg_total_norm,
        "total_grad_norm_sq": avg_total_norm_sq,
        "loss": avg_loss,
        "first_target_loss": avg_first_target_loss,
        "component_grad_norms": avg_component_norms,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute gradient norms at checkpoints for entropic barrier analysis"
    )
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base output directory")
    parser.add_argument("--n-batches", type=int, default=4, help="Batches to average over")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for gradient computation")
    parser.add_argument("--every-n", type=int, default=1, help="Evaluate every N-th checkpoint")
    args = parser.parse_args()

    experiment_dir = Path(args.output_dir) / args.experiment
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = _to_namespace(yaml.safe_load(f))

    device = _select_device()
    print(f"Using device: {device}")

    # Reproducibility
    torch.manual_seed(cfg.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.experiment.seed)

    # Create tokenizer and dataset
    tokenizer = create_tokenizer_from_config(cfg)
    train_dataset, probe_dataset, mapping_data = create_datasets_from_config(cfg, tokenizer)
    if len(probe_dataset) == 0:
        probe_dataset = train_dataset

    data_loader = DataLoader(
        probe_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # List and subsample checkpoints
    checkpoint_dir = experiment_dir / "checkpoints"
    all_steps = list_checkpoints(checkpoint_dir)
    if not all_steps:
        print(f"Error: No checkpoints found in {checkpoint_dir}")
        sys.exit(1)

    steps = _select_steps(all_steps, args.every_n)
    print(f"Computing gradient norms for {len(steps)} checkpoints")
    print(f"  First: {steps[0]}, Last: {steps[-1]}")
    print(f"  Averaging over {args.n_batches} batches of size {args.batch_size}")

    # Results storage
    results = {
        "steps": [],
        "total_grad_norm": [],
        "total_grad_norm_sq": [],
        "loss_at_checkpoint": [],
        "first_target_loss_at_checkpoint": [],
        "component_grad_norms": {},
    }

    for i, step in enumerate(steps):
        print(f"  [{i + 1}/{len(steps)}] Step {step}...", end=" ", flush=True)

        # Create fresh model and load weights
        model = create_model_from_config(cfg, tokenizer)
        load_checkpoint(model, None, checkpoint_dir, step)
        model.to(device)

        # Compute gradient norms
        norms = compute_gradient_norms_at_checkpoint(
            model, data_loader, args.n_batches, device
        )

        results["steps"].append(step)
        results["total_grad_norm"].append(norms["total_grad_norm"])
        results["total_grad_norm_sq"].append(norms["total_grad_norm_sq"])
        results["loss_at_checkpoint"].append(norms["loss"])
        results["first_target_loss_at_checkpoint"].append(norms["first_target_loss"])

        # Component norms (initialise lists on first checkpoint)
        for bucket, norm_val in norms["component_grad_norms"].items():
            if bucket not in results["component_grad_norms"]:
                results["component_grad_norms"][bucket] = []
            results["component_grad_norms"][bucket].append(norm_val)

        print(
            f"||∇L||={norms['total_grad_norm']:.4f}  "
            f"||∇L||²={norms['total_grad_norm_sq']:.4f}  "
            f"loss={norms['loss']:.4f}"
        )

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if device == "mps":
            torch.mps.empty_cache()

    # Add metadata
    results["n_batches_averaged"] = args.n_batches
    results["batch_size"] = args.batch_size
    results["config"] = {
        "learning_rate": float(cfg.training.learning_rate),
        "batch_size": int(cfg.training.batch_size),
        "k": int(cfg.data.k),
        "experiment_name": str(cfg.experiment.name),
    }

    # Save
    results_path = experiment_dir / "gradient_norm_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved gradient norm results to: {results_path}")
    print(f"  Steps evaluated: {len(results['steps'])}")
    print(f"  Component buckets: {list(results['component_grad_norms'].keys())}")


if __name__ == "__main__":
    main()
