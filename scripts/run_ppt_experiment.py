"""
Phase 1 Runner: C0 × C1 × C2 × 6 seeds at K=10.

Usage:
    # Full Phase 1
    python scripts/run_ppt_experiment.py

    # Single run (debug)
    python scripts/run_ppt_experiment.py --condition C2 --seed 0

    # Custom device
    python scripts/run_ppt_experiment.py --device mps
"""

import argparse
import json
import math
import time
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tokenizer import CharTokenizer
from src.data.dataset import generate_mappings, DisambiguationDataset, collate_fn
from src.model.hooked_transformer import create_hooked_transformer
from src.training.trainer import compute_loss, shuffle_z_in_batch
from src.training.checkpoint import save_checkpoint

from src.ppt.generators import MarkovBigramGenerator, ShuffleDyckGenerator
from src.ppt.ppt_trainer import pre_pre_train
from src.ppt.transfer import transfer_weights

from src.analysis.candidate_eval import run_candidate_eval


def load_config(config_path: str = "configs/ppt/phase1.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def detect_tau_z_from_trace(
    steps: list,
    z_gaps: list,
    threshold: float = 0.1,
    consecutive: int = 3,
) -> int | None:
    """
    Detect τ_z: first step where z_gap > threshold for `consecutive` evals.

    z_gap = loss_z_shuffled - loss_clean.
    When z_gap > threshold, the model is using z.
    """
    count = 0
    start_idx = None
    for i, gap in enumerate(z_gaps):
        if gap > threshold:
            if count == 0:
                start_idx = i
            count += 1
            if count >= consecutive and start_idx is not None:
                return steps[start_idx]
        else:
            count = 0
            start_idx = None
    return None


def detect_tau_z_from_candidate_loss(
    steps: list,
    candidate_losses: list,
    k: int,
    drop_fraction: float = 0.2,
    consecutive: int = 2,
) -> int | None:
    """
    Detect τ_z from candidate loss: first step where candidate loss drops
    below (1 - drop_fraction) * log(K).

    Candidate loss floor is log(K) when model ignores z.
    """
    log_k = math.log(k)
    threshold = log_k * (1 - drop_fraction)

    count = 0
    start_idx = None
    for i, cl in enumerate(candidate_losses):
        if cl < threshold:
            if count == 0:
                start_idx = i
            count += 1
            if count >= consecutive and start_idx is not None:
                return steps[start_idx]
        else:
            count = 0
            start_idx = None
    return None


def run_single(condition: str, seed: int, cfg: dict, device: str) -> dict:
    """Run one experiment: one condition, one seed."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"  {condition} | seed={seed} | K={cfg['data']['k']} | device={device}")
    print(f"{'='*60}")

    mcfg = cfg["model"]
    pcfg = cfg["ppt"]
    dcfg = cfg["data"]
    tcfg = cfg["training"]
    ecfg = cfg["eval"]

    # ── 1. Setup tokenizer and data ──
    tokenizer = CharTokenizer(vocab_chars=dcfg["vocab_chars"])

    mapping_data = generate_mappings(
        n_unique_b=dcfg["n_unique_b"],
        k=dcfg["k"],
        b_length=dcfg["b_length"],
        a_length=dcfg["a_length"],
        z_length=dcfg["z_length"],
        vocab_chars=dcfg["vocab_chars"],
        seed=seed,
        task=dcfg["task"],
    )

    train_dataset = DisambiguationDataset(
        mapping_data=mapping_data,
        tokenizer=tokenizer,
        split="train",
        probe_fraction=0.0,
        seed=seed,
        task=dcfg["task"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    # ── 2. Pre-pre-training (if applicable) ──
    ppt_model = None
    ppt_loss_curve = []

    if condition == "C1":
        print("  [PPT] Markov Bigram...")
        gen = MarkovBigramGenerator(vocab_size=pcfg["markov_vocab"], seed=seed)
        ppt_model, ppt_loss_curve = pre_pre_train(
            generator=gen,
            n_steps=pcfg["n_steps"],
            batch_size=pcfg["batch_size"],
            seq_len=pcfg["seq_len"],
            lr=pcfg["lr"],
            weight_decay=pcfg["weight_decay"],
            n_layers=mcfg["n_layers"],
            n_heads=mcfg["n_heads"],
            d_model=mcfg["d_model"],
            d_head=mcfg["d_head"],
            d_mlp=mcfg["d_mlp"],
            device=device,
            seed=seed,
            log_every=pcfg["log_every"],
        )

    elif condition == "C2":
        print("  [PPT] Shuffle-Dyck...")
        gen = ShuffleDyckGenerator(k=pcfg["dyck_k"], max_depth=pcfg["dyck_max_depth"])
        ppt_model, ppt_loss_curve = pre_pre_train(
            generator=gen,
            n_steps=pcfg["n_steps"],
            batch_size=pcfg["batch_size"],
            seq_len=pcfg["seq_len"],
            lr=pcfg["lr"],
            weight_decay=pcfg["weight_decay"],
            n_layers=mcfg["n_layers"],
            n_heads=mcfg["n_heads"],
            d_model=mcfg["d_model"],
            d_head=mcfg["d_head"],
            d_mlp=mcfg["d_mlp"],
            device=device,
            seed=seed,
            log_every=pcfg["log_every"],
        )

    # ── 3. Create target model and transfer ──
    target_model = create_hooked_transformer(
        tokenizer=tokenizer,
        n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"],
        d_model=mcfg["d_model"],
        d_head=mcfg["d_head"],
        d_mlp=mcfg["d_mlp"],
        device=device,
    )

    transfer_stats = None
    if ppt_model is not None:
        transfer_stats = transfer_weights(ppt_model, target_model, mode="full")
        del ppt_model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── 4. Target task training with z-shuffle logging ──
    optimizer = torch.optim.AdamW(
        target_model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )

    # Output directory
    exp_name = f"ppt_{condition}_seed{seed}_k{dcfg['k']}"
    output_dir = Path("outputs") / "ppt_phase1" / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Traces
    loss_trace = []
    first_target_loss_trace = []
    z_shuffle_loss_trace = []  # This IS the τ_z signal
    z_gap_trace = []  # loss_shuffled - loss_clean
    steps_trace = []
    candidate_loss_trace = []
    candidate_acc_trace = []
    candidate_steps = []

    target_model.train()
    step = 0
    epoch = 0
    start_time = time.time()

    pbar = tqdm(total=tcfg["max_steps"], desc=f"{condition}/s{seed}")

    while step < tcfg["max_steps"]:
        epoch += 1
        for batch in train_loader:
            if step >= tcfg["max_steps"]:
                break

            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            optimizer.zero_grad()
            loss, acc, first_loss = compute_loss(target_model, batch_dev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(target_model.parameters(), 1.0)
            optimizer.step()
            step += 1

            # ── Evaluation checkpoint ──
            if step % tcfg["eval_every"] == 0:
                target_model.eval()
                with torch.no_grad():
                    # Z-shuffle diagnostic (REUSE existing infrastructure)
                    shuffled_batch = shuffle_z_in_batch(batch_dev)
                    _, _, shuffled_first_loss = compute_loss(
                        target_model, shuffled_batch
                    )
                    z_gap = shuffled_first_loss - first_loss

                loss_trace.append(loss.item())
                first_target_loss_trace.append(first_loss)
                z_shuffle_loss_trace.append(shuffled_first_loss)
                z_gap_trace.append(z_gap)
                steps_trace.append(step)

                if step % 500 == 0:
                    print(
                        f"    step {step:>6d} | loss={loss.item():.4f} | "
                        f"first={first_loss:.4f} | z_shuf={shuffled_first_loss:.4f} | "
                        f"Δz={z_gap:.4f}"
                    )

                target_model.train()

            # ── Candidate eval (less frequent — more expensive) ──
            if step % ecfg["candidate_eval_every"] == 0:
                target_model.eval()
                cand_result = run_candidate_eval(
                    model=target_model,
                    tokenizer=tokenizer,
                    mapping_data=mapping_data,
                    n_examples=ecfg["candidate_eval_n_examples"],
                    task=dcfg["task"],
                    device=device,
                    seed=seed,
                )
                candidate_loss_trace.append(cand_result["candidate_loss"])
                candidate_acc_trace.append(cand_result["candidate_accuracy"])
                candidate_steps.append(step)
                target_model.train()

            # ── Save checkpoint ──
            if step % tcfg["checkpoint_every"] == 0:
                save_checkpoint(
                    model=target_model,
                    optimizer=optimizer,
                    step=step,
                    train_loss=loss.item(),
                    train_accuracy=acc,
                    checkpoint_dir=checkpoint_dir,
                )

            pbar.update(1)

    pbar.close()
    elapsed = time.time() - start_time

    # ── 5. Detect τ_z ──
    tau_z = detect_tau_z_from_trace(
        steps=steps_trace,
        z_gaps=z_gap_trace,
        threshold=ecfg["z_shuffle_threshold"],
    )

    tau_z_candidate = detect_tau_z_from_candidate_loss(
        steps=candidate_steps,
        candidate_losses=candidate_loss_trace,
        k=dcfg["k"],
    )

    # ── 6. Save results ──
    result = {
        "condition": condition,
        "seed": seed,
        "K": dcfg["k"],
        "tau_z_shuffle": tau_z,
        "tau_z_candidate": tau_z_candidate,
        "final_loss": float(np.mean(loss_trace[-10:])) if loss_trace else None,
        "final_z_gap": float(np.mean(z_gap_trace[-10:])) if z_gap_trace else None,
        "elapsed_seconds": elapsed,
        "ppt_converged": len(ppt_loss_curve) == 0
        or (
            len(ppt_loss_curve) > 100
            and np.mean(ppt_loss_curve[-100:]) < np.mean(ppt_loss_curve[:100]) * 0.5
        ),
        "transfer_stats": transfer_stats,
        # Traces
        "steps": steps_trace,
        "loss_trace": loss_trace,
        "first_target_loss_trace": first_target_loss_trace,
        "z_shuffle_loss_trace": z_shuffle_loss_trace,
        "z_gap_trace": z_gap_trace,
        "candidate_steps": candidate_steps,
        "candidate_loss_trace": candidate_loss_trace,
        "candidate_acc_trace": candidate_acc_trace,
        "ppt_loss_curve": ppt_loss_curve,
    }

    result_path = output_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(
        f"  [DONE] τ_z(shuffle)={tau_z} | τ_z(candidate)={tau_z_candidate} | "
        f"time={elapsed:.0f}s"
    )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppt/phase1.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--condition", default=None, help="Single condition (debug)")
    parser.add_argument("--seed", type=int, default=None, help="Single seed (debug)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device if args.device != "auto" else detect_device()

    # Single run mode
    if args.condition and args.seed is not None:
        result = run_single(args.condition, args.seed, cfg, device)
        return

    # Full Phase 1
    all_results = []
    total_start = time.time()

    for condition in cfg["experiment"]["conditions"]:
        for seed in cfg["experiment"]["seeds"]:
            result = run_single(condition, seed, cfg, device)
            all_results.append(result)

    total_time = time.time() - total_start
    print(f"\nTotal Phase 1 time: {total_time/3600:.1f} hours")

    # Save combined results
    combined_path = Path("outputs/ppt_phase1/all_results.json")
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    from scripts.analyze_ppt import print_summary

    print_summary(all_results)


if __name__ == "__main__":
    main()
