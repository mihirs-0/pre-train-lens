"""
Single-condition PPT runner for debugging.

Usage:
    python scripts/run_ppt_single.py --condition C0 --seed 0
    python scripts/run_ppt_single.py --condition C2 --seed 0 --device cpu --max-steps 200
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_ppt_experiment import load_config, detect_device, run_single


def main():
    parser = argparse.ArgumentParser(description="Run a single PPT condition/seed")
    parser.add_argument("--condition", required=True, choices=["C0", "C1", "C2"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", default="configs/ppt/phase1.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps (for quick debugging)",
    )
    parser.add_argument(
        "--ppt-steps",
        type=int,
        default=None,
        help="Override PPT steps (for quick debugging)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device if args.device != "auto" else detect_device()

    # Apply debug overrides
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
        cfg["training"]["eval_every"] = min(
            cfg["training"]["eval_every"], args.max_steps // 4
        )
        cfg["training"]["checkpoint_every"] = args.max_steps  # one at end
        cfg["eval"]["candidate_eval_every"] = max(
            1, args.max_steps // 4
        )

    if args.ppt_steps is not None:
        cfg["ppt"]["n_steps"] = args.ppt_steps
        cfg["ppt"]["log_every"] = max(1, args.ppt_steps // 5)

    result = run_single(args.condition, args.seed, cfg, device)

    print(f"\n--- Result Summary ---")
    print(f"  Condition: {result['condition']}")
    print(f"  Seed: {result['seed']}")
    print(f"  τ_z (shuffle): {result['tau_z_shuffle']}")
    print(f"  τ_z (candidate): {result['tau_z_candidate']}")
    print(f"  Final loss: {result['final_loss']}")
    print(f"  Final z_gap: {result['final_z_gap']}")
    print(f"  PPT converged: {result['ppt_converged']}")
    if result["transfer_stats"]:
        ts = result["transfer_stats"]
        print(f"  Transfer: {ts['transferred']} transferred, {ts['skipped']} skipped")
    print(f"  Time: {result['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
