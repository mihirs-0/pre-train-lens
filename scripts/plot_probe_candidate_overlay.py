#!/usr/bin/env python
"""
Overlay probe metrics with candidate loss for a single experiment.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 2,
        "font.size": 10,
    })


def _extract_attention_mean(results: Dict) -> Tuple[List[int], List[float]]:
    steps = results["steps"]
    attn_results = results["probe_results"]["attention_to_z"]
    values = []
    for step in steps:
        attn = np.array(attn_results[str(step)]["attention_to_z"])
        values.append(float(attn.mean()))
    return steps, values


def _extract_logit_lens_final_prob(results: Dict) -> Tuple[List[int], List[float]]:
    steps = results["steps"]
    ll_results = results["probe_results"]["logit_lens"]
    values = []
    for step in steps:
        probs = np.array(ll_results[str(step)]["correct_prob_by_layer"])
        values.append(float(probs[-1]))
    return steps, values


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay probes with candidate loss")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name in outputs/")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base output directory")
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional explicit output path for the figure",
    )
    args = parser.parse_args()

    _apply_style()
    output_dir = Path(args.output_dir)
    experiment_dir = output_dir / args.experiment

    candidate_path = experiment_dir / "candidate_eval_results.json"
    probe_path = experiment_dir / "probe_results" / "all_probes.json"

    if not candidate_path.exists():
        raise FileNotFoundError(f"Missing candidate results: {candidate_path}")
    if not probe_path.exists():
        raise FileNotFoundError(f"Missing probe results: {probe_path}")

    candidate = _load_json(candidate_path)
    probes = _load_json(probe_path)

    cand_steps = candidate["steps"]
    cand_loss = candidate["candidate_loss"]
    log_k = candidate.get("log_k")
    onset_step = candidate.get("binding_onset_step")

    attn_steps, attn_mean = _extract_attention_mean(probes)
    ll_steps, ll_final = _extract_logit_lens_final_prob(probes)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    cand_line = ax.plot(cand_steps, cand_loss, color="red", label="candidate_loss")[0]
    if log_k is not None:
        ax.axhline(log_k, color="red", linestyle="--", alpha=0.4, label=f"log(K)={log_k:.2f}")
    if onset_step is not None:
        ax.axvline(onset_step, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Candidate Loss (nats)", color="red")
    ax.tick_params(axis="y", labelcolor="red")

    ax_right = ax.twinx()
    attn_line = ax_right.plot(attn_steps, attn_mean, color="green", label="attention_to_z (mean)")[0]
    ll_line = ax_right.plot(ll_steps, ll_final, color="purple", label="logit_lens P(correct) (final layer)")[0]
    ax_right.set_ylabel("Probe Metrics (attention / probability)", color="black")

    handles = [cand_line, attn_line, ll_line]
    labels = ["candidate_loss", "attention_to_z (mean)", "logit_lens P(correct)"]
    ax.legend(handles, labels, fontsize=8, loc="upper right")

    fig.suptitle(f"{args.experiment}: Candidate Loss vs Probe Metrics", fontsize=12)
    fig.tight_layout()

    if args.output_path:
        save_path = Path(args.output_path)
    else:
        figures_dir = experiment_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        save_path = figures_dir / "probe_candidate_overlay.png"

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
