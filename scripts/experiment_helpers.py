#!/usr/bin/env python
"""Shared utilities for overnight experiment suites."""

import sys
import json
import math
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config, collate_fn
from src.data.dataset import MappingData, DisambiguationDataset, generate_mappings
from src.model import create_model_from_config
from src.training import train


def make_config(
    experiment_name: str,
    task: str = "bz_to_a",
    k: int = 10,
    seed: int = 42,
    lr: float = 1e-3,
    bs: int = 128,
    max_steps: int = 50000,
    eval_every: int = 50,
    checkpoint_every: int = 5000,
    label_noise_prob: float = 0.0,
    early_stop_frac: Optional[float] = None,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
):
    """Create an OmegaConf config matching the Hydra config schema."""
    return OmegaConf.create({
        "experiment": {"name": experiment_name, "seed": seed},
        "data": {
            "n_unique_b": 1000, "k": k, "task": task,
            "b_length": 6, "a_length": 4, "z_length": 2,
            "vocab_chars": "abcdefghijklmnopqrstuvwxyz0123456789",
            "probe_fraction": 0.0, "split_by_base": False,
            "enforce_unique_a_first_char_per_b": False,
            "disambiguation_prefix_length": 1,
            "label_noise_prob": label_noise_prob,
        },
        "tokenizer": {
            "pad_token": "<PAD>", "bos_token": "<BOS>",
            "eos_token": "<EOS>", "sep_token": "<SEP>",
        },
        "model": {
            "n_layers": 4, "n_heads": 4, "d_model": 128,
            "d_head": 32, "d_mlp": 512, "act_fn": "gelu",
        },
        "training": {
            "batch_size": bs, "learning_rate": lr,
            "weight_decay": weight_decay,
            "max_steps": max_steps, "warmup_steps": warmup_steps,
            "scheduler": "cosine",
            "checkpoint_every": checkpoint_every,
            "eval_every": eval_every,
            "early_stop_convergence_frac": early_stop_frac,
        },
        "output": {"base_dir": "outputs"},
    })


def run_exists(experiment_name, min_steps=0, output_dir="outputs"):
    """Check if a training run exists and reached min_steps."""
    p = Path(output_dir) / experiment_name / "training_history.json"
    if not p.exists():
        return False
    if min_steps <= 0:
        return True
    try:
        with open(p) as f:
            h = json.load(f)
        return len(h.get("steps", [])) > 0 and h["steps"][-1] >= min_steps
    except Exception:
        return False


def save_mapping(mapping_data, path):
    """Save MappingData to JSON for reuse across tasks."""
    d = {
        "mappings": {b: pairs for b, pairs in mapping_data.mappings.items()},
        "examples": mapping_data.examples,
        "n_unique_b": mapping_data.n_unique_b,
        "n_unique_a": mapping_data.n_unique_a,
        "k": mapping_data.k,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f)


def load_mapping(path):
    """Load MappingData from JSON."""
    with open(path) as f:
        d = json.load(f)
    return MappingData(
        mappings={b: [tuple(p) for p in pairs] for b, pairs in d["mappings"].items()},
        examples=d["examples"],
        n_unique_b=d["n_unique_b"],
        n_unique_a=d["n_unique_a"],
        k=d["k"],
        task="bz_to_a",
    )


def run_single_experiment(cfg, mapping_data=None, output_dir="outputs", model=None):
    """
    Full training pipeline.

    If mapping_data is provided, datasets are created from it with cfg.data.task
    controlling tokenization. If model is provided, reuses it (for Transfer task).

    Returns (model, history, mapping_data, tokenizer).
    """
    torch.manual_seed(cfg.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.experiment.seed)

    tokenizer = create_tokenizer_from_config(cfg)

    if mapping_data is None:
        from src.data import create_datasets_from_config
        train_ds, probe_ds, mapping_data = create_datasets_from_config(cfg, tokenizer)
    else:
        train_ds = DisambiguationDataset(
            mapping_data=mapping_data,
            tokenizer=tokenizer,
            split="train",
            probe_fraction=0.0,
            seed=cfg.experiment.seed,
            task=cfg.data.task,
            label_noise_prob=float(getattr(cfg.data, "label_noise_prob", 0.0)),
        )
        # Empty probe set (probe_fraction=0.0 means probe split is empty)
        probe_ds = DisambiguationDataset(
            mapping_data=mapping_data,
            tokenizer=tokenizer,
            split="probe",
            probe_fraction=0.0,
            seed=cfg.experiment.seed,
            task=cfg.data.task,
        )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    probe_loader = DataLoader(
        probe_ds, batch_size=cfg.training.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    if model is None:
        model = create_model_from_config(cfg, tokenizer)
    else:
        model.train()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config_path = out / cfg.experiment.name / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Only do candidate eval for bz_to_a (requires correct mapping structure)
    if cfg.data.task == "bz_to_a":
        history = train(
            model=model, train_loader=train_loader, probe_loader=probe_loader,
            cfg=cfg, output_dir=out, grad_clip=1.0,
            mapping_data=mapping_data, tokenizer=tokenizer,
        )
    else:
        history = train(
            model=model, train_loader=train_loader, probe_loader=probe_loader,
            cfg=cfg, output_dir=out, grad_clip=1.0,
        )

    return model, history, mapping_data, tokenizer


def detect_tau(history, log_k, key="candidate_loss", threshold_frac=0.5):
    """
    Detect transition time tau from training history.

    Returns the first step where history[key] drops below threshold_frac * log_k.
    Returns None if transition not detected.
    """
    if key not in history or not history[key]:
        # Fallback to first_target_loss
        key = "first_target_loss"
    if key not in history or not history[key]:
        return None
    threshold = threshold_frac * log_k
    steps = history["steps"]
    vals = history[key]
    for s, v in zip(steps, vals):
        if v is not None and v < threshold:
            return s
    return None


def detect_convergence(history, threshold=0.01, key="train_loss"):
    """Detect convergence: first step where loss drops below threshold."""
    if key not in history or not history[key]:
        return None
    steps = history["steps"]
    vals = history[key]
    for s, v in zip(steps, vals):
        if v < threshold:
            return s
    return None


def load_history(experiment_name, output_dir="outputs"):
    """Load training history from JSON."""
    p = Path(output_dir) / experiment_name / "training_history.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)
