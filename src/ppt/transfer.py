"""
Weight transfer between HookedTransformers with different vocab sizes.

Transfers attention + MLP weights. Re-initializes:
- embed.W_E (token embeddings)
- pos_embed.W_pos (positional embeddings) — KEPT if same n_ctx
- unembed.W_U (unembedding/lm_head)
- unembed.b_U (unembedding bias)

HookedTransformer weight names follow this pattern:
    embed.W_E                           — token embedding
    pos_embed.W_pos                     — positional embedding
    blocks.{i}.ln1.w / .b               — pre-attention LayerNorm
    blocks.{i}.attn.W_Q / W_K / W_V     — attention weights
    blocks.{i}.attn.W_O                  — attention output projection
    blocks.{i}.attn.b_Q / b_K / b_V     — attention biases
    blocks.{i}.attn.b_O                  — attention output bias
    blocks.{i}.ln2.w / .b               — pre-MLP LayerNorm
    blocks.{i}.mlp.W_in / b_in          — MLP input projection
    blocks.{i}.mlp.W_out / b_out        — MLP output projection
    ln_final.w / .b                     — final LayerNorm
    unembed.W_U / b_U                  — unembedding
"""

from transformer_lens import HookedTransformer
from typing import Literal


# Keys that depend on vocab size and MUST be re-initialized
EMBEDDING_KEYS = {"embed.W_E", "unembed.W_U", "unembed.b_U"}

# Keys that depend on context length (transfer if same n_ctx, else re-init)
POSITIONAL_KEYS = {"pos_embed.W_pos"}


def _is_attention_key(key: str) -> bool:
    """Keys that are attention-specific."""
    return ".attn." in key or ".ln1." in key


def _is_mlp_key(key: str) -> bool:
    """Keys that are MLP-specific."""
    return ".mlp." in key or ".ln2." in key


def _is_final_ln_key(key: str) -> bool:
    """Keys that are final LayerNorm."""
    return key.startswith("ln_final.")


def transfer_weights(
    source_model: HookedTransformer,
    target_model: HookedTransformer,
    mode: Literal["full", "attn_only", "mlp_only"] = "full",
) -> dict:
    """
    Transfer weights from source (PPT) to target (disambiguation task) model.

    Args:
        source_model: Pre-pre-trained HookedTransformer
        target_model: Fresh HookedTransformer for target task
        mode: Which components to transfer
            "full"      — transfer attention + MLP + final LN
            "attn_only" — transfer attention weights only (+ ln1)
            "mlp_only"  — transfer MLP weights only (+ ln2 + final LN)

    Returns:
        dict with transfer statistics
    """
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()

    transferred = []
    skipped = []
    shape_mismatch = []

    for key in target_state:
        # Always skip embeddings (different vocab)
        if key in EMBEDDING_KEYS:
            skipped.append(key)
            continue

        # Positional embeddings: transfer if same shape
        if key in POSITIONAL_KEYS:
            if (
                key in source_state
                and source_state[key].shape == target_state[key].shape
            ):
                target_state[key] = source_state[key].clone()
                transferred.append(key)
            else:
                skipped.append(key)
            continue

        # Check if key exists in source
        if key not in source_state:
            skipped.append(key)
            continue

        # Check shape match
        if source_state[key].shape != target_state[key].shape:
            shape_mismatch.append(key)
            continue

        # Apply transfer mode filter
        should_transfer = False
        if mode == "full":
            should_transfer = True
        elif mode == "attn_only":
            should_transfer = _is_attention_key(key)
        elif mode == "mlp_only":
            should_transfer = _is_mlp_key(key) or _is_final_ln_key(key)

        if should_transfer:
            target_state[key] = source_state[key].clone()
            transferred.append(key)
        else:
            skipped.append(key)

    target_model.load_state_dict(target_state)

    stats = {
        "transferred": len(transferred),
        "skipped": len(skipped),
        "shape_mismatch": len(shape_mismatch),
        "mode": mode,
        "transferred_keys": transferred,
        "shape_mismatch_keys": shape_mismatch,
    }

    print(
        f"    [Transfer:{mode}] {len(transferred)} transferred, "
        f"{len(skipped)} skipped, {len(shape_mismatch)} shape mismatch"
    )

    return stats
