"""
Pre-pre-training loop.

Creates a temporary HookedTransformer with the PPT generator's vocab size,
trains it on next-token prediction over synthetic sequences,
and returns the trained state dict for weight transfer.

IMPORTANT: The PPT model has a DIFFERENT vocab size than the target model.
Only attention + MLP weights transfer. Embeddings are discarded.
"""

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig
from typing import List, Tuple
from tqdm import tqdm


def create_ppt_model(
    vocab_size: int,
    n_layers: int = 4,
    n_heads: int = 4,
    d_model: int = 128,
    d_head: int = 32,
    d_mlp: int = 512,
    n_ctx: int = 128,
    device: str = "cpu",
    seed: int = 42,
) -> HookedTransformer:
    """
    Create a HookedTransformer for pre-pre-training.

    Architecture MUST match the target model exactly (same n_layers,
    n_heads, d_model, d_head, d_mlp) so weights can transfer.
    Only vocab_size differs.
    """
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_head=d_head,
        d_mlp=d_mlp,
        d_vocab=vocab_size,
        n_ctx=n_ctx,
        act_fn="gelu",
        positional_embedding_type="standard",
        normalization_type="LN",
        attn_only=False,
        device=device,
        seed=seed,
        init_weights=True,
    )
    return HookedTransformer(cfg)


def pre_pre_train(
    generator,  # MarkovBigramGenerator or ShuffleDyckGenerator
    n_steps: int = 5000,
    batch_size: int = 32,
    seq_len: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    # Architecture params (must match target model)
    n_layers: int = 4,
    n_heads: int = 4,
    d_model: int = 128,
    d_head: int = 32,
    d_mlp: int = 512,
    device: str = "cpu",
    seed: int = 42,
    log_every: int = 500,
) -> Tuple[HookedTransformer, List[float]]:
    """
    Run pre-pre-training and return the trained model + loss curve.

    The caller will extract transferable weights via transfer.py.
    """
    model = create_ppt_model(
        vocab_size=generator.vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_head=d_head,
        d_mlp=d_mlp,
        n_ctx=seq_len,
        device=device,
        seed=seed,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    model.train()
    loss_curve = []

    for step in range(n_steps):
        # Generate batch on-the-fly
        batch = generator.generate_batch(batch_size, seq_len).to(device)

        # Next-token prediction
        logits = model(batch)  # (B, T, V)
        # Shift: logits[:, :-1] predicts batch[:, 1:]
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            batch[:, 1:].reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_curve.append(loss.item())

        if step % log_every == 0:
            print(f"    PPT step {step}/{n_steps} | loss={loss.item():.4f}")

    # Verify convergence
    if len(loss_curve) > 100:
        early = sum(loss_curve[:100]) / 100
        late = sum(loss_curve[-100:]) / 100
        if late >= early * 0.95:
            print(
                f"    WARNING: PPT may not have converged "
                f"(early={early:.4f}, late={late:.4f})"
            )

    return model, loss_curve
