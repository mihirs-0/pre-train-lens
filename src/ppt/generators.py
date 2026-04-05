"""
Pre-pre-training data generators.

These produce integer sequences for next-token prediction during
the pre-pre-training phase. They use their OWN vocabulary
(independent of CharTokenizer) because the PPT vocab is
re-initialized before target task training.

Each generator must implement:
    - vocab_size: int
    - generate_batch(batch_size, seq_len) -> torch.LongTensor of shape (B, T)
"""

import numpy as np
import torch


class MarkovBigramGenerator:
    """
    C1: Order-1 Markov chain.
    No long-range dependencies. Controls for generic warm-start.

    Limitation (pre-registered): This is NOT a fully matched structural
    placebo. C1 and C2 differ in compression ratio, effective context
    burden, token reuse patterns, and sequence grammar.
    """

    def __init__(self, vocab_size: int = 50, seed: int = 0):
        rng = np.random.RandomState(seed)
        # Random stochastic transition matrix
        self.transition = rng.dirichlet(np.ones(vocab_size), size=vocab_size)
        self.vocab_size = vocab_size

    def generate_batch(self, batch_size: int, seq_len: int) -> torch.LongTensor:
        sequences = np.zeros((batch_size, seq_len), dtype=np.int64)
        sequences[:, 0] = np.random.randint(0, self.vocab_size, size=batch_size)
        for t in range(1, seq_len):
            for b in range(batch_size):
                sequences[b, t] = np.random.choice(
                    self.vocab_size, p=self.transition[sequences[b, t - 1]]
                )
        return torch.from_numpy(sequences)


class ShuffleDyckGenerator:
    """
    C2: k-Shuffle Dyck language.

    Interleaved bracket types requiring hierarchical tracking.
    This is the distribution Hu et al. (ACL 2025) identified as most
    effective for natural language transfer.

    vocab_size = 2 * k (one open + one close per bracket type)
    """

    def __init__(self, k: int = 5, max_depth: int = 8):
        self.k = k
        self.max_depth = max_depth
        self.vocab_size = 2 * k

    def generate_batch(self, batch_size: int, seq_len: int) -> torch.LongTensor:
        sequences = np.zeros((batch_size, seq_len), dtype=np.int64)
        for b in range(batch_size):
            stack = []
            for t in range(seq_len):
                can_open = len(stack) < self.max_depth
                can_close = len(stack) > 0

                if can_open and (not can_close or np.random.random() < 0.55):
                    bracket_type = np.random.randint(self.k)
                    sequences[b, t] = 2 * bracket_type  # open
                    stack.append(bracket_type)
                elif can_close:
                    bracket_type = stack.pop()
                    sequences[b, t] = 2 * bracket_type + 1  # close
                else:
                    bracket_type = np.random.randint(self.k)
                    sequences[b, t] = 2 * bracket_type
                    stack.append(bracket_type)
        return torch.from_numpy(sequences)
