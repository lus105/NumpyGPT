from contextlib import contextmanager
from typing import Any, Generator

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import src.gpt2 as _gpt2

Array = NDArray[Any]


@contextmanager
def collect_attention() -> Generator[list[Array], None, None]:
    _gpt2._attn_collector = []
    try:
        yield _gpt2._attn_collector
    finally:
        _gpt2._attn_collector = None


def plot_attention(attn_weights: list[Array], tokens: list[str], layer: int = 0) -> None:
    """Heatmap grid of attention weights for one layer. attn_weights from collect_attention()."""
    weights = attn_weights[layer]  # [n_head, n_seq, n_seq]
    n_head = weights.shape[0]
    fig, axes = plt.subplots(1, n_head, figsize=(3 * n_head, 3.5))
    if n_head == 1:
        axes = [axes]
    for h, ax in enumerate(axes):
        im = ax.imshow(weights[h], vmin=0, vmax=1, cmap='Blues')
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_title(f'Head {h}', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'Attention weights — layer {layer}')
    plt.tight_layout()
    plt.show()


def plot_top_k_logits(logits: Array, encoder: Any, k: int = 10) -> None:
    """Bar chart of top-k next-token probabilities. Pass logits[-1] from gpt2()."""
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    top_ids = np.argsort(probs)[-k:][::-1]
    top_probs = probs[top_ids]
    top_tokens = [repr(encoder.decode([int(i)])) for i in top_ids]

    _, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(k), top_probs)
    ax.set_xticks(range(k))
    ax.set_xticklabels(top_tokens, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Probability')
    ax.set_title(f'Top-{k} next-token probabilities')
    plt.tight_layout()
    plt.show()
