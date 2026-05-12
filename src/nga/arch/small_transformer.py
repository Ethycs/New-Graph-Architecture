"""Small causal transformer for Phase 23d.

A minimal `nn.TransformerEncoderLayer`-based causal LM intended as
**the arbitrary frozen base network** in Phase 23d's PCG-X test.
Train it from scratch on a grammar's token sequences via causal
next-token prediction, then harvest mid-layer activations as the
substrate for the predictive-projection pipeline.

The architecture is intentionally bare:

* token embedding + learned positional embedding
* ``n_layers`` of ``TransformerEncoderLayer`` (gelu, norm-first)
* causal attention mask
* linear LM head producing next-token logits

Mid-layer harvesting: ``encode(x, harvest_layer=k)`` returns
``(final_hidden, harvested_at_layer_k)`` so the runner can pluck any
layer's output without re-running the model. ``forward(x)`` is the
training-time path.

Sized for CPU: with ``d_model=64``, ``n_heads=4``, ``n_layers=2``,
``max_seq_len=100`` this is ~50K parameters and trains in a couple of
seconds on a small grammar corpus.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

__all__ = ["SmallTransformer", "train_small_transformer"]


class SmallTransformer(nn.Module if nn is not None else object):
    """Bare causal transformer LM with mid-layer harvest.

    Parameters
    ----------
    vocab_size:
        Token alphabet cardinality.
    d_model:
        Hidden dimension; also the embedding dimension.
    n_heads:
        Multi-head attention heads.
    n_layers:
        Number of encoder layers stacked under causal masking.
    max_seq_len:
        Maximum sequence length; used to pre-allocate position embeddings.
    seed:
        Drives ``torch.manual_seed`` for deterministic init.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 128,
        seed: int = 0,
    ) -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for SmallTransformer")
        super().__init__()
        torch.manual_seed(int(seed))
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=4 * d_model,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, T: int, device) -> "torch.Tensor":
        return torch.triu(
            torch.full((T, T), float("-inf"), device=device), diagonal=1
        )

    def encode(
        self, x: "torch.Tensor", harvest_layer: int | None = None
    ) -> tuple["torch.Tensor", "torch.Tensor | None"]:
        """Run the transformer; return ``(final_hidden, harvested_layer)``.

        ``x`` is integer token IDs of shape ``(B, T)``. The harvested
        layer (0-indexed) is the **output** of that layer (post-residual,
        post-layer-norm in norm-first config).
        """
        T = int(x.size(1))
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token_embed(x) + self.pos_embed(positions)
        mask = self._causal_mask(T, x.device)
        harvested: "torch.Tensor | None" = None
        for i, layer in enumerate(self.layers):
            h = layer(h, src_mask=mask, is_causal=True)
            if harvest_layer is not None and i == harvest_layer:
                harvested = h.clone()
        return h, harvested

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Next-token logits, shape ``(B, T, vocab_size)``."""
        h, _ = self.encode(x)
        return self.head(h)


def train_small_transformer(
    model: SmallTransformer,
    program_token_ids: list[np.ndarray],
    *,
    epochs: int = 10,
    lr: float = 3e-3,
    seed: int = 0,
    pad_token_id: int = 0,
) -> dict[str, float]:
    """Train ``model`` on causal next-token prediction over programs.

    Each program contributes one (T,) token-id array; we feed positions
    ``[0, T-2]`` and target positions ``[1, T-1]``. Returns a dict with
    ``final_loss`` and ``train_token_accuracy``.

    Programs of unequal length are stacked into a padded batch each epoch
    (left-padded with ``pad_token_id``); cross-entropy ignores the pad
    positions via ``ignore_index``.
    """
    if torch is None or nn is None:  # pragma: no cover
        raise ImportError("torch is required")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    max_T = max(int(t.size) for t in program_token_ids)
    n_progs = len(program_token_ids)
    rng = np.random.default_rng(int(seed))
    final = {"final_loss": float("nan"), "train_token_accuracy": float("nan")}
    # Pre-pad arrays so we can shuffle indices.
    padded = np.full((n_progs, max_T), pad_token_id, dtype=np.int64)
    lengths = np.zeros(n_progs, dtype=np.int64)
    for i, t in enumerate(program_token_ids):
        T = int(t.size)
        padded[i, :T] = t
        lengths[i] = T
    padded_t = torch.from_numpy(padded.astype(np.int64))
    lengths_t = torch.from_numpy(lengths)
    for _epoch in range(int(epochs)):
        perm = rng.permutation(n_progs)
        # Train one program at a time (small dataset; faster than batching
        # heterogeneous-length sequences at this scale).
        for i in perm:
            T = int(lengths_t[i].item())
            if T < 2:
                continue
            x = padded_t[i : i + 1, :T]
            logits = model(x)  # (1, T, V)
            target = x[:, 1:].contiguous()  # (1, T-1)
            pred = logits[:, :-1, :].contiguous()  # (1, T-1, V)
            opt.zero_grad()
            loss = ce(pred.view(-1, model.vocab_size), target.view(-1))
            loss.backward()
            opt.step()
            final["final_loss"] = float(loss.item())
    # Train accuracy.
    model.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for i in range(n_progs):
            T = int(lengths_t[i].item())
            if T < 2:
                continue
            x = padded_t[i : i + 1, :T]
            logits = model(x)
            preds = logits[:, :-1, :].argmax(dim=-1)
            target = x[:, 1:]
            n_correct += int((preds == target).sum().item())
            n_total += int(target.numel())
    final["train_token_accuracy"] = (
        float(n_correct) / float(n_total) if n_total > 0 else float("nan")
    )
    return final
