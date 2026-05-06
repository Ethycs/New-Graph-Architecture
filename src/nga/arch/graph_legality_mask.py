"""Graph legality mask: zeroes out illegal transitions before softmax.

This module is a thin functional layer over GraphFSM.apply_mask. The actual
mask matrix and lookup live in nga.arch.graph_fsm.GraphFSM. This file adds:
  - softmax_with_mask(logits, fsm, current_state, *, enabled, temperature)
  - batch_apply_mask(logits_batch, fsm, current_states, *, enabled)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nga.arch.graph_fsm import GraphFSM

__all__ = ["softmax_with_mask", "batch_apply_mask", "stable_softmax"]


def stable_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with a temperature parameter.

    Handles both 1-D (V,) and 2-D (B, V) inputs. Subtracts the max along the
    last axis before exponentiating to prevent overflow, then normalises.

    Args:
        logits: Shape (V,) or (B, V) array of unnormalised log-scores.
        temperature: Positive scalar; values > 1.0 flatten, < 1.0 sharpen.

    Returns:
        Array of the same shape as logits, summing to 1 along the last axis.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = logits / temperature
    # keepdims so subtraction broadcasts correctly for both 1-D and 2-D.
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def softmax_with_mask(
    logits: np.ndarray,
    fsm: GraphFSM,
    current_state: str | None,
    *,
    enabled: bool,
    temperature: float = 1.0,
) -> np.ndarray:
    """Apply legality mask, then numerically-stable softmax.

    Args:
        logits: Shape (V,) unnormalised log-scores over next states.
        fsm: GraphFSM instance that owns the adjacency mask.
        current_state: Current FSM state label. When None the mask is skipped.
        enabled: If False the mask is not applied (ablation mode).
        temperature: Softmax temperature forwarded to stable_softmax.

    Returns:
        Shape (V,) probability distribution over next states.
    """
    # Lazy import so the module loads even before graph_fsm.py is written.
    from nga.arch.graph_fsm import GraphFSM as _GraphFSM  # noqa: F401

    masked = fsm.apply_mask(logits, current_state) if (enabled and current_state is not None) else logits
    return stable_softmax(masked, temperature=temperature)


def batch_apply_mask(
    logits_batch: np.ndarray,          # shape (B, V)
    fsm: GraphFSM,
    current_states: list[str | None],  # length B
    *,
    enabled: bool,
) -> np.ndarray:
    """Apply mask row-by-row over a batch. Returns shape (B, V).

    Each row i of logits_batch is masked using current_states[i]. When
    enabled is False the input is returned unchanged (zero-copy if possible).

    Args:
        logits_batch: Shape (B, V) array of unnormalised log-scores.
        fsm: GraphFSM instance.
        current_states: Per-sample current state labels; None skips masking.
        enabled: Master switch - if False, no masking is applied.

    Returns:
        Shape (B, V) array with illegal transitions suppressed.
    """
    # Lazy import so the module loads even before graph_fsm.py is written.
    from nga.arch.graph_fsm import GraphFSM as _GraphFSM  # noqa: F401

    if not enabled:
        return logits_batch

    b = logits_batch.shape[0]
    if len(current_states) != b:
        raise ValueError(
            f"current_states length {len(current_states)} does not match "
            f"batch size {b}"
        )

    result = np.empty_like(logits_batch)
    for i, state in enumerate(current_states):
        if state is not None:
            result[i] = fsm.apply_mask(logits_batch[i], state)
        else:
            result[i] = logits_batch[i]
    return result
