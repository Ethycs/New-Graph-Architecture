"""Reproducibility helper: set deterministic seeds across random, numpy, and torch."""

from __future__ import annotations

import random
import warnings


def set_seed(seed: int) -> int:
    """Set seeds across python random, numpy.random, torch.manual_seed,
    and torch.cuda.manual_seed_all. Set torch.use_deterministic_algorithms(True)
    and cuDNN deterministic flags. Returns the seed for chaining.

    torch is imported lazily; if torch is not installed (Phase 0 has no torch),
    the torch-side calls are skipped silently. PYTHONHASHSEED must be set in the
    environment before Python starts; this function does NOT mutate os.environ
    because it would be too late to affect hash randomization.
    """
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")

    # stdlib random
    random.seed(seed)

    # numpy (lazy - optional dep in Phase 0)
    try:
        import numpy as np  # type: ignore[import-untyped]

        np.random.seed(seed)
    except ImportError:
        warnings.warn(
            "numpy is not installed; numpy random state was not seeded.",
            UserWarning,
            stacklevel=2,
        )

    # torch (lazy - not present in Phase 0)
    try:
        import torch  # type: ignore[import-untyped]

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def current_seed_signature() -> dict[str, int | str | bool]:
    """Return a snapshot of seeded state for logging into JSONL records."""
    torch_available = False
    numpy_available = False

    try:
        import torch  # type: ignore[import-untyped]  # noqa: F401

        torch_available = True
    except ImportError:
        pass

    try:
        import numpy  # type: ignore[import-untyped]  # noqa: F401

        numpy_available = True
    except ImportError:
        pass

    return {
        "seed_set": True,
        "torch_available": torch_available,
        "numpy_available": numpy_available,
    }


__all__ = ["set_seed", "current_seed_signature"]
