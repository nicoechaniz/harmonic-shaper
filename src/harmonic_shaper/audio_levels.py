"""Shared level handling for real-time and standalone Shaper rendering."""

from __future__ import annotations

import numpy as np


OUTPUT_LIMIT = 0.95
LIMITER_DRIVE = 1.05


def soft_limit(samples: np.ndarray) -> np.ndarray:
    """Apply the Shaper output limiter, with a declared ±0.95 range."""

    return np.tanh(np.asarray(samples) * LIMITER_DRIVE) * OUTPUT_LIMIT
