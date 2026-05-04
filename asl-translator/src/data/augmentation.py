"""Training-time augmentations applied to landmark sequences.

All transforms operate on numpy arrays of shape ``(T, D)`` where
``D = num_hands * 21 * 3``. Transforms are no-ops at evaluation time.

Implemented:
  * horizontal flip       — mirrors x-coordinates (and swaps left/right hand
                            slots), matching the proposal's "coordinate flipping".
  * gaussian noise        — small per-coordinate jitter for robustness.
  * time warp             — speed up / slow down by linearly resampling time.
  * random frame drop     — Bernoulli drop per frame (with re-interpolation).
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3


def _per_hand_size() -> int:
    return NUM_LANDMARKS * COORDS_PER_LANDMARK


def horizontal_flip(seq: np.ndarray, num_hands: int) -> np.ndarray:
    """Negate the x-coordinate (every 3rd value starting from index 0) and
    swap the left and right hand slots if there are two hands."""
    out = seq.copy()
    D = out.shape[1]
    # x-coords are at indices 0, 3, 6, ... ; negate them.
    out[:, 0:D:3] *= -1.0
    if num_hands == 2:
        ph = _per_hand_size()
        left = out[:, :ph].copy()
        right = out[:, ph:2 * ph].copy()
        out[:, :ph] = right
        out[:, ph:2 * ph] = left
    return out


def gaussian_noise(seq: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std <= 0:
        return seq
    return seq + rng.normal(0.0, std, size=seq.shape).astype(seq.dtype)


def time_warp(seq: np.ndarray, factor: float) -> np.ndarray:
    """Resample the time axis. ``factor < 1`` slows the clip down (more frames),
    ``factor > 1`` speeds it up."""
    T = seq.shape[0]
    new_T = max(2, int(round(T / factor)))
    src = np.linspace(0, T - 1, num=new_T)
    base = np.arange(T)
    out = np.empty((new_T, seq.shape[1]), dtype=seq.dtype)
    for d in range(seq.shape[1]):
        out[:, d] = np.interp(src, base, seq[:, d])
    return out


def random_drop_frames(seq: np.ndarray, drop_prob: float,
                       rng: np.random.Generator) -> np.ndarray:
    if drop_prob <= 0 or seq.shape[0] <= 4:
        return seq
    keep = rng.random(seq.shape[0]) > drop_prob
    if keep.sum() < 2:
        return seq
    return seq[keep]


class SequenceAugmenter:
    """Composes the configured augmentations. Use as a callable per-sample."""

    def __init__(self, aug_cfg, num_hands: int, seed: int | None = None) -> None:
        self.cfg = aug_cfg
        self.num_hands = num_hands
        self.rng = np.random.default_rng(seed)

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        if not bool(self.cfg.enable):
            return seq
        out = seq

        if self.rng.random() < float(self.cfg.horizontal_flip_prob):
            out = horizontal_flip(out, self.num_hands)

        if self.rng.random() < float(self.cfg.time_warp_prob):
            lo, hi = self.cfg.time_warp_factor_range
            factor = float(self.rng.uniform(float(lo), float(hi)))
            out = time_warp(out, factor)

        out = random_drop_frames(
            out, float(self.cfg.random_drop_frame_prob), self.rng,
        )

        out = gaussian_noise(out, float(self.cfg.gaussian_noise_std), self.rng)

        return out
