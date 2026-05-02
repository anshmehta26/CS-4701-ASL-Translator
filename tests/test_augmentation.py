"""Tests for sequence augmentation."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.augmentation import (
    SequenceAugmenter, gaussian_noise, horizontal_flip,
    random_drop_frames, time_warp,
)


def _dummy_seq(T=20, num_hands=2):
    D = num_hands * 21 * 3
    return np.random.RandomState(0).randn(T, D).astype(np.float32)


def test_horizontal_flip_negates_x_one_hand():
    seq = _dummy_seq(T=5, num_hands=1)
    flipped = horizontal_flip(seq, num_hands=1)
    # x-coordinates are at indices 0,3,6,...
    np.testing.assert_allclose(flipped[:, 0::3], -seq[:, 0::3])
    # y and z untouched
    np.testing.assert_allclose(flipped[:, 1::3], seq[:, 1::3])
    np.testing.assert_allclose(flipped[:, 2::3], seq[:, 2::3])


def test_horizontal_flip_swaps_hands_two_hands():
    seq = _dummy_seq(T=4, num_hands=2)
    flipped = horizontal_flip(seq, num_hands=2)
    per_hand = 21 * 3
    # The "right" hand block should now hold (negated x of) the original "left".
    np.testing.assert_allclose(
        flipped[:, :per_hand][:, 1::3], seq[:, per_hand:][:, 1::3],
    )
    np.testing.assert_allclose(
        flipped[:, per_hand:][:, 1::3], seq[:, :per_hand][:, 1::3],
    )


def test_gaussian_noise_zero_std_is_identity():
    seq = _dummy_seq()
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(gaussian_noise(seq, 0.0, rng), seq)


def test_time_warp_changes_length():
    seq = _dummy_seq(T=20)
    out = time_warp(seq, factor=2.0)         # speed up -> fewer frames
    assert out.shape[0] < seq.shape[0]
    out = time_warp(seq, factor=0.5)         # slow down -> more frames
    assert out.shape[0] > seq.shape[0]


def test_random_drop_keeps_at_least_two_frames():
    seq = _dummy_seq(T=30)
    rng = np.random.default_rng(0)
    out = random_drop_frames(seq, drop_prob=0.99, rng=rng)
    assert out.shape[0] >= 2


def test_augmenter_disabled_is_passthrough():
    class Cfg:
        enable = False
        horizontal_flip_prob = 0.5
        gaussian_noise_std = 0.01
        time_warp_prob = 0.5
        time_warp_factor_range = [0.8, 1.2]
        random_drop_frame_prob = 0.1
    aug = SequenceAugmenter(Cfg(), num_hands=2, seed=0)
    seq = _dummy_seq()
    np.testing.assert_array_equal(aug(seq), seq)


if __name__ == "__main__":
    test_horizontal_flip_negates_x_one_hand()
    test_horizontal_flip_swaps_hands_two_hands()
    test_gaussian_noise_zero_std_is_identity()
    test_time_warp_changes_length()
    test_random_drop_keeps_at_least_two_frames()
    test_augmenter_disabled_is_passthrough()
    print("All augmentation tests passed.")
