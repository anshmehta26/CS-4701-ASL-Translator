"""Tests for landmark preprocessing helpers (no MediaPipe required)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.landmarks import (
    _interpolate_missing,
    _moving_average,
    _wrist_relative_normalize,
)


def test_interpolation_fills_nans():
    seq = np.array([
        [1.0, 10.0],
        [np.nan, np.nan],
        [3.0, 30.0],
    ], dtype=np.float32)
    out = _interpolate_missing(seq)
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out[1, 0], 2.0)
    np.testing.assert_allclose(out[1, 1], 20.0)


def test_interpolation_handles_all_nan_column():
    seq = np.array([
        [np.nan, 1.0],
        [np.nan, 2.0],
    ], dtype=np.float32)
    out = _interpolate_missing(seq)
    np.testing.assert_array_equal(out[:, 0], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(out[:, 1], np.array([1.0, 2.0], dtype=np.float32))


def test_moving_average_smooths_edges():
    seq = np.arange(10, dtype=np.float32).reshape(-1, 1)
    sm = _moving_average(seq, window=3)
    assert sm.shape == seq.shape
    # Interior values should be averages of neighbours
    np.testing.assert_allclose(sm[1, 0], 1.0)
    np.testing.assert_allclose(sm[5, 0], 5.0)


def test_moving_average_window_one_is_identity():
    seq = np.random.RandomState(0).randn(5, 3).astype(np.float32)
    np.testing.assert_array_equal(_moving_average(seq, window=1), seq)


def test_wrist_relative_centers_first_joint_at_origin():
    # Build a trivial single-frame, single-hand vector where every joint sits
    # offset from a known wrist position.
    frame = np.zeros(21 * 3, dtype=np.float32)
    for i in range(21):
        frame[3*i:3*i + 3] = [1.0 + i, 2.0 + i, 3.0 + i]
    out = _wrist_relative_normalize(frame, num_hands=1)
    # Wrist (joint 0) should now be at the origin
    np.testing.assert_allclose(out[0:3], [0.0, 0.0, 0.0], atol=1e-6)


if __name__ == "__main__":
    test_interpolation_fills_nans()
    test_interpolation_handles_all_nan_column()
    test_moving_average_smooths_edges()
    test_moving_average_window_one_is_identity()
    test_wrist_relative_centers_first_joint_at_origin()
    print("All landmark preprocessing tests passed.")
