"""Tests for placing frames on the grid of a nominal rate."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from aind_video_utils.frame_grid import fit_nominal_grid


def _ms_rounded(n_frames: int, rate: int) -> np.ndarray:
    """PTS of frames at *rate*, rounded to 1 ms and stored at 16000 ticks per second, as a Matroska round trip does."""
    return np.round(np.arange(n_frames) * 1000 / rate).astype(np.int64) * 16


def test_ms_rounded_timestamps_fit_their_rate():
    grid = fit_nominal_grid(_ms_rounded(1000, 120), 16000, Fraction(120))
    assert np.array_equal(grid.slots, np.arange(1000))
    assert grid.gap_after.size == 0
    assert grid.max_residual == pytest.approx(0.04)


def test_gaps_are_reported_by_display_index_and_excess():
    pts = np.delete(_ms_rounded(20, 120), [5, 6, 12])
    grid = fit_nominal_grid(pts[::-1], 16000, Fraction(120))
    assert grid.gap_after.tolist() == [4, 9]
    assert grid.gap_excess.tolist() == [2, 1]


@pytest.mark.parametrize(
    ("pts", "rate", "match"),
    [
        # Every step is within 0.25 of a period at 125, but frames drift off the grid.
        (_ms_rounded(100, 120), 125, "periods from its slot"),
        # 99 ms steps at 10 fps: 1% slow, so frame 26 is past the tolerance.
        (np.arange(100) * 99 * 16, 10, "frame 27 in display order"),
        (np.array([0, 128, 144, 400]), 120, "share a slot"),
        (_ms_rounded(100, 120), 240, "median step is 2"),
        (_ms_rounded(100, 120), 0, "must be positive"),
        (np.array([0]), 120, "at least two"),
    ],
)
def test_fit_refuses(pts, rate, match):
    with pytest.raises(ValueError, match=match):
        fit_nominal_grid(pts, 16000, Fraction(rate))


def test_a_frame_too_far_from_its_slot_is_refused():
    pts = _ms_rounded(10, 120)
    pts[5] += 53  # 0.4 period
    with pytest.raises(ValueError, match="frame 6 in display order is 0.4"):
        fit_nominal_grid(pts, 16000, Fraction(120))
