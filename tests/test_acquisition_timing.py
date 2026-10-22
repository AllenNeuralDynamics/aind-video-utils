"""Tests for reading and repairing per-frame acquisition timing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aind_video_utils.acquisition_timing import read_frame_timing, repair_frame_times, video_frame_rows
from aind_video_utils.mp4_index import Mp4FrameIndex

PERIOD = 0.002  # seconds between triggers at 500 Hz


def test_reads_the_legacy_headerless_layout(tmp_path: Path):
    csv = tmp_path / "bottom_camera.csv"
    csv.write_text("100.000,7,14593876902543944,1000,0\n100.002,8,14593876904543944,1000,0\n")
    table = read_frame_timing(csv)
    assert table.n_frames == 2
    assert table.reference_time.tolist() == [100.0, 100.002]
    assert table.frame_number.tolist() == [7, 8]
    # Beyond float64's exact integers, so it must not pass through a float.
    assert table.camera_time.tolist() == [14593876902543944, 14593876904543944]


def test_reads_the_file_standard_layout_by_column_name(tmp_path: Path):
    csv = tmp_path / "metadata.csv"
    csv.write_text("CameraFrameNumber,Gain,ReferenceTime,CameraFrameTime\n3,1.5,10.5,99\n4,1.5,10.502,101\n")
    table = read_frame_timing(csv)
    assert table.reference_time.tolist() == [10.5, 10.502]
    assert table.frame_number.tolist() == [3, 4]
    assert table.camera_time.tolist() == [99, 101]


def test_a_header_missing_a_standard_column_is_refused(tmp_path: Path):
    csv = tmp_path / "metadata.csv"
    csv.write_text("ReferenceTime,CameraFrameNumber\n10.5,3\n")
    with pytest.raises(ValueError, match="CameraFrameTime"):
        read_frame_timing(csv)


def test_reads_a_single_row(tmp_path: Path):
    csv = tmp_path / "metadata.csv"
    csv.write_text("ReferenceTime,CameraFrameNumber,CameraFrameTime\n10.5,3,99\n")
    assert read_frame_timing(csv).n_frames == 1


def _received(exposures: list[int], start: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """True trigger times of every exposure up to the last received, and the counter of those received."""
    triggers = start + PERIOD * np.arange(exposures[-1] + 1)
    return triggers, np.array(exposures, dtype=np.int64) + 1000


def test_no_dropped_frame_needs_no_repair():
    triggers, counter = _received(list(range(6)))
    result = repair_frame_times(triggers, counter)
    assert np.array_equal(result.time, triggers)
    assert result.dropped_before is not None and not result.dropped_before.any()
    assert result.repairs == ()
    assert result.pairing is None


def test_arrival_pairing_is_detected_and_repaired():
    exposures = [0, 1, 2, 5, 6, 9, 10]  # two bursts of two lost exposures
    triggers, counter = _received(exposures)
    recorded = triggers[: len(exposures)]  # the k-th frame received beside the k-th trigger
    result = repair_frame_times(recorded, counter)
    assert result.pairing == "by_arrival"
    assert result.repairs == ("zip_pairing_repaired",)
    # Exposures inside the recorded span take their own trigger; later ones extrapolate onto it.
    np.testing.assert_allclose(result.time, triggers[exposures])
    assert result.dropped_before is not None
    assert result.dropped_before.tolist() == [False, False, False, True, False, True, False]


def test_exposure_pairing_is_detected_and_kept():
    exposures = [0, 1, 2, 5, 6, 9, 10]
    triggers, counter = _received(exposures)
    result = repair_frame_times(triggers[exposures], counter)
    assert result.pairing == "by_exposure"
    assert result.repairs == ()
    assert np.array_equal(result.time, triggers[exposures])


def test_a_stated_pairing_is_used_without_detection():
    exposures = [0, 1, 2, 5, 6]
    triggers, counter = _received(exposures)
    result = repair_frame_times(triggers[exposures], counter, pairing="by_arrival")
    assert result.pairing == "by_arrival"
    assert result.repairs == ("zip_pairing_repaired",)


def test_pairing_is_refused_when_the_skips_fit_neither():
    exposures = [0, 1, 4, 5, 8, 9]
    triggers, counter = _received(exposures)
    recorded = triggers[exposures].copy()
    recorded[2:4] -= 2 * PERIOD  # first skip steps one period, the second three
    with pytest.raises(ValueError, match="pass pairing="):
        repair_frame_times(recorded, counter)


def test_a_counter_that_steps_backwards_is_distrusted():
    recorded = 100.0 + PERIOD * np.arange(6)
    counter = np.array([0, 1, 1050, 3, 4, 5])
    result = repair_frame_times(recorded, counter)
    assert result.repairs == ("frame_counter_distrusted",)
    assert result.dropped_before is None
    assert result.pairing is None
    assert np.array_equal(result.time, recorded)


def test_a_time_out_of_line_with_its_neighbours_takes_their_midpoint():
    recorded = 100.0 + PERIOD * np.arange(6)
    recorded[3] -= 0.98  # a second behind, as at a HARP second boundary
    result = repair_frame_times(recorded, np.arange(6))
    assert result.repairs == ("trigger_time_out_of_line",)
    assert result.time[3] == pytest.approx(100.0 + 3 * PERIOD)
    assert recorded[3] != result.time[3], "the input must not be modified"


def test_an_adjacent_pair_out_of_order_is_replaced_together():
    ticks = np.array([0, 62, 125, 190, 187, 250, 312], dtype=np.int64)
    result = repair_frame_times(ticks, np.arange(7))
    assert result.time.tolist() == [0, 62, 125, 156, 220, 250, 312]
    assert result.time.dtype == np.int64


def test_integer_ticks_extrapolate_to_integers():
    exposures = [0, 1, 2, 5, 6]
    ticks = np.arange(len(exposures), dtype=np.int64) * 63
    result = repair_frame_times(ticks, np.array(exposures), pairing="by_arrival")
    assert result.time.dtype == np.int64
    assert result.time.tolist() == [0, 63, 126, 315, 378]


@pytest.mark.parametrize(
    ("times", "counter", "match"),
    [
        (np.zeros(3), np.zeros(2, dtype=np.int64), "must match"),
        (np.zeros(1), np.zeros(1, dtype=np.int64), "at least two rows"),
    ],
)
def test_malformed_input_is_refused(times, counter, match):
    with pytest.raises(ValueError, match=match):
        repair_frame_times(times, counter)


def _video(pts: list[int]) -> Mp4FrameIndex:
    """An all-intra index, 32 ticks a frame, with *pts* in display order."""
    n = len(pts)
    return Mp4FrameIndex(
        media_timescale=16000,
        media_duration=0,
        movie_timescale=1000,
        dts=np.array(pts, dtype=np.int64),
        pts=np.array(pts, dtype=np.int64),
        is_keyframe=np.ones(n, dtype=np.bool_),
        byte_offset=np.zeros(n, dtype=np.int64),
        size=np.ones(n, dtype=np.int64),
        edits=(),
    )


def test_frames_after_a_gap_skip_the_rows_lost_in_it():
    # A step of 3 periods after frame 1 loses 2 frames, and of 2 after frame 3, 1.
    video = _video([0, 32, 128, 160, 224, 256])
    assert video_frame_rows(video, 9).tolist() == [0, 1, 4, 5, 7, 8]


def test_a_notch_without_extra_rows_is_a_stamping_fault():
    video = _video([0, 32, 128, 160])
    assert video_frame_rows(video, 4).tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("pts", "n_rows", "match"),
    [
        ([0, 32, 64], 2, "2 timing rows for 3 frames"),
        ([0, 32, 64, 96], 6, "hold 0 lost frames"),
        ([0, 32, 128, 160], 7, "hold 2 lost frames"),
        ([0, 32, 112, 144, 176], 8, "not whole periods"),
    ],
)
def test_rows_the_timeline_cannot_place_are_refused(pts, n_rows, match):
    with pytest.raises(ValueError, match=match):
        video_frame_rows(_video(pts), n_rows)
