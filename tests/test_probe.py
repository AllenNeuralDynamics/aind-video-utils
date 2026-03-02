"""Tests for probe accessor functions."""

from aind_video_utils.probe import get_color_transfer, get_nb_frames

# ---------------------------------------------------------------------------
# get_color_transfer
# ---------------------------------------------------------------------------


def test_get_color_transfer_present():
    probe_json = {"streams": [{"color_transfer": "bt709"}]}
    assert get_color_transfer(probe_json) == "bt709"


def test_get_color_transfer_linear():
    probe_json = {"streams": [{"color_transfer": "linear"}]}
    assert get_color_transfer(probe_json) == "linear"


def test_get_color_transfer_unknown():
    probe_json = {"streams": [{"color_transfer": "unknown"}]}
    assert get_color_transfer(probe_json) is None


def test_get_color_transfer_missing():
    probe_json = {"streams": [{"pix_fmt": "yuv420p"}]}
    assert get_color_transfer(probe_json) is None


# ---------------------------------------------------------------------------
# get_nb_frames
# ---------------------------------------------------------------------------


def test_get_nb_frames_direct():
    probe_json = {"streams": [{"nb_frames": "1000"}]}
    assert get_nb_frames(probe_json) == 1000


def test_get_nb_frames_na():
    probe_json = {"streams": [{"nb_frames": "N/A", "duration": "10.0", "r_frame_rate": "30/1"}]}
    assert get_nb_frames(probe_json) == 300


def test_get_nb_frames_from_duration():
    probe_json = {"streams": [{"duration": "10.0", "r_frame_rate": "30/1"}]}
    assert get_nb_frames(probe_json) == 300


def test_get_nb_frames_from_duration_fractional():
    probe_json = {"streams": [{"duration": "10.0", "r_frame_rate": "24000/1001"}]}
    # 10.0 * (24000/1001) = 239.76... -> ceil = 240
    assert get_nb_frames(probe_json) == 240


def test_get_nb_frames_unavailable():
    probe_json = {"streams": [{"pix_fmt": "yuv420p"}]}
    assert get_nb_frames(probe_json) is None
