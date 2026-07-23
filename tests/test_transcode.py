"""Tests for encoding profiles and transcode module."""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from aind_video_utils import transcode as transcode_mod
from aind_video_utils.encoding import (
    OFFLINE_8BIT,
    OFFLINE_10BIT,
    ONLINE_8BIT,
    ONLINE_10BIT,
    PROFILES,
    SPEC_VERSION,
    EncodingProfile,
    with_setparams,
)
from aind_video_utils.probe import get_r_frame_rate
from aind_video_utils.transcode import VIDEO_EXTENSIONS, transcode_video

ffmpeg_required = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)

# ---------------------------------------------------------------------------
# SPEC_VERSION
# ---------------------------------------------------------------------------


def test_spec_version_is_string():
    assert isinstance(SPEC_VERSION, str)
    assert SPEC_VERSION == "1.0"


# ---------------------------------------------------------------------------
# EncodingProfile basics
# ---------------------------------------------------------------------------


def test_profile_is_frozen():
    import dataclasses

    assert dataclasses.fields(EncodingProfile)  # is a dataclass
    try:
        OFFLINE_8BIT.codec = "libx265"  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except dataclasses.FrozenInstanceError:
        pass


def test_replace_returns_new_instance():
    fast = OFFLINE_8BIT.replace(codec_params=("-preset", "veryfast", "-crf", "18"))
    assert fast is not OFFLINE_8BIT
    assert fast.codec_params == ("-preset", "veryfast", "-crf", "18")
    # Original unchanged
    assert OFFLINE_8BIT.codec_params == ("-preset", "veryslow", "-crf", "18")


def test_replace_preserves_other_fields():
    modified = OFFLINE_8BIT.replace(codec="libx265")
    assert modified.codec == "libx265"
    assert modified.video_filters == OFFLINE_8BIT.video_filters
    assert modified.pixel_format == OFFLINE_8BIT.pixel_format
    assert modified.container == OFFLINE_8BIT.container


# ---------------------------------------------------------------------------
# OFFLINE_8BIT
# ---------------------------------------------------------------------------


def test_offline_8bit_output_args():
    args = OFFLINE_8BIT.ffmpeg_output_args()
    assert args == [
        "-vf",
        "scale=out_color_matrix=bt709:out_range=full"
        ":flags=accurate_rnd+full_chroma_int+full_chroma_inp:sws_dither=none,"
        "format=yuv420p10le,"
        "colorspace=all=bt709:dither=none,"
        "scale=out_range=tv:flags=accurate_rnd+full_chroma_int:sws_dither=ed,"
        "format=yuv420p",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryslow",
        "-crf",
        "18",
        "-metadata",
        "author=Allen Institute for Neural Dynamics",
        "-movflags",
        "+faststart+write_colr",
    ]


def test_offline_8bit_input_args():
    assert OFFLINE_8BIT.ffmpeg_input_args() == []


def test_offline_8bit_container():
    assert OFFLINE_8BIT.container == "mp4"


# ---------------------------------------------------------------------------
# OFFLINE_10BIT
# ---------------------------------------------------------------------------


def test_offline_10bit_output_args():
    args = OFFLINE_10BIT.ffmpeg_output_args()
    assert args == [
        "-vf",
        "colorspace=all=bt709:dither=none,"
        "scale=out_range=tv:flags=accurate_rnd+full_chroma_int:sws_dither=none,"
        "format=yuv420p10le",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p10le",
        "-preset",
        "veryslow",
        "-crf",
        "18",
        "-metadata",
        "author=Allen Institute for Neural Dynamics",
        "-movflags",
        "+faststart+write_colr",
    ]


def test_offline_10bit_input_args():
    assert OFFLINE_10BIT.ffmpeg_input_args() == []


def test_offline_10bit_container():
    assert OFFLINE_10BIT.container == "mp4"


# ---------------------------------------------------------------------------
# ONLINE_8BIT
# ---------------------------------------------------------------------------


def test_online_8bit_output_args():
    args = ONLINE_8BIT.ffmpeg_output_args()
    assert args == [
        "-vf",
        "scale=out_range=full,setparams=range=full:colorspace=bt709:color_primaries=bt709:color_trc=linear",
        "-c:v",
        "h264_nvenc",
        "-pix_fmt",
        "yuv420p",
        "-tune",
        "hq",
        "-preset",
        "p3",
        "-rc",
        "vbr",
        "-cq",
        "18",
        "-b:v",
        "0M",
        "-metadata",
        "author=Allen Institute for Neural Dynamics",
        "-color_range",
        "full",
        "-colorspace",
        "bt709",
        "-color_trc",
        "linear",
        "-maxrate",
        "700M",
        "-bufsize",
        "350M",
        "-f",
        "matroska",
        "-write_crc32",
        "0",
    ]


def test_online_8bit_input_args():
    assert ONLINE_8BIT.ffmpeg_input_args() == [
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_range",
        "full",
        "-color_trc",
        "linear",
    ]


def test_online_8bit_container():
    assert ONLINE_8BIT.container == "mkv"


# ---------------------------------------------------------------------------
# ONLINE_10BIT
# ---------------------------------------------------------------------------


def test_online_10bit_output_args():
    args = ONLINE_10BIT.ffmpeg_output_args()
    assert args == [
        "-vf",
        "format=yuv420p10le,"
        "scale=out_range=full,"
        "setparams=range=full:colorspace=bt709:color_primaries=bt709:color_trc=linear",
        "-c:v",
        "hevc_nvenc",
        "-pix_fmt",
        "p010le",
        "-tune",
        "hq",
        "-preset",
        "p4",
        "-rc",
        "vbr",
        "-cq",
        "12",
        "-b:v",
        "0M",
        "-metadata",
        "author=Allen Institute for Neural Dynamics",
        "-color_range",
        "full",
        "-colorspace",
        "bt709",
        "-color_trc",
        "linear",
        "-maxrate",
        "700M",
        "-bufsize",
        "350M",
        "-f",
        "matroska",
        "-write_crc32",
        "0",
    ]


def test_online_10bit_input_args():
    assert ONLINE_10BIT.ffmpeg_input_args() == []


def test_online_10bit_container():
    assert ONLINE_10BIT.container == "mkv"


# ---------------------------------------------------------------------------
# with_setparams
# ---------------------------------------------------------------------------


def _probe_json(**stream_fields: object) -> dict:
    """Minimal probe_json with given stream-level color fields. Defaults to
    untagged (mimics the AIND mpeg4 yuv420p production shape)."""
    return {"streams": [{"pix_fmt": stream_fields.get("pix_fmt", "yuv420p"), **stream_fields}]}


def test_with_setparams_no_probe_uses_aind_defaults():
    """Without probe_json, with_setparams fills every field with AIND defaults
    (colorspace=smpte170m, matching the bitstream truth for untagged YUV)."""
    modified = with_setparams(OFFLINE_8BIT)
    expected_prefix = "setparams=color_primaries=bt709:color_trc=linear:colorspace=smpte170m:range=pc,"
    assert modified.video_filters.startswith(expected_prefix)
    assert modified.video_filters == expected_prefix + OFFLINE_8BIT.video_filters


def test_with_setparams_probe_aware_fully_tagged_source_unchanged():
    """If the source already tags all four color fields, with_setparams returns
    the profile unchanged."""
    probe_json = _probe_json(
        pix_fmt="yuv420p",
        color_primaries="bt709",
        color_transfer="bt709",
        color_space="bt709",
        color_range="tv",
    )
    modified = with_setparams(OFFLINE_8BIT, probe_json)
    assert modified.video_filters == OFFLINE_8BIT.video_filters  # no setparams prepended


def test_with_setparams_probe_aware_yuv420p_untagged_fills_smpte170m():
    """Untagged yuv420p sources get colorspace=smpte170m (ffmpeg encoder
    default for untagged YUV)."""
    probe_json = _probe_json(pix_fmt="yuv420p")
    modified = with_setparams(OFFLINE_8BIT, probe_json)
    assert "colorspace=smpte170m" in modified.video_filters
    assert "color_trc=linear" in modified.video_filters
    assert "range=pc" in modified.video_filters


def test_with_setparams_probe_aware_gbrp_untagged_fills_gbr():
    """gbrp sources missing color_space get colorspace=gbr (truthfully RGB,
    no YUV matrix yet applied)."""
    probe_json = _probe_json(pix_fmt="gbrp")
    modified = with_setparams(OFFLINE_8BIT, probe_json)
    assert "colorspace=gbr" in modified.video_filters


def test_with_setparams_probe_aware_gbrp_with_tags_preserves_them():
    """gbrp sources with color_range=pc and color_space=gbr already tagged
    only get color_trc and color_primaries added — the existing tags are not
    overridden."""
    probe_json = _probe_json(
        pix_fmt="gbrp",
        color_space="gbr",
        color_range="pc",
    )
    modified = with_setparams(OFFLINE_8BIT, probe_json)
    assert "color_trc=linear" in modified.video_filters
    assert "color_primaries=bt709" in modified.video_filters
    # Source-tagged fields must NOT be re-asserted (would be lying-or-redundant)
    assert "colorspace=" not in modified.video_filters.split(",")[0]
    assert "range=" not in modified.video_filters.split(",")[0]


def test_with_setparams_probe_aware_unknown_treated_as_missing():
    """ffprobe sometimes emits 'unknown' for fields that aren't tagged.  Those
    should be treated identically to missing — filled in by setparams."""
    probe_json = _probe_json(
        pix_fmt="yuv420p",
        color_primaries="unknown",
        color_transfer="unknown",
        color_space="unknown",
        color_range="unknown",
    )
    modified = with_setparams(OFFLINE_8BIT, probe_json)
    assert "color_primaries=bt709" in modified.video_filters
    assert "color_trc=linear" in modified.video_filters
    assert "colorspace=smpte170m" in modified.video_filters
    assert "range=pc" in modified.video_filters


def test_with_setparams_range_override_no_probe_uses_override():
    """Without probe_json, range_override='tv' replaces the default range=pc."""
    modified = with_setparams(OFFLINE_8BIT, range_override="tv")
    expected_prefix = "setparams=color_primaries=bt709:color_trc=linear:colorspace=smpte170m:range=tv,"
    assert modified.video_filters.startswith(expected_prefix)


def test_with_setparams_range_override_overrides_untagged_default():
    """For an untagged source, range_override='tv' wins over the range=pc default."""
    probe_json = _probe_json(pix_fmt="yuv420p")
    modified = with_setparams(OFFLINE_8BIT, probe_json, range_override="tv")
    assert "range=tv" in modified.video_filters
    assert "range=pc" not in modified.video_filters


def test_with_setparams_range_override_overrides_source_tag():
    """range_override forces the range field even when the source declares one
    (the source's tag is wrong for AIND mpeg4 TV-range files; the override is
    the authoritative manifest signal)."""
    probe_json = _probe_json(
        pix_fmt="yuv420p",
        color_primaries="bt709",
        color_transfer="bt709",
        color_space="bt709",
        color_range="pc",
    )
    modified = with_setparams(OFFLINE_8BIT, probe_json, range_override="tv")
    assert "range=tv" in modified.video_filters


def test_with_setparams_range_override_fully_tagged_source_still_adds_setparams():
    """When all source fields are tagged AND a range_override is requested, the
    setparams clause is still prepended (it carries only the range= field)."""
    probe_json = _probe_json(
        pix_fmt="yuv420p",
        color_primaries="bt709",
        color_transfer="bt709",
        color_space="bt709",
        color_range="pc",
    )
    modified = with_setparams(OFFLINE_8BIT, probe_json, range_override="tv")
    assert modified.video_filters.startswith("setparams=range=tv,")


def test_with_setparams_range_override_pc_explicit():
    """range_override='pc' is a valid explicit value (equivalent to current default)."""
    modified = with_setparams(OFFLINE_8BIT, range_override="pc")
    assert "range=pc" in modified.video_filters


def test_with_setparams_does_not_mutate_original():
    original_vf = OFFLINE_8BIT.video_filters
    with_setparams(OFFLINE_8BIT)
    assert OFFLINE_8BIT.video_filters == original_vf


def test_with_setparams_preserves_other_fields():
    modified = with_setparams(OFFLINE_8BIT)
    assert modified.codec == OFFLINE_8BIT.codec
    assert modified.pixel_format == OFFLINE_8BIT.pixel_format
    assert modified.container == OFFLINE_8BIT.container
    assert modified.codec_params == OFFLINE_8BIT.codec_params


# ---------------------------------------------------------------------------
# PROFILES dict
# ---------------------------------------------------------------------------


def test_profiles_lookup():
    assert PROFILES["offline-8bit"] is OFFLINE_8BIT
    assert PROFILES["offline-10bit"] is OFFLINE_10BIT
    assert PROFILES["online-8bit"] is ONLINE_8BIT
    assert PROFILES["online-10bit"] is ONLINE_10BIT


def test_profiles_has_four_entries():
    assert len(PROFILES) == 4


# ---------------------------------------------------------------------------
# End-to-end regression: untagged yuv420p must not crush full-range content
# ---------------------------------------------------------------------------


def _encode_untagged_yuv420p(out_path: Path, luma_values: list[int]) -> None:
    """Write a yuv420p mpeg4 source with given Y values and NO color_range tag.

    Mimics the AIND Bonsai production format: full-range yuv420p without
    explicit range/space/transfer/primaries metadata.
    """
    W, H = 64, 64
    raw = out_path.parent / "untagged.yuv"
    with raw.open("wb") as f:
        for y in luma_values:
            Y = np.full((H, W), y, dtype=np.uint8)
            U = np.full((H // 2, W // 2), 128, dtype=np.uint8)
            V = np.full((H // 2, W // 2), 128, dtype=np.uint8)
            f.write(Y.tobytes())
            f.write(U.tobytes())
            f.write(V.tobytes())
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "yuv420p",
            "-video_size",
            f"{W}x{H}",
            "-framerate",
            "10",
            "-i",
            str(raw),
            "-c:v",
            "mpeg4",
            "-q:v",
            "1",
            "-pix_fmt",
            "yuv420p",
            # Deliberately NO -color_range, -colorspace, -color_trc, -color_primaries
            # — match the production-file metadata shape.
            str(out_path),
        ],
        check=True,
    )


def _decode_center_luma(path: Path, n_frames: int) -> list[int]:
    """Return the center-pixel Y value of each frame in the file."""
    W, H = 64, 64
    raw = path.parent / f"{path.stem}.y"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "extractplanes=y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            str(raw),
        ],
        check=True,
    )
    arr = np.fromfile(raw, dtype=np.uint8).reshape(n_frames, H, W)
    return [int(arr[i, H // 2, W // 2]) for i in range(n_frames)]


@ffmpeg_required
def test_offline_8bit_preserves_full_range_yuv420p_shadows_and_highlights(tmp_path: Path) -> None:
    """Untagged full-range yuv420p input must round-trip through OFFLINE_8BIT
    without crushing shadow (Y<16) or highlight (Y>235) detail.

    Regression guard for the ``range=pc`` term in ``_SETPARAMS``. Without it,
    every Y in [0, 16] crushes to output 16 (shadows lost) and every Y in
    [235, 255] crushes to output 236 (highlights lost). With it, shadow and
    highlight values map to distinguishable output luma.
    """
    sentinels = [8, 16, 245, 255]
    src = tmp_path / "src.mp4"
    dst = tmp_path / "dst.mp4"
    _encode_untagged_yuv420p(src, sentinels)

    profile = with_setparams(OFFLINE_8BIT)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(src),
            *profile.ffmpeg_output_args(),
            str(dst),
        ],
        check=True,
    )

    out_y = _decode_center_luma(dst, len(sentinels))

    # The two shadow values (8, 16) and two highlight values (245, 255) must
    # produce DIFFERENT output luma. With the limited-range-default bug both
    # pairs collapse to a single value; the fix is what spreads them apart.
    assert out_y[0] != out_y[1], (
        f"shadow detail crushed: Y=8 and Y=16 both mapped to {out_y[0]} — "
        "untagged yuv420p is being treated as limited-range. "
        "Check that range=pc is set in _SETPARAMS."
    )
    assert out_y[2] != out_y[3], (
        f"highlight detail crushed: Y=245 and Y=255 both mapped to {out_y[2]} — "
        "untagged yuv420p is being treated as limited-range. "
        "Check that range=pc is set in _SETPARAMS."
    )

    # And the values should be in the right direction: low-source-Y → low-output-Y,
    # high-source-Y → high-output-Y. (Sanity check that no inversion crept in.)
    assert out_y[0] < out_y[1] < out_y[2] < out_y[3]


# ---------------------------------------------------------------------------
# get_r_frame_rate
# ---------------------------------------------------------------------------


def test_get_r_frame_rate_parses_integer_rate():
    assert get_r_frame_rate({"streams": [{"r_frame_rate": "500/1"}]}) == (500, 1)


def test_get_r_frame_rate_parses_ntsc_rate():
    assert get_r_frame_rate({"streams": [{"r_frame_rate": "30000/1001"}]}) == (30000, 1001)


@pytest.mark.parametrize("rate", [None, "N/A", "0/0", "0/1", "500", "abc/1"])
def test_get_r_frame_rate_returns_none_on_bad_input(rate):
    stream: dict = {} if rate is None else {"r_frame_rate": rate}
    assert get_r_frame_rate({"streams": [stream]}) is None


# ---------------------------------------------------------------------------
# transcode_video: CFR normalization + frame-drop guard
#
# These are deterministic unit tests over the command construction and the
# progress-counter check; they fake the ffmpeg subprocess so they need neither
# ffmpeg nor a (hard-to-synthesize) drop-prone source. A drop-prone h264-in-AVI
# fixture isn't portable to CI, so the frame loss is covered here by injecting a
# drop_frames line into the faked progress stream.
# ---------------------------------------------------------------------------


class _FakePopen:
    """Stand-in for subprocess.Popen that replays a fixed progress stream."""

    def __init__(self, cmd, *, stdout_bytes: bytes, returncode: int, capture: list):
        self.args = cmd
        capture.append(cmd)
        self.stdout = io.BytesIO(stdout_bytes)
        self.stderr = io.BytesIO(b"")
        self._returncode = returncode

    def wait(self):
        return self._returncode


def _patch_ffmpeg(monkeypatch, *, stdout_bytes: bytes, returncode: int = 0, rate: str | None = "500/1"):
    """Fake out probe() and subprocess.Popen; return the list capturing argv."""
    captured: list = []

    def fake_probe(_path):
        return {"streams": [{"pix_fmt": "gbrp", "color_space": "gbr", "color_range": "pc", "r_frame_rate": rate}]}

    def fake_popen(cmd, stdout=None, stderr=None):
        return _FakePopen(cmd, stdout_bytes=stdout_bytes, returncode=returncode, capture=captured)

    monkeypatch.setattr(transcode_mod, "probe", fake_probe)
    monkeypatch.setattr(transcode_mod.subprocess, "Popen", fake_popen)
    return captured


def _vf_value(cmd: list) -> str:
    return cmd[cmd.index("-vf") + 1]


_CLEAN_PROGRESS = b"frame=100\ndrop_frames=0\ndup_frames=0\nprogress=end\n"
_DROP_PROGRESS = b"frame=94\ndrop_frames=6\ndup_frames=0\nprogress=end\n"


def test_normalize_cfr_prepends_setpts(monkeypatch, tmp_path):
    captured = _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS, rate="500/1")
    transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4", normalize_cfr=True)
    assert _vf_value(captured[0]).startswith("setpts=N/(500/1)/TB,")


def test_normalize_cfr_uses_exact_rational_rate(monkeypatch, tmp_path):
    captured = _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS, rate="30000/1001")
    transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4", normalize_cfr=True)
    assert _vf_value(captured[0]).startswith("setpts=N/(30000/1001)/TB,")


def test_normalize_cfr_off_omits_setpts(monkeypatch, tmp_path):
    captured = _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS)
    transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4", normalize_cfr=False)
    assert "setpts" not in _vf_value(captured[0])


def test_normalize_cfr_raises_without_readable_rate(monkeypatch, tmp_path):
    _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS, rate=None)
    with pytest.raises(RuntimeError, match="no readable r_frame_rate"):
        transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4", normalize_cfr=True)


def test_fail_on_frame_drop_raises_on_dropped_frames(monkeypatch, tmp_path):
    _patch_ffmpeg(monkeypatch, stdout_bytes=_DROP_PROGRESS)
    with pytest.raises(RuntimeError, match="drop_frames=6"):
        transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4", fail_on_frame_drop=True)


def test_fail_on_frame_drop_off_allows_dropped_frames(monkeypatch, tmp_path):
    _patch_ffmpeg(monkeypatch, stdout_bytes=_DROP_PROGRESS)
    out = transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4", fail_on_frame_drop=False)
    assert out == tmp_path / "out.mp4"


def test_frame_exact_output_does_not_raise(monkeypatch, tmp_path):
    _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS)
    out = transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4", fail_on_frame_drop=True)
    assert out == tmp_path / "out.mp4"


@ffmpeg_required
def test_transcode_video_preserves_frame_count_and_zero_start(tmp_path: Path) -> None:
    """End-to-end happy path: a clean CFR source transcodes with defaults
    (normalize_cfr + fail_on_frame_drop on) without dropping frames, lands at
    PTS 0, and keeps every source frame."""
    src = tmp_path / "src.mp4"
    dst = tmp_path / "dst.mp4"
    _encode_untagged_yuv420p(src, [20, 40, 60, 80, 100, 120])

    profile = OFFLINE_8BIT.replace(codec_params=("-preset", "ultrafast", "-crf", "18"))
    transcode_video(src, dst, profile=profile)  # defaults: normalize_cfr=True, fail_on_frame_drop=True

    def _count(path: Path) -> int:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nk=1:nw=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(out.stdout.strip())

    start = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=start_time",
            "-of",
            "default=nk=1:nw=1",
            str(dst),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert _count(dst) == _count(src) == 6
    assert float(start) == 0.0


# ---------------------------------------------------------------------------
# VIDEO_EXTENSIONS
# ---------------------------------------------------------------------------


def test_video_extensions_is_frozenset():
    assert isinstance(VIDEO_EXTENSIONS, frozenset)


def test_video_extensions_contains_expected():
    expected = {".avi", ".flv", ".mkv", ".mov", ".mp4", ".webm", ".wmv"}
    assert VIDEO_EXTENSIONS == expected
