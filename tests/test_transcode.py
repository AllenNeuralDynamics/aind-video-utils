"""Tests for encoding profiles and transcode module."""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
from fractions import Fraction
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
    Derivative,
    EncodingProfile,
    preview_decimation,
    with_poster,
    with_preview,
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
    assert SPEC_VERSION == "0.3.0"


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
    assert OFFLINE_8BIT.codec_params == ("-preset", "slow", "-crf", "18")


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
        "scale=out_range=tv:flags=accurate_rnd+full_chroma_int:sws_dither=bayer,"
        "format=yuv420p",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "slow",
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
        "scale=out_color_matrix=bt709:out_range=full:flags=accurate_rnd+full_chroma_int+full_chroma_inp:sws_dither=none,"
        "format=yuv420p10le,"
        "colorspace=all=bt709:dither=none,"
        "scale=out_range=tv:flags=accurate_rnd+full_chroma_int:sws_dither=none,"
        "format=yuv420p10le",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p10le",
        "-preset",
        "slow",
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
    # Same tags as ONLINE_8BIT: an untagged rig stream would otherwise reach
    # scale=out_range=full as limited range and be stretched.
    assert ONLINE_10BIT.ffmpeg_input_args() == ONLINE_8BIT.ffmpeg_input_args()
    assert ONLINE_10BIT.ffmpeg_input_args() == [
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_range",
        "full",
        "-color_trc",
        "linear",
    ]


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
    expected = "setparams=color_primaries=bt709:color_trc=linear:colorspace=smpte170m:range=pc"
    assert modified.source_filters == expected
    assert modified.video_filters == OFFLINE_8BIT.video_filters  # the encoding chain is untouched


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
    assert modified.source_filters == ""  # no conditioning needed


def test_with_setparams_probe_aware_yuv420p_untagged_fills_smpte170m():
    """Untagged yuv420p sources get colorspace=smpte170m (ffmpeg encoder
    default for untagged YUV)."""
    probe_json = _probe_json(pix_fmt="yuv420p")
    modified = with_setparams(OFFLINE_8BIT, probe_json)
    assert "colorspace=smpte170m" in modified.source_filters
    assert "color_trc=linear" in modified.source_filters
    assert "range=pc" in modified.source_filters


def test_with_setparams_probe_aware_gbrp_untagged_fills_gbr():
    """gbrp sources missing color_space get colorspace=gbr (truthfully RGB,
    no YUV matrix yet applied)."""
    probe_json = _probe_json(pix_fmt="gbrp")
    modified = with_setparams(OFFLINE_8BIT, probe_json)
    assert "colorspace=gbr" in modified.source_filters


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
    assert "color_trc=linear" in modified.source_filters
    assert "color_primaries=bt709" in modified.source_filters
    # Source-tagged fields must NOT be re-asserted (would be lying-or-redundant)
    assert "colorspace=" not in modified.source_filters
    assert "range=" not in modified.source_filters


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
    assert "color_primaries=bt709" in modified.source_filters
    assert "color_trc=linear" in modified.source_filters
    assert "colorspace=smpte170m" in modified.source_filters
    assert "range=pc" in modified.source_filters


def test_with_setparams_range_override_no_probe_uses_override():
    """Without probe_json, range_override='tv' replaces the default range=pc."""
    modified = with_setparams(OFFLINE_8BIT, range_override="tv")
    assert modified.source_filters == "setparams=color_primaries=bt709:color_trc=linear:colorspace=smpte170m:range=tv"


def test_with_setparams_range_override_overrides_untagged_default():
    """For an untagged source, range_override='tv' wins over the range=pc default."""
    probe_json = _probe_json(pix_fmt="yuv420p")
    modified = with_setparams(OFFLINE_8BIT, probe_json, range_override="tv")
    assert "range=tv" in modified.source_filters
    assert "range=pc" not in modified.source_filters


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
    assert "range=tv" in modified.source_filters


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
    assert modified.source_filters == "setparams=range=tv"


def test_with_setparams_range_override_pc_explicit():
    """range_override='pc' is a valid explicit value (equivalent to current default)."""
    modified = with_setparams(OFFLINE_8BIT, range_override="pc")
    assert "range=pc" in modified.source_filters


def test_with_setparams_does_not_mutate_original():
    original = (OFFLINE_8BIT.source_filters, OFFLINE_8BIT.video_filters)
    with_setparams(OFFLINE_8BIT)
    assert (OFFLINE_8BIT.source_filters, OFFLINE_8BIT.video_filters) == original


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


def _encode_untagged_yuv420p(out_path: Path, luma_values: list[int], framerate: int = 10) -> None:
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
            str(framerate),
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
# transcode_video: CFR normalization + frame-count check
#
# These are deterministic unit tests over the command construction and the
# frame-count check; they fake the ffmpeg subprocess so they need neither
# ffmpeg nor a (hard-to-synthesize) drop-prone source. A drop-prone h264-in-AVI
# fixture isn't portable to CI, so the frame loss is covered here by feeding
# ffmpeg's end-of-run totals into the faked stderr.
# ---------------------------------------------------------------------------


class _FakePopen:
    """Stand-in for subprocess.Popen that replays fixed progress and log streams."""

    def __init__(self, cmd, *, stdout_bytes: bytes, stderr_bytes: bytes, returncode: int, capture: list):
        self.args = cmd
        capture.append(cmd)
        self.stdout = io.BytesIO(stdout_bytes)
        self.stderr = io.BytesIO(stderr_bytes)
        self._returncode = returncode

    def wait(self):
        return self._returncode


def _summary(decoded: int, *encoded: int, decode_errors: int = 0) -> bytes:
    """ffmpeg's end-of-run totals, one output per *encoded* count, as ``-loglevel level+verbose`` logs them."""
    lines = [
        f"[out#{i}/mp4 @ 0x1] [verbose]   Output stream #{i}:0 (video): "
        f"{n} frames encoded; {n} packets muxed (1 bytes); \n"
        for i, n in enumerate(encoded)
    ]
    lines.append(
        f"[in#0/avi @ 0x2] [verbose]   Input stream #0:0 (video): {decoded} packets read (1 bytes); "
        f"{decoded} frames decoded; {decode_errors} decode errors; \n"
    )
    return "".join(lines).encode()


_CLEAN_SUMMARY = _summary(100, 100)


def _patch_ffmpeg(
    monkeypatch,
    *,
    stdout_bytes: bytes,
    stderr_bytes: bytes = _CLEAN_SUMMARY,
    returncode: int = 0,
    rate: str | None = "500/1",
):
    """Fake out probe() and subprocess.Popen; return the list capturing argv."""
    captured: list = []

    def fake_probe(_path):
        return {"streams": [{"pix_fmt": "gbrp", "color_space": "gbr", "color_range": "pc", "r_frame_rate": rate}]}

    def fake_popen(cmd, stdout=None, stderr=None):
        return _FakePopen(
            cmd, stdout_bytes=stdout_bytes, stderr_bytes=stderr_bytes, returncode=returncode, capture=captured
        )

    monkeypatch.setattr(transcode_mod, "probe", fake_probe)
    monkeypatch.setattr(transcode_mod.subprocess, "Popen", fake_popen)
    return captured


def _vf_value(cmd: list) -> str:
    return cmd[cmd.index("-vf") + 1]


_CLEAN_PROGRESS = b"frame=100\nprogress=end\n"


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


def test_normalize_cfr_is_off_by_default(monkeypatch, tmp_path):
    captured = _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS)
    transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4")
    assert "setpts" not in _vf_value(captured[0])


def test_ffmpeg_logs_level_tagged_verbose(monkeypatch, tmp_path):
    captured = _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS)
    transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4")
    assert captured[0][1:4] == ["-hide_banner", "-loglevel", "level+verbose"]


def test_fail_on_frame_drop_raises_when_frames_go_missing(monkeypatch, tmp_path):
    _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS, stderr_bytes=_summary(1000, 994))
    with pytest.raises(RuntimeError, match=r"994 frames but ffmpeg decoded 1000 .*normalize_cfr=True"):
        transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4")


def test_fail_on_frame_drop_raises_on_duplicated_frames(monkeypatch, tmp_path):
    _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS, stderr_bytes=_summary(100, 101))
    with pytest.raises(RuntimeError, match="101 frames but ffmpeg decoded 100"):
        transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4")


def test_fail_on_frame_drop_raises_on_decode_errors(monkeypatch, tmp_path):
    _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS, stderr_bytes=_summary(100, 100, decode_errors=2))
    with pytest.raises(RuntimeError, match="2 decode error"):
        transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4")


def test_fail_on_frame_drop_raises_without_frame_totals(monkeypatch, tmp_path):
    _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS, stderr_bytes=b"")
    with pytest.raises(RuntimeError, match="no video frame totals"):
        transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4")


def test_fail_on_frame_drop_off_accepts_missing_frames(monkeypatch, tmp_path):
    _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS, stderr_bytes=_summary(1000, 994))
    out = transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4", fail_on_frame_drop=False)
    assert out == tmp_path / "out.mp4"


def test_frame_exact_output_does_not_raise(monkeypatch, tmp_path):
    _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS)
    out = transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4", fail_on_frame_drop=True)
    assert out == tmp_path / "out.mp4"


def test_ffmpeg_failure_reports_warnings_and_errors_only(monkeypatch, tmp_path):
    log = (
        b"[info] Stream mapping:\n"
        b"[verbose] filter graph chatter\n"
        b"[AVFilterGraph @ 0x1] [error] No such filter: 'x'\n"
        b"[fatal] Error opening output files: Filter not found\n"
    )
    _patch_ffmpeg(monkeypatch, stdout_bytes=b"", stderr_bytes=log, returncode=8)
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4")
    assert excinfo.value.stderr == b"".join(log.splitlines(keepends=True)[2:])


@ffmpeg_required
def test_transcode_video_preserves_frame_count_and_zero_start(tmp_path: Path) -> None:
    """End-to-end happy path: a clean CFR source transcodes with CFR
    normalization and the frame-count check on, lands at PTS 0, and keeps every
    source frame."""
    src = tmp_path / "src.mp4"
    dst = tmp_path / "dst.mp4"
    _encode_untagged_yuv420p(src, [20, 40, 60, 80, 100, 120])

    profile = OFFLINE_8BIT.replace(codec_params=("-preset", "ultrafast", "-crf", "18"))
    transcode_video(src, dst, profile=profile, normalize_cfr=True)

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


@ffmpeg_required
def test_frame_check_catches_a_frame_a_filter_drops(tmp_path: Path) -> None:
    """A frame dropped inside the filter graph never registers as a vsync drop,
    but ffmpeg's end-of-run totals still come up one short."""
    src = tmp_path / "src.mp4"
    _encode_untagged_yuv420p(src, [20, 40, 60, 80, 100, 120])
    profile = OFFLINE_8BIT.replace(
        video_filters="select=not(eq(n\\,2)),format=yuv420p",
        codec_params=("-preset", "ultrafast", "-crf", "18"),
    )
    with pytest.raises(RuntimeError, match="5 frames but ffmpeg decoded 6"):
        transcode_video(src, tmp_path / "dst.mp4", profile=profile)


@ffmpeg_required
def test_ffmpeg_failure_carries_the_error_line(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _encode_untagged_yuv420p(src, [20, 40])
    profile = OFFLINE_8BIT.replace(video_filters="nosuchfilter")
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        transcode_video(src, tmp_path / "dst.mp4", profile=profile)
    assert b"[error] No such filter: 'nosuchfilter'" in excinfo.value.stderr


# ---------------------------------------------------------------------------
# VIDEO_EXTENSIONS
# ---------------------------------------------------------------------------


def test_video_extensions_is_frozenset():
    assert isinstance(VIDEO_EXTENSIONS, frozenset)


def test_video_extensions_contains_expected():
    expected = {".avi", ".flv", ".mkv", ".mov", ".mp4", ".webm", ".wmv"}
    assert VIDEO_EXTENSIONS == expected


# ---------------------------------------------------------------------------
# Derivatives: multi-output profiles
# ---------------------------------------------------------------------------


def _dummy_derivative(**kwargs) -> Derivative:
    defaults = dict(suffix="_preview", codec="libx264", pixel_format="yuv420p", container="mp4")
    return Derivative(**{**defaults, **kwargs})


def test_plain_profile_still_emits_vf():
    assert OFFLINE_8BIT.derivatives == ()
    assert OFFLINE_8BIT.ffmpeg_graph_args() == ["-vf", OFFLINE_8BIT.video_filters]


def test_plain_profile_output_groups_have_no_map():
    groups = OFFLINE_8BIT.ffmpeg_output_groups()
    assert len(groups) == 1
    assert "-map" not in groups[0]


def test_output_args_equals_graph_plus_single_group():
    assert OFFLINE_8BIT.ffmpeg_output_args() == [
        *OFFLINE_8BIT.ffmpeg_graph_args(),
        *OFFLINE_8BIT.ffmpeg_output_groups()[0],
    ]


def test_output_args_raises_when_derivatives_present():
    profile = OFFLINE_8BIT.replace(derivatives=(_dummy_derivative(),))
    with pytest.raises(ValueError, match="1 derivative output"):
        profile.ffmpeg_output_args()


def test_graph_args_split_and_tail_chain():
    profile = OFFLINE_8BIT.replace(derivatives=(_dummy_derivative(filters="select=not(mod(n\\,20))"),))
    flag, graph = profile.ffmpeg_graph_args()
    assert flag == "-filter_complex"
    assert graph == f"[0:v]{OFFLINE_8BIT.video_filters},split=2[main][d0];[d0]select=not(mod(n\\,20))[d0out]"


def test_graph_args_derivative_without_filters_has_no_tail_segment():
    profile = OFFLINE_8BIT.replace(derivatives=(_dummy_derivative(),))
    _, graph = profile.ffmpeg_graph_args()
    assert graph.endswith("split=2[main][d0]")
    assert ";" not in graph


def test_map_label_follows_presence_of_tail_chain():
    tailed = OFFLINE_8BIT.replace(derivatives=(_dummy_derivative(filters="select=not(mod(n\\,2))"),))
    bare = OFFLINE_8BIT.replace(derivatives=(_dummy_derivative(),))
    assert tailed.ffmpeg_output_groups()[1][:2] == ["-map", "[d0out]"]
    assert bare.ffmpeg_output_groups()[1][:2] == ["-map", "[d0]"]
    assert tailed.ffmpeg_output_groups()[0][:2] == ["-map", "[main]"]


def test_multiple_derivatives_get_distinct_labels():
    profile = OFFLINE_8BIT.replace(
        derivatives=(
            _dummy_derivative(suffix="_a", filters="select=not(mod(n\\,2))"),
            _dummy_derivative(suffix="_b"),
        )
    )
    _, graph = profile.ffmpeg_graph_args()
    assert "split=3[main][d0][d1]" in graph
    labels = [g[1] for g in profile.ffmpeg_output_groups()]
    assert labels == ["[main]", "[d0out]", "[d1]"]


def test_derivative_emits_fps_mode_before_codec():
    group = OFFLINE_8BIT.replace(derivatives=(_dummy_derivative(),)).ffmpeg_output_groups()[1]
    assert group[2:6] == ["-fps_mode", "passthrough", "-c:v", "libx264"]


def test_output_paths_name_derivatives_from_primary_stem():
    profile = OFFLINE_8BIT.replace(
        derivatives=(_dummy_derivative(suffix="_preview"), _dummy_derivative(suffix="_thumb", container="mkv"))
    )
    assert profile.output_paths(Path("/data/clip.mp4")) == [
        Path("/data/clip.mp4"),
        Path("/data/clip_preview.mp4"),
        Path("/data/clip_thumb.mkv"),
    ]


def test_replace_preserves_derivatives():
    """with_setparams and the CFR clause both go through replace(video_filters=...)."""
    profile = OFFLINE_8BIT.replace(derivatives=(_dummy_derivative(),))
    assert profile.replace(video_filters="null").derivatives == profile.derivatives


def test_with_setparams_preserves_derivatives():
    probe_json = {"streams": [{"pix_fmt": "yuv420p"}]}
    profile = OFFLINE_8BIT.replace(derivatives=(_dummy_derivative(),))
    assert with_setparams(profile, probe_json).derivatives == profile.derivatives


# ---------------------------------------------------------------------------
# Source conditioning vs the encoding chain
# ---------------------------------------------------------------------------


def test_vf_is_conditioning_then_chain():
    """A single-output profile emits the two zones concatenated, which is what
    it emitted before they were separate fields."""
    profile = OFFLINE_8BIT.replace(source_filters="COND", video_filters="CHAIN")
    assert profile.ffmpeg_graph_args() == ["-vf", "COND,CHAIN"]


def test_vf_omits_the_separator_when_there_is_no_conditioning():
    assert OFFLINE_8BIT.ffmpeg_graph_args() == ["-vf", OFFLINE_8BIT.video_filters]


def test_prepend_conditioning_stacks_at_the_head():
    profile = OFFLINE_8BIT.replace(source_filters="B").prepend_conditioning("A")
    assert profile.source_filters == "A,B"


def test_prepend_conditioning_onto_nothing_adds_no_separator():
    assert OFFLINE_8BIT.prepend_conditioning("A").source_filters == "A"


def test_conditioning_precedes_the_source_split():
    """The point of the split field: a source tap reads the conditioned source,
    not the raw decode."""
    profile = OFFLINE_8BIT.replace(
        source_filters="COND",
        video_filters="CHAIN",
        derivatives=(_dummy_derivative(filters="OWN", tap="source"),),
    )
    assert profile.ffmpeg_graph_args()[1] == "[0:v]COND,split=2[chain][d0];[chain]CHAIN[main];[d0]OWN[d0out]"


def test_source_tap_inherits_range_override():
    """A hand-written setparams in a derivative's own chain could not see
    range_override, so the archive and the still would disagree on black level
    for exactly the sources the override exists for."""
    probe_json = _probe_json(pix_fmt="yuv420p")
    profile = OFFLINE_8BIT.replace(video_filters="CHAIN", derivatives=(_dummy_derivative(filters="OWN", tap="source"),))
    graph = with_setparams(profile, probe_json, range_override="tv").ffmpeg_graph_args()[1]
    conditioning, _, branches = graph.partition("split=2")
    assert "range=tv" in conditioning
    assert "setparams" not in branches  # the derivative restates nothing


def test_conditioning_reaches_both_taps():
    profile = OFFLINE_8BIT.replace(
        source_filters="COND",
        video_filters="CHAIN",
        derivatives=(
            _dummy_derivative(suffix="_a", filters="TAIL"),
            _dummy_derivative(suffix="_b", filters="OWN", tap="source"),
        ),
    )
    assert profile.ffmpeg_graph_args()[1] == (
        "[0:v]COND,split=2[chain][d1];[chain]CHAIN,split=2[main][d0];[d0]TAIL[d0out];[d1]OWN[d1out]"
    )


# ---------------------------------------------------------------------------
# Derivative tap point
# ---------------------------------------------------------------------------


def test_tap_defaults_to_shared():
    assert _dummy_derivative().tap == "shared"


def test_source_tap_splits_ahead_of_the_shared_chain():
    profile = OFFLINE_8BIT.replace(
        video_filters="SHARED", derivatives=(_dummy_derivative(filters="OWN", tap="source"),)
    )
    assert profile.ffmpeg_graph_args()[1] == "[0:v]split=2[chain][d0];[chain]SHARED[main];[d0]OWN[d0out]"


def test_source_tap_alone_needs_no_second_split():
    """With nothing tapping the shared chain, its output goes straight to [main]."""
    profile = OFFLINE_8BIT.replace(
        video_filters="SHARED", derivatives=(_dummy_derivative(filters="OWN", tap="source"),)
    )
    assert ",split=" not in profile.ffmpeg_graph_args()[1]


def test_mixed_taps_split_at_both_points():
    profile = OFFLINE_8BIT.replace(
        video_filters="SHARED",
        derivatives=(
            _dummy_derivative(suffix="_a", filters="TAIL"),
            _dummy_derivative(suffix="_b", filters="OWN", tap="source"),
        ),
    )
    assert profile.ffmpeg_graph_args()[1] == (
        "[0:v]split=2[chain][d1];[chain]SHARED,split=2[main][d0];[d0]TAIL[d0out];[d1]OWN[d1out]"
    )
    assert [g[1] for g in profile.ffmpeg_output_groups()] == ["[main]", "[d0out]", "[d1out]"]


def test_several_source_taps_share_one_split():
    profile = OFFLINE_8BIT.replace(
        video_filters="SHARED",
        derivatives=(
            _dummy_derivative(suffix="_a", filters="A", tap="source"),
            _dummy_derivative(suffix="_b", filters="B", tap="source"),
        ),
    )
    assert profile.ffmpeg_graph_args()[1].startswith("[0:v]split=3[chain][d0][d1];")


def test_shared_tap_graph_is_unchanged_by_tap_support():
    """Adding source taps must not perturb the graph a shared-only profile emits."""
    profile = OFFLINE_8BIT.replace(video_filters="SHARED", derivatives=(_dummy_derivative(filters="TAIL"),))
    assert profile.ffmpeg_graph_args()[1] == "[0:v]SHARED,split=2[main][d0];[d0]TAIL[d0out]"


@ffmpeg_required
def test_archive_preview_and_poster_from_one_invocation(tmp_path: Path) -> None:
    """One transcode writes archive, preview and an sRGB still accurate to two codes.

    Converting the archive's BT.709 output to sRGB after the fact measures worse
    than not converting at all, because zimg treats BT.709 as BT.1886.  Tapping
    the conditioned source and encoding sRGB from linear light is what makes the
    still match the video a viewer sees beside it -- and the still needs no
    setparams of its own, because the conditioning runs ahead of its tap.
    """

    def srgb(linear: float) -> float:
        return 12.92 * linear if linear <= 0.0031308 else 1.055 * linear ** (1 / 2.4) - 0.055

    width = height = 64
    bands = 16
    levels = [16 * i for i in range(bands)]
    plane = np.repeat(np.array(levels, dtype=np.uint8), height // bands)[:, None].repeat(width, 1)

    raw = tmp_path / "src.yuv"
    with raw.open("wb") as handle:
        for _ in range(100):
            handle.write(plane.tobytes())
            handle.write(np.full((height // 2, width // 2), 128, dtype=np.uint8).tobytes() * 2)
    src = tmp_path / "src.avi"
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
            f"{width}x{height}",
            "-framerate",
            "500",
            "-i",
            str(raw),
            "-c:v",
            "mpeg4",
            "-q:v",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ],
        check=True,
    )

    profile = OFFLINE_8BIT.replace(codec_params=("-preset", "ultrafast", "-crf", "18"))
    dst = tmp_path / "v.mp4"
    # One invocation, three outputs: archive, decimated preview, sRGB still.
    assert transcode_video(src, dst, profile=profile, preview_fps=30.0, poster_at_seconds=0.1) == dst
    assert (tmp_path / "v_preview.mp4").exists()

    jpg = dst.with_name("v_poster.jpg")
    assert jpg.exists()
    gray = tmp_path / "poster.gray"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(jpg),
            "-vf",
            "extractplanes=y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            str(gray),
        ],
        check=True,
    )
    decoded = np.fromfile(gray, dtype=np.uint8)[: width * height].reshape(height, width)
    for i, level in enumerate(levels):
        expected = round(255 * srgb(level / 255))
        got = int(decoded[i * (height // bands) + 1, width // 2])
        assert abs(got - expected) <= 2, f"band {i}: sRGB {expected}, poster {got}"


# ---------------------------------------------------------------------------
# with_preview
# ---------------------------------------------------------------------------


def _preview_of(rate: str, **kwargs) -> Derivative:
    return with_preview(OFFLINE_8BIT, {"streams": [{"r_frame_rate": rate}]}, **kwargs).derivatives[0]


def _decimation(rate: str, **kwargs) -> tuple[int, Fraction]:
    return preview_decimation({"streams": [{"r_frame_rate": rate}]}, **kwargs)


@pytest.mark.parametrize(
    ("rate", "factor", "preview_fps"),
    [
        ("500/1", 20, 25),  # the AIND behavior rate; 20 is its only in-band whole divisor
        ("240/1", 8, 30),  # lands exactly on the target
        ("120/1", 4, 30),
        ("60/1", 2, 30),
        ("300/1", 10, 30),  # 30 and 25 are both whole; the target breaks the tie
        ("600/1", 20, 30),
        ("1000/1", 40, 25),
    ],
)
def test_decimation_prefers_a_whole_preview_rate(rate, factor, preview_fps):
    assert _decimation(rate) == (factor, Fraction(preview_fps))


@pytest.mark.parametrize("rate", ["500/1", "499/1", "997/1", "240/1", "1000/1", "120/1"])
def test_decimation_stays_inside_the_band(rate):
    """The band is a hard constraint, and it is what makes the whole-rate
    preference safe: unbounded, 499 fps would decimate to 1 fps."""
    _, preview_fps = _decimation(rate)
    assert 25 <= preview_fps <= 35


def test_decimation_ranks_by_target_when_no_whole_rate_is_available():
    factor, preview_fps = _decimation("499/1")
    assert factor == 17
    assert preview_fps.denominator != 1


@pytest.mark.parametrize("rate", ["25/1", "30000/1001", "10/1"])
def test_decimation_keeps_every_frame_at_or_below_the_band(rate):
    """Dropping frames cannot speed a video up, so slow sources get factor 1."""
    assert _decimation(rate)[0] == 1


def test_decimation_respects_a_custom_band():
    """Excluding 25 forces 500 fps off its whole-rate factor onto 500/17."""
    assert _decimation("500/1", target_fps=30.0, fps_band=(28.0, 35.0))[0] == 17


def test_decimation_falls_back_out_of_band_rather_than_refusing():
    factor, preview_fps = _decimation("500/1", target_fps=30.0, fps_band=(29.9, 30.1))
    assert factor == 17
    assert not 29.9 <= preview_fps <= 30.1


def test_decimation_rejects_target_outside_band():
    with pytest.raises(ValueError, match="falls outside"):
        _decimation("500/1", target_fps=60.0)


def test_decimation_rejects_unordered_band():
    with pytest.raises(ValueError, match="positive"):
        _decimation("500/1", target_fps=30.0, fps_band=(35.0, 25.0))


def test_decimation_raises_without_readable_rate():
    with pytest.raises(RuntimeError, match="r_frame_rate"):
        preview_decimation({"streams": [{}]})


def test_with_preview_filters_match_the_selected_factor():
    factor, _ = _decimation("500/1")
    assert _preview_of("500/1").filters == f"select=not(mod(n\\,{factor}))"


@pytest.mark.parametrize("rate", ["25/1", "30000/1001", "10/1"])
def test_with_preview_omits_select_when_source_is_at_or_below_the_band(rate):
    """A factor of 1 keeps every frame, so the chain segment is dropped entirely."""
    assert _preview_of(rate).filters == ""


def test_with_preview_gop_is_two_seconds_of_preview_frames():
    params = _preview_of("500/1").codec_params
    assert params[params.index("-g") + 1] == "50"


def test_with_preview_defaults_to_passthrough_fps_mode():
    assert _preview_of("500/1").fps_mode == "passthrough"


def test_with_preview_tags_colour_and_faststart():
    assert _preview_of("500/1").output_flags == ("-movflags", "+faststart+write_colr")


def test_with_preview_inherits_profile_metadata():
    assert _preview_of("500/1").metadata == OFFLINE_8BIT.metadata


def test_with_preview_raises_without_readable_rate():
    with pytest.raises(RuntimeError, match="r_frame_rate"):
        with_preview(OFFLINE_8BIT, {"streams": [{}]})


def test_with_preview_rejects_nonpositive_target():
    with pytest.raises(ValueError, match="must be positive"):
        _preview_of("500/1", target_fps=0.0)


def test_with_preview_rejects_duplicate_suffix():
    once = with_preview(OFFLINE_8BIT, {"streams": [{"r_frame_rate": "500/1"}]})
    with pytest.raises(ValueError, match="same path"):
        with_preview(once, {"streams": [{"r_frame_rate": "500/1"}]})


def test_with_preview_appends_to_existing_derivatives():
    base = OFFLINE_8BIT.replace(derivatives=(_dummy_derivative(suffix="_other"),))
    grown = with_preview(base, {"streams": [{"r_frame_rate": "500/1"}]})
    assert [d.suffix for d in grown.derivatives] == ["_other", "_preview"]


# ---------------------------------------------------------------------------
# with_poster
# ---------------------------------------------------------------------------


def _poster_of(rate: str = "500/1", nb_frames: str | None = None, **kwargs) -> Derivative:
    stream: dict = {"r_frame_rate": rate}
    if nb_frames is not None:
        stream["nb_frames"] = nb_frames
    return with_poster(OFFLINE_8BIT, {"streams": [stream]}, **kwargs).derivatives[0]


def _poster_frame(derivative: Derivative) -> int:
    match = re.search(r"eq\(n\\,(\d+)\)", derivative.filters)
    assert match is not None, derivative.filters
    return int(match.group(1))


@pytest.mark.parametrize(("at_seconds", "frame"), [(1.0, 500), (0.1, 50), (0.0, 0), (2.5, 1250)])
def test_with_poster_turns_seconds_into_a_frame_index(at_seconds, frame):
    assert _poster_frame(_poster_of(nb_frames="42000", at_seconds=at_seconds)) == frame


def test_with_poster_clamps_to_the_last_frame():
    """Past the end, select would match nothing and ffmpeg would write no still
    while still exiting zero."""
    assert _poster_frame(_poster_of(nb_frames="30", at_seconds=1.0)) == 29


def test_with_poster_cannot_clamp_without_a_frame_count():
    assert _poster_frame(_poster_of(at_seconds=1.0)) == 500


def test_with_poster_taps_the_conditioned_source():
    assert _poster_of().tap == "source"


def test_with_poster_carries_no_setparams_of_its_own():
    """The conditioning upstream of the tap supplies it, range_override included."""
    assert "setparams" not in _poster_of().filters


def test_with_poster_encodes_srgb_from_linear():
    filters = _poster_of().filters
    assert filters.startswith("select=")  # select first, so the colour work runs on one frame
    assert filters.endswith("zscale=t=iec61966-2-1:r=full")


def test_with_poster_writes_one_image():
    poster = _poster_of()
    assert poster.output_flags == ("-frames:v", "1", "-update", "1")
    assert poster.codec == "mjpeg"
    assert poster.container == "jpg"


def test_with_poster_rejects_negative_seconds():
    with pytest.raises(ValueError, match="must not be negative"):
        _poster_of(at_seconds=-1.0)


def test_with_poster_rejects_duplicate_suffix():
    once = with_poster(OFFLINE_8BIT, {"streams": [{"r_frame_rate": "500/1"}]})
    with pytest.raises(ValueError, match="same path"):
        with_poster(once, {"streams": [{"r_frame_rate": "500/1"}]})


def test_with_poster_raises_without_readable_rate():
    with pytest.raises(RuntimeError, match="r_frame_rate"):
        with_poster(OFFLINE_8BIT, {"streams": [{}]})


def test_preview_and_poster_compose():
    probe_json = {"streams": [{"r_frame_rate": "500/1", "nb_frames": "42000"}]}
    profile = with_poster(with_preview(OFFLINE_8BIT, probe_json), probe_json)
    assert [d.suffix for d in profile.derivatives] == ["_preview", "_poster"]
    assert [d.tap for d in profile.derivatives] == ["shared", "source"]
    assert profile.output_paths(Path("/d/video.mp4")) == [
        Path("/d/video.mp4"),
        Path("/d/video_preview.mp4"),
        Path("/d/video_poster.jpg"),
    ]


# ---------------------------------------------------------------------------
# transcode_video with a preview attached
# ---------------------------------------------------------------------------


def _filter_complex(cmd: list) -> str:
    return cmd[cmd.index("-filter_complex") + 1]


def test_preview_command_uses_filter_complex_not_vf(monkeypatch, tmp_path):
    captured = _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS, rate="500/1")
    transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4", preview_fps=25.0)
    assert "-vf" not in captured[0]
    assert "split=2[main][d0]" in _filter_complex(captured[0])


def test_preview_command_keeps_cfr_setpts_at_the_head_of_the_shared_chain(monkeypatch, tmp_path):
    captured = _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS, rate="500/1")
    transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4", preview_fps=25.0, normalize_cfr=True)
    assert _filter_complex(captured[0]).startswith("[0:v]setpts=N/(500/1)/TB,")


def test_preview_command_writes_both_paths(monkeypatch, tmp_path):
    captured = _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS, rate="500/1")
    transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4", preview_fps=25.0)
    assert captured[0][-1] == str(tmp_path / "out_preview.mp4")
    assert str(tmp_path / "out.mp4") in captured[0]


def test_preview_command_strips_audio_from_every_output(monkeypatch, tmp_path):
    captured = _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS, rate="500/1")
    transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4", preview_fps=25.0, no_audio=True)
    assert captured[0].count("-an") == 2


def test_transcode_video_returns_primary_path_only(monkeypatch, tmp_path):
    _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS, rate="500/1")
    assert transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4", preview_fps=25.0) == tmp_path / "out.mp4"


def test_no_preview_by_default(monkeypatch, tmp_path):
    captured = _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS, rate="500/1")
    transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4")
    assert "-filter_complex" not in captured[0]
    assert "-vf" in captured[0]


def test_frame_check_reads_the_primary_output_only(monkeypatch, tmp_path):
    _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS, stderr_bytes=_summary(500, 500, 25), rate="500/1")
    assert transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4", preview_fps=25.0) == tmp_path / "out.mp4"


def test_non_passthrough_derivative_passes_the_frame_check(monkeypatch, tmp_path):
    """Totals are per output, so a derivative's fps_mode cannot indict the primary encode."""
    stderr = _summary(100, 100, 250)
    captured = _patch_ffmpeg(monkeypatch, stdout_bytes=_CLEAN_PROGRESS, stderr_bytes=stderr, rate="500/1")
    profile = OFFLINE_8BIT.replace(derivatives=(_dummy_derivative(fps_mode="cfr"),))
    transcode_video(tmp_path / "in.avi", tmp_path / "out.mp4", profile=profile)
    assert "-fps_mode" in captured[0]


@ffmpeg_required
def test_preview_is_decimated_and_frame_aligned(tmp_path: Path) -> None:
    """A 500 fps source yields a 25 fps preview whose frame k is source frame 20k.

    Covers the whole mechanism end-to-end: ffmpeg accepts the split graph, the
    escaped comma in ``mod(n\\,20)`` parses, ``fps_mode=passthrough`` rules out a
    CFR stage duplicating the retained frames back up to 500 fps, and the frame
    check reads the archive's totals rather than the preview's.
    """
    src = tmp_path / "src.avi"
    dst = tmp_path / "out.mp4"
    preview = tmp_path / "out_preview.mp4"
    # Widely separated luma so a one-frame misalignment cannot hide inside
    # encoder noise: adjacent retained frames differ by >= 20 after the chain.
    _encode_untagged_yuv420p(src, [30 + 2 * n for n in range(100)], framerate=500)

    fast = OFFLINE_8BIT.replace(codec_params=("-preset", "ultrafast", "-crf", "18"))
    assert transcode_video(src, dst, profile=fast, preview_fps=25.0) == dst
    assert preview.exists()

    def _probe_stream(path: Path) -> dict:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames,r_frame_rate,width,height,color_transfer",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(out.stdout)["streams"][0]

    main_info, preview_info = _probe_stream(dst), _probe_stream(preview)
    assert int(main_info["nb_read_frames"]) == 100
    assert int(preview_info["nb_read_frames"]) == 5
    assert preview_info["r_frame_rate"] == "25/1"
    # Geometry is untouched: frame rate, not resolution, is what blocks playback.
    assert (preview_info["width"], preview_info["height"]) == (main_info["width"], main_info["height"])
    # write_colr must tag the preview too, or it renders unlike the file it stands for.
    assert preview_info["color_transfer"] == main_info["color_transfer"] == "bt709"

    main_y = _decode_center_luma(dst, 100)
    preview_y = _decode_center_luma(preview, 5)
    for k, y in enumerate(preview_y):
        assert abs(y - main_y[20 * k]) <= 2, f"preview frame {k} is not source frame {20 * k}"
