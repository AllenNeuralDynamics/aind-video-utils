"""Tests for encoding profiles and transcode module."""

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
from aind_video_utils.transcode import VIDEO_EXTENSIONS

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
        "scale=out_color_matrix=bt709:out_range=full:sws_dither=none,"
        "format=yuv420p10le,"
        "colorspace=ispace=bt709:all=bt709:dither=none,"
        "scale=out_range=tv:sws_dither=none,"
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
        "colorspace=ispace=bt709:all=bt709:dither=none,scale=out_range=tv:sws_dither=none,format=yuv420p10le",
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


def test_with_setparams_prepends_filter():
    modified = with_setparams(OFFLINE_8BIT)
    expected_prefix = "setparams=color_primaries=bt709:color_trc=linear:colorspace=bt709,"
    assert modified.video_filters.startswith(expected_prefix)
    assert modified.video_filters == expected_prefix + OFFLINE_8BIT.video_filters


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
# VIDEO_EXTENSIONS
# ---------------------------------------------------------------------------


def test_video_extensions_is_frozenset():
    assert isinstance(VIDEO_EXTENSIONS, frozenset)


def test_video_extensions_contains_expected():
    expected = {".avi", ".flv", ".mkv", ".mov", ".mp4", ".webm", ".wmv"}
    assert VIDEO_EXTENSIONS == expected
