"""Encoding profiles for the AIND behavior video file standard.

Defines frozen dataclass profiles that map 1-to-1 with the encoding profiles
in the `aind-file-standards behavior video spec
<https://allenneuraldynamics.github.io/aind-file-standards/file_formats/behavior_videos/>`_.

Each profile bundles every ffmpeg flag needed to produce a compliant file.
Select a preset constant and optionally customise it with
:meth:`EncodingProfile.replace`::

    from aind_video_utils.encoding import OFFLINE_8BIT
    fast = OFFLINE_8BIT.replace(codec_params=("-preset", "veryfast", "-crf", "18"))
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

SPEC_VERSION: str = "1.0"
"""Tracks which revision of the aind-file-standards behavior video spec
the profiles implement.  Independent of the package version."""

# ---------------------------------------------------------------------------
# Setparams filter for sources missing colour metadata (used by auto-fix)
# ---------------------------------------------------------------------------

_SETPARAMS = "setparams=color_primaries=bt709:color_trc=linear:colorspace=bt709"


@dataclass(frozen=True)
class EncodingProfile:
    """Immutable bundle of ffmpeg parameters for a single encoding profile.

    Parameters
    ----------
    video_filters : str
        Value for ``-vf``.
    codec : str
        Value for ``-c:v``.
    pixel_format : str
        Value for ``-pix_fmt``.
    container : str
        File extension without dot (``"mp4"``, ``"mkv"``).
    codec_params : tuple[str, ...]
        Rate-control, tuning, preset flags (e.g. ``("-crf", "18")``).
    input_flags : tuple[str, ...]
        Flags placed before ``-i``.
    output_flags : tuple[str, ...]
        Container/muxer flags placed after codec options.
    metadata : tuple[tuple[str, str], ...]
        ``-metadata key=value`` pairs to embed.
    """

    video_filters: str
    codec: str
    pixel_format: str
    container: str
    codec_params: tuple[str, ...] = ()
    input_flags: tuple[str, ...] = ()
    output_flags: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def replace(self, **kwargs: Any) -> EncodingProfile:
        """Return a copy with the given fields replaced."""
        return dataclasses.replace(self, **kwargs)

    def ffmpeg_input_args(self) -> list[str]:
        """Return the argument list to insert before ``-i``."""
        return list(self.input_flags)

    def ffmpeg_output_args(self) -> list[str]:
        """Return the argument list to insert after ``-i <input>``.

        Order: ``-vf``, ``-c:v``, ``-pix_fmt``, codec params, metadata,
        output flags.
        """
        args: list[str] = [
            "-vf",
            self.video_filters,
            "-c:v",
            self.codec,
            "-pix_fmt",
            self.pixel_format,
        ]
        args.extend(self.codec_params)
        for key, value in self.metadata:
            args.extend(["-metadata", f"{key}={value}"])
        args.extend(self.output_flags)
        return args


# ---------------------------------------------------------------------------
# Canonical profiles — directly from the spec
# ---------------------------------------------------------------------------

_AIND_METADATA: tuple[tuple[str, str], ...] = (("author", "Allen Institute for Neural Dynamics"),)

OFFLINE_8BIT = EncodingProfile(
    video_filters=(
        "scale=out_color_matrix=bt709:out_range=full:sws_dither=none,"
        "format=yuv420p10le,"
        "colorspace=ispace=bt709:all=bt709:dither=none,"
        "scale=out_range=tv:sws_dither=none,"
        "format=yuv420p"
    ),
    codec="libx264",
    pixel_format="yuv420p",
    container="mp4",
    codec_params=("-preset", "veryslow", "-crf", "18"),
    input_flags=(),
    output_flags=("-movflags", "+faststart+write_colr"),
    metadata=_AIND_METADATA,
)

OFFLINE_10BIT = EncodingProfile(
    video_filters=(
        "colorspace=ispace=bt709:all=bt709:dither=none,scale=out_range=tv:sws_dither=none,format=yuv420p10le"
    ),
    codec="libx264",
    pixel_format="yuv420p10le",
    container="mp4",
    codec_params=("-preset", "veryslow", "-crf", "18"),
    input_flags=(),
    output_flags=("-movflags", "+faststart+write_colr"),
    metadata=_AIND_METADATA,
)

ONLINE_8BIT = EncodingProfile(
    video_filters=("scale=out_range=full,setparams=range=full:colorspace=bt709:color_primaries=bt709:color_trc=linear"),
    codec="h264_nvenc",
    pixel_format="yuv420p",
    container="mkv",
    codec_params=("-tune", "hq", "-preset", "p3", "-rc", "vbr", "-cq", "18", "-b:v", "0M"),
    input_flags=(
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_range",
        "full",
        "-color_trc",
        "linear",
    ),
    output_flags=(
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
    ),
    metadata=_AIND_METADATA,
)

ONLINE_10BIT = EncodingProfile(
    video_filters=(
        "format=yuv420p10le,"
        "scale=out_range=full,"
        "setparams=range=full:colorspace=bt709:color_primaries=bt709:color_trc=linear"
    ),
    codec="hevc_nvenc",
    pixel_format="p010le",
    container="mkv",
    codec_params=("-tune", "hq", "-preset", "p4", "-rc", "vbr", "-cq", "12", "-b:v", "0M"),
    input_flags=(),
    output_flags=(
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
    ),
    metadata=_AIND_METADATA,
)

# ---------------------------------------------------------------------------
# Profile lookup by name (used by CLI)
# ---------------------------------------------------------------------------

PROFILES: dict[str, EncodingProfile] = {
    "offline-8bit": OFFLINE_8BIT,
    "offline-10bit": OFFLINE_10BIT,
    "online-8bit": ONLINE_8BIT,
    "online-10bit": ONLINE_10BIT,
}


def with_setparams(profile: EncodingProfile) -> EncodingProfile:
    """Prepend the ``setparams`` colour-metadata filter to *profile*.

    Useful for sources missing ``color_trc`` metadata.  Returns a new profile
    via :meth:`EncodingProfile.replace`.
    """
    return profile.replace(
        video_filters=f"{_SETPARAMS},{profile.video_filters}",
    )
