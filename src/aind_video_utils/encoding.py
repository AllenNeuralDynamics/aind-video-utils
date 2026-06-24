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

from aind_video_utils.probe import (
    ProbeDict,
    get_color_primaries,
    get_color_range,
    get_color_space,
    get_color_transfer,
    get_yuv_format,
)

SPEC_VERSION: str = "1.0"
"""Tracks which revision of the aind-file-standards behavior video spec
the profiles implement.  Independent of the package version."""

# ---------------------------------------------------------------------------
# Setparams filter — fill missing color metadata only
# ---------------------------------------------------------------------------
#
# AIND sources carry partial or no color metadata in the bitstream:
#   - h264 gbrp files (camera-linear RGB) tag color_range=pc and color_space=gbr;
#     color_trc and color_primaries are absent.
#   - mpeg4 yuv420p files carry NO color metadata at all (the bitstream has no
#     VUI for it); the underlying RGB→YUV matrix is whatever libswscale used
#     when Bonsai/ffmpeg created the file, which empirically is smpte170m
#     (BT.601) — the universal default for untagged conversion.
#
# Defensive defaults injected here (only for fields the source doesn't tag):
#   color_primaries=bt709    — Modern camera sensors, and ffmpeg doesn't apply
#                              a primary rotation during RGB→YUV. Declaring
#                              bt709 means "no primary rotation" downstream.
#   color_trc=linear         — Bonsai stores raw linear scene light (no OETF
#                              applied at capture time).
#   colorspace=gbr / smpte170m — Match the bitstream truth: gbr for RGB-planar
#                              sources; smpte170m (BT.601) for untagged YUV,
#                              the matrix libswscale uses by default.
#   range=pc                 — Bonsai stores full-range Y (luma reaches 0 and
#                              255 in production files, incompatible with
#                              tv-range floor at 16). Without this, untagged
#                              yuv420p defaults to tv-range and crushes
#                              shadows/highlights at the chain's first scale.


def setparams_filter_for_source(probe_json: ProbeDict) -> str | None:
    """Build a ``setparams`` filter string with only the fields missing on the source.

    Returns ``None`` if the source has color_primaries, color_trc, color_space,
    and color_range all tagged.  Otherwise returns ``setparams=<a=b:c=d:...>``
    for use as a leading filter in the chain.

    Defaults follow the AIND Bonsai capture conventions documented above.
    """
    parts: list[str] = []
    if get_color_primaries(probe_json) is None:
        parts.append("color_primaries=bt709")
    if get_color_transfer(probe_json) is None:
        parts.append("color_trc=linear")
    if get_color_space(probe_json) is None:
        pix_fmt = get_yuv_format(probe_json)
        if pix_fmt and pix_fmt.startswith("gbr"):
            parts.append("colorspace=gbr")
        else:
            parts.append("colorspace=smpte170m")
    if get_color_range(probe_json) is None:
        parts.append("range=pc")
    if not parts:
        return None
    return f"setparams={':'.join(parts)}"


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
        "colorspace=all=bt709:dither=none,"
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
    video_filters=("colorspace=all=bt709:dither=none,scale=out_range=tv:sws_dither=none,format=yuv420p10le"),
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


def with_setparams(profile: EncodingProfile, probe_json: ProbeDict | None = None) -> EncodingProfile:
    """Prepend a ``setparams`` colour-metadata filter to *profile*.

    Parameters
    ----------
    profile : EncodingProfile
        Base profile to extend.
    probe_json : ProbeDict | None
        Result of :func:`aind_video_utils.probe.probe` on the source.  When
        provided, the prepended setparams clause includes only fields the
        source has tagged as missing (per :func:`setparams_filter_for_source`),
        respecting any color metadata the source already declares.  When
        ``None``, every field is set to the AIND default — useful when the
        caller knows the source is fully untagged or doesn't want to probe.

    Returns
    -------
    EncodingProfile
        A new profile with the setparams clause prepended to ``video_filters``,
        or the original profile unchanged when ``probe_json`` indicates the
        source already carries all four color fields.
    """
    if probe_json is not None:
        sp = setparams_filter_for_source(probe_json)
        if sp is None:
            return profile
    else:
        # No probe — default to filling every field. ``colorspace=smpte170m``
        # matches the bitstream truth for the typical untagged-yuv420p caller;
        # for gbrp callers without a probe, ``setparams`` is harmless metadata
        # (the scale step does RGB→YUV explicitly via ``out_color_matrix=``).
        sp = "setparams=color_primaries=bt709:color_trc=linear:colorspace=smpte170m:range=pc"
    return profile.replace(
        video_filters=f"{sp},{profile.video_filters}",
    )
