"""FFprobe wrapper and video metadata accessors."""

from __future__ import annotations

import json
import math
import subprocess as sp
from pathlib import Path
from typing import Any

from aind_video_utils._rawvideo import pix_format_bit_depth
from aind_video_utils.utils import http_input_flags

ProbeDict = dict[str, Any]


def probe(video_path: str | Path) -> ProbeDict:
    """Run ffprobe on a video file and return the parsed JSON output.

    Parameters
    ----------
    video_path : str | Path
        Path to the video file.

    Returns
    -------
    ProbeDict
        Parsed ffprobe JSON containing stream and format information.
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        *http_input_flags(video_path),
        str(video_path),
    ]
    result = sp.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)  # type: ignore[no-any-return]


def get_yuv_format(probe_json: ProbeDict) -> str:
    """Return the pixel format string from the first video stream.

    Parameters
    ----------
    probe_json : ProbeDict
        Parsed ffprobe output.

    Returns
    -------
    str
        Pixel format, e.g. ``"yuv420p"`` or ``"gbrp"``.
    """
    return str(probe_json["streams"][0]["pix_fmt"])


def get_color_range(probe_json: ProbeDict) -> str | None:
    """Return the color range from the first video stream.

    Parameters
    ----------
    probe_json : ProbeDict
        Parsed ffprobe output.

    Returns
    -------
    str | None
        Color range string (``"pc"`` full, ``"tv"`` limited), or ``None`` when
        absent or ``"unknown"``.
    """
    color_range = probe_json["streams"][0].get("color_range")
    if color_range in (None, "unknown"):
        return None
    return str(color_range)


def get_color_space(probe_json: ProbeDict) -> str | None:
    """Return the colorspace (matrix) tag from the first video stream.

    Parameters
    ----------
    probe_json : ProbeDict
        Parsed ffprobe output.

    Returns
    -------
    str | None
        Matrix string (e.g. ``"bt709"``, ``"smpte170m"``, ``"gbr"``), or
        ``None`` when absent or ``"unknown"``.
    """
    color_space = probe_json["streams"][0].get("color_space")
    if color_space in (None, "unknown"):
        return None
    return str(color_space)


def get_color_primaries(probe_json: ProbeDict) -> str | None:
    """Return the color primaries from the first video stream.

    Parameters
    ----------
    probe_json : ProbeDict
        Parsed ffprobe output.

    Returns
    -------
    str | None
        Primaries string (e.g. ``"bt709"``, ``"smpte170m"``), or ``None`` when
        absent or ``"unknown"``.
    """
    color_primaries = probe_json["streams"][0].get("color_primaries")
    if color_primaries in (None, "unknown"):
        return None
    return str(color_primaries)


def get_frame_dimensions(probe_json: ProbeDict) -> tuple[int, int]:
    """Return the (width, height) of the first video stream.

    Parameters
    ----------
    probe_json : ProbeDict
        Parsed ffprobe output.

    Returns
    -------
    tuple[int, int]
        ``(width, height)`` in pixels.
    """
    vidstream = probe_json["streams"][0]
    return vidstream["width"], vidstream["height"]


def get_video_range_info(probe_json: ProbeDict) -> tuple[str, int]:
    """Return the color range and bit depth of the first video stream.

    Parameters
    ----------
    probe_json : ProbeDict
        Parsed ffprobe output.

    Returns
    -------
    color_range : str
        ``"pc"`` (full), ``"tv"`` (limited), or ``"unknown"`` when the source
        bitstream doesn't carry the tag (e.g., mpeg4, which has no VUI for
        color range).
    bit_depth : int
        Bits per component (8 or 10).
    """
    vidstream = probe_json["streams"][0]
    pix_fmt = vidstream["pix_fmt"]
    color_range = vidstream.get("color_range", "unknown")
    bit_depth = pix_format_bit_depth(pix_fmt)
    return color_range, bit_depth


def get_color_transfer(probe_json: ProbeDict) -> str | None:
    """Return the transfer characteristic from the first video stream.

    Parameters
    ----------
    probe_json : ProbeDict
        Parsed ffprobe output.

    Returns
    -------
    str | None
        Transfer characteristic string (e.g. ``"bt709"``, ``"linear"``),
        or ``None`` when absent or ``"unknown"``.
    """
    color_trc = probe_json["streams"][0].get("color_transfer")
    if color_trc in (None, "unknown"):
        return None
    return str(color_trc)


def get_duration_seconds(probe_json: ProbeDict) -> float | None:
    """Return the first video stream's duration in seconds.

    Tries the stream's ``duration`` field directly; falls back to
    ``nb_frames / r_frame_rate`` when only the frame count is known.

    Parameters
    ----------
    probe_json : ProbeDict
        Parsed ffprobe output.

    Returns
    -------
    float | None
        Duration in seconds, or ``None`` when unavailable.
    """
    stream = probe_json["streams"][0]
    raw = stream.get("duration")
    if raw is not None and raw != "N/A":
        try:
            return float(raw)
        except ValueError:
            pass
    nb = stream.get("nb_frames")
    if nb is not None and nb != "N/A" and stream.get("r_frame_rate"):
        try:
            num, den = stream["r_frame_rate"].split("/")
            fps = int(num) / int(den)
            return int(nb) / fps
        except (ValueError, ZeroDivisionError):
            pass
    return None


def get_r_frame_rate(probe_json: ProbeDict) -> tuple[int, int] | None:
    """Return the base frame rate ``(num, den)`` of the first video stream.

    Parses ffprobe's ``r_frame_rate`` field (the stream's real base frame rate,
    a rational like ``"500/1"`` or ``"30000/1001"``) into its numerator and
    denominator.  Returned as an exact fraction so callers can build a
    precision-preserving ``setpts=N/(num/den)/TB`` expression for CFR rates that
    aren't integers.

    Parameters
    ----------
    probe_json : ProbeDict
        Parsed ffprobe output.

    Returns
    -------
    tuple[int, int] | None
        ``(numerator, denominator)``, or ``None`` when the field is absent,
        unparsable, or degenerate (``"0/0"``).
    """
    rate = probe_json["streams"][0].get("r_frame_rate")
    if not rate or rate == "N/A":
        return None
    try:
        num_str, den_str = rate.split("/")
        num, den = int(num_str), int(den_str)
    except ValueError:
        return None
    if num <= 0 or den <= 0:
        return None
    return num, den


def get_nb_frames(probe_json: ProbeDict) -> int | None:
    """Return the frame count from the first video stream.

    Tries ``nb_frames`` directly, then falls back to
    ``duration * r_frame_rate``.

    Parameters
    ----------
    probe_json : ProbeDict
        Parsed ffprobe output.

    Returns
    -------
    int | None
        Total frame count, or ``None`` when unavailable.
    """
    stream = probe_json["streams"][0]
    raw = stream.get("nb_frames")
    if raw is not None and raw != "N/A":
        return int(raw)

    if stream.get("duration") and stream.get("r_frame_rate"):
        try:
            dur = float(stream["duration"])
            num, den = stream["r_frame_rate"].split("/")
            fps = int(num) / int(den)
            return math.ceil(dur * fps)
        except (ValueError, ZeroDivisionError):
            pass

    return None
