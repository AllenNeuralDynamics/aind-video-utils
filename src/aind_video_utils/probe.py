"""FFprobe wrapper and video metadata accessors."""

from __future__ import annotations

import json
import math
import subprocess as sp
from pathlib import Path
from typing import Any

from aind_video_utils._rawvideo import pix_format_bit_depth

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


def get_color_range(probe_json: ProbeDict) -> str:
    """Return the color range from the first video stream.

    Parameters
    ----------
    probe_json : ProbeDict
        Parsed ffprobe output.

    Returns
    -------
    str
        Color range, typically ``"pc"`` (full) or ``"tv"`` (limited).
    """
    return str(probe_json["streams"][0]["color_range"])


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
        ``"pc"`` (full) or ``"tv"`` (limited).
    bit_depth : int
        Bits per component (8 or 10).
    """
    vidstream = probe_json["streams"][0]
    pix_fmt = vidstream["pix_fmt"]
    color_range = vidstream["color_range"]
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
