"""Shared utility helpers."""

from __future__ import annotations

import datetime
from pathlib import Path


def get_millisecond_string(seconds: float) -> str:
    """Convert a time in seconds to an ffmpeg-compatible millisecond string.

    Parameters
    ----------
    seconds : float
        Time value in seconds.

    Returns
    -------
    str
        Formatted string, e.g. ``"1500ms"``.
    """
    ms = datetime.timedelta(seconds=seconds) / datetime.timedelta(milliseconds=1)
    return f"{ms:f}".rstrip("0").rstrip(".") + "ms"


def http_input_flags(source: str | Path) -> list[str]:
    """Return ffmpeg/ffprobe protocol flags for HTTP(S) sources.

    For local file paths, returns an empty list.  For HTTP(S) URLs,
    returns flags for connection reuse and reconnection resilience.

    Parameters
    ----------
    source : str | Path
        Video path or URL.

    Returns
    -------
    list[str]
        Flags to splice before the input in an ffmpeg/ffprobe command.
    """
    if str(source).startswith(("http://", "https://")):
        return [
            "-reconnect",
            "1",
            "-reconnect_on_network_error",
            "1",
            "-reconnect_delay_max",
            "10",
            "-multiple_requests",
            "1",
        ]
    return []
