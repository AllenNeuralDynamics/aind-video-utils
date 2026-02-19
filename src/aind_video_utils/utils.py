"""Shared utility helpers."""

import datetime


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
