"""Transcode videos using encoding profiles from the AIND behavior video spec.

Provides the single-video transcode function used by the ``aind-transcode``
CLI and available as a Python API.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from aind_video_utils.encoding import OFFLINE_8BIT, EncodingProfile, with_setparams
from aind_video_utils.probe import get_color_transfer, probe
from aind_video_utils.utils import http_input_flags

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".avi",
        ".flv",
        ".mkv",
        ".mov",
        ".mp4",
        ".webm",
        ".wmv",
    }
)


def transcode_video(
    input_path: Path,
    output_path: Path,
    *,
    profile: EncodingProfile = OFFLINE_8BIT,
    auto_fix_colorspace: bool = True,
    no_audio: bool = True,
    on_progress: Callable[[int], None] | None = None,
) -> Path:
    """Transcode a single video using an :class:`EncodingProfile`.

    Parameters
    ----------
    input_path : Path
        Source video file.
    output_path : Path
        Destination file.
    profile : EncodingProfile
        Encoding profile to use.  Defaults to :data:`OFFLINE_8BIT`.
    auto_fix_colorspace : bool
        When ``True`` (the default), probe the source and prepend a
        ``setparams`` filter if ``color_trc`` metadata is absent.
        Set to ``False`` for exact control over filters.
    no_audio : bool
        If ``True``, strip audio (``-an``).
    on_progress : Callable[[int], None] | None
        Called with the current frame number as ffmpeg reports progress.

    Returns
    -------
    Path
        *output_path* on success.

    Raises
    ------
    subprocess.CalledProcessError
        If ffmpeg exits with a non-zero return code.
    """
    effective = profile
    if auto_fix_colorspace:
        probe_json = probe(input_path)
        color_trc = get_color_transfer(probe_json)
        if color_trc is None:
            effective = with_setparams(profile)

    cmd: list[str] = ["ffmpeg"]
    cmd.extend(http_input_flags(input_path))
    cmd.extend(effective.ffmpeg_input_args())
    cmd.extend(["-i", str(input_path)])
    cmd.extend(effective.ffmpeg_output_args())

    if no_audio:
        cmd.append("-an")

    cmd.extend(
        [
            "-progress",
            "pipe:1",
            "-nostats",
            "-y",
            str(output_path),
        ]
    )

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None

    frame_prefix = b"frame="
    for raw_line in proc.stdout:
        if on_progress and raw_line.startswith(frame_prefix):
            try:
                on_progress(int(raw_line[6:].strip()))
            except ValueError:
                pass

    returncode = proc.wait()
    if returncode != 0:
        stderr = proc.stderr.read() if proc.stderr else b""
        raise subprocess.CalledProcessError(returncode, cmd, stderr=stderr)

    return output_path
