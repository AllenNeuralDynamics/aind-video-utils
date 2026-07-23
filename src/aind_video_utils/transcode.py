"""Transcode videos using encoding profiles from the AIND behavior video spec.

Provides the single-video transcode function used by the ``aind-transcode``
CLI and available as a Python API.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from aind_video_utils.encoding import OFFLINE_8BIT, EncodingProfile, RangeOverride, with_setparams
from aind_video_utils.probe import get_r_frame_rate, probe
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


def _effective_profile(
    profile: EncodingProfile,
    input_path: Path,
    *,
    auto_fix_colorspace: bool,
    range_override: RangeOverride | None,
    normalize_cfr: bool,
) -> EncodingProfile:
    """Apply the probe-driven filter adjustments to *profile*.

    Prepends the per-source ``setparams`` colour clause (when
    ``auto_fix_colorspace``) and the CFR-normalizing ``setpts`` clause (when
    ``normalize_cfr``).  Probes the source at most once.
    """
    effective = profile
    probe_json = probe(input_path) if (auto_fix_colorspace or normalize_cfr) else None
    if auto_fix_colorspace:
        assert probe_json is not None
        effective = with_setparams(profile, probe_json, range_override=range_override)
    if normalize_cfr:
        assert probe_json is not None
        rate = get_r_frame_rate(probe_json)
        if rate is None:
            raise RuntimeError(
                f"normalize_cfr=True but {input_path} has no readable r_frame_rate; "
                "pass normalize_cfr=False for variable-frame-rate sources."
            )
        num, den = rate
        setpts = f"setpts=N/({num}/{den})/TB"
        effective = effective.replace(video_filters=f"{setpts},{effective.video_filters}")
    return effective


def _run_ffmpeg(cmd: list[str], *, on_progress: Callable[[int], None] | None) -> tuple[int, int]:
    """Run *cmd*, forwarding frame progress, and return ``(drop, dup)`` counts.

    ffmpeg's ``-progress`` stream emits cumulative ``key=value`` lines; the last
    value seen for each counter is the running total.  ``drop_frames`` /
    ``dup_frames`` report what the implicit vsync stage did to the frame count.

    Raises
    ------
    subprocess.CalledProcessError
        If ffmpeg exits with a non-zero return code.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None

    drop_frames = 0
    dup_frames = 0
    for raw_line in proc.stdout:
        if raw_line.startswith(b"frame="):
            if on_progress:
                try:
                    on_progress(int(raw_line[6:].strip()))
                except ValueError:
                    pass
        elif raw_line.startswith(b"drop_frames="):
            try:
                drop_frames = int(raw_line.split(b"=", 1)[1].strip())
            except ValueError:
                pass
        elif raw_line.startswith(b"dup_frames="):
            try:
                dup_frames = int(raw_line.split(b"=", 1)[1].strip())
            except ValueError:
                pass

    returncode = proc.wait()
    if returncode != 0:
        stderr = proc.stderr.read() if proc.stderr else b""
        raise subprocess.CalledProcessError(returncode, cmd, stderr=stderr)

    return drop_frames, dup_frames


def transcode_video(
    input_path: Path,
    output_path: Path,
    *,
    profile: EncodingProfile = OFFLINE_8BIT,
    auto_fix_colorspace: bool = True,
    range_override: RangeOverride | None = None,
    normalize_cfr: bool = True,
    fail_on_frame_drop: bool = True,
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
        ``setparams`` filter that fills in only the color-metadata fields the
        source has tagged as missing.  Set to ``False`` for exact control
        over filters (no setparams prepended at all).
    range_override : {"pc", "tv"} | None
        When set, force the ``range=`` field of the prepended setparams clause
        to this value, overriding both the default ``range=pc`` and any range
        tag carried by the source.  Use ``"tv"`` for AIND mpeg4 yuv420p
        sources that are TV-range encoded.  Ignored when
        ``auto_fix_colorspace=False``.
    normalize_cfr : bool
        When ``True`` (the default), probe the source frame rate and prepend a
        ``setpts=N/(num/den)/TB`` filter that re-stamps every frame's
        presentation timestamp from its display-order index at the source's
        base frame rate.  This yields a clean constant-frame-rate timeline
        starting at PTS 0 and — critically — prevents ffmpeg's implicit vsync
        stage from silently dropping frames whose *reconstructed* timestamps
        are non-monotonic.  That reconstruction failure is common for
        h264-in-AVI sources (AVI cannot store composition offsets, so a
        declared reorder/DPB delay makes the leading frames look
        "in the past"), where the naive path drops the first several frames.
        Assumes a constant-frame-rate source with a readable ``r_frame_rate``;
        set to ``False`` for genuinely variable-frame-rate input whose original
        timing must be preserved.
    fail_on_frame_drop : bool
        When ``True`` (the default), raise :class:`RuntimeError` if ffmpeg's
        progress stream reports any dropped or duplicated frames
        (``drop_frames``/``dup_frames``), i.e. the output is not a frame-exact
        copy of the decoded source.  This turns ffmpeg's silent
        ``*** dropping frame`` warning into a hard error.  Set to ``False`` when
        legitimately resampling variable-frame-rate input to CFR (where
        duplicated frames are expected).
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
    RuntimeError
        If ``fail_on_frame_drop`` is set and ffmpeg dropped or duplicated
        frames, or if ``normalize_cfr`` is set but the source has no readable
        base frame rate.
    """
    effective = _effective_profile(
        profile,
        input_path,
        auto_fix_colorspace=auto_fix_colorspace,
        range_override=range_override,
        normalize_cfr=normalize_cfr,
    )

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

    drop_frames, dup_frames = _run_ffmpeg(cmd, on_progress=on_progress)

    if fail_on_frame_drop and (drop_frames or dup_frames):
        raise RuntimeError(
            f"ffmpeg altered the frame stream transcoding {input_path} "
            f"(drop_frames={drop_frames}, dup_frames={dup_frames}); {output_path} "
            "is not a frame-exact copy of the decoded source. This usually means "
            "the source has non-monotonic timestamps (e.g. h264-in-AVI); "
            "normalize_cfr=True should prevent it. Pass fail_on_frame_drop=False "
            "to allow non-frame-exact output."
        )

    return output_path
