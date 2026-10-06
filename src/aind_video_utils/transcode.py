"""Transcode videos using encoding profiles from the AIND behavior video spec.

Provides the single-video transcode function used by the ``aind-transcode``
CLI and available as a Python API.
"""

from __future__ import annotations

import re
import subprocess
import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aind_video_utils.encoding import (
    OFFLINE_8BIT,
    EncodingProfile,
    RangeOverride,
    preview_decimation,
    with_poster,
    with_preview,
    with_setparams,
)
from aind_video_utils.preview_metadata import PREVIEW_METADATA_FILENAME, read_frame_metadata, write_decimated
from aind_video_utils.probe import get_duration_seconds, get_exact_nb_frames, get_r_frame_rate, probe
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

# ffmpeg logs its end-of-run per-stream totals only at verbose, and the level
# tag on every line keeps warnings and errors separable from that chatter.
_FFMPEG_LOG_ARGS: tuple[str, ...] = ("-hide_banner", "-loglevel", "level+verbose")
_ENCODED_RE = re.compile(rb"Output stream #(\d+):\d+ \(video\): (\d+) frames encoded")
_DECODED_RE = re.compile(
    rb"Input stream #\d+:\d+ \(video\): \d+ packets read \(\d+ bytes\); "
    rb"(\d+) frames decoded(?:; (\d+) decode errors)?"
)
_PROBLEM_RE = re.compile(rb"\[(?:warning|error|fatal|panic)\]")
_STDERR_TAIL_LINES = 200


class FfmpegError(subprocess.CalledProcessError):
    """ffmpeg exited non-zero; the message carries its warning and error lines.

    ``CalledProcessError`` prints only the exit status, which hides why ffmpeg
    failed from a traceback.  ``stderr`` holds the same lines as bytes.
    """

    def __str__(self) -> str:
        detail = self.stderr.decode(errors="replace").rstrip() if self.stderr else ""
        return f"{super().__str__()}\n{detail}" if detail else super().__str__()


@dataclass
class _FrameCounts:
    """Video frame totals from ffmpeg's end-of-run summary.

    ``decoded`` has one entry per input video stream; ``encoded`` is keyed by
    output file index, the order outputs appear on the command line.
    """

    decoded: list[int] = field(default_factory=list)
    decode_errors: int = 0
    encoded: dict[int, int] = field(default_factory=dict)


def _effective_profile(
    profile: EncodingProfile,
    input_path: Path,
    *,
    auto_fix_colorspace: bool,
    range_override: RangeOverride | None,
    normalize_cfr: bool,
    preview_fps: float | None,
    poster: bool,
    poster_at_seconds: float | None,
    count_frames: bool,
) -> tuple[EncodingProfile, int | None]:
    """Apply the probe-driven adjustments to *profile*.

    Prepends the per-source ``setparams`` colour clause (when
    ``auto_fix_colorspace``) and the CFR-normalizing ``setpts`` clause (when
    ``normalize_cfr``), then appends the preview and poster derivatives (when
    ``preview_fps`` / ``poster``).  Probes the source once, and a second time
    to count packets only when the poster needs the middle frame of a source
    that records no frame count and either ``count_frames`` asks for the exact
    one or the source records no duration to estimate it from.

    Returns the profile and the preview's decimation factor, ``None`` without
    a preview.
    """
    effective = profile
    poster = poster or poster_at_seconds is not None
    needs_probe = auto_fix_colorspace or normalize_cfr or preview_fps is not None or poster
    probe_json = probe(input_path) if needs_probe else None
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
        effective = effective.prepend_conditioning(setpts)
    factor = None
    if preview_fps is not None:
        assert probe_json is not None
        factor, _ = preview_decimation(probe_json, target_fps=preview_fps)
        # The same factor thins the preview and preview_metadata.parquet, so their rows cannot drift apart.
        effective = with_preview(effective, probe_json, factor=factor)
    if poster:
        assert probe_json is not None
        if (
            poster_at_seconds is None
            and get_exact_nb_frames(probe_json) is None
            and (count_frames or get_duration_seconds(probe_json) is None)
        ):
            probe_json = probe(input_path, count_packets=True)
        effective = with_poster(effective, probe_json, at_seconds=poster_at_seconds)
    return effective, factor


def _check_derivatives_written(counts: _FrameCounts, paths: list[Path]) -> None:
    """Raise if ffmpeg encoded no frame for a derivative, which some ffmpeg versions do without failing."""
    for index, path in enumerate(paths, 1):
        if counts.encoded.get(index) == 0:
            raise RuntimeError(
                f"ffmpeg wrote no frames to {path}: its frames lie past the end of the source. A poster placed from "
                "a duration that overstates the source does this; pass count_frames=True to place it exactly."
            )


def _check_metadata_rows(frame_metadata: Any, counts: _FrameCounts, metadata_csv: Path, output_path: Path) -> None:
    """Raise unless *metadata_csv* has one row per frame of *output_path*.

    Without that, row ``k * N`` of the decimated table does not describe preview
    frame ``k``.
    """
    encoded = counts.encoded.get(0)
    if encoded is None:
        raise RuntimeError(
            f"ffmpeg reported no frame total for {output_path}, so {metadata_csv} cannot be matched to its frames."
        )
    if frame_metadata.num_rows != encoded:
        raise RuntimeError(
            f"{metadata_csv} has {frame_metadata.num_rows} rows but {output_path} has {encoded} frames, so "
            f"{PREVIEW_METADATA_FILENAME} would not line up with the preview; it was not written."
        )


def _read_stderr(stream: Iterable[bytes], counts: _FrameCounts, problems: deque[bytes]) -> None:
    """Collect frame totals and warning/error lines from ffmpeg's stderr.

    Runs on its own thread, since a stderr pipe left unread until exit fills
    and stalls ffmpeg.
    """
    for line in stream:
        if encoded := _ENCODED_RE.search(line):
            counts.encoded[int(encoded[1])] = int(encoded[2])
        elif decoded := _DECODED_RE.search(line):
            counts.decoded.append(int(decoded[1]))
            counts.decode_errors += int(decoded[2] or 0)
        elif _PROBLEM_RE.search(line):
            problems.append(line)


def _run_ffmpeg(cmd: list[str], *, on_progress: Callable[[int], None] | None) -> _FrameCounts:
    """Run *cmd*, forwarding frame progress, and return ffmpeg's frame totals.

    Raises
    ------
    FfmpegError
        If ffmpeg exits with a non-zero return code.  Its message and
        ``stderr`` carry the last warning and error lines rather than the
        verbose log.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None

    counts = _FrameCounts()
    problems: deque[bytes] = deque(maxlen=_STDERR_TAIL_LINES)
    reader = threading.Thread(target=_read_stderr, args=(proc.stderr, counts, problems), daemon=True)
    reader.start()
    for raw_line in proc.stdout:
        if raw_line.startswith(b"frame=") and on_progress:
            try:
                on_progress(int(raw_line[6:].strip()))
            except ValueError:
                pass
    returncode = proc.wait()
    reader.join()

    if returncode != 0:
        raise FfmpegError(returncode, cmd, stderr=b"".join(problems))
    return counts


def _check_frame_count(counts: _FrameCounts, input_path: Path, output_path: Path) -> None:
    """Raise unless *output_path* holds exactly the frames ffmpeg decoded from *input_path*.

    Both totals come from the same run, so the check costs no second decode and
    sees a frame lost anywhere between decoder and muxer.  Frames missing from
    the source itself are beyond it.
    """
    encoded = counts.encoded.get(0)
    if encoded is None or not counts.decoded:
        raise RuntimeError(
            f"ffmpeg reported no video frame totals transcoding {input_path}, so {output_path} cannot be "
            "checked for frame exactness. Pass fail_on_frame_drop=False to skip the check."
        )
    decoded = [n for n in counts.decoded if n]
    if len(decoded) > 1:
        raise RuntimeError(
            f"ffmpeg decoded {len(decoded)} video streams from {input_path}, so {output_path} has no single "
            "frame count to match. Pass fail_on_frame_drop=False to skip the check."
        )
    if counts.decode_errors:
        raise RuntimeError(
            f"ffmpeg hit {counts.decode_errors} decode error(s) reading {input_path}, so {output_path} may not "
            "hold every recorded frame intact. Pass fail_on_frame_drop=False to accept it."
        )
    source_frames = decoded[0] if decoded else 0
    if encoded != source_frames:
        raise RuntimeError(
            f"{output_path} has {encoded} frames but ffmpeg decoded {source_frames} from {input_path}. "
            "h264 in AVI loses frames this way unless its timestamps are re-stamped: pass normalize_cfr=True. "
            "Pass fail_on_frame_drop=False to accept a non-frame-exact output."
        )


def transcode_video(
    input_path: Path,
    output_path: Path,
    *,
    profile: EncodingProfile = OFFLINE_8BIT,
    auto_fix_colorspace: bool = True,
    range_override: RangeOverride | None = None,
    normalize_cfr: bool = False,
    fail_on_frame_drop: bool = True,
    preview_fps: float | None = None,
    metadata_csv: Path | None = None,
    poster: bool = False,
    poster_at_seconds: float | None = None,
    count_frames: bool = False,
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
        When ``True``, probe the source frame rate and add to the profile's
        conditioning a ``setpts=N/(num/den)/TB`` filter that re-stamps every
        frame's presentation timestamp from its display-order index at the
        source's base frame rate, starting at PTS 0.  Legacy h264-in-AVI
        sources need it: AVI cannot store composition offsets, so ffmpeg
        reconstructs non-monotonic timestamps for some frames and drops them.
        Off by default because it rewrites timing, which only sources with
        broken timestamps call for.  Requires a readable ``r_frame_rate``; leave
        it off for variable-frame-rate input whose timing must survive.
    fail_on_frame_drop : bool
        When ``True`` (the default), raise :class:`RuntimeError` unless the
        primary output holds exactly as many frames as ffmpeg decoded from the
        source, with no decode errors.  Both totals come from ffmpeg's own
        end-of-run summary, so the check needs no second decode and ignores
        derivatives, which drop frames by design.  Set to ``False`` when
        legitimately resampling variable-frame-rate input to CFR.
    preview_fps : float | None
        When set, also write ``preview.mp4`` beside *output_path*, a
        frame-decimated preview encoded as a second output of the same ffmpeg
        process, sharing the decode and the colour chain.  The value is a
        target rather than an exact rate: the integer decimation factor comes
        from :func:`aind_video_utils.encoding.preview_decimation`, so every
        preview frame is a real source frame.  See
        :func:`aind_video_utils.encoding.with_preview`.
    metadata_csv : Path | None
        The camera's ``metadata.csv``.  With ``preview_fps`` set, its
        ``ReferenceTime`` column, decimated as the preview is, is written to
        ``preview_metadata.parquet`` beside *output_path*, which the spec
        requires of every preview.  It is read before encoding, so a missing
        file, column or the ``parquet`` extra fails fast.  See
        :mod:`aind_video_utils.preview_metadata`.
    poster : bool
        When ``True``, also write ``poster.jpg`` beside *output_path*: the
        middle frame, branched off the conditioned source and encoded as sRGB
        so it matches what a browser shows for the video beside it.  Where the
        container records no frame count, as Matroska does not, the middle is
        estimated from the duration.  See
        :func:`aind_video_utils.encoding.with_poster`.
    poster_at_seconds : float | None
        When set, write the poster from this time into the video instead of
        the middle frame.  Implies ``poster``.
    count_frames : bool
        When ``True``, place the poster on the exact middle frame of a source
        that records no frame count by counting its packets first, which reads
        the whole source once more without decoding it -- a second download
        for a URL.  Without it, packets are counted only when the source
        records no duration either, as a recording that was killed mid-write
        does not.
    no_audio : bool
        If ``True``, strip audio (``-an``).
    on_progress : Callable[[int], None] | None
        Called with the current frame number as ffmpeg reports progress.  With
        a preview attached the progress stream covers both encoders, so treat
        the count as approximate.

    Returns
    -------
    Path
        *output_path* on success.  Derivative outputs are written but not
        returned; ask ``profile.output_paths(output_path)`` for every path a
        profile writes.

    Raises
    ------
    FfmpegError
        If ffmpeg exits with a non-zero return code, its warning and error
        lines in the message.  A subclass of ``subprocess.CalledProcessError``.
        ffmpeg 8.1 fails this way when a poster estimated from the
        duration lies past the end of a truncated source.
    RuntimeError
        If ``fail_on_frame_drop`` is set and the primary output's frame count
        differs from the decoded source's, or ffmpeg reported decode errors or
        no frame totals; if ``normalize_cfr`` is set but the source has no
        readable base frame rate; if ``metadata_csv`` has a row count other
        than the primary output's frame count; or if ffmpeg exited zero
        without writing a frame to a derivative.
    ValueError
        If ``metadata_csv`` is given without ``preview_fps``, or lacks the
        ``ReferenceTime`` column.
    ImportError
        If ``metadata_csv`` is given without the ``parquet`` extra installed.
    """
    if metadata_csv is not None and preview_fps is None:
        raise ValueError("metadata_csv describes a preview's frames, so it needs preview_fps")
    frame_metadata = read_frame_metadata(metadata_csv) if metadata_csv is not None else None
    effective, factor = _effective_profile(
        profile,
        input_path,
        auto_fix_colorspace=auto_fix_colorspace,
        range_override=range_override,
        normalize_cfr=normalize_cfr,
        preview_fps=preview_fps,
        poster=poster,
        poster_at_seconds=poster_at_seconds,
        count_frames=count_frames,
    )

    cmd: list[str] = ["ffmpeg", *_FFMPEG_LOG_ARGS, "-progress", "pipe:1", "-nostats", "-y"]
    cmd.extend(http_input_flags(input_path))
    cmd.extend(effective.ffmpeg_input_args())
    cmd.extend(["-i", str(input_path)])
    cmd.extend(effective.ffmpeg_graph_args())
    for group, path in zip(effective.ffmpeg_output_groups(), effective.output_paths(output_path), strict=True):
        cmd.extend(group)
        if no_audio:
            cmd.append("-an")
        cmd.append(str(path))

    counts = _run_ffmpeg(cmd, on_progress=on_progress)
    if fail_on_frame_drop:
        _check_frame_count(counts, input_path, output_path)
    _check_derivatives_written(counts, effective.output_paths(output_path)[1:])
    if frame_metadata is not None:
        assert metadata_csv is not None and factor is not None
        _check_metadata_rows(frame_metadata, counts, metadata_csv, output_path)
        write_decimated(frame_metadata, output_path.with_name(PREVIEW_METADATA_FILENAME), factor)
    return output_path
