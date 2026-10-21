"""Put an MP4's timestamps on a uniform grid without re-encoding.

A concat ``-c copy`` of separately encoded chunks can leave forward notches in
the timeline: every frame after a seam is stamped late, so tools that address a
frame by ``index / rate`` land on the wrong one.  :func:`restamp_uniform` removes
the notches with ffmpeg's ``setts`` bitstream filter: DTS are renumbered onto
the grid and every PTS displayed after a notch moves back by its excess.  Keeping
each sample's ``PTS - DTS`` instead does not work, because the composition
offsets at a seam carry the notch too.
"""

from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

from aind_video_utils.mp4_index import (
    Mp4FrameIndex,
    TimelineNotches,
    _find_child,
    _find_video_stbl,
    _iter_child_boxes,
    _read_moov,
    read_mp4_frame_index,
)
from aind_video_utils.probe import ProbeDict, probe
from aind_video_utils.transcode import FfmpegError
from aind_video_utils.utils import http_input_flags

# The muxer stamps its own version here whatever the source said.
_TAGS_THE_MUXER_REWRITES = frozenset({"encoder"})
_STREAM_FIELDS = (
    "codec_name",
    "profile",
    "pix_fmt",
    "width",
    "height",
    "sample_aspect_ratio",
    "color_range",
    "color_space",
    "color_transfer",
    "color_primaries",
)
# ISO 14496-12 VisualSampleEntry fields ahead of its child boxes.
_VISUAL_SAMPLE_ENTRY_HEADER = 78


def _media_time(index: Mp4FrameIndex) -> int:
    return index.edits[0].media_time if index.edits else 0


def _check_repairable(index: Mp4FrameIndex, notches: TimelineNotches) -> None:
    """Raise ValueError unless :func:`restamp_uniform` can repair *index* exactly."""
    if len(index.edits) > 1:
        raise ValueError(f"{len(index.edits)} edits in the edit list; only a single edit can be restamped")
    if index.edits and (index.edits[0].media_rate != 1.0 or index.edits[0].media_time < 0):
        raise ValueError("the edit is empty or plays at a non-unit rate")
    if (np.diff(index.dts) <= 0).any():
        raise ValueError("DTS step backwards or repeat")
    if (notches.pts_steps <= 0).any():
        raise ValueError("two frames share a PTS")
    if (bad := notches.pts_steps % notches.period != 0).any():
        raise ValueError(
            f"a presentation step of {int(notches.pts_steps[bad][0])} ticks is not a whole number of "
            f"{notches.period}-tick periods"
        )


def _setts_expression(index: Mp4FrameIndex, notches: TimelineNotches) -> str:
    """Build the ``setts`` filter that removes *notches*.

    Thresholds are in ffmpeg's timestamps, which have the edit's ``media_time``
    subtracted from the moov's.
    """
    first_after = np.sort(index.pts)[notches.pts_after + 1] - _media_time(index)
    excess = notches.pts_steps - notches.period
    shifts = "".join(f"-{int(e)}*gte(PTS\\,{int(t)})" for e, t in zip(excess, first_after, strict=True))
    return f"setts=dts=STARTDTS+N*{notches.period}:pts=PTS{shifts}"


def _uses_mdta_keys(moov: bytes) -> bool:
    """Whether the movie's tags are QuickTime ``mdta`` keys, which only ``+use_metadata_tags`` writes back."""
    metas = []
    if (meta := _find_child(moov, 0, len(moov), b"meta")) is not None:
        metas.append(meta)
    if (udta := _find_child(moov, 0, len(moov), b"udta")) is not None:
        if (meta := _find_child(moov, udta[0], udta[1], b"meta")) is not None:
            metas.append(meta)
    for start, end in metas:
        # A QuickTime meta has no version/flags field and an ISO one does.
        for body in (start, start + 4):
            hdlr = _find_child(moov, body, end, b"hdlr")
            if hdlr is not None and moov[hdlr[0] + 8 : hdlr[0] + 12] == b"mdta":
                return True
    return False


def _has_colr(moov: bytes) -> bool:
    """Whether the video's sample entry carries a ``colr`` box, which a stream copy drops unless asked."""
    stbl_start, stbl_end, *_ = _find_video_stbl(moov)
    stsd = _find_child(moov, stbl_start, stbl_end, b"stsd")
    if stsd is None:
        return False
    for _type, entry_start, entry_end in _iter_child_boxes(moov, stsd[0] + 8, stsd[1]):
        children = entry_start + _VISUAL_SAMPLE_ENTRY_HEADER
        if _find_child(moov, children, entry_end, b"colr") is not None:
            return True
    return False


def _verify_timeline(before: Mp4FrameIndex, after: Mp4FrameIndex, period: int) -> None:
    """Raise RuntimeError unless *after* is *before* with every timestamp on the grid.

    Pixels cannot show a timing fault, so the check reads only the index.
    """
    problems = []
    if after.n_samples != before.n_samples:
        problems.append(f"{after.n_samples} samples, not {before.n_samples}")
    else:
        if not np.array_equal(after.size, before.size):
            problems.append("sample sizes differ")
        if not np.array_equal(after.is_keyframe, before.is_keyframe):
            problems.append("keyframes differ")
        if not np.array_equal(after.display_order, before.display_order):
            problems.append("display order differs")
    if after.media_timescale != before.media_timescale:
        problems.append(f"timescale {after.media_timescale}, not {before.media_timescale}")
    if (np.diff(after.dts) != period).any():
        problems.append(f"DTS do not step by {period}")
    if (np.diff(np.sort(after.pts)) != period).any():
        problems.append(f"PTS do not step by {period}")
    if (after.pts < after.dts).any():
        problems.append("a PTS precedes its DTS")
    if len(after.edits) > 1:
        problems.append(f"{len(after.edits)} edits")
    elif _media_time(after) != _media_time(before):
        problems.append(f"edit media_time {_media_time(after)}, not {_media_time(before)}")
    elif after.edits:
        expected = Fraction(after.n_samples * period * after.movie_timescale, after.media_timescale)
        if abs(after.edits[0].segment_duration - expected) >= 1:
            problems.append(f"edit lasts {after.edits[0].segment_duration} movie ticks, not {float(expected):g}")
    if problems:
        raise RuntimeError("; ".join(problems))


def _verify_metadata(before: ProbeDict, after: ProbeDict) -> None:
    """Raise RuntimeError unless *after* keeps the container and stream metadata of *before*."""

    def tags(section: dict[str, object], rewritten: frozenset[str] = frozenset()) -> dict[str, object]:
        found = section.get("tags", {})
        assert isinstance(found, dict)
        return {k: v for k, v in found.items() if k not in rewritten}

    problems = []
    if tags(after["format"], _TAGS_THE_MUXER_REWRITES) != tags(before["format"], _TAGS_THE_MUXER_REWRITES):
        problems.append("container tags differ")
    stream_before, stream_after = before["streams"][0], after["streams"][0]
    if tags(stream_after) != tags(stream_before):
        problems.append("stream tags differ")
    problems.extend(
        f"{field} is {stream_after.get(field)}, not {stream_before.get(field)}"
        for field in _STREAM_FIELDS
        if stream_after.get(field) != stream_before.get(field)
    )
    if problems:
        raise RuntimeError("; ".join(problems))


def restamp_uniform(src: str | Path, dst: Path) -> Path:
    """Copy *src* to *dst* with its PTS and DTS on a uniform grid at its frame period.

    Each notch, a presentation step longer than the frame period, is removed by
    moving every frame displayed after it back by the excess; DTS are
    renumbered onto the grid from the first.  No frame is re-encoded, so the
    payload is untouched, and a notch-free source comes back with the same
    timeline.  The repair is verified from *dst*'s own index before it is
    returned: same samples, sizes, keyframes and display order; PTS and DTS
    both on the grid, PTS never before DTS; one edit with the same
    ``media_time``, lasting the frame count times the period; and the same
    container and stream tags and colour description.  ffmpeg writes to a
    temporary file beside *dst*, so *dst* never holds an unverified repair.

    Erasing a notch is right when the frame times live elsewhere, as in AIND's
    ``metadata.csv``.  Where a gap in the PTS marks a real dropped exposure, the
    repair erases that record too.

    Parameters
    ----------
    src : str | Path
        A non-fragmented MP4 or MOV with one video stream and nothing else;
        local or an ``http(s)://`` URL.
    dst : Path
        Local destination, overwritten if it exists.

    Returns
    -------
    Path
        *dst*.

    Raises
    ------
    ValueError
        If *src* has a stream besides its video, more than one edit, an empty
        or non-unit-rate edit, DTS that step backwards, two frames sharing a
        PTS, or a notch that is not a whole number of frame periods.
    FfmpegError
        If ffmpeg fails.
    RuntimeError
        If the output fails verification; *dst* is left as it was.
    """
    source_probe = probe(src)
    if len(source_probe["streams"]) != 1 or source_probe["streams"][0].get("codec_type") != "video":
        raise ValueError(f"{src} has streams besides its video, which a restamp would drop")
    index = read_mp4_frame_index(src)
    notches = index.timeline_notches()
    _check_repairable(index, notches)
    moov = _read_moov(src)
    movflags = (
        "+faststart"
        + ("+write_colr" if _has_colr(moov) else "")
        + ("+use_metadata_tags" if _uses_mdta_keys(moov) else "")
    )

    partial = dst.with_name(f".{dst.stem}.restamping{dst.suffix}")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *http_input_flags(src), "-i", str(src),
        "-map", "0:v", "-c", "copy", "-map_metadata", "0",
        "-bsf:v", _setts_expression(index, notches),
        "-video_track_timescale", str(index.media_timescale),
        "-movflags", movflags,
        str(partial),
    ]  # fmt: skip
    try:
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise FfmpegError(result.returncode, cmd, stderr=result.stderr)
        try:
            _verify_timeline(index, read_mp4_frame_index(partial), notches.period)
            _verify_metadata(source_probe, probe(partial))
        except RuntimeError as err:
            raise RuntimeError(f"restamping {src} failed verification: {err}") from None
        partial.replace(dst)
    finally:
        partial.unlink(missing_ok=True)
    return dst
