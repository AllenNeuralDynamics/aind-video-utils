"""Put an MP4's timestamps on a uniform grid without re-encoding.

A concat ``-c copy`` of separately encoded chunks can leave forward notches in
the timeline: every frame after a seam is stamped late, so tools that address a
frame by ``index / rate`` land on the wrong one.  :func:`restamp_uniform` removes
the notches with ffmpeg's ``setts`` bitstream filter: DTS are renumbered onto
the grid and every PTS displayed after a notch moves back by its excess.  Keeping
each sample's ``PTS - DTS`` instead does not work, because the composition
offsets at a seam carry the notch too.

That repair needs every step to be whole periods in the file's own ticks.
:func:`restamp_nominal` covers a timeline that a coarser clock rounded, given
the nominal rate: it moves each frame to its slot on that rate's grid, in a
timescale where one period is whole.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aind_video_utils.frame_grid import DEFAULT_TOLERANCE, NominalGrid, fit_nominal_grid, ticks_to_periods
from aind_video_utils.mp4_index import (
    EditListEntry,
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


def _presenting_edits(index: Mp4FrameIndex) -> tuple[EditListEntry, ...]:
    """Return the edit list past a leading empty edit, which only delays the start."""
    edits = index.edits
    return edits[1:] if len(edits) > 1 and edits[0].media_time == -1 else edits


def _media_time(index: Mp4FrameIndex) -> int:
    edits = _presenting_edits(index)
    return edits[0].media_time if edits else 0


def _check_single_edit_and_dts(index: Mp4FrameIndex) -> None:
    """Raise ValueError unless *index* has at most one presenting edit, at unit rate, and rising DTS."""
    edits = _presenting_edits(index)
    if len(edits) > 1:
        raise ValueError(f"{len(index.edits)} edits in the edit list; only a single edit can be restamped")
    if edits and (edits[0].media_rate != 1.0 or edits[0].media_time < 0):
        raise ValueError("the edit is empty or plays at a non-unit rate")
    if (np.diff(index.dts) <= 0).any():
        raise ValueError("DTS step backwards or repeat")


def _check_repairable(index: Mp4FrameIndex, notches: TimelineNotches) -> None:
    """Raise ValueError unless :func:`restamp_uniform` can repair *index* exactly."""
    _check_single_edit_and_dts(index)
    if (notches.pts_steps <= 0).any():
        raise ValueError("two frames share a PTS")
    if (bad := notches.pts_steps % notches.period != 0).any():
        raise ValueError(
            f"a presentation step of {int(notches.pts_steps[bad][0])} ticks is not a whole number of "
            f"{notches.period}-tick periods"
        )


def _setts_expression(index: Mp4FrameIndex, notches: TimelineNotches) -> str:
    """Build the ``setts`` filter that removes *notches*.

    Thresholds count from the first frame, since ffmpeg shifts its input by a
    start time that the edit list moves.
    """
    sorted_pts = np.sort(index.pts)
    first_after = sorted_pts[notches.pts_after + 1] - sorted_pts[0]
    excess = notches.pts_steps - notches.period
    shifts = "".join(f"-{int(e)}*gte(PTS-STARTPTS\\,{int(t)})" for e, t in zip(excess, first_after, strict=True))
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


def _probe_single_video(src: str | Path) -> ProbeDict:
    """Probe *src*, raising ValueError if it holds anything but one video stream."""
    source_probe = probe(src)
    if len(source_probe["streams"]) != 1 or source_probe["streams"][0].get("codec_type") != "video":
        raise ValueError(f"{src} has streams besides its video, which a restamp would drop")
    return source_probe


def _movflags(src: str | Path) -> str:
    """Return the ``-movflags`` that keep *src*'s colour box and tag style through a stream copy."""
    moov = _read_moov(src)
    return (
        "+faststart"
        + ("+write_colr" if _has_colr(moov) else "")
        + ("+use_metadata_tags" if _uses_mdta_keys(moov) else "")
    )


def _partial_path(dst: Path, stage: str) -> Path:
    return dst.with_name(f".{dst.stem}.{stage}{dst.suffix}")


def _copy_cmd(src: str | Path, dst: Path, timescale: int, movflags: str, bsf: str | None = None) -> list[str]:
    """Build a stream copy of *src*'s video and metadata to *dst* at *timescale*."""
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *http_input_flags(src), "-i", str(src),
        "-map", "0:v", "-c", "copy", "-map_metadata", "0",
        *(["-bsf:v", bsf] if bsf else []),
        "-video_track_timescale", str(timescale),
        "-movflags", movflags,
        str(dst),
    ]  # fmt: skip


def _run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise FfmpegError(result.returncode, cmd, stderr=result.stderr)


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
    container and stream tags and colour description.  A leading empty edit,
    which only delays the first frame, is dropped, so *dst* starts at 0.  ffmpeg
    writes to a temporary file beside *dst*, so *dst* never holds an unverified
    repair.

    Erasing a notch is right when the frame times live elsewhere, as in AIND's
    ``metadata.csv``.  Where a gap in the PTS marks a real dropped exposure, the
    repair erases that record too: match frames to timing rows with
    :func:`~aind_video_utils.acquisition_timing.video_frame_rows` first.

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
        If *src* has a stream besides its video, more than one edit after any
        leading empty one, an empty or non-unit-rate edit, DTS that step
        backwards, two frames sharing a PTS, or a notch that is not a whole
        number of frame periods.
    FfmpegError
        If ffmpeg fails.
    RuntimeError
        If the output fails verification; *dst* is left as it was.
    """
    source_probe = _probe_single_video(src)
    index = read_mp4_frame_index(src)
    notches = index.timeline_notches()
    _check_repairable(index, notches)
    movflags = _movflags(src)

    partial = _partial_path(dst, "restamping")
    try:
        _run_ffmpeg(_copy_cmd(src, partial, index.media_timescale, movflags, _setts_expression(index, notches)))
        try:
            _verify_timeline(index, read_mp4_frame_index(partial), notches.period)
            _verify_metadata(source_probe, probe(partial))
        except RuntimeError as err:
            raise RuntimeError(f"restamping {src} failed verification: {err}") from None
        partial.replace(dst)
    finally:
        partial.unlink(missing_ok=True)
    return dst


@dataclass(frozen=True)
class RestampResult:
    """What :func:`restamp_nominal` changed, for a provenance record.

    Attributes
    ----------
    path : Path
        The restamped file.
    frame_rate : Fraction
        The nominal rate the timeline was put on.
    source_timescale : int
        The source's media timescale, in ticks per second.
    timescale : int
        The output's media timescale, in which one period is a whole number of ticks.
    keep_gaps : bool
        Whether gaps were kept as missing slots rather than erased.
    notch_after_frame : tuple[int, ...]
        Frames, counted from 1 in display order, after which the source skipped slots.
    notch_excess_frames : tuple[int, ...]
        The slots skipped after each of ``notch_after_frame``.
    max_residual : float
        The largest distance of a source frame from its slot, in periods.
    """

    path: Path
    frame_rate: Fraction
    source_timescale: int
    timescale: int
    keep_gaps: bool
    notch_after_frame: tuple[int, ...]
    notch_excess_frames: tuple[int, ...]
    max_residual: float

    def as_record(self) -> dict[str, object]:
        """Return the result as JSON-ready values, the rate as ``"num/den"``."""
        record = asdict(self)
        record["path"] = str(self.path)
        record["frame_rate"] = f"{self.frame_rate.numerator}/{self.frame_rate.denominator}"
        record["notch_after_frame"] = list(self.notch_after_frame)
        record["notch_excess_frames"] = list(self.notch_excess_frames)
        return record


@dataclass(frozen=True, eq=False)
class _NominalPlan:
    """A checked :func:`restamp_nominal` of one index: the grid, the timescales and the reorder delay."""

    grid: NominalGrid
    source_timescale: int
    timescale: int
    reorder_delay: int  # whole periods between the first DTS and the first PTS

    @property
    def frame_rate(self) -> Fraction:
        return self.grid.frame_rate

    def period(self, timescale: int) -> int:
        """One period of the rate in *timescale* ticks; whole for the timescales the plan uses."""
        return timescale * self.frame_rate.denominator // self.frame_rate.numerator

    @property
    def working_timescale(self) -> int:
        """A multiple of the source timescale in which one period is whole, so ``setts`` can work exactly."""
        return math.lcm(self.source_timescale, self.frame_rate.numerator)

    def output_slots(self, keep_gaps: bool) -> npt.NDArray[np.int64]:
        """Each frame's slot in display order after the restamp."""
        return self.grid.slots if keep_gaps else np.arange(self.grid.slots.size, dtype=np.int64)


def _nominal_timescale(frame_rate: Fraction) -> int:
    """Return the timescale ffmpeg's MP4 muxer picks for *frame_rate*, so restamped files match fresh encodes.

    The muxer doubles the encoder's ``1/rate`` time base until it reaches 10000
    ticks per second, which keeps one period whole.
    """
    timescale = frame_rate.numerator
    while timescale < 10000:
        timescale *= 2
    return timescale


def _plan_nominal(index: Mp4FrameIndex, frame_rate: Fraction) -> _NominalPlan:
    """Raise ValueError unless :func:`restamp_nominal` can put *index* on the grid of *frame_rate*."""
    _check_single_edit_and_dts(index)
    first_pts = int(index.pts.min())
    if (edits := _presenting_edits(index)) and edits[0].media_time != first_pts:
        raise ValueError(f"the edit starts at {edits[0].media_time} ticks, not at the first frame's {first_pts}")
    grid = fit_nominal_grid(index.pts, index.media_timescale, frame_rate)
    delay = ticks_to_periods(first_pts - int(index.dts[0]), index.media_timescale, frame_rate)
    if abs(delay - round(delay)) > DEFAULT_TOLERANCE:
        raise ValueError(
            f"the first frame is presented {float(delay):.2f} periods after it is decoded, not a whole number"
        )
    return _NominalPlan(
        grid=grid,
        source_timescale=index.media_timescale,
        timescale=_nominal_timescale(frame_rate),
        reorder_delay=round(delay),
    )


def _nominal_setts_expression(index: Mp4FrameIndex, plan: _NominalPlan, keep_gaps: bool) -> str:
    """Build the ``setts`` filter that stamps each frame of *index*, in the working timescale, at its output slot.

    ffmpeg's stream copy reads the result in the input's time base whatever
    ``time_base`` the filter declares, so the input must already be in a
    timescale where one period is whole.  Times count from the first frame
    presented, which need not be the first decoded, on which ``STARTPTS`` sits.
    """
    if index.media_timescale != plan.working_timescale:
        raise ValueError(f"the index is at {index.media_timescale} ticks per second, not {plan.working_timescale}")
    period = plan.period(plan.working_timescale)
    first_pts = int(index.pts.min())
    since_first = f"(PTS-STARTPTS+{int(index.pts[0]) - first_pts})"
    rate = plan.frame_rate
    slot = f"round({since_first}*{rate.numerator}/{plan.working_timescale * rate.denominator})"
    shifts = ""
    if not keep_gaps:
        sorted_pts = np.sort(index.pts)
        first_after = sorted_pts[plan.grid.gap_after + 1] - first_pts
        shifts = "".join(
            f"-{int(e)}*gte({since_first}\\,{int(t)})" for e, t in zip(plan.grid.gap_excess, first_after, strict=True)
        )
    return f"setts=pts=({slot}{shifts})*{period}:dts=(N-{plan.reorder_delay})*{period}:duration={period}"


def _verify_rescale(before: Mp4FrameIndex, after: Mp4FrameIndex, scale: int) -> None:
    """Raise RuntimeError unless *after* is *before* with every timestamp multiplied by *scale*."""
    for name in ("pts", "dts"):
        old, new = getattr(before, name), getattr(after, name)
        if old.shape != new.shape or not np.array_equal(new - new[0], (old - old[0]) * scale):
            raise RuntimeError(f"rescaling to {after.media_timescale} changed the {name.upper()}")


def _verify_nominal(before: Mp4FrameIndex, after: Mp4FrameIndex, plan: _NominalPlan, keep_gaps: bool) -> None:
    """Raise RuntimeError unless *after* is *before* with each frame at its slot on the nominal grid."""
    problems = []
    if after.n_samples != before.n_samples:
        raise RuntimeError(f"{after.n_samples} samples, not {before.n_samples}")
    if not np.array_equal(after.size, before.size):
        problems.append("sample sizes differ")
    if not np.array_equal(after.is_keyframe, before.is_keyframe):
        problems.append("keyframes differ")
    if not np.array_equal(after.display_order, before.display_order):
        problems.append("display order differs")
    if after.media_timescale != plan.timescale:
        problems.append(f"timescale {after.media_timescale}, not {plan.timescale}")
    period = plan.period(plan.timescale)
    slots = plan.output_slots(keep_gaps)
    first_pts = int(after.dts[0]) + plan.reorder_delay * period
    if not np.array_equal(np.sort(after.pts), first_pts + slots * period):
        problems.append(f"PTS are not at their slots of {period} ticks")
    if (np.diff(after.dts) != period).any():
        problems.append(f"DTS do not step by {period}")
    if (after.pts < after.dts).any():
        problems.append("a PTS precedes its DTS")
    if len(after.edits) > 1:
        problems.append(f"{len(after.edits)} edits")
    elif after.edits:
        if after.edits[0].media_time != first_pts:
            problems.append(f"edit media_time {after.edits[0].media_time}, not {first_pts}")
        expected = Fraction((int(slots[-1]) + 1) * period * after.movie_timescale, after.media_timescale)
        if abs(after.edits[0].segment_duration - expected) >= 1:
            problems.append(f"edit lasts {after.edits[0].segment_duration} movie ticks, not {float(expected):g}")
    if problems:
        raise RuntimeError("; ".join(problems))


def restamp_nominal(
    src: str | Path, dst: Path, *, frame_rate: Fraction | int, keep_gaps: bool = False
) -> RestampResult:
    """Copy *src* to *dst* with every frame on the grid of a known nominal rate.

    For a timeline that was exact at *frame_rate* until a coarser clock
    rounded it, such as a Matroska file's 1 ms, where a period need not be a
    whole number of the file's ticks.  Each frame goes to its slot,
    ``round((T - T0) * frame_rate)``; with ``keep_gaps=False`` the frames are
    then renumbered by display rank, erasing skipped slots as
    :func:`restamp_uniform` does.  The output timescale is ffmpeg's own choice
    for the rate (15360 for 120, 60 and 30 fps), so one period is whole; DTS
    step by one period from the source's reorder delay, which stays in the
    edit.  No frame is re-encoded.

    Before anything is written, every frame must sit within 0.25 period of its
    slot, so a wrong rate is refused rather than forced onto the grid.  The
    result is verified from *dst*'s own index: same samples, sizes, keyframes
    and display order; each PTS at its slot; DTS one period apart and never
    after their PTS; one edit, lasting to the end of the last slot; and the
    same container and stream tags and colour description.  ffmpeg writes to
    temporary files beside *dst*, so *dst* never holds an unverified repair.

    Parameters
    ----------
    src : str | Path
        A non-fragmented MP4 or MOV with one video stream and nothing else;
        local or an ``http(s)://`` URL.
    dst : Path
        Local destination, overwritten if it exists.
    frame_rate : Fraction | int
        The acquisition's nominal rate.  Never measure it from *src*: the
        rounding it repairs is what makes such a measurement wrong.
    keep_gaps : bool
        Keep a skipped slot as a gap, for a timeline whose gaps are real
        dropped exposures.  The default erases them, which is right when the
        notches are concat artefacts and the frame times live elsewhere.

    Returns
    -------
    RestampResult

    Raises
    ------
    ValueError
        If *src* has a stream besides its video, more than one edit after any
        leading empty one, an edit that is empty, plays at a non-unit rate or
        starts past the first frame, DTS that step backwards, a frame further
        than 0.25 period from its slot, two frames in one slot, a median step
        other than one slot, or a reorder delay that is not whole periods.
    FfmpegError
        If ffmpeg fails.
    RuntimeError
        If the output fails verification; *dst* is left as it was.
    """
    source_probe = _probe_single_video(src)
    index = read_mp4_frame_index(src)
    plan = _plan_nominal(index, Fraction(frame_rate))
    movflags = _movflags(src)

    rescaled = _partial_path(dst, "rescaling")
    partial = _partial_path(dst, "restamping")
    try:
        try:
            working_source: str | Path = src
            working_index = index
            if plan.working_timescale != plan.source_timescale:
                _run_ffmpeg(_copy_cmd(src, rescaled, plan.working_timescale, movflags))
                working_source, working_index = rescaled, read_mp4_frame_index(rescaled)
                _verify_rescale(index, working_index, plan.working_timescale // plan.source_timescale)
            setts = _nominal_setts_expression(working_index, plan, keep_gaps)
            _run_ffmpeg(_copy_cmd(working_source, partial, plan.timescale, movflags, setts))
            _verify_nominal(index, read_mp4_frame_index(partial), plan, keep_gaps)
            _verify_metadata(source_probe, probe(partial))
        except RuntimeError as err:
            raise RuntimeError(f"restamping {src} failed verification: {err}") from None
        partial.replace(dst)
    finally:
        rescaled.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    return RestampResult(
        path=dst,
        frame_rate=plan.frame_rate,
        source_timescale=plan.source_timescale,
        timescale=plan.timescale,
        keep_gaps=keep_gaps,
        notch_after_frame=tuple(int(k) + 1 for k in plan.grid.gap_after),
        notch_excess_frames=tuple(int(e) for e in plan.grid.gap_excess),
        max_residual=plan.grid.max_residual,
    )
