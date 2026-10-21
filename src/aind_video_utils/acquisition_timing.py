"""Read a camera's per-frame timing CSV and repair the times it records.

AIND behavior videos come with one CSV row per frame received: the trigger time
on the rig's reference clock, the camera's frame counter, and the camera's own
timestamp.  Two layouts exist, read by :func:`read_frame_timing` into one
:class:`FrameTiming`:

- the file standard's ``metadata.csv``, with a header naming ``ReferenceTime``,
  ``CameraFrameNumber`` and ``CameraFrameTime``;
- the legacy dynamic-foraging ``<camera>.csv``, headerless, whose first three
  columns hold the same in that order and any further ones exposure and gain.

The recorded times cannot always be taken at face value.
:func:`repair_frame_times` applies three repairs and names each one it applies,
so a caller can record the provenance:

- ``trigger_time_out_of_line``: a time outside the span of its two neighbours,
  where those are themselves in order, is replaced by their midpoint.  Seen at a
  HARP second boundary.
- ``zip_pairing_repaired``: a workflow that pairs the k-th frame received with
  the k-th trigger writes every frame after a dropped one beside an earlier
  trigger's time.  Every recorded time is still a real trigger time, so exposure
  k, counted by the frame counter from the first frame, takes the k-th one.
- ``frame_counter_distrusted``: a counter that ever steps backwards is not
  counting exposures, so each frame keeps the time on its own row, and whether
  frames were dropped is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

Pairing = Literal["by_arrival", "by_exposure"]
"""How the acquiring workflow paired frames with trigger times.

``by_arrival`` pairs the k-th frame received with the k-th trigger, as
``foraging.bonsai``'s ``Zip`` does, so a dropped frame shifts every later time;
``by_exposure`` writes each frame beside its own trigger's time.
"""

# Float seconds or integer clock ticks; repairs keep whichever they are given.
_Times = npt.NDArray[Any]

Repair = Literal["trigger_time_out_of_line", "zip_pairing_repaired", "frame_counter_distrusted"]

STANDARD_COLUMNS: tuple[str, str, str] = ("ReferenceTime", "CameraFrameNumber", "CameraFrameTime")
"""The ``metadata.csv`` columns the file standard requires, in the order :class:`FrameTiming` holds them."""


@dataclass(frozen=True, eq=False)
class FrameTiming:
    """One camera's timing CSV, one entry per row, as recorded.

    Attributes
    ----------
    reference_time : NDArray[np.float64]
        Trigger time on the rig's reference clock, in seconds.
    frame_number : NDArray[np.int64]
        The camera's frame counter.
    camera_time : NDArray[np.int64]
        The camera's own timestamp, in its own units (nanoseconds on the
        cameras AIND uses).  Integer, since those exceed float64's exact range.
    """

    reference_time: npt.NDArray[np.float64]
    frame_number: npt.NDArray[np.int64]
    camera_time: npt.NDArray[np.int64]

    @property
    def n_frames(self) -> int:
        """Number of rows, one per frame received."""
        return int(self.reference_time.shape[0])


@dataclass(frozen=True, eq=False)
class RepairedFrameTimes:
    """Frame times after :func:`repair_frame_times`.

    Attributes
    ----------
    time : NDArray
        One time per row, in the units and dtype of the times given.
    dropped_before : NDArray[np.bool_] | None
        ``True`` where at least one exposure was lost just before the row's
        frame; never for the first row.  ``None`` where the frame counter is
        distrusted, since nothing then shows whether a frame was lost.
    repairs : tuple[Repair, ...]
        The repairs applied, in the order applied; empty when the recorded
        times stand.
    pairing : Pairing | None
        The pairing the times were read under: as given, or detected.  ``None``
        where it made no difference, because the counter skips no exposure or
        is distrusted.
    """

    time: _Times
    dropped_before: npt.NDArray[np.bool_] | None
    repairs: tuple[Repair, ...]
    pairing: Pairing | None


def _has_header(path: Path) -> list[str] | None:
    """Return the CSV's column names, or ``None`` when its first line is data."""
    with path.open() as handle:
        fields = handle.readline().strip().split(",")
    try:
        float(fields[0])
    except ValueError:
        return [f.strip() for f in fields]
    return None


def read_frame_timing(path: Path) -> FrameTiming:
    """Read a camera's timing CSV in either AIND layout.

    A first line that is not numeric is a header, which must name the file
    standard's three columns, wherever they sit; otherwise the first three
    columns are read as the legacy layout's.

    Parameters
    ----------
    path : Path
        ``metadata.csv`` or a legacy ``<camera>.csv``.

    Returns
    -------
    FrameTiming
        The recorded columns.

    Raises
    ------
    ValueError
        If the header lacks a required column, or a value does not parse.
    """
    header = _has_header(path)
    if header is None:
        columns, skip = [0, 1, 2], 0
    else:
        if missing := [c for c in STANDARD_COLUMNS if c not in header]:
            raise ValueError(f"{path} has no column {missing}")
        columns, skip = [header.index(c) for c in STANDARD_COLUMNS], 1
    # Read as text first: the camera timestamp loses precision through float64.
    raw = np.loadtxt(path, delimiter=",", skiprows=skip, usecols=columns, dtype=str, ndmin=2)
    return FrameTiming(
        reference_time=raw[:, 0].astype(np.float64),
        frame_number=raw[:, 1].astype(np.int64),
        camera_time=raw[:, 2].astype(np.int64),
    )


def _counts_exposures(frame_number: npt.NDArray[np.integer]) -> bool:
    """Whether a frame counter can be counting exposures: one that ever steps backwards cannot."""
    return bool((np.diff(frame_number) >= 0).all())


def _replace_out_of_line(times: _Times) -> npt.NDArray[np.intp]:
    """Replace, in place, each time outside the span of its two neighbours, and return where.

    Only a value whose neighbours are themselves in order is replaced, by their
    midpoint, which is within a trigger period of the true time.  Two adjacent
    values out of order with each other both qualify, since neither can be
    singled out.  Any other disorder is left alone.
    """
    before, value, after = times[:-2], times[1:-1], times[2:]
    where = np.flatnonzero((before <= after) & ((value < before) | (value > after))) + 1
    total = times[where - 1] + times[where + 1]
    times[where] = total // 2 if np.issubdtype(times.dtype, np.integer) else total / 2
    return where


def _detect_pairing(times: _Times, exposures: npt.NDArray[np.int64]) -> Pairing:
    """Read the pairing from the time steps where the counter skips.

    By arrival, a skip of any length steps the time by one trigger period; by
    exposure, by as many periods as it skips.  Refuses when the skips fit
    neither throughout.
    """
    time_steps = np.diff(times.astype(np.float64))
    period = float(np.median(time_steps))
    skipped = np.flatnonzero(np.diff(exposures) > 1)
    steps_in_periods = time_steps[skipped] / period
    if (np.abs(steps_in_periods - 1) < 0.5).all():
        return "by_arrival"
    if (np.abs(steps_in_periods - np.diff(exposures)[skipped]) < 0.5).all():
        return "by_exposure"
    raise ValueError(
        f"the frame counter skips {skipped.size} times, but the recorded times step neither by one trigger period "
        "at every skip nor by the exposures skipped; pass pairing= to say how the workflow paired them"
    )


def repair_frame_times(
    reference_time: _Times,
    frame_number: npt.NDArray[np.integer],
    *,
    pairing: Pairing | None = None,
) -> RepairedFrameTimes:
    """Return the trigger time of each row's exposure, repairing what the recording got wrong.

    Out-of-line times are replaced first.  A counter that steps backwards is
    then distrusted, and the times stand.  Otherwise, where it skips, the
    pairing decides: by exposure, the times stand; by arrival, exposure k,
    counted from the first row's, takes the k-th recorded time, and exposures
    past the last recorded time extrapolate at the period the recording spans.

    Parameters
    ----------
    reference_time : NDArray
        One recorded trigger time per row, in any units: float seconds, or
        integer ticks, which keep integer arithmetic throughout.
    frame_number : NDArray[np.integer]
        The camera's frame counter, one per row.
    pairing : Pairing | None
        How the workflow paired frames with triggers.  ``None`` detects it from
        the steps at the counter's skips, and refuses when they are ambiguous.

    Returns
    -------
    RepairedFrameTimes
        The times, the dropped-frame flags and the repairs applied.

    Raises
    ------
    ValueError
        If the arrays differ in length or hold fewer than two rows, or the
        pairing is not given and cannot be detected.
    """
    if reference_time.shape != frame_number.shape or reference_time.ndim != 1:
        raise ValueError(f"reference_time {reference_time.shape} and frame_number {frame_number.shape} must match")
    if reference_time.shape[0] < 2:
        raise ValueError("repairing frame times needs at least two rows")
    times = reference_time.copy()
    repairs: list[Repair] = []
    if _replace_out_of_line(times).size:
        repairs.append("trigger_time_out_of_line")
    if not _counts_exposures(frame_number):
        repairs.append("frame_counter_distrusted")
        return RepairedFrameTimes(times, None, tuple(repairs), None)

    exposures = (frame_number - frame_number[0]).astype(np.int64)
    dropped_before = np.zeros(times.shape[0], dtype=np.bool_)
    dropped_before[1:] = np.diff(exposures) > 1
    if not dropped_before.any():
        return RepairedFrameTimes(times, dropped_before, tuple(repairs), None)
    if pairing is None:
        pairing = _detect_pairing(times, exposures)
    if pairing == "by_exposure":
        return RepairedFrameTimes(times, dropped_before, tuple(repairs), pairing)

    last = times.shape[0] - 1
    period = (times[last] - times[0]) / last
    repaired = np.empty_like(times)
    known = exposures <= last
    repaired[known] = times[exposures[known]]
    offset = (exposures[~known] - last) * period
    # Rounded before adding: the sum's magnitude costs float64 the half-tick.
    repaired[~known] = times[last] + (np.rint(offset) if np.issubdtype(times.dtype, np.integer) else offset)
    repairs.append("zip_pairing_repaired")
    return RepairedFrameTimes(repaired, dropped_before, tuple(repairs), pairing)
