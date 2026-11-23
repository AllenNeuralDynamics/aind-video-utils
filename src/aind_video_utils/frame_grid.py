"""Place each frame on the grid of a known nominal frame rate.

A container that stores timestamps coarser than the frame period, such as
Matroska's 1 ms, leaves them jittering around the true grid.  When the nominal
rate is known from outside the file, each frame's slot on that grid is
``round((T - T0) * R)``, provided every frame sits close to its slot.
:func:`fit_nominal_grid` is that rule and its refusals; the MP4 restamp and the
transcode snap share it so they accept the same files.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import numpy.typing as npt

# A frame further than this from its slot is not rounding jitter.  1 ms rounding
# moves a frame under 0.1 period at 120 fps; a rate wrong by 1% passes it within 25 frames.
DEFAULT_TOLERANCE = 0.25


@dataclass(frozen=True, eq=False)
class NominalGrid:
    """Frames placed on the grid of a nominal rate, in display order.

    Attributes
    ----------
    frame_rate : Fraction
        The nominal rate the grid was fitted at.
    slots : NDArray[np.int64]
        Each frame's slot, in periods from the first frame's.
    max_residual : float
        The largest distance of a frame from its slot, in periods.
    gap_after : NDArray[np.intp]
        Display indices after which the slot steps by more than one.
    gap_excess : NDArray[np.int64]
        The missing slots at each of ``gap_after``.
    """

    frame_rate: Fraction
    slots: npt.NDArray[np.int64]
    max_residual: float
    gap_after: npt.NDArray[np.intp]
    gap_excess: npt.NDArray[np.int64]


def ticks_to_periods(ticks: int, timescale: int, frame_rate: Fraction) -> Fraction:
    """Return *ticks* of a *timescale* clock as a whole and fractional number of periods of *frame_rate*."""
    return Fraction(ticks * frame_rate.numerator, timescale * frame_rate.denominator)


def fit_nominal_grid(
    pts: npt.ArrayLike, timescale: int, frame_rate: Fraction, *, tolerance: float = DEFAULT_TOLERANCE
) -> NominalGrid:
    """Place presentation timestamps on the grid of *frame_rate*.

    The arithmetic is exact integer arithmetic, so a residual is never a
    floating-point artefact.

    Parameters
    ----------
    pts : array_like
        Presentation timestamps in ticks, in any order; they are sorted.
    timescale : int
        Ticks per second.
    frame_rate : Fraction
        The nominal rate, from the acquisition rather than guessed from *pts*.
    tolerance : float
        The largest distance from its slot a frame may sit, in periods.

    Returns
    -------
    NominalGrid

    Raises
    ------
    ValueError
        If fewer than two frames are given, the rate is not positive, a frame
        sits further than *tolerance* from its slot, two frames share a slot,
        or the median step is not one slot (the rate is a multiple of the
        true one).
    """
    if frame_rate <= 0:
        raise ValueError(f"frame rate must be positive, got {frame_rate}")
    ticks = np.sort(np.asarray(pts, dtype=np.int64))
    if ticks.size < 2:
        raise ValueError("a grid needs at least two frames")
    # Frame k sits offset_numerator[k] / offset_denominator periods after the first.
    offset_numerator = (ticks - ticks[0]) * frame_rate.numerator
    offset_denominator = timescale * frame_rate.denominator
    slots = (2 * offset_numerator + offset_denominator) // (2 * offset_denominator)
    residuals = np.abs(offset_numerator - slots * offset_denominator) / offset_denominator
    if (off := residuals > tolerance).any():
        k = int(np.argmax(off))
        raise ValueError(
            f"frame {k + 1} in display order is {residuals[k]:.2f} periods from its slot at {frame_rate} fps, "
            f"more than {tolerance}: the rate is wrong or the timestamps are not rounding jitter"
        )
    steps = np.diff(slots)
    if (shared := steps < 1).any():
        k = int(np.argmax(shared))
        raise ValueError(f"frames {k + 1} and {k + 2} in display order share a slot at {frame_rate} fps")
    if round(float(np.median(steps))) != 1:
        raise ValueError(f"the median step is {np.median(steps):g} slots at {frame_rate} fps, not 1: the rate is wrong")
    gap_after = np.flatnonzero(steps > 1)
    return NominalGrid(
        frame_rate=frame_rate,
        slots=slots,
        max_residual=float(residuals.max()),
        gap_after=gap_after,
        gap_excess=steps[gap_after] - 1,
    )
