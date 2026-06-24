"""Headless batch exposure QC for ranking videos at scale.

Per-video output is a compact set of summary statistics computed from
``n_samples`` evenly-spaced frames. Designed for ranking many videos
by exposure issues (clipping at black/white, narrow dynamic range,
color-range mismatch) rather than producing per-video plots.

The module depends only on numpy and the package's own probe / frame
extraction helpers, so it does not require the ``[plotting]`` or
``[transcode]`` extras.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aind_video_utils.color_spaces import luma_range
from aind_video_utils.frames import extract_luma_frame
from aind_video_utils.probe import (
    ProbeDict,
    get_frame_dimensions,
    get_video_range_info,
    probe,
)

LumaFrame = npt.NDArray[np.uint8] | npt.NDArray[np.uint16]


@dataclass(frozen=True)
class FrameExposureStats:
    """Exposure statistics for a single video frame.

    All percentages are 0-100. Percentiles and mean/std are in raw luma
    units (0-255 for 8-bit, 0-1023 for 10-bit).
    """

    mean: float
    std: float
    p1: float
    p5: float
    p50: float
    p95: float
    p99: float
    pct_at_min: float
    pct_at_max: float
    pct_below_floor: float
    pct_above_ceiling: float
    pct_outside_tagged: float
    entropy_bits: float


@dataclass(frozen=True)
class VideoExposureQc:
    """Per-video exposure QC summary aggregated across sampled frames.

    Output columns suitable for sorting/ranking. ``*_max`` fields capture
    worst-case behavior across samples; ``*_med`` capture typical
    behavior; ``*_min`` are used where smaller = worse (dynamic range).
    """

    n_samples: int
    width: int
    height: int
    bit_depth: int
    color_range: str
    duration_s: float | None
    luma_mean_med: float
    luma_std_med: float
    p1_min: float
    p99_max: float
    dynamic_range_min: float
    dynamic_range_med: float
    pct_at_min_max: float
    pct_at_max_max: float
    pct_below_floor_max: float
    pct_above_ceiling_max: float
    pct_outside_tagged_max: float
    pct_outside_tagged_med: float
    entropy_bits_min: float
    entropy_bits_med: float


def compute_frame_stats(
    luma: LumaFrame,
    color_range: str,
    bit_depth: int,
) -> FrameExposureStats:
    """Compute exposure statistics for a single luma frame.

    Parameters
    ----------
    luma : NDArray[np.uint8] | NDArray[np.uint16]
        Luma plane (shape ``(h, w)``).
    color_range : str
        ``"pc"`` for full range, ``"tv"`` for limited range. Used to
        compute the tagged floor/ceiling for clipping percentages.
    bit_depth : int
        Bits per sample (8 or 10).

    Returns
    -------
    FrameExposureStats
        Frame-level summary statistics.
    """
    max_val = (1 << bit_depth) - 1
    is_full = color_range == "pc"
    floor, ceiling = luma_range(bit_depth, is_full)
    flat = luma.ravel()
    total = flat.size
    p1, p5, p50, p95, p99 = (float(v) for v in np.percentile(flat, [1, 5, 50, 95, 99]))
    n_below = int((flat < floor).sum())
    n_above = int((flat > ceiling).sum())
    n_min = int((flat == 0).sum())
    n_max = int((flat == max_val).sum())
    bins = 256 if bit_depth == 8 else 1024
    counts, _ = np.histogram(flat, bins=bins, range=(0, max_val))
    probs = counts.astype(np.float64) / total
    nz = probs[probs > 0]
    entropy_bits = float(-(nz * np.log2(nz)).sum())
    return FrameExposureStats(
        mean=float(flat.mean()),
        std=float(flat.std()),
        p1=p1,
        p5=p5,
        p50=p50,
        p95=p95,
        p99=p99,
        pct_at_min=100.0 * n_min / total,
        pct_at_max=100.0 * n_max / total,
        pct_below_floor=100.0 * n_below / total,
        pct_above_ceiling=100.0 * n_above / total,
        pct_outside_tagged=100.0 * (n_below + n_above) / total,
        entropy_bits=entropy_bits,
    )


def _format_duration(probe_json: ProbeDict) -> float | None:
    """Extract the video duration in seconds from a probe dict."""
    fmt = probe_json.get("format", {})
    if "duration" in fmt:
        try:
            return float(fmt["duration"])
        except (TypeError, ValueError):
            return None
    return None


def _sample_times(duration: float, n: int, skip_edge_fraction: float) -> list[float]:
    """Generate ``n`` evenly-spaced sample times, skipping the edges.

    Parameters
    ----------
    duration : float
        Total video duration in seconds.
    n : int
        Number of samples.
    skip_edge_fraction : float
        Fraction of duration to skip at start and end (each).

    Returns
    -------
    list[float]
        Sample times in seconds.
    """
    edge = duration * skip_edge_fraction
    if n == 1:
        return [duration / 2.0]
    return [float(t) for t in np.linspace(edge, max(edge, duration - edge), n)]


def qc_video(
    video_path: str | Path,
    n_samples: int = 10,
    skip_edge_fraction: float = 0.01,
    probe_json: ProbeDict | None = None,
) -> VideoExposureQc:
    """Sample frames evenly across a video and summarize exposure.

    The video is probed once; each sampled frame extraction reuses that
    probe dict to avoid redundant ffprobe roundtrips (important when
    extracting many frames from S3-hosted videos).

    Parameters
    ----------
    video_path : str | Path
        Local path or HTTPS URL to the video.
    n_samples : int, optional
        Number of frames to sample (default 10).
    skip_edge_fraction : float, optional
        Fraction of duration to skip at start and end (default 0.01).
    probe_json : ProbeDict, optional
        Pre-computed probe output. Probed if omitted.

    Returns
    -------
    VideoExposureQc
        Per-video aggregate exposure stats.
    """
    pj: ProbeDict = probe_json if probe_json is not None else probe(video_path)
    color_range, bit_depth = get_video_range_info(pj)
    width, height = get_frame_dimensions(pj)
    duration = _format_duration(pj)
    times = _sample_times(duration if duration else 1.0, n_samples, skip_edge_fraction)

    frame_stats: list[FrameExposureStats] = []
    for t in times:
        luma, _, _ = extract_luma_frame(video_path, t, probe_json=pj)
        frame_stats.append(compute_frame_stats(luma, color_range, bit_depth))
    return _aggregate(frame_stats, color_range, bit_depth, width, height, duration)


def _aggregate(
    frame_stats: list[FrameExposureStats],
    color_range: str,
    bit_depth: int,
    width: int,
    height: int,
    duration: float | None,
) -> VideoExposureQc:
    """Aggregate per-frame stats into a per-video QC summary."""
    means = np.array([f.mean for f in frame_stats])
    stds = np.array([f.std for f in frame_stats])
    p1s = np.array([f.p1 for f in frame_stats])
    p99s = np.array([f.p99 for f in frame_stats])
    drs = p99s - p1s
    pct_min = np.array([f.pct_at_min for f in frame_stats])
    pct_max = np.array([f.pct_at_max for f in frame_stats])
    pct_below = np.array([f.pct_below_floor for f in frame_stats])
    pct_above = np.array([f.pct_above_ceiling for f in frame_stats])
    pct_outside = np.array([f.pct_outside_tagged for f in frame_stats])
    entropy = np.array([f.entropy_bits for f in frame_stats])
    return VideoExposureQc(
        n_samples=len(frame_stats),
        width=width,
        height=height,
        bit_depth=bit_depth,
        color_range=color_range,
        duration_s=duration,
        luma_mean_med=float(np.median(means)),
        luma_std_med=float(np.median(stds)),
        p1_min=float(p1s.min()),
        p99_max=float(p99s.max()),
        dynamic_range_min=float(drs.min()),
        dynamic_range_med=float(np.median(drs)),
        pct_at_min_max=float(pct_min.max()),
        pct_at_max_max=float(pct_max.max()),
        pct_below_floor_max=float(pct_below.max()),
        pct_above_ceiling_max=float(pct_above.max()),
        pct_outside_tagged_max=float(pct_outside.max()),
        pct_outside_tagged_med=float(np.median(pct_outside)),
        entropy_bits_min=float(entropy.min()),
        entropy_bits_med=float(np.median(entropy)),
    )


def qc_result_fieldnames() -> list[str]:
    """Return the ordered field names of ``VideoExposureQc`` for CSV output."""
    return [f.name for f in fields(VideoExposureQc)]
