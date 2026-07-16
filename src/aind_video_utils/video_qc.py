"""QC functions comparing video frames before and after encoding."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

from aind_video_utils.color_spaces import linear_to_rec_709_trc, luma_range, rec_709_trc_to_linear
from aind_video_utils.frames import extract_luma_frame, extract_srgb_frame
from aind_video_utils.plotting import (
    _apply_tufte_style,
    apply_sparse_ticklabels,
    bivariate_intensity_histogram,
    bivariate_with_marginals,
    imshow_clipping,
    intensity_histogram,
    luma_comparison_figure,
)
from aind_video_utils.probe import get_duration_seconds, get_video_range_info, probe

NOISE_FRAMES_DEFAULT = 8
_TOE_CUTOFF_DN = 12.0
_UNDER_COLOR = "#2166ac"
_OVER_COLOR = "#b2182b"
_TRC_COLOR = "C2"

LumaFrame = NDArray[np.uint8] | NDArray[np.uint16]
sRGBFrame = NDArray[np.uint8]


def get_frame_pair_from_video(
    video_path: str | Path,
    frame_time: float,
    coerce_color_space: bool = False,
) -> tuple[LumaFrame, sRGBFrame, int, bool]:
    """Extract luma and sRGB frames from a video at a given time.

    Parameters
    ----------
    video_path : str | Path
        Path to the video file.
    frame_time : float
        Time in seconds at which to extract the frame.
    coerce_color_space : bool, optional
        If True, override the stream's transfer characteristic metadata.

    Returns
    -------
    luma : LumaFrame
        Luma plane.
    srgb : sRGBFrame
        sRGB image with shape ``(h, w, 3)``.
    bit_depth : int
        Bits per component (8 or 10).
    is_full_range : bool
        Whether the video uses full-range (pc) color range.
    """
    probe_json = probe(video_path)
    color_range, bit_depth = get_video_range_info(probe_json)
    luma = extract_luma_frame(video_path, frame_time)[0]
    srgb = extract_srgb_frame(video_path, frame_time, coerce_color_space)
    # coerce_color_space implies AIND linear-light convention, which is PC
    # range by definition (linear photon counts have no headroom/footroom).
    # Without this override, an untagged source (color_range='unknown') is
    # treated as TV, which propagates to luma_range_input=(16,235) and
    # mislabels the bivariate plot's "standard range" lines + the luma
    # clip title. Caught 2026-06-25 on the smoke QC pass: untagged mpeg4
    # source's bivariate showed input min appearing at x~2 because the
    # axis was clamped to TV range while real source values straddled 16.
    is_full_range = coerce_color_space or color_range == "pc"
    return luma, srgb, bit_depth, is_full_range


def compare_linear_to_bt709(
    input_video_path: str | Path,
    output_video_path: str | Path,
    frame_time: float,
    coerce_input_color_space: bool = False,
) -> Figure:
    """Compare linear-light input against BT.709-encoded output.

    Assumes the input video contains linear light and the output has been
    encoded with the BT.709 transfer characteristic. The bivariate
    histogram overlay shows the expected BT.709 TRC curve for reference.

    .. note:: Only 8-bit videos are currently supported.

    Parameters
    ----------
    input_video_path : str | Path
        Path to the linear-light input video.
    output_video_path : str | Path
        Path to the BT.709-encoded output video.
    frame_time : float
        Time in seconds to extract the frames.
    coerce_input_color_space : bool, optional
        Override the input stream's transfer characteristic metadata
        (assume linear light).

    Returns
    -------
    Figure
        Comparison figure with sRGB frames, luma highlights, and bivariate histogram.
    """
    luma_input, srgb_input, depth_input, is_full_range_input = get_frame_pair_from_video(
        input_video_path, frame_time, coerce_input_color_space
    )
    luma_output, srgb_output, depth_output, is_full_range_output = get_frame_pair_from_video(
        output_video_path, frame_time, False
    )
    luma_range_input = luma_range(depth_input, is_full_range_input)
    luma_range_output = luma_range(depth_output, is_full_range_output)
    (
        fig,
        ax_input_srgb,
        ax_output_srgb,
        ax_input_luma,
        ax_output_luma,
        ax_bivariate,
        ax_top_marginal,
        ax_right_marginal,
        gs,
    ) = luma_comparison_figure(
        luma_input,
        luma_output,
        srgb_input,
        srgb_output,
        intensity_range=luma_range_input,
        output_clip=luma_range_output,
        input_limits=luma_range_input,
        input_srgb_title="Frame interpreted as sRGB",
        output_srgb_title="Frame interpreted as sRGB",
        title="Video compression QC",
    )
    y_range = ax_bivariate.get_ylim()
    v_space = 0.02 * (y_range[1] - y_range[0])
    ax_bivariate.text(
        0.5,
        luma_range_output[0],
        "minimum, standard range",
        fontsize=8,
        fontstyle="italic",
        color="#2166ac",
        ha="center",
        va="bottom",
        transform=ax_bivariate.get_yaxis_transform(),
    )
    ax_bivariate.text(
        0.5,
        luma_range_output[1] - v_space,
        "maximum, standard range",
        fontsize=8,
        fontstyle="italic",
        color="#b2182b",
        ha="center",
        va="top",
        transform=ax_bivariate.get_yaxis_transform(),
    )

    in_lo, in_hi = luma_range_input
    out_lo, out_hi = luma_range_output

    def bt709_trc_fcn(x: float) -> float:
        lin = min(max((x - in_lo) / (in_hi - in_lo), 0.0), 1.0)
        return out_lo + (out_hi - out_lo) * linear_to_rec_709_trc(lin)

    max_input = (1 << depth_input) - 1
    luma_space = np.linspace(0, max_input, max_input + 1)
    bt709_trc_values = [bt709_trc_fcn(v) for v in luma_space]
    bt709_color = "C2"
    ax_bivariate.plot(
        luma_space,
        bt709_trc_values,
        color=bt709_color,
        linestyle="--",
        linewidth=1,
        zorder=2,
        label="BT.709 TRC",
    )
    ax_bivariate.text(
        50,
        120,
        "BT.709 TRC",
        rotation=45,
        rotation_mode="anchor",  # rotate around the anchor point
        color=bt709_color,
        fontsize=8,
        fontstyle="italic",
        ha="center",
        va="bottom",
    )
    ax_bivariate.set_xlabel("Input luma value")
    ax_bivariate.set_ylabel("Output luma value")
    return fig


def compare_luma_opencv_frames(
    input_video_path: str | Path,
    frame_time: float = 0,
) -> Figure:
    """Compare ffmpeg luma extraction with OpenCV's first-frame decode.

    Produces a figure with the luma and OpenCV frames side-by-side above
    a bivariate histogram showing how the two sets of values relate.

    .. note:: Only 8-bit videos are currently supported.

    Parameters
    ----------
    input_video_path : str | Path
        Path to the video file.
    frame_time : float, optional
        Time in seconds at which to extract the frame.

    Returns
    -------
    Figure
        Comparison figure.
    """
    import cv2

    vidcap = cv2.VideoCapture(input_video_path)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
    _, image = vidcap.read()
    opencv_frame = image
    luma_frame, color_range, bit_depth = extract_luma_frame(input_video_path, frame_time)
    is_full_range = color_range == "pc"
    luma_low, luma_high = luma_range(bit_depth, is_full_range)
    fig = plt.figure(figsize=(8, 8))

    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1, 1],
        height_ratios=[0.75, 1],
        hspace=0.12,
        wspace=0.08,
    )
    ax_luma = fig.add_subplot(gs[0, 0])
    ax_opencv = fig.add_subplot(gs[0, 1])
    ax_biv = fig.add_subplot(gs[1, :])
    imshow_clipping(opencv_frame[:, :, 0], vmin=0, vmax=255, ax=ax_opencv)
    ax_opencv.axis("off")
    imshow_clipping(luma_frame, vmin=luma_low, vmax=luma_high, ax=ax_luma)
    ax_luma.axis("off")
    range_title = "full-range" if is_full_range else "limited-range"
    ax_opencv.set_title("OpenCV (full-range)")
    ax_luma.set_title(f"Luma ({range_title})")
    _, ax_biv, _, _ = bivariate_with_marginals(
        luma_frame,
        opencv_frame[:, :, 0],
        x_limits=(luma_low, luma_high),
        y_limits=None,
        ax=ax_biv,
    )
    ax_biv.set_ylabel("OpenCV values (full-range)")
    ax_biv.set_xlabel(f"Actual luma ({range_title})")
    return fig


def check_color_range(
    video_path: str | Path,
    frame_time: float = 0,
    mismatch_threshold: float = 1.0,
) -> Figure:
    """Check whether a video's pixel data matches its color range metadata.

    Produces a figure with two frame interpretations (tagged range vs
    opposite range) and a histogram showing the luma distribution relative
    to the limited-range boundaries.

    Parameters
    ----------
    video_path : str | Path
        Path to the video file.
    frame_time : float, optional
        Time in seconds at which to extract the frame.
    mismatch_threshold : float, optional
        Percentage of pixels outside the tagged range above which the
        verdict reports a possible mismatch (default 1.0%).

    Returns
    -------
    Figure
        Diagnostic figure.
    """
    luma, color_range, bit_depth = extract_luma_frame(video_path, frame_time)
    is_full_range = color_range == "pc"
    max_val = (1 << bit_depth) - 1

    tagged_lo, tagged_hi = luma_range(bit_depth, is_full_range)
    opposite_lo, opposite_hi = luma_range(bit_depth, not is_full_range)

    # Limited-range boundaries (always shown on histogram)
    limited_lo, limited_hi = luma_range(bit_depth, False)

    # Percentage of pixels outside limited range
    total = luma.size
    outside_limited = int(np.sum(luma < limited_lo) + np.sum(luma > limited_hi))
    pct_outside_limited = 100.0 * outside_limited / total

    tagged_label = "full-range" if is_full_range else "limited-range"
    opposite_label = "full-range" if not is_full_range else "limited-range"

    if is_full_range:
        if pct_outside_limited > mismatch_threshold:
            verdict = f"PASS — {pct_outside_limited:.1f}% outside limited range"
        else:
            verdict = "FAIL — no pixels outside limited range"
    else:
        if pct_outside_limited > mismatch_threshold:
            verdict = f"FAIL — {pct_outside_limited:.1f}% outside limited range"
        else:
            verdict = "PASS"

    # --- Figure layout ---
    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1, 1],
        height_ratios=[1, 0.6],
        hspace=0.30,
        wspace=0.08,
    )

    # Top row: two imshow_clipping panels
    ax_tagged = fig.add_subplot(gs[0, 0])
    ax_opposite = fig.add_subplot(gs[0, 1])

    imshow_clipping(luma, vmin=tagged_lo, vmax=tagged_hi, ax=ax_tagged)
    ax_tagged.axis("off")
    ax_tagged.set_title(
        f"Metadata says {tagged_label}\nclip @ {tagged_lo}–{tagged_hi}",
        fontsize=9,
    )

    imshow_clipping(luma, vmin=opposite_lo, vmax=opposite_hi, ax=ax_opposite)
    ax_opposite.axis("off")
    ax_opposite.set_title(
        f"Interpreted as {opposite_label}\nclip @ {opposite_lo}–{opposite_hi}",
        fontsize=9,
    )

    # Bottom: histogram spanning both columns
    ax_hist = fig.add_subplot(gs[1, :])
    intensity_histogram(
        luma,
        clip_vmin=limited_lo,
        clip_vmax=limited_hi,
        intensity_range=(0, max_val),
        extreme_values=(0, max_val),
        ax=ax_hist,
    )
    ax_hist.set_xlabel("Luma value")

    fig.suptitle(
        f"Color Range Check: {verdict}",
        fontsize=11,
        fontweight="bold",
    )
    return fig


def _oetf_derivative(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerical derivative of the BT.709 OETF at linear inputs ``x`` in [0, 1]."""
    h = 1e-3
    up = np.clip(x + h, 0.0, 1.0)
    dn = np.clip(x - h, 0.0, 1.0)
    fwd = np.array([linear_to_rec_709_trc(float(v)) for v in up], dtype=np.float64)
    bwd = np.array([linear_to_rec_709_trc(float(v)) for v in dn], dtype=np.float64)
    deriv: NDArray[np.float64] = (fwd - bwd) / np.maximum(up - dn, 1e-9)
    return deriv


def bt709_noise_shape(mean_dn: NDArray[np.float64]) -> NDArray[np.float64]:
    """CALCULATED (unnormalized) BT.709-encoded shot-noise variance vs mean.

    If a source were BT.709-encoded, shot noise in the linear scene ``x`` transforms
    as ``Var ∝ (OETF'(x))²·x`` with ``x = OETF⁻¹(mean/255)``. Pure calculation from the
    transfer function (BT.601 shares this OETF); a single gain is applied by the
    caller's anchor. Assumes 8-bit input means.
    """
    v = np.clip(mean_dn / 255.0, 1e-4, 1.0)
    x = np.array([rec_709_trc_to_linear(float(vi)) for vi in v], dtype=np.float64)
    shape: NDArray[np.float64] = _oetf_derivative(x) ** 2 * x
    return shape


def _box_mean(img: NDArray[np.float64], radius: int = 2) -> NDArray[np.float64]:
    """Fast ``(2·radius+1)`` box mean via an integral image (edge-replicated)."""
    pad = np.pad(img, radius + 1, mode="edge")
    integral = pad.cumsum(0).cumsum(1)
    k = 2 * radius + 1
    h, w = img.shape
    box: NDArray[np.float64] = (
        integral[k : k + h, k : k + w] + integral[0:h, 0:w] - integral[k : k + h, 0:w] - integral[0:h, k : k + w]
    ) / (k * k)
    return box


def _robust_variance(values: NDArray[np.float64]) -> float:
    """MAD-based variance estimate (resistant to residual scene structure)."""
    if values.size < 32:
        return float("nan")
    mad = float(np.median(np.abs(values - np.median(values))))
    return (1.4826 * mad) ** 2


def noise_transfer_curve(
    frames: list[NDArray[np.float64]],
    n_bins: int = 24,
    min_count: int = 400,
    box_radius: int = 2,
    flat_gradient_max: float = 8.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Spatial photon-transfer curve: per-intensity noise variance from flat regions.

    A high-pass residual (frame minus local box-mean) in low-gradient regions isolates
    pixel noise; residuals are pooled across ``frames``, binned by local mean intensity
    (quantile-spaced), and a robust variance is computed per populated bin.

    Returns ``(mean, variance)`` arrays, one point per bin.
    """
    means: list[NDArray[np.float64]] = []
    resids: list[NDArray[np.float64]] = []
    for frame in frames:
        local = _box_mean(frame, box_radius)
        highpass = frame - local
        gx = np.abs(np.diff(frame, axis=1, prepend=frame[:, :1]))
        gy = np.abs(np.diff(frame, axis=0, prepend=frame[:1, :]))
        flat = (gx + gy) < flat_gradient_max
        means.append(local[flat])
        resids.append(highpass[flat])
    if not means:  # all frames failed to read / no flat regions -> no curve
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    mean_all = np.concatenate(means)
    resid_all = np.concatenate(resids)
    edges = np.unique(np.quantile(mean_all, np.linspace(0.02, 0.98, n_bins + 1)))
    idx = np.digitize(mean_all, edges) - 1
    mus: list[float] = []
    vars_: list[float] = []
    for b in range(len(edges) - 1):
        sel = idx == b
        if int(sel.sum()) < min_count:
            continue
        v = _robust_variance(resid_all[sel])
        if np.isfinite(v) and v > 0:
            mus.append(float(np.median(mean_all[sel])))
            vars_.append(v)
    return np.array(mus, dtype=np.float64), np.array(vars_, dtype=np.float64)


def classify_gamma(
    mean: NDArray[np.float64],
    variance: NDArray[np.float64],
    toe_cutoff: float = _TOE_CUTOFF_DN,
) -> dict[str, Any]:
    """Verdict by scoring the PTC against CALCULATED linear and BT.709 curves.

    No model is fitted. The linear (``Var ∝ mean``) and BT.709 (``bt709_noise_shape``)
    curves are anchored to the data at the brightest reliable bin (where the two nearly
    coincide) and the data is scored against each by summed log-residual. Bins below
    ``toe_cutoff`` (h264-flattened dark flats) are excluded from the verdict.

    Returns a dict with ``call`` (LINEAR/GAMMA/AMBIGUOUS), ``resid_linear``,
    ``resid_bt709``, ``ok_mask``, and the anchor point.
    """
    ok = mean >= toe_cutoff
    out: dict[str, Any] = {
        "call": "AMBIGUOUS",
        "ok_mask": ok,
        "resid_linear": float("nan"),
        "resid_bt709": float("nan"),
    }
    if int(ok.sum()) < 3:
        return out
    ok_mean, ok_var = mean[ok], variance[ok]
    anchor_i = int(np.argmax(ok_mean))
    anchor_mean = float(ok_mean[anchor_i])
    anchor_var = float(ok_var[anchor_i])
    shape_anchor = float(bt709_noise_shape(np.array([anchor_mean]))[0])
    lin_pred = anchor_var * (ok_mean / anchor_mean)
    bt709_pred = anchor_var * bt709_noise_shape(ok_mean) / shape_anchor
    resid_lin = float(np.sum((np.log(ok_var) - np.log(lin_pred)) ** 2))
    resid_gam = float(np.sum((np.log(ok_var) - np.log(bt709_pred)) ** 2))
    out.update(
        call="LINEAR" if resid_lin <= resid_gam else "GAMMA",
        resid_linear=resid_lin,
        resid_bt709=resid_gam,
        anchor_mean=anchor_mean,
        anchor_var=anchor_var,
    )
    return out


def _noise_timestamps(input_probe: Any, n_frames: int) -> NDArray[np.float64]:
    """``n_frames`` timestamps spread across the source (fallback to a 60 s guess)."""
    duration = get_duration_seconds(input_probe) or 60.0
    return np.linspace(0.05 * duration, 0.95 * duration, max(n_frames, 3), dtype=np.float64)


def transcode_qc_figure(
    input_video_path: str | Path,
    output_video_path: str | Path,
    frame_time: float,
    *,
    range_override: str | None = None,
    rig_group: str | None = None,
    noise_frames: int = NOISE_FRAMES_DEFAULT,
    coerce_input_color_space: bool = True,
) -> Figure:
    """Build the transcode QC figure for one linear→BT.709 (video) pair.

    Panels: (1) before/after sRGB frames, (2) cliff-range histogram [raw-pixel range
    evidence], (3) noise-transfer gamma test [data vs calculated linear & BT.709
    curves], (4) transfer characteristic [observed input→output vs the BT.709 TRC].
    A summary table reports the Range / Gamma / Rig calls.

    ``range_override`` (``"pc"``/``"tv"``/``None``) is the authority for the input
    range and threads through the sRGB interpretation, the cliff clip lines, and the
    expected-TRC input normalization. ``rig_group`` is metadata shown in the table.

    .. note:: The cliff + noise + TRC calculations assume 8-bit source luma.
    """
    in_probe = probe(input_video_path)
    _, bit_depth = get_video_range_info(in_probe)
    if range_override == "pc":
        is_full = True
    elif range_override == "tv":
        is_full = False
    else:
        color_range, _ = get_video_range_info(in_probe)
        is_full = coerce_input_color_space or color_range == "pc"
    in_lo, in_hi = luma_range(bit_depth, is_full)
    tv_lo, tv_hi = luma_range(bit_depth, False)
    max_val = (1 << bit_depth) - 1

    before_luma, _, _ = extract_luma_frame(input_video_path, frame_time, in_probe)
    after_luma, _, _ = extract_luma_frame(output_video_path, frame_time)
    srgb_before = extract_srgb_frame(input_video_path, frame_time, coerce_input_color_space, input_is_full=is_full)
    srgb_after = extract_srgb_frame(output_video_path, frame_time, False)

    noise_ts = _noise_timestamps(in_probe, noise_frames)
    noise_planes: list[NDArray[np.float64]] = []
    for t in noise_ts:
        try:
            noise_planes.append(extract_luma_frame(input_video_path, float(t), in_probe)[0].astype(np.float64))
        except Exception:  # a deep remote seek can fail transiently; drop that sample rather than crash
            continue
    mean, variance = noise_transfer_curve(noise_planes)
    verdict = classify_gamma(mean, variance)
    gamma_call = str(verdict["call"])

    footroom = 100.0 * float(np.mean(before_luma < tv_lo))
    headroom = 100.0 * float(np.mean(before_luma > tv_hi))
    cliff_full = (footroom + headroom) > 1.0
    if range_override == "tv":
        range_call = "TV (forced)"
        range_ev = (
            f"{footroom:.0f}%↓ in reserved ⇒ really FULL ⚠"
            if cliff_full
            else f"{footroom:.0f}%↓ {headroom:.0f}%↑ — clean TV"
        )
    elif is_full:
        range_call = "FULL (PC)"
        range_ev = f"{footroom:.0f}%↓ + {headroom:.0f}%↑ fill reserved zones"
    else:
        range_call = "LIMITED (TV)"
        range_ev = f"{footroom:.0f}%↓ {headroom:.0f}%↑ outside band"
    gamma_ev = (
        f"data ~ linear; log-resid {verdict['resid_linear']:.1f} vs BT.709 {verdict['resid_bt709']:.1f}"
        if np.isfinite(verdict["resid_linear"])
        else "insufficient bins"
    )

    fig = plt.figure(figsize=(12, 15))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[0.42, 1.05, 1.0, 1.15], hspace=0.38, wspace=0.2)
    rig_txt = f" · rig {rig_group}" if rig_group else ""
    fig.suptitle(f"Transcode QC{rig_txt}", fontsize=13, fontweight="bold")

    ax_tbl = fig.add_subplot(gs[0, :])
    ax_tbl.axis("off")
    tbl = ax_tbl.table(
        cellText=[["Range", range_call, range_ev], ["Gamma", gamma_call, gamma_ev], ["Rig", rig_group or "—", "N/A"]],
        colLabels=["", "Call", "Method / evidence"],
        colWidths=[0.1, 0.22, 0.68],
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    for col, (srgb, title) in enumerate(
        [(srgb_before, "BEFORE — frame as sRGB\n(interpreted with calls)"), (srgb_after, "AFTER — BT.709 MP4 as sRGB")]
    ):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(srgb)
        ax.axis("off")
        ax.set_title(title, fontsize=9)

    ax_cliff = fig.add_subplot(gs[2, 0])
    intensity_histogram(
        before_luma,
        clip_vmin=tv_lo,
        clip_vmax=tv_hi,
        extreme_values=(0, max_val),
        intensity_range=(0, max_val),
        log_scale=True,
        ax=ax_cliff,
        show_stems=True,
        shade_regions=[(-0.5, tv_lo - 0.5, _UNDER_COLOR), (tv_hi + 0.5, max_val + 0.5, _OVER_COLOR)],
    )
    cliff_verdict = "FULL" if cliff_full else "LIMITED"
    ax_cliff.set_title(
        f"Cliff range test — TV valid band {tv_lo}–{tv_hi} dashed\ndata reaching reserved zones ⇒ {cliff_verdict}",
        fontsize=9,
    )

    ax_noise = fig.add_subplot(gs[2, 1])
    _apply_tufte_style(ax_noise)
    ok = np.asarray(verdict["ok_mask"], dtype=bool)
    ax_noise.plot(mean[~ok], variance[~ok], "o", color="0.78", ms=3)
    ax_noise.plot(mean[ok], variance[ok], "o", color="0.25", ms=3)
    if int(ok.sum()) >= 3:
        grid = np.linspace(float(mean[ok].min()), float(mean[ok].max()), 60, dtype=np.float64)
        a_mean, a_var = float(verdict["anchor_mean"]), float(verdict["anchor_var"])
        lin_curve = a_var * (grid / a_mean)
        gam_curve = a_var * bt709_noise_shape(grid) / float(bt709_noise_shape(np.array([a_mean]))[0])
        ax_noise.plot(grid, lin_curve, "--", color=_UNDER_COLOR, lw=1.2)
        ax_noise.plot(grid, gam_curve, ":", color=_OVER_COLOR, lw=1.5)
        ax_noise.text(
            grid[-1], lin_curve[-1], " linear", color=_UNDER_COLOR, fontsize=8, va="center", fontstyle="italic"
        )
        gj = len(grid) // 4
        ax_noise.text(grid[gj], gam_curve[gj], " BT709", color=_OVER_COLOR, fontsize=8, va="bottom", fontstyle="italic")
    ax_noise.set_xscale("log")
    ax_noise.set_yscale("log")
    ax_noise.set_xlabel("intensity mean (DN)")
    ax_noise.set_ylabel("noise variance (DN²)")
    ax_noise.set_title(
        f"Noise-transfer gamma test ⇒ {gamma_call}\n(data vs calculated curves; toe grayed = h264-flattened)",
        fontsize=9,
    )

    ax_trc = fig.add_subplot(gs[3, :])
    bivariate_intensity_histogram(
        before_luma,
        after_luma,
        ax=ax_trc,
        intensity_range=(0, max_val),
        input_limits=(in_lo, in_hi),
        output_limits=(tv_lo, tv_hi),
        show_identity=False,
        log_scale=True,
    )
    span = np.linspace(0, max_val, max_val + 1)
    lin_in = np.clip((span - in_lo) / (in_hi - in_lo), 0.0, 1.0)
    trc = tv_lo + (tv_hi - tv_lo) * np.array([linear_to_rec_709_trc(float(v)) for v in lin_in])
    ax_trc.plot(span, trc, "--", color=_TRC_COLOR, lw=1.4)
    label_x = int(0.35 * (in_lo + in_hi))
    ax_trc.text(
        label_x,
        trc[label_x] + 6,
        "expected BT.709 TRC",
        color=_TRC_COLOR,
        fontsize=8,
        rotation=38,
        rotation_mode="anchor",
        fontstyle="italic",
        ha="left",
        va="bottom",
    )
    ax_trc.set_xlabel(f"BEFORE luma  (input range {in_lo}–{in_hi}, {'full' if is_full else 'TV'})")
    ax_trc.set_ylabel("AFTER luma (BT.709, TV)")
    ax_trc.set_xlim(0, max_val)
    ax_trc.set_ylim(0, max_val)
    ax_trc.set_aspect("equal", adjustable="box")
    ax_trc.set_title(
        "Transfer characteristic\nobserved input→output vs expected BT.709 (curve anchored to input range)",
        fontsize=9,
    )
    apply_sparse_ticklabels(ax_trc.xaxis)
    apply_sparse_ticklabels(ax_trc.yaxis)
    return fig
