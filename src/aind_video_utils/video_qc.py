"""QC functions comparing video frames before and after encoding."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

from aind_video_utils.color_spaces import linear_to_rec_709_trc, luma_range
from aind_video_utils.frames import extract_luma_frame, extract_srgb_frame
from aind_video_utils.plotting import (
    bivariate_with_marginals,
    imshow_clipping,
    intensity_histogram,
    luma_comparison_figure,
)
from aind_video_utils.probe import get_video_range_info, probe

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
    return luma, srgb, bit_depth, color_range == "pc"


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

    def bt709_trc_fcn(x: float) -> float:
        return 16 + 219 * linear_to_rec_709_trc(x / 255)

    luma_space = np.linspace(0, 255, 256)
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
