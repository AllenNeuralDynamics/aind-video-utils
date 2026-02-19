import cv2
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from aind_video_utils.color_spaces import linear_to_rec_709_trc, luma_range
from aind_video_utils.ffmpeg_utils import (
    PathLike,
    extract_luma_frame,
    extract_srgb_frame,
    get_video_range_info,
)
from aind_video_utils.plotting import (
    bivariate_with_marginals,
    imshow_clipping,
    luma_comparison_figure,
)

LumaFrame = NDArray[np.uint8] | NDArray[np.uint16]
sRGBFrame = NDArray[np.uint8]


def get_frame_pair_from_video(
    video_path: str,
    frame_time: float,
    coerce_color_space: bool = False,
) -> tuple[LumaFrame, sRGBFrame, int, bool]:
    probe_json = ffmpeg.probe(video_path)
    color_range, bit_depth = get_video_range_info(probe_json)
    luma = extract_luma_frame(video_path, frame_time)[0]
    srgb = extract_srgb_frame(video_path, frame_time, coerce_color_space)
    return luma, srgb, bit_depth, color_range == "pc"


def compare_input_output_frames(
    input_video_path: PathLike,
    output_video_path: PathLike,
    frame_time: float,
    coerce_input_color_space: bool = False,
) -> Figure:
    """Extract and compare frames from input and output videos at a specific time.

    Args:
        input_video_path (PathLike): Path to the input video file.
        output_video_path (PathLike): Path to the output video file.
        frame_time (float): Time in seconds to extract the frames.
        coerce_input_color_space (bool, optional): Whether to coerce the input video
            color space to sRGB. Defaults to False.
    """
    luma_input, srgb_input, depth_input, is_full_range_input = (
        get_frame_pair_from_video(
            input_video_path, frame_time, coerce_input_color_space
        )
    )
    luma_output, srgb_output, depth_output, is_full_range_output = (
        get_frame_pair_from_video(output_video_path, frame_time, False)
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
    input_video_path: PathLike,
) -> Figure:
    vidcap = cv2.VideoCapture(input_video_path)
    _, image = vidcap.read()
    opencv_frame = image
    luma_frame, color_range, bit_depth = extract_luma_frame(input_video_path, 0)
    is_full_range = color_range == "pc"
    luma_low, luma_high = luma_range(bit_depth, is_full_range)
    fig = plt.figure(figsize=(8, 8))

    gs = plt.GridSpec(
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
