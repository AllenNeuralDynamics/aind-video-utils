"""High-level functions for extracting video frames via ffmpeg."""

from __future__ import annotations

import subprocess as sp
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aind_video_utils import utils
from aind_video_utils._rawvideo import (
    _ALL_SUPPORTED_FORMATS_8BIT,
    _ALL_SUPPORTED_FORMATS_10BIT,
    _is_gbr_format,
    luma_from_rawvideo_yuv420p10le_buff,
    luma_from_rawvideo_yuvp420_buff,
    luma_from_yuv420p_buff_eltype,
    rgb_from_rawvideo_rgb24_buff,
)
from aind_video_utils.probe import get_frame_dimensions, get_video_range_info, get_yuv_format, probe
from aind_video_utils.utils import http_input_flags


def extract_srgb_frame(
    video_path: str | Path, frame_time: float, coerce_input_color_space: bool = False
) -> npt.NDArray[np.uint8]:
    """Extract a single frame from a video, converted to sRGB RGB24.

    Handles both YUV and GBR pixel formats. For YUV, uses the ffmpeg
    ``colorspace`` filter; for GBR, uses ``zscale``.

    Parameters
    ----------
    video_path : str | Path
        Path to the video file.
    frame_time : float
        Time in seconds at which to extract the frame.
    coerce_input_color_space : bool, optional
        If True, override the stream's transfer characteristic metadata
        (assume linear light input).

    Returns
    -------
    NDArray[np.uint8]
        RGB image with shape ``(h, w, 3)``.
    """
    probe_json = probe(video_path)
    pix_fmt = get_yuv_format(probe_json)
    w, h = get_frame_dimensions(probe_json)
    ms_string = utils.get_millisecond_string(frame_time)
    base_colorspace_filter = "colorspace=trc=srgb:space=bt709:primaries=bt709:range=pc,format=rgb24"
    if _is_gbr_format(pix_fmt):
        # Use zscale for GBR formats (colorspace filter requires YCbCr input)
        base_zscale = "zscale=matrixin=gbr:matrix=gbr:transfer=iec61966-2-1:range=full"
        if coerce_input_color_space:
            video_filter = base_zscale + ":transferin=linear,format=rgb24"
        else:
            video_filter = base_zscale + ",format=rgb24"
    elif coerce_input_color_space:
        video_filter = "setparams=color_primaries=bt709:color_trc=linear:colorspace=bt709," + base_colorspace_filter
    else:
        video_filter = base_colorspace_filter
    cmd_parts = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        ms_string,
        *http_input_flags(video_path),
        "-i",
        str(video_path),
        "-vf",
        video_filter,
        "-vframes",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = sp.run(cmd_parts, stdout=sp.PIPE, stderr=sp.DEVNULL, text=False, check=True)
    img_arr = rgb_from_rawvideo_rgb24_buff(result.stdout, w, h)
    return img_arr


def extract_luma_frame(
    video_path: str | Path, frame_time: float
) -> tuple[npt.NDArray[np.uint8] | npt.NDArray[np.uint16], str, int]:
    """Extract the luma (Y) plane from a single video frame.

    For YUV formats the Y plane is read directly; for GBR formats,
    ffmpeg converts to grayscale using BT.709 luminance coefficients.

    Parameters
    ----------
    video_path : str | Path
        Path to the video file.
    frame_time : float
        Time in seconds at which to extract the frame.

    Returns
    -------
    luma : NDArray[np.uint8] | NDArray[np.uint16]
        Luma plane with shape ``(h, w)``.
    color_range : str
        ``"pc"`` (full) or ``"tv"`` (limited).
    bit_depth : int
        Bits per component (8 or 10).
    """
    probe_json = probe(video_path)
    pix_fmt = get_yuv_format(probe_json)
    format_is_8_bit = pix_fmt in _ALL_SUPPORTED_FORMATS_8BIT
    if not (format_is_8_bit or pix_fmt in _ALL_SUPPORTED_FORMATS_10BIT):
        raise ValueError(f"Unsupported pixel format: {pix_fmt}")
    w, h = get_frame_dimensions(probe_json)
    ms_string = utils.get_millisecond_string(frame_time)

    if _is_gbr_format(pix_fmt):
        # For GBR formats, use ffmpeg to compute BT.709 luminance
        gray_fmt = "gray" if format_is_8_bit else "gray16le"
        cmd_parts = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            ms_string,
            *http_input_flags(video_path),
            "-i",
            str(video_path),
            "-vf",
            f"format={gray_fmt}",
            "-vframes",
            "1",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    else:
        cmd_parts = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            ms_string,
            *http_input_flags(video_path),
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-f",
            "rawvideo",
            "pipe:1",
        ]

    result = sp.run(cmd_parts, stdout=sp.PIPE, stderr=sp.DEVNULL, text=False, check=True)
    y: npt.NDArray[np.uint8] | npt.NDArray[np.uint16]
    if _is_gbr_format(pix_fmt):
        # Output is a single gray plane (w*h)
        if format_is_8_bit:
            y = luma_from_yuv420p_buff_eltype(result.stdout, w, h, np.uint8)  # type: ignore[assignment]
        else:
            y = luma_from_yuv420p_buff_eltype(result.stdout, w, h, np.uint16)  # type: ignore[assignment]
    elif format_is_8_bit:
        y = luma_from_rawvideo_yuvp420_buff(result.stdout, w, h)
    else:
        y = luma_from_rawvideo_yuv420p10le_buff(result.stdout, w, h)
    color_range, bit_depth = get_video_range_info(probe_json)
    return y, color_range, bit_depth
