"""High-level functions for extracting video frames via ffmpeg."""

from __future__ import annotations

import logging
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
from aind_video_utils.mp4_index import Mp4FrameIndex, read_mp4_frame_index
from aind_video_utils.probe import (
    ProbeDict,
    get_color_transfer,
    get_duration_seconds,
    get_frame_dimensions,
    get_video_range_info,
    get_yuv_format,
    probe,
)
from aind_video_utils.utils import http_input_flags

logger = logging.getLogger(__name__)


def _effective_frame_time(probe_json: ProbeDict, requested: float) -> float:
    """Clamp ``requested`` to lie within the video's actual duration.

    Very short sources (test clips) can be shorter than a QC's target
    timestamp; without a clamp ffmpeg seeks past EOF and returns rc=1
    with no output. Falls back to ``requested`` unchanged when duration
    cannot be resolved from probe metadata.
    """
    duration = get_duration_seconds(probe_json)
    if duration is None or requested <= duration:
        return requested
    # Back off one frame-period-ish from the end. r_frame_rate gives fps;
    # if unavailable, use a 10 ms epsilon (finer grain isn't meaningful for
    # frame seeking).
    stream = probe_json["streams"][0]
    epsilon = 0.010
    rate = stream.get("r_frame_rate")
    if rate:
        try:
            num, den = rate.split("/")
            fps = int(num) / int(den)
            if fps > 0:
                epsilon = 1.0 / fps
        except (ValueError, ZeroDivisionError):
            pass
    return max(0.0, duration - epsilon)


def extract_srgb_frame(
    video_path: str | Path,
    frame_time: float,
    coerce_input_color_space: bool = False,
    input_is_full: bool = True,
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
    input_is_full : bool, optional
        Declared luma range of the *input* when coercing color space: ``True``
        (default) tags it full/PC, ``False`` tags it limited/TV. Only affects the
        coerced YUV path (the ``setparams`` range); a limited-range source read as
        PC would lift its blacks. AIND linear sources are PC, but mpeg4 yuv420p
        sources are actually TV (see the mpeg4 TV-range fix) — pass ``False`` there.

    Returns
    -------
    NDArray[np.uint8]
        RGB image with shape ``(h, w, 3)``.
    """
    probe_json = probe(video_path)
    pix_fmt = get_yuv_format(probe_json)
    w, h = get_frame_dimensions(probe_json)
    effective_time = _effective_frame_time(probe_json, frame_time)
    ms_string = utils.get_millisecond_string(effective_time)
    base_colorspace_filter = "colorspace=trc=srgb:space=bt709:primaries=bt709:range=pc,format=rgb24"
    # Both the GBR (zscale) and YUV (colorspace filter) branches need the
    # source's transfer characteristic to resolve a path. AIND mpeg4 and
    # h264_nvenc sources are typically untagged (no `color_transfer` in
    # metadata) — without an explicit `transferin=...` or `setparams=...`
    # the filters fail with "no path between colorspaces" (exit 187 / -22
    # EINVAL). Treat missing tag as linear (AIND convention) and warn so
    # callers know we inferred; passing coerce_input_color_space=True
    # silences the warning.
    source_transfer = get_color_transfer(probe_json)
    must_coerce = coerce_input_color_space or source_transfer is None
    if must_coerce and source_transfer is None and not coerce_input_color_space:
        logger.warning(
            "%s has no color_transfer in metadata; defaulting to "
            "linear-light input assumption. Pass "
            "coerce_input_color_space=True to silence this warning.",
            video_path,
        )
    if _is_gbr_format(pix_fmt):
        # zscale for GBR formats (the colorspace filter requires YCbCr).
        base_zscale = "zscale=matrixin=gbr:matrix=gbr:transfer=iec61966-2-1:range=full"
        if must_coerce:
            video_filter = base_zscale + ":transferin=linear,format=rgb24"
        else:
            video_filter = base_zscale + ",format=rgb24"
    elif must_coerce:
        # range=pc is critical here: without it, the subsequent
        # `colorspace=range=pc` filter sees an unspecified input range and
        # defaults BT.709 to limited (TV), then does a bogus TV→PC
        # expansion that crushes linear sub-16 source values toward 0.
        # AIND linear-light sources ARE PC range (no headroom/footroom),
        # so tag it explicitly. Caught 2026-06-25 while QCing the smoke
        # batch — input sRGB extraction was 30 codes darker than the
        # otherwise-identical output sRGB, all from this missing tag.
        setparams_range = "pc" if input_is_full else "tv"
        video_filter = (
            f"setparams=color_primaries=bt709:color_trc=linear:colorspace=bt709:range={setparams_range},"
            + base_colorspace_filter
        )
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


def extract_frame_by_index(
    video_path: str | Path,
    display_index: int,
    *,
    index: Mp4FrameIndex | None = None,
    probe_json: ProbeDict | None = None,
) -> npt.NDArray[np.uint8]:
    """Extract the frame at presentation-order *display_index* from an MP4, frame-exact.

    Addresses frames by *number* rather than by time.  Reads the MP4 ``moov``
    sample tables (see :func:`~aind_video_utils.read_mp4_frame_index`) to find
    the keyframe beginning *display_index*'s GOP, seeks there with input-side
    ``-ss``, and decodes forward to the target — the recipe verified to be
    frame-exact, including the edit-list ``media_time`` compensation that a
    naive time seek gets wrong.

    Requires a local, seekable, non-fragmented MP4 whose edit list is
    frame-addressing-safe.  Returns a fast preview decode using ffmpeg's default
    color conversion; for color-managed sRGB output use
    :func:`extract_srgb_frame` at the seconds returned by the index's
    ``seek_plan``.

    Parameters
    ----------
    video_path : str | Path
        Path to a local MP4/MOV file.
    display_index : int
        0-based frame index in presentation order.
    index : Mp4FrameIndex, optional
        Pre-parsed frame index.  Pass it when extracting several frames from the
        same file to avoid re-reading the ``moov`` each call.
    probe_json : ProbeDict, optional
        Pre-computed ffprobe output (used only for frame dimensions).

    Returns
    -------
    NDArray[np.uint8]
        RGB image with shape ``(h, w, 3)``.

    Raises
    ------
    ValueError
        If *display_index* is out of range or the file's edit list is not
        frame-addressing-safe.
    """
    if index is None:
        index = read_mp4_frame_index(video_path)
    seek_seconds, frames_after_keyframe = index.seek_plan(display_index)
    if probe_json is None:
        probe_json = probe(video_path)
    w, h = get_frame_dimensions(probe_json)
    cmd_parts = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        # 9 decimals rounds to the exact media tick, so -ss lands on the target
        # keyframe rather than its predecessor.
        "-ss",
        f"{seek_seconds:.9f}",
        *http_input_flags(video_path),
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{frames_after_keyframe}),format=rgb24",
        "-frames:v",
        "1",
        # passthrough so the implicit CFR vsync stage can't drop the selected
        # frame based on its post-seek timestamp.
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = sp.run(cmd_parts, stdout=sp.PIPE, stderr=sp.DEVNULL, text=False, check=True)
    return rgb_from_rawvideo_rgb24_buff(result.stdout, w, h)


def extract_luma_frame(
    video_path: str | Path,
    frame_time: float,
    probe_json: ProbeDict | None = None,
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
    probe_json : ProbeDict, optional
        Pre-computed ffprobe output. Avoids re-probing when extracting
        many frames from the same video.

    Returns
    -------
    luma : NDArray[np.uint8] | NDArray[np.uint16]
        Luma plane with shape ``(h, w)``.
    color_range : str
        ``"pc"`` (full) or ``"tv"`` (limited).
    bit_depth : int
        Bits per component (8 or 10).
    """
    if probe_json is None:
        probe_json = probe(video_path)
    pix_fmt = get_yuv_format(probe_json)
    format_is_8_bit = pix_fmt in _ALL_SUPPORTED_FORMATS_8BIT
    if not (format_is_8_bit or pix_fmt in _ALL_SUPPORTED_FORMATS_10BIT):
        raise ValueError(f"Unsupported pixel format: {pix_fmt}")
    w, h = get_frame_dimensions(probe_json)
    effective_time = _effective_frame_time(probe_json, frame_time)
    ms_string = utils.get_millisecond_string(effective_time)

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
