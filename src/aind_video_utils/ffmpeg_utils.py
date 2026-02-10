from __future__ import annotations

import itertools
import shlex
import subprocess as sp
from pathlib import Path
from typing import Any

import ffmpeg  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from . import utils

ProbeDict = dict[str, Any]
PathLike = str | Path


_SUPPORTED_YUV_FORMATS_8BIT = {
    "".join(["yuv", r, chroma, "p"])
    for r, chroma in itertools.product(["j", ""], ["420", "422", "444"])
}

_SUPPORTED_YUV_FORMATS_10BIT = {p + "10le" for p in _SUPPORTED_YUV_FORMATS_8BIT}

_SUPPORTED_GBR_FORMATS_8BIT = {"gbrp"}
_SUPPORTED_GBR_FORMATS_10BIT = {"gbrp10le"}

_ALL_SUPPORTED_FORMATS_8BIT = _SUPPORTED_YUV_FORMATS_8BIT | _SUPPORTED_GBR_FORMATS_8BIT
_ALL_SUPPORTED_FORMATS_10BIT = (
    _SUPPORTED_YUV_FORMATS_10BIT | _SUPPORTED_GBR_FORMATS_10BIT
)


def _is_gbr_format(pix_fmt: str) -> bool:
    return pix_fmt in _SUPPORTED_GBR_FORMATS_8BIT | _SUPPORTED_GBR_FORMATS_10BIT


def get_yuv_format(probe_json: ProbeDict) -> str:
    return str(probe_json["streams"][0]["pix_fmt"])


def get_color_range(probe_json: ProbeDict) -> str:
    return str(probe_json["streams"][0]["color_range"])


def get_frame_dimensions(probe_json: ProbeDict) -> tuple[int, int]:
    vidstream = probe_json["streams"][0]
    return vidstream["width"], vidstream["height"]


def get_video_range_info(probe_json: ProbeDict) -> tuple[str, int]:
    vidstream = probe_json["streams"][0]
    pix_fmt = vidstream["pix_fmt"]
    color_range = vidstream["color_range"]
    bit_depth = pix_format_bit_depth(pix_fmt)
    return color_range, bit_depth


def luma_from_yuv420p_buff_eltype(
    pxdata: bytes, w: int, h: int, eltype: type[np.uint8 | np.uint16]
) -> npt.NDArray[np.uint8 | np.uint16]:
    y_len = w * h
    yarr = np.frombuffer(pxdata, dtype=eltype, count=y_len).reshape(h, w)
    return yarr  # type: ignore[return-value]


def rgb_from_rawvideo_rgb24_buff(
    pxdata: bytes, w: int, h: int
) -> npt.NDArray[np.uint8]:
    rgb_len = w * h * 3
    rgbarr = np.frombuffer(pxdata, dtype=np.uint8, count=rgb_len).reshape(h, w, 3)
    return rgbarr  # type: ignore[return-value]


def luma_from_rawvideo_yuvp420_buff(
    pxdata: bytes, w: int, h: int
) -> npt.NDArray[np.uint8]:
    return luma_from_yuv420p_buff_eltype(pxdata, w, h, np.uint8)  # type: ignore[return-value]


def luma_from_rawvideo_yuv420p10le_buff(
    pxdata: bytes, w: int, h: int
) -> npt.NDArray[np.uint16]:
    return luma_from_yuv420p_buff_eltype(pxdata, w, h, np.uint16)  # type: ignore[return-value]


def pix_format_bit_depth(pix_fmt: str) -> int:
    if pix_fmt in _ALL_SUPPORTED_FORMATS_10BIT:
        return 10
    elif pix_fmt in _ALL_SUPPORTED_FORMATS_8BIT:
        return 8
    raise ValueError(f"Unsupported pixel format: {pix_fmt}")


def extract_srgb_frame(
    video_path: PathLike, frame_time: float, coerce_input_color_space: bool = False
) -> npt.NDArray[np.uint8]:
    probe_json = ffmpeg.probe(video_path)
    pix_fmt = get_yuv_format(probe_json)
    w, h = get_frame_dimensions(probe_json)
    ms_string = utils.get_millisecond_string(frame_time)
    base_colorspace_filter = (
        "colorspace=trc=srgb:space=bt709:primaries=bt709:range=pc,format=rgb24"
    )
    if _is_gbr_format(pix_fmt):
        # Use zscale for GBR formats (colorspace filter requires YCbCr input)
        base_zscale = "zscale=matrixin=gbr:matrix=gbr:transfer=iec61966-2-1:range=full"
        if coerce_input_color_space:
            video_filter = base_zscale + ":transferin=linear,format=rgb24"
        else:
            video_filter = base_zscale + ",format=rgb24"
    elif coerce_input_color_space:
        video_filter = (
            "setparams=color_primaries=bt709:color_trc=linear:colorspace=bt709,"
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
    result = sp.run(
        cmd_parts, stdout=sp.PIPE, stderr=sp.DEVNULL, text=False, check=True
    )
    img_arr = rgb_from_rawvideo_rgb24_buff(result.stdout, w, h)
    return img_arr


def extract_luma_frame(
    video_path: PathLike, frame_time: float
) -> npt.NDArray[np.uint8] | npt.NDArray[np.uint16]:
    probe_json = ffmpeg.probe(video_path)
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
        cmd_parts = shlex.split(
            f"ffmpeg -hide_banner -loglevel error -y  -ss {ms_string} "
            f"-i {video_path} -vframes 1 -f rawvideo pipe:1"
        )

    result = sp.run(
        cmd_parts, stdout=sp.PIPE, stderr=sp.DEVNULL, text=False, check=True
    )
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
    return y


def capture_ffmpeg_command_output(cmd_parts: list[str]) -> str:
    result = sp.run(cmd_parts, stdout=sp.DEVNULL, stderr=sp.PIPE, text=True, check=True)
    return result.stderr
