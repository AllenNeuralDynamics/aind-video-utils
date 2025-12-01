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


def pix_format_bit_depth(yuv_format: str) -> int:
    if yuv_format in _SUPPORTED_YUV_FORMATS_10BIT:
        return 10
    elif yuv_format in _SUPPORTED_YUV_FORMATS_8BIT:
        return 8
    raise ValueError(f"Unsupported yuv format: {yuv_format}")


def extract_srgb_frame(
    video_path: PathLike, frame_time: float, coerce_input_color_space: bool = False
) -> npt.NDArray[np.uint8]:
    w, h = get_frame_dimensions(ffmpeg.probe(video_path))
    ms_string = utils.get_millisecond_string(frame_time)
    base_video_filter = (
        "colorspace=trc=srgb:space=bt709:primaries=bt709:range=pc,format=rgb24"
    )
    if coerce_input_color_space:
        video_filter = (
            "setparams=color_primaries=bt709:color_trc=linear:colorspace=bt709,"
            + base_video_filter
        )
    else:
        video_filter = base_video_filter
    cmd_parts = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        ms_string,
        "-i",
        video_path,
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
    yuv_format = get_yuv_format(probe_json)
    format_is_8_bit = yuv_format in _SUPPORTED_YUV_FORMATS_8BIT
    if not (format_is_8_bit or yuv_format in _SUPPORTED_YUV_FORMATS_10BIT):
        raise ValueError(f"Unsupported yuv format: {yuv_format}")
    w, h = get_frame_dimensions(probe_json)
    ms_string = utils.get_millisecond_string(frame_time)
    cmd_parts = shlex.split(
        f"ffmpeg -hide_banner -loglevel error -y  -ss {ms_string} -i {video_path} "
        f"-vframes 1 -f rawvideo pipe:1"
    )
    result = sp.run(
        cmd_parts, stdout=sp.PIPE, stderr=sp.DEVNULL, text=False, check=True
    )
    y: npt.NDArray[np.uint8] | npt.NDArray[np.uint16]
    if format_is_8_bit:
        y = luma_from_rawvideo_yuvp420_buff(result.stdout, w, h)
    else:
        y = luma_from_rawvideo_yuv420p10le_buff(result.stdout, w, h)
    return y


def capture_ffmpeg_command_output(cmd_parts: list[str]) -> str:
    result = sp.run(cmd_parts, stdout=sp.DEVNULL, stderr=sp.PIPE, text=True, check=True)
    return result.stderr
