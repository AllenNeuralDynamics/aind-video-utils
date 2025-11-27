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


def get_yuv_format(probe_json: ProbeDict) -> str:
    return str(probe_json["streams"][0]["pix_fmt"])


def get_color_range(probe_json: ProbeDict) -> str:
    return str(probe_json["streams"][0]["color_range"])


def get_frame_dimensions(probe_json: ProbeDict) -> tuple[int, int]:
    vidstream = probe_json["streams"][0]
    return vidstream["width"], vidstream["height"]


def extract_yuv420p_eltype(
    pxdata: bytes, w: int, h: int, eltype: type[np.uint8 | np.uint16]
) -> npt.NDArray[np.uint8 | np.uint16]:
    y_len = w * h
    yarr = np.frombuffer(pxdata, dtype=eltype, count=y_len).reshape(h, w)
    return yarr  # type: ignore[return-value]


def extract_yuv420p_y(pxdata: bytes, w: int, h: int) -> npt.NDArray[np.uint8]:
    return extract_yuv420p_eltype(pxdata, w, h, np.uint8)  # type: ignore[return-value]


def extract_yuv420p10le_y(pxdata: bytes, w: int, h: int) -> npt.NDArray[np.uint16]:
    return extract_yuv420p_eltype(pxdata, w, h, np.uint16)  # type: ignore[return-value]


def extract_yuv_frame(
    video_path: PathLike, frame_time: float
) -> npt.NDArray[np.uint8] | npt.NDArray[np.uint16]:
    ms_string = utils.get_millisecond_string(frame_time)
    cmd_parts = shlex.split(
        f"ffmpeg -y -hide_banner -ss {ms_string} -i {video_path} "
        f"-vframes 1 -f rawvideo pipe:1"
    )
    result = sp.run(
        cmd_parts, stdout=sp.PIPE, stderr=sp.DEVNULL, text=False, check=True
    )
    probe_json = ffmpeg.probe(video_path)
    yuv_format = get_yuv_format(probe_json)
    w, h = get_frame_dimensions(probe_json)
    pix_bases = [
        "".join(["yuv", r, chroma, "p"])
        for r, chroma in itertools.product(["j", ""], ["420", "422", "444"])
    ]
    y: npt.NDArray[np.uint8] | npt.NDArray[np.uint16]
    if yuv_format in pix_bases:
        y = extract_yuv420p_y(result.stdout, w, h)
    elif yuv_format in [p + "10le" for p in pix_bases]:
        y = extract_yuv420p10le_y(result.stdout, w, h)
    else:
        raise ValueError(f"Unsupported yuv format: {yuv_format}")
    return y


def capture_ffmpeg_command_output(cmd_parts: list[str]) -> str:
    result = sp.run(cmd_parts, stdout=sp.DEVNULL, stderr=sp.PIPE, text=True, check=True)
    return result.stderr
