import ffmpeg
import numpy as np
import subprocess as sp
import shlex

from . import utils
import itertools


def get_yuv_format(probe_json):
    return probe_json["streams"][0]["pix_fmt"]


def get_frame_dimensions(probe_json):
    vidstream = probe_json["streams"][0]
    return vidstream["width"], vidstream["height"]


def extract_yuv420p_eltype(pxdata, w, h, eltype):
    y_len = w * h
    yarr = np.frombuffer(pxdata, dtype=eltype, count=y_len).reshape(h, w)
    return yarr


def extract_yuv420p_y(pxdata, w, h):
    return extract_yuv420p_eltype(pxdata, w, h, np.uint8)


def extract_yuv420p10le_y(pxdata, w, h):
    return extract_yuv420p_eltype(pxdata, w, h, np.uint16)


def extract_yuv_frame(video_path, frame_time):
    ms_string = utils.get_millisecond_string(frame_time)
    cmd_parts = shlex.split(
        f"ffmpeg -y -hide_banner -ss {ms_string} -i {video_path} -vframes 1 -f rawvideo pipe:1"
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
    if yuv_format in pix_bases:
        y = extract_yuv420p_y(result.stdout, w, h)
    elif yuv_format in [p + "10le" for p in pix_bases]:
        y = extract_yuv420p10le_y(result.stdout, w, h)
    else:
        raise ValueError(f"Unsupported yuv format: {yuv_format}")
    return y


def capture_ffmpeg_command_output(cmd_parts):
    result = sp.run(
        cmd_parts, stdout=sp.DEVNULL, stderr=sp.PIPE, text=True, check=True
    )
    return result.stderr
