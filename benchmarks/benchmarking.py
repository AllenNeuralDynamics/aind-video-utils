"""Benchmarking utilities for measuring encode performance and VMAF quality."""

import re
import shlex
import subprocess as sp
import tempfile
import xml.etree.ElementTree as ET
from typing import Any

from aind_video_utils import probe

EncodeStats = dict[str, float | int]


def _capture_ffmpeg_stderr(cmd_parts: list[str]) -> str:
    """Run an ffmpeg command and return its stderr output."""
    result = sp.run(cmd_parts, stdout=sp.DEVNULL, stderr=sp.PIPE, text=True, check=True)
    return result.stderr


def get_vmaf_score_from_xml(log_filename: str) -> float:
    """Parse the mean VMAF score from a libvmaf XML log file.

    Parameters
    ----------
    log_filename : str
        Path to the XML log produced by ffmpeg's ``libvmaf`` filter.

    Returns
    -------
    float
        Mean VMAF score.
    """
    tree = ET.parse(log_filename)
    root = tree.getroot()
    pooled_node = root.find("pooled_metrics")
    if pooled_node is None:
        raise ValueError("Could not find pooled_metrics node in VMAF log")
    vmaf_metric = pooled_node.find("metric[@name='vmaf']")
    if vmaf_metric is None:
        raise ValueError("Could not find vmaf metric in VMAF log")
    return float(vmaf_metric.attrib["mean"])


def get_vmaf_score(
    raw_filename: str,
    distorted_filename: str,
    log_filename: str,
    range: str = "tv",
    format: str = "yuv420p10le",
    n_threads: int = 4,
) -> float:
    """Compute VMAF between a reference and distorted video.

    Runs ffmpeg with the ``libvmaf`` filter, writes an XML log, then
    parses and returns the mean score.

    Parameters
    ----------
    raw_filename : str
        Path to the reference (undistorted) video.
    distorted_filename : str
        Path to the distorted (encoded) video.
    log_filename : str
        Path where the VMAF XML log will be written.
    range : str
        Output color range for comparison (``"tv"`` or ``"pc"``).
    format : str
        Pixel format for comparison.
    n_threads : int
        Number of threads for libvmaf.

    Returns
    -------
    float
        Mean VMAF score.
    """
    vmaf_str = (
        f"ffmpeg -i {raw_filename} "
        f"-i {distorted_filename} "
        f'-lavfi "[0:v]setpts=PTS-STARTPTS,scale=out_range={range},'
        f"format={format}[reference]; "
        f"        [1:v]setpts=PTS-STARTPTS,scale=out_range={range},"
        f"format={format}[distorted]; "
        f"        [distorted][reference]libvmaf=log_fmt=xml:"
        f'log_path={log_filename}:n_threads={n_threads}" '
        "-f null -"
    )
    sp.run(
        shlex.split(vmaf_str),
        stderr=sp.DEVNULL,
        stdout=sp.DEVNULL,
        check=True,
        capture_output=False,
    )
    return get_vmaf_score_from_xml(log_filename)


def get_psnr_score(
    raw_filename: str,
    distorted_filename: str,
    log_filename: str | None = None,
    range: str = "tv",
    format: str = "yuv420p10le",
) -> float:
    """Compute average luma PSNR (dB) between reference and distorted videos.

    Runs ffmpeg with the ``psnr`` filter and parses the summary line printed
    to stderr.  Returns the luma (Y) channel average, which is the meaningful
    number for grayscale / scientific content where chroma is approximately
    constant.

    Parameters
    ----------
    raw_filename : str
        Path to the reference (undistorted) video.
    distorted_filename : str
        Path to the distorted (encoded) video.
    log_filename : str | None
        If provided, per-frame PSNR stats are written here.
    range : str
        Output color range for comparison (``"tv"`` or ``"pc"``).
    format : str
        Pixel format both inputs are converted to before comparison.

    Returns
    -------
    float
        Average luma PSNR in dB across all frames.
    """
    psnr_filter = "psnr"
    if log_filename is not None:
        psnr_filter += f"=stats_file={log_filename}"
    cmd = (
        f"ffmpeg -i {raw_filename} "
        f"-i {distorted_filename} "
        f'-lavfi "[0:v]setpts=PTS-STARTPTS,scale=out_range={range},'
        f"format={format}[reference]; "
        f"        [1:v]setpts=PTS-STARTPTS,scale=out_range={range},"
        f"format={format}[distorted]; "
        f'        [distorted][reference]{psnr_filter}" '
        "-f null -"
    )
    result = sp.run(
        shlex.split(cmd),
        stderr=sp.PIPE,
        stdout=sp.DEVNULL,
        text=True,
        check=True,
    )
    # Summary line: "PSNR y:42.654 u:48.123 v:48.456 average:43.890 min:... max:..."
    match = re.search(r"PSNR\s+y:([\d.]+)", result.stderr)
    if not match:
        raise ValueError("Could not find PSNR y: value in ffmpeg output")
    return float(match.group(1))


def get_ssim_score(
    raw_filename: str,
    distorted_filename: str,
    log_filename: str | None = None,
    range: str = "tv",
    format: str = "yuv420p10le",
) -> float:
    """Compute average luma SSIM between reference and distorted videos.

    Runs ffmpeg with the ``ssim`` filter and parses the summary line printed
    to stderr.  Returns the luma (Y) channel SSIM; on grayscale content the
    ``All`` weighted value would be inflated by near-constant chroma planes.

    Parameters
    ----------
    raw_filename : str
        Path to the reference video.
    distorted_filename : str
        Path to the distorted video.
    log_filename : str | None
        If provided, per-frame SSIM stats are written here.
    range : str
        Output color range for comparison.
    format : str
        Pixel format both inputs are converted to before comparison.

    Returns
    -------
    float
        Average luma SSIM (0-1, higher is better; >0.95 is typical "high
        quality," >0.99 typically transparent).
    """
    ssim_filter = "ssim"
    if log_filename is not None:
        ssim_filter += f"=stats_file={log_filename}"
    cmd = (
        f"ffmpeg -i {raw_filename} "
        f"-i {distorted_filename} "
        f'-lavfi "[0:v]setpts=PTS-STARTPTS,scale=out_range={range},'
        f"format={format}[reference]; "
        f"        [1:v]setpts=PTS-STARTPTS,scale=out_range={range},"
        f"format={format}[distorted]; "
        f'        [distorted][reference]{ssim_filter}" '
        "-f null -"
    )
    result = sp.run(
        shlex.split(cmd),
        stderr=sp.PIPE,
        stdout=sp.DEVNULL,
        text=True,
        check=True,
    )
    # Summary line: "SSIM Y:0.969 (15.19) U:0.992 (21.44) V:0.991 (20.85) All:0.978 (16.62)"
    match = re.search(r"SSIM\s+Y:([\d.]+)", result.stderr)
    if not match:
        raise ValueError("Could not find SSIM Y: value in ffmpeg output")
    return float(match.group(1))


def parse_ffmpeg_output_for_perf_stats(cmd_output: str) -> tuple[float, float]:
    """Parse encoding FPS and bitrate from ffmpeg stderr output.

    Parameters
    ----------
    cmd_output : str
        The stderr text captured from an ffmpeg encode run.

    Returns
    -------
    fps : float
        Encoding speed in frames per second.
    bitrate : float
        Output bitrate in bits per second.
    """
    outlines = cmd_output.splitlines()
    for lineno in reversed(range(len(outlines))):
        if outlines[lineno].startswith("frame="):
            break
    if lineno == 0 and not outlines[lineno].startswith("frame="):
        raise ValueError("Could not find frame= line in ffmpeg output")
    line = outlines[lineno]
    eq_reg = r" ?= ?"
    numeric_reg = r"([\d\.]+)"
    fps_m = re.search(f"fps{eq_reg}{numeric_reg}", line)
    if not fps_m:
        raise ValueError("Could not find fps in ffmpeg output")
    fps = float(fps_m.group(1))
    bitrate_m = re.search(f"bitrate{eq_reg}{numeric_reg} ?kbits/s", line)
    if not bitrate_m:
        raise ValueError("Could not find bitrate in ffmpeg output")
    bitrate_kbs = float(bitrate_m.group(1))
    return fps, bitrate_kbs * 1e3


def encode_stats(raw_filename: str, out_filename: str, preamble: str, filter_and_encode: str) -> EncodeStats:
    """Run an ffmpeg encode and return performance statistics.

    Parameters
    ----------
    raw_filename : str
        Path to the input video.
    out_filename : str
        Path for the encoded output.
    preamble : str
        ffmpeg flags before ``-i`` (e.g. ``"ffmpeg -hide_banner"``).
    filter_and_encode : str
        Filter and codec flags after the input (e.g. ``"-vf ... -c:v libx264"``).

    Returns
    -------
    EncodeStats
        Dict with ``"fps"`` and ``"bitrate"`` keys.
    """
    cmd_parts = shlex.split(preamble + f" -i {raw_filename} {filter_and_encode} {out_filename}")
    cmd_output = _capture_ffmpeg_stderr(cmd_parts)
    out_probe = probe(out_filename)
    bitrate = int(out_probe["streams"][0]["bit_rate"])
    fps, _ = parse_ffmpeg_output_for_perf_stats(cmd_output)
    return {"fps": fps, "bitrate": bitrate}


def encode_stats_and_vmaf(
    raw_filename: str,
    preamble: str,
    filter_and_encode: str,
    original_filename: str | None = None,
    dir: str | None = None,
    delete: bool = True,
    vmaf_kwargs: dict[str, Any] | None = None,
) -> tuple[EncodeStats, str | None, str | None]:
    """Encode a video, measure performance, and compute VMAF.

    Parameters
    ----------
    raw_filename : str
        Path to the input video.
    preamble : str
        ffmpeg flags before ``-i``.
    filter_and_encode : str
        Filter and codec flags after the input.
    original_filename : str | None
        Reference video for VMAF. Defaults to *raw_filename*.
    dir : str | None
        Directory for temporary files.
    delete : bool
        If True, delete temporary output and log files after scoring.
    vmaf_kwargs : dict[str, Any] | None
        Extra keyword arguments passed to :func:`get_vmaf_score`.

    Returns
    -------
    estats : EncodeStats
        Dict with ``"fps"``, ``"bitrate"``, and ``"vmaf"`` keys.
    out_filename : str | None
        Path to encoded file (None if *delete* is True).
    log_filename : str | None
        Path to VMAF log (None if *delete* is True).
    """
    if original_filename is None:
        original_filename = raw_filename
    if vmaf_kwargs is None:
        vmaf_kwargs = {}

    out_filename: str | None
    log_filename: str | None

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=delete, delete_on_close=False, dir=dir) as fp:
        out_filename = fp.name
        fp.close()
        estats = encode_stats(raw_filename, out_filename, preamble, filter_and_encode)
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=delete, delete_on_close=False, dir=dir) as log_fp:
            log_fp.close()
            log_filename = log_fp.name
            vmaf_score = get_vmaf_score(original_filename, out_filename, log_filename, **vmaf_kwargs)
        estats["vmaf"] = vmaf_score
    if not delete:
        out_filename = None
        log_filename = None
    return estats, out_filename, log_filename
