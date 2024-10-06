from . import ffmpeg_utils
import shlex
import tempfile
import re
import xml.etree.ElementTree as ET
import subprocess as sp
import ffmpeg


def get_vmaf_score_from_xml(log_filename):
    tree = ET.parse(log_filename)
    root = tree.getroot()
    pooled_node = root.find("pooled_metrics")
    if pooled_node is None:
        raise ValueError("Could not find pooled_metrics node in VMAF log")
    vmaf_metric = pooled_node.find("metric[@name='vmaf']")
    return float(vmaf_metric.attrib["mean"])


def get_vmaf_score(
    raw_filename,
    distorted_filename,
    log_filename,
    range="tv",
    format="yuv420p10le",
    n_threads=4,
):
    vmaf_str = (
        f"ffmpeg -i {raw_filename} "
        f"-i {distorted_filename} "
        f'-lavfi "[0:v]setpts=PTS-STARTPTS,scale=out_range={range},format={format}[reference]; '
        f"        [1:v]setpts=PTS-STARTPTS,scale=out_range={range},format={format}[distorted]; "
        f'        [distorted][reference]libvmaf=log_fmt=xml:log_path={log_filename}:n_threads={n_threads}" '
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


def parse_ffmpeg_output_for_perf_stats(cmd_output):
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


def encode_stats(raw_filename, out_filename, preamble, filter_and_encode):
    cmd_parts = shlex.split(
        preamble + f" -i {raw_filename} {filter_and_encode} {out_filename}"
    )
    cmd_output = ffmpeg_utils.capture_ffmpeg_command_output(cmd_parts)
    out_probe = ffmpeg.probe(out_filename)
    bitrate = int(out_probe["streams"][0]["bit_rate"])
    fps, _ = parse_ffmpeg_output_for_perf_stats(cmd_output)
    return {"fps": fps, "bitrate": bitrate}


def encode_stats_and_vmaf(
    raw_filename,
    preamble,
    filter_and_encode,
    original_filename=None,
    dir=None,
    delete=True,
    vmaf_kwargs={},
):
    if original_filename is None:
        original_filename = raw_filename
    with tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=delete, delete_on_close=False, dir=dir
    ) as fp:
        out_filename = fp.name
        fp.close()
        estats = encode_stats(
            raw_filename, out_filename, preamble, filter_and_encode
        )
        with tempfile.NamedTemporaryFile(
            suffix=".xml", delete=delete, delete_on_close=False, dir=dir
        ) as log_fp:
            log_fp.close()
            log_filename = log_fp.name
            vmaf_score = get_vmaf_score(
                original_filename, out_filename, log_filename, **vmaf_kwargs
            )
        estats["vmaf"] = vmaf_score
    if not delete:
        out_filename = None
        log_filename = None
    return estats, out_filename, log_filename
