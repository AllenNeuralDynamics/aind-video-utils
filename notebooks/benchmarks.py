# %%
from aind_video_utils import benchmarking as avbench
import ffmpeg
import pandas as pd
import copy
import re

# %%
raw_filename = "/home/galen.lynch/encode-testing/raw/testing_videos/gamma_no-05282024161822-0000.avi"
# raw_filename = "/mnt/Data/encodes/lossless_raw.mp4"
out_filename = "/home/galen.lynch/encode-testing/benchmarks.csv"

# %%

ffmpeg_preamble = "ffmpeg -y -hide_banner -hwaccel cuda -colorspace rgb -color_primaries bt709 -color_trc linear"

h264_common_scale = (
    '-vf "scale=out_range=full:sws_dither=none:out_color_matrix=bt709"'
)
h265_common_scale = (
    '-vf "scale=out_range=full:sws_dither=none:out_color_matrix=bt709"'
)
pipeline_encode = f"{h264_common_scale} -vcodec h264_nvenc -pix_fmt yuv420p -crf 23 -preset fast -b:v 50M"
pipeline_encode_info = {
    "encode_str": pipeline_encode,
    "nickname": "pipeline_encode",
    "codec": "h264_nvenc",
    "compute_type": "GPU",
    "rate_control": "crf",
    "rate_parameter": 23,
    "preset": "fast",
}
lili_encode = f"{h264_common_scale} -c:v h264_nvenc -pix_fmt yuv420p -vsync 0 -2pass 0 -bf:v 0 -qp 24 -preset fast"
lili_encode_info = {
    "encode_str": lili_encode,
    "nickname": "lili_encode",
    "codec": "h264_nvenc",
    "compute_type": "GPU",
    "rate_control": "qp",
    "rate_parameter": 24,
    "preset": "fast",
}
h264_slow_encode_base = f"{h264_common_scale} -c:v libx264 -pix_fmt yuv420p -preset veryslow -vsync passthrough -crf "
h264_slow_encode_info = {
    "encode_str": h264_slow_encode_base,
    "nickname": "h264_slow_crf_",
    "codec": "libx264",
    "compute_type": "CPU",
    "rate_control": "crf",
    "preset": "veryslow",
}

h264_fast_encode_base = f"{h264_common_scale} -c:v libx264 -preset fast -pix_fmt yuv420p -vsync passthrough -crf "
h264_fast_encode_info = copy.deepcopy(h264_slow_encode_info)
h264_fast_encode_info["encode_str"] = h264_fast_encode_base
h264_fast_encode_info["nickname"] = "h264_fast_crf_"
h264_fast_encode_info["preset"] = "fast"

h264_nvenc_encode_cq_base = (
    f"{h264_common_scale}  -c:v h264_nvenc "
    "-pix_fmt nv12 -fps_mode passthrough "
    "-tune hq -preset p4 -rc vbr -b:v 0M  "
    "-maxrate 700M -bufsize 350M -rc vbr -cq "
)
h264_nvenc_encode_cq_info = {
    "encode_str": h264_nvenc_encode_cq_base,
    "nickname": "h264_nvenc_cq_",
    "codec": "h264_nvenc",
    "compute_type": "GPU",
    "rate_control": "crf",
    "preset": "p4",
}

h264_nvenc_encode_constrained = (
    f"{h264_common_scale}  -c:v h264_nvenc -pix_fmt nv12 -fps_mode passthrough "
    "-tune hq -preset p4 -rc vbr -cq 16 -qmin 0 -qmax 10"
)
h264_nvenc_encode_constrained_info = {
    "encode_str": h264_nvenc_encode_constrained,
    "nickname": "h264_nvenc_constrained",
    "codec": "h264_nvenc",
    "compute_type": "GPU",
    "rate_control": "cq",
    "rate_parameter": 10,
    "preset": "p4",
}

h265_fast_encode_base = f"{h265_common_scale} -c:v libx265 -pix_fmt yuv420p10le -preset fast -vsync passthrough -crf "
h265_fast_encode_info = {
    "encode_str": h265_fast_encode_base,
    "nickname": "h265_fast_crf_",
    "codec": "libx265",
    "compute_type": "CPU",
    "rate_control": "crf",
    "preset": "fast",
}
h265_slow_encode_base = f"{h265_common_scale} -c:v libx265 -pix_fmt yuv420p10le -preset veryslow -vsync passthrough -crf "
h265_slow_encode_info = copy.deepcopy(h265_fast_encode_info)
h265_slow_encode_info["encode_str"] = h265_slow_encode_base
h265_slow_encode_info["nickname"] = "h265_slow_crf_"
h265_slow_encode_info["preset"] = "veryslow"

h265_nvenc_encode_cq_base = (
    f"{h265_common_scale}  -c:v hevc_nvenc "
    "-pix_fmt p010le -fps_mode passthrough "
    "-tune hq -preset p4 -rc vbr -b:v 0M  "
    "-maxrate 700M -bufsize 350M -rc vbr -cq "
)
h265_nvenc_encode_cq_info = {
    "encode_str": h265_nvenc_encode_cq_base,
    "nickname": "h265_nvenc_cq_",
    "codec": "hevc_nvenc",
    "compute_type": "GPU",
    "rate_control": "crf",
    "preset": "p4",
}

h265_nvenc_encode_constrained = (
    f"{h265_common_scale}  -c:v hevc_nvenc "
    "-pix_fmt p010le -fps_mode passthrough "
    "-tune hq -preset p4 -rc vbr -cq 16 -qmin 0 -qmax 10"
)
h265_nvenc_encode_constrained_info = {
    "encode_str": h265_nvenc_encode_constrained,
    "nickname": "h265_nvenc_constrained",
    "codec": "hevc_nvenc",
    "compute_type": "GPU",
    "rate_control": "cq",
    "rate_parameter": 10,
    "preset": "p4",
}

twostage_encode_info = copy.deepcopy(h264_slow_encode_info)
crf_str = "18"
twostage_encode_info["encode_str"] = h264_slow_encode_base + crf_str
twostage_encode_info["nickname"] = "twostage_encode"
twostage_encode_info["rate_parameter"] = 18

h264_crfs = [12, 16, 18, 20, 22, 24]
h265_crfs = [14, 18, 20, 22, 24, 26]
nvenc_cqs = [4, 10, 15, 20, 26, 30, 35, 40]
h264_vmaf_kwargs = dict(format="yuv420p10le", range="full")
h265_vmaf_kwargs = dict(format="yuv420p10le", range="full")

# %%
rawprobe = ffmpeg.probe(raw_filename)
rawvidstreamprobe = rawprobe["streams"][0]
raw_bitrate = int(rawvidstreamprobe["bit_rate"])

benchmarks = []


# %%
def run_benchmark(
    raw_filename,
    ffmpeg_preamble,
    encode_info,
    vmaf_kwargs,
    raw_bitrate,
    orig_filename=None,
):
    estats, _, _ = avbench.encode_stats_and_vmaf(
        raw_filename,
        ffmpeg_preamble,
        encode_info["encode_str"],
        original_filename=orig_filename,
        vmaf_kwargs=vmaf_kwargs,
    )
    return {
        "nickname": encode_info["nickname"],
        "codec": encode_info["codec"],
        "compute_type": encode_info["compute_type"],
        "rate_control": encode_info["rate_control"],
        "rate_parameter": encode_info["rate_parameter"],
        "preset": encode_info["preset"],
        "bitrate": estats["bitrate"],
        "fps": estats["fps"],
        "vmaf": estats["vmaf"],
        "compression_ratio": raw_bitrate / estats["bitrate"],
        "encode_str": encode_info["encode_str"],
        "raw_filename": raw_filename,
    }


def fill_encode_info(encode_info, crf):
    filled_info = copy.deepcopy(encode_info)
    filled_info["encode_str"] = filled_info["encode_str"] + str(crf)
    filled_info["nickname"] = filled_info["nickname"] + str(crf)
    filled_info["rate_parameter"] = crf
    return filled_info


def fill_nvenc_encode_info(base_info, cq, preset):
    """Fill NVENC encoder config with both CQ and preset parameters.

    Parameters
    ----------
    base_info : dict
        Base encoder configuration with encode_str containing '-preset {default}'
    cq : int
        CQ (Constant Quality) value to append
    preset : str
        NVENC preset to use (e.g., 'p2', 'p3', 'p4')

    Returns
    -------
    dict
        Filled encoder configuration with updated encode_str, nickname, and metadata
    """
    filled_info = copy.deepcopy(base_info)

    # Replace preset in encode_str
    encode_str = filled_info["encode_str"]
    encode_str = re.sub(r'-preset \w+', f'-preset {preset}', encode_str)

    # Append CQ value (base string ends with '-cq ')
    encode_str += str(cq)

    # Update all fields
    filled_info["encode_str"] = encode_str
    filled_info["nickname"] = f"h264_nvenc_{preset}_cq{cq}"
    filled_info["rate_parameter"] = cq
    filled_info["preset"] = preset

    return filled_info


def run_benchmark_sweep(
    raw_filename: str,
    ffmpeg_preamble: str,
    encode_info_base: dict,
    param_values: list,
    vmaf_kwargs: dict,
    raw_bitrate: int,
) -> list[dict]:
    """Run benchmarks across multiple parameter values (CRF/CQ sweep).

    Parameters
    ----------
    raw_filename : str
        Input video file path
    ffmpeg_preamble : str
        FFmpeg command preamble
    encode_info_base : dict
        Base encoder configuration (encode_str should end with parameter placeholder)
    param_values : list
        List of CRF/CQ values to test
    vmaf_kwargs : dict
        VMAF calculation parameters
    raw_bitrate : int
        Raw video bitrate for compression ratio calculation

    Returns
    -------
    list[dict]
        List of benchmark result dictionaries
    """
    results = []
    for param in param_values:
        filled_info = fill_encode_info(encode_info_base, param)
        results.append(
            run_benchmark(
                raw_filename,
                ffmpeg_preamble,
                filled_info,
                vmaf_kwargs,
                raw_bitrate,
            )
        )
    return results


# %%
benchmarks.append(
    run_benchmark(
        raw_filename,
        ffmpeg_preamble,
        pipeline_encode_info,
        h264_vmaf_kwargs,
        raw_bitrate,
    )
)

benchmarks.append(
    run_benchmark(
        raw_filename,
        ffmpeg_preamble,
        lili_encode_info,
        h264_vmaf_kwargs,
        raw_bitrate,
    )
)
# %%
for crf in [12, 35, 40]:
    filled_info = fill_encode_info(h264_nvenc_encode_cq_info, crf)
    benchmarks.append(
        run_benchmark(
            raw_filename,
            ffmpeg_preamble,
            filled_info,
            h264_vmaf_kwargs,
            raw_bitrate,
        )
    )
    filled_info = fill_encode_info(h265_nvenc_encode_cq_info, crf)
    benchmarks.append(
        run_benchmark(
            raw_filename,
            ffmpeg_preamble,
            filled_info,
            h265_vmaf_kwargs,
            raw_bitrate,
        )
    )

# %%
for crf in [12]:
    for encode_info in [
        h264_fast_encode_info,
        h264_slow_encode_info,
    ]:
        filled_info = fill_encode_info(encode_info, crf)
        benchmarks.append(
            run_benchmark(
                raw_filename,
                ffmpeg_preamble,
                filled_info,
                h264_vmaf_kwargs,
                raw_bitrate,
            )
        )

for crf in [14]:
    for encode_info in [
        h265_fast_encode_info,
        h265_slow_encode_info,
    ]:
        filled_info = fill_encode_info(encode_info, crf)
        benchmarks.append(
            run_benchmark(
                raw_filename,
                ffmpeg_preamble,
                filled_info,
                h265_vmaf_kwargs,
                raw_bitrate,
            )
        )

# %%
# Create a dictionary to store first-stage files for two-stage testing
first_stage_files = {}  # {(cq_value, preset): (filename, benchmark_result)}

# Define presets to test for first-stage encoding
nvenc_presets = ["p2", "p3", "p4"]  # Medium-range presets for finer granularity

for preset in nvenc_presets:
    for cq in nvenc_cqs:
        # H.264 NVENC - preserve for two-stage testing
        filled_info = fill_nvenc_encode_info(h264_nvenc_encode_cq_info, cq, preset)

        estats, out_file, log_file = avbench.encode_stats_and_vmaf(
            raw_filename,
            ffmpeg_preamble,
            filled_info["encode_str"],
            vmaf_kwargs=h264_vmaf_kwargs,
            delete=False,  # Preserve for two-stage testing
        )

        benchmark_result = {
            "nickname": filled_info["nickname"],
            "codec": filled_info["codec"],
            "compute_type": filled_info["compute_type"],
            "rate_control": filled_info["rate_control"],
            "rate_parameter": cq,
            "preset": preset,
            "bitrate": estats["bitrate"],
            "fps": estats["fps"],
            "vmaf": estats["vmaf"],
            "compression_ratio": raw_bitrate / estats["bitrate"],
            "encode_str": filled_info["encode_str"],
            "raw_filename": raw_filename,
        }
        benchmarks.append(benchmark_result)

        # Store with (cq, preset) tuple key
        first_stage_files[(cq, preset)] = (out_file, benchmark_result)

    # H.265 NVENC - keep existing behavior (delete=True, single preset)
    # Only run once per preset to avoid redundancy
    if preset == "p4":  # Only test h265 with default preset
        for cq in nvenc_cqs:
            filled_info = fill_encode_info(h265_nvenc_encode_cq_info, cq)
            benchmarks.append(
                run_benchmark(
                    raw_filename,
                    ffmpeg_preamble,
                    filled_info,
                    h265_vmaf_kwargs,
                    raw_bitrate,
                )
            )
benchmarks.append(
    run_benchmark(
        raw_filename,
        ffmpeg_preamble,
        h264_nvenc_encode_constrained_info,
        h264_vmaf_kwargs,
        raw_bitrate,
    )
)
benchmarks.append(
    run_benchmark(
        raw_filename,
        ffmpeg_preamble,
        h265_nvenc_encode_constrained_info,
        h265_vmaf_kwargs,
        raw_bitrate,
    )
)

# %%
# H.264 CRF sweeps (CPU encoders)
for encode_info in [h264_fast_encode_info, h264_slow_encode_info]:
    benchmarks.extend(
        run_benchmark_sweep(
            raw_filename,
            ffmpeg_preamble,
            encode_info,
            h264_crfs,
            h264_vmaf_kwargs,
            raw_bitrate,
        )
    )

# H.265 CRF sweeps (CPU encoders)
for encode_info in [h265_fast_encode_info, h265_slow_encode_info]:
    benchmarks.extend(
        run_benchmark_sweep(
            raw_filename,
            ffmpeg_preamble,
            encode_info,
            h265_crfs,
            h265_vmaf_kwargs,
            raw_bitrate,
        )
    )
# %%
# Original two-stage example (keep for comparison with new approach)
# Uses a specific CQ=12, preset=p4 first stage for legacy comparison
if (12, "p4") in first_stage_files:
    legacy_first_stage, _ = first_stage_files[(12, "p4")]
    benchmarks.append(
        run_benchmark(
            legacy_first_stage,
            ffmpeg_preamble,
            twostage_encode_info,
            h264_vmaf_kwargs,
            raw_bitrate,
            orig_filename=raw_filename,
        )
    )

# %%
# Two-stage encoding: h264_nvenc (various presets/CQ) -> h264_slow (fixed CRF 18)
# Sweep various preset and CQ combinations for first stage

second_stage_crf = 18  # Fixed second-stage quality

for (first_cq, first_preset), (first_stage_file, first_stage_benchmark) in first_stage_files.items():
    # Create second-stage config
    second_stage_info = copy.deepcopy(h264_slow_encode_info)
    second_stage_info["encode_str"] = h264_slow_encode_base + str(second_stage_crf)
    second_stage_info["nickname"] = f"twostage_nvenc_{first_preset}_cq{first_cq}_slow_crf{second_stage_crf}"
    second_stage_info["rate_parameter"] = second_stage_crf

    # Run second-stage benchmark
    estats, _, _ = avbench.encode_stats_and_vmaf(
        first_stage_file,  # Input: first-stage output
        ffmpeg_preamble,
        second_stage_info["encode_str"],
        original_filename=raw_filename,  # VMAF against original
        vmaf_kwargs=h264_vmaf_kwargs,
    )

    # Record results with two-stage metadata
    benchmarks.append({
        "nickname": second_stage_info["nickname"],
        "codec": second_stage_info["codec"],
        "compute_type": "CPU",
        "rate_control": "two_stage",
        "rate_parameter": second_stage_crf,
        "preset": second_stage_info["preset"],
        "bitrate": estats["bitrate"],
        "fps": estats["fps"],
        "vmaf": estats["vmaf"],
        "compression_ratio": raw_bitrate / estats["bitrate"],
        "encode_str": second_stage_info["encode_str"],
        "raw_filename": raw_filename,
        "first_stage_cq": first_cq,
        "first_stage_preset": first_preset,  # NEW: Track preset
        "first_stage_codec": "h264_nvenc",
        "first_stage_bitrate": first_stage_benchmark["bitrate"],
        "first_stage_vmaf": first_stage_benchmark["vmaf"],
    })


# %%
df = pd.DataFrame(data=benchmarks)
df.to_csv(out_filename, index=False, sep="\t")

# %%
# Cleanup: Remove temporary first-stage files
import os

for cq, (filepath, _) in first_stage_files.items():
    try:
        os.remove(filepath)
        print(f"Removed temporary file: {filepath}")
    except OSError as e:
        print(f"Warning: Could not remove {filepath}: {e}")

# %%
