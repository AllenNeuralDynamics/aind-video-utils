# %%
from aind_video_utils import benchmarking as avbench
import ffmpeg
import pandas as pd
import copy

# %%
raw_filename = "/home/galen.lynch/encode-testing/raw/testing_videos/gamma_no-05282024161822-0000.avi"
first_stage = "/mnt/Data/encodes/stream1.mp4"
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
for crf in nvenc_cqs:
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
for crf in h264_crfs:
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

for crf in h265_crfs:
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
# two stage
benchmarks.append(
    run_benchmark(
        first_stage,
        ffmpeg_preamble,
        twostage_encode_info,
        h264_vmaf_kwargs,
        raw_bitrate,
        orig_filename=raw_filename,
    )
)


# %%
df = pd.DataFrame(data=benchmarks)
df.to_csv(out_filename, index=False, sep="\t")

# %%
