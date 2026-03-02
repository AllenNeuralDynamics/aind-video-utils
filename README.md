# AIND Video Utils

![CI](https://github.com/AllenNeuralDynamics/aind-video-utils/actions/workflows/ci-call.yml/badge.svg)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Tools for working with video files using ffmpeg.

## Prerequisites

This package requires **ffmpeg** and **ffprobe** to be installed and available on your `PATH`.

- **Linux**: `sudo apt install ffmpeg` (Debian/Ubuntu) or `sudo dnf install ffmpeg` (Fedora)
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

## Installation

```bash
# uv
uv add aind-video-utils                        # core only
uv add "aind-video-utils[transcode]"            # with transcode CLI
uv add "aind-video-utils[plotting]"             # with QC plotting

# pip
pip install aind-video-utils
pip install "aind-video-utils[transcode]"
pip install "aind-video-utils[plotting]"
```

| Extra | Adds |
|-------|------|
| `transcode` | pydantic-settings, rich — required for `aind-transcode` CLI |
| `plotting` | matplotlib, opencv — required for `aind-video-qc` CLI |

## Usage

```python
from aind_video_utils import probe, extract_luma_frame, extract_srgb_frame

# Probe video metadata
info = probe("video.mp4")
print(info["streams"][0]["pix_fmt"])  # e.g. "yuv420p", "gbrp"

# Extract luma (Y) plane at t=1.0s
luma, color_range, bit_depth = extract_luma_frame("video.mp4", 1.0)

# Extract an sRGB frame at t=1.0s
srgb = extract_srgb_frame("video.mp4", 1.0)
```

### Encoding Profiles

This package is the canonical Python source for the [AIND behavior video file standard](https://allenneuraldynamics.github.io/aind-file-standards/file_formats/behavior_videos/) encoding profiles. Four profiles are provided as frozen dataclass constants:

| Constant | Codec | Pixel Format | Container | Use Case |
|----------|-------|-------------|-----------|----------|
| `OFFLINE_8BIT` | libx264 | yuv420p | mp4 | Long-term storage (default) |
| `OFFLINE_10BIT` | libx264 | yuv420p10le | mp4 | Long-term storage, 10-bit |
| `ONLINE_8BIT` | h264_nvenc | yuv420p | mkv | Real-time acquisition |
| `ONLINE_10BIT` | hevc_nvenc | p010le | mkv | Real-time acquisition, 10-bit |

```python
from aind_video_utils import OFFLINE_8BIT, ONLINE_8BIT

# Inspect the exact ffmpeg args
OFFLINE_8BIT.ffmpeg_output_args()
ONLINE_8BIT.ffmpeg_input_args()

# Customize with replace()
fast = OFFLINE_8BIT.replace(codec_params=("-preset", "veryfast", "-crf", "18"))
```

For online acquisition pipelines that build their own ffmpeg command:

```python
from aind_video_utils import ONLINE_8BIT

profile = ONLINE_8BIT
cmd = [
    "ffmpeg",
    *profile.ffmpeg_input_args(),
    "-f", "rawvideo", "-pix_fmt", "bgr24",
    "-s", f"{w}x{h}", "-r", str(fps),
    "-i", "pipe:0",
    *profile.ffmpeg_output_args(),
    str(output_path),
]
```

### Transcoding Python API

```python
from aind_video_utils import transcode_video, OFFLINE_8BIT

# Simplest form — offline 8-bit with automatic colorspace fix
transcode_video(input_path, output_path)

# Explicit profile
transcode_video(input_path, output_path, profile=OFFLINE_8BIT)

# Custom profile with speed override
fast = OFFLINE_8BIT.replace(codec_params=("-preset", "veryfast", "-crf", "18"))
transcode_video(input_path, output_path, profile=fast)

# Skip automatic setparams probing
transcode_video(input_path, output_path, auto_fix_colorspace=False)
```

### Transcode CLI

With the `transcode` extra installed, the `aind-transcode` command is available:

```bash
aind-transcode videos/                                # defaults: offline-8bit, auto-fix
aind-transcode videos/ --profile offline-10bit        # explicit profile
aind-transcode videos/ --preset veryfast              # override speed
aind-transcode videos/ --crf 20 --preset veryfast     # override quality + speed
aind-transcode videos/ --no-auto-fix-colorspace       # skip setparams probing
aind-transcode videos/ --jobs 4                       # parallel workers
```

Settings can also be stored in `aind-transcode.toml` in the working directory:

```toml
input = ["videos/"]
output_dir = "transcoded"
profile = "offline-8bit"
overwrite = false
```

### QC CLI

With the `plotting` extra installed, the `aind-video-qc` command is available:

```bash
# Compare linear-light input against BT.709-encoded output
aind-video-qc linear-to-bt709 input.mp4 output.mp4

# Compare ffmpeg luma extraction with OpenCV decode
aind-video-qc opencv input.mp4

# Options
aind-video-qc linear-to-bt709 input.mp4 output.mp4 --frame-time 1.5 --dpi 300 -o qc.png
```

### QC Python API

```python
from aind_video_utils.video_qc import compare_linear_to_bt709

fig = compare_linear_to_bt709("input.mp4", "output.mp4", frame_time=0)
fig.savefig("qc.png")
```

## Development

```bash
uv sync                              # install all dev dependencies
./scripts/run_linters_and_checks.sh -c  # run full lint + test suite
```

## Contributing

We use [Conventional Commits](https://www.conventionalcommits.org/):
```text
<type>(<scope>): <short summary>
```

Types: **feat**, **fix**, **docs**, **ci**, **build**, **perf**, **refactor**, **test**

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
