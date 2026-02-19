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

This package is not published on PyPI. Install from GitHub directly:

```bash
# uv
uv add git+https://github.com/AllenNeuralDynamics/aind-video-utils.git
uv add "aind-video-utils[plotting] @ git+https://github.com/AllenNeuralDynamics/aind-video-utils.git"

# pip
pip install git+https://github.com/AllenNeuralDynamics/aind-video-utils.git
pip install "aind-video-utils[plotting] @ git+https://github.com/AllenNeuralDynamics/aind-video-utils.git"
```

Or from a local clone:

```bash
# uv
uv add --editable .           # core only
uv add --editable ".[plotting]" # with plotting

# pip
pip install -e .           # core only
pip install -e ".[plotting]" # with plotting
```

The `plotting` extra adds matplotlib and opencv, required for QC visualization and the `aind-video-qc` CLI.

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
