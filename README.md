# AIND Video Utils

![CI](https://github.com/AllenNeuralDynamics/aind-video-utils/actions/workflows/ci-call.yml/badge.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/aind-video-utils)](https://pypi.org/project/aind-video-utils/)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border.json)](https://github.com/copier-org/copier)

Tools for working with video files using ffmpeg.

## Prerequisites

This package requires **ffmpeg** and **ffprobe** to be installed and available on your `PATH`.

- **Linux**: `sudo apt install ffmpeg` (Debian/Ubuntu) or `sudo dnf install ffmpeg` (Fedora)
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

## Installation

```bash
pip install aind-video-utils
```

For QC plotting and visualization features (matplotlib, opencv):

```bash
pip install aind-video-utils[plotting]
```

Or from source:

```bash
pip install .           # core only
pip install .[plotting] # with plotting
```

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

### CLI

With the `plotting` extra installed, the `aind-video-qc` command is available:

```bash
# Compare linear-light input against BT.709-encoded output
aind-video-qc linear-to-bt709 input.mp4 output.mp4

# Compare ffmpeg luma extraction with OpenCV decode
aind-video-qc opencv input.mp4

# Options
aind-video-qc linear-to-bt709 input.mp4 output.mp4 --frame-time 1.5 --dpi 300 -o qc.png
```

### Python API

```python
from aind_video_utils.video_qc import compare_linear_to_bt709

fig = compare_linear_to_bt709("input.mp4", "output.mp4", frame_time=0)
fig.savefig("qc.png")
```

To set up for development:
```bash
uv sync
```

## Development

Run the full linting and testing suite:

```bash
./scripts/run_linters_and_checks.sh -c
```

Or run individual commands:
```bash
uv run --frozen ruff format          # Code formatting
uv run --frozen ruff check           # Linting
uv run --frozen mypy                 # Type checking
uv run --frozen interrogate -v       # Documentation coverage
uv run --frozen codespell --check-filenames  # Spell checking
uv run --frozen pytest --cov aind_video_utils # Tests with coverage
```

## Contributing

We use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style commit messages:
```text
<type>(<scope>): <short summary>
```

Types: **feat**, **fix**, **docs**, **ci**, **build**, **perf**, **refactor**, **test**

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
