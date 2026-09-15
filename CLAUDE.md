# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Runtime requirement

`ffmpeg` and `ffprobe` must be on `PATH` — every code path in this package shells out to them. Local dev assumes a working system install.

## Common commands

```bash
uv sync                                       # install all dev dependencies (incl. plotting + transcode extras)
./scripts/run_linters_and_checks.sh           # ruff format only
./scripts/run_linters_and_checks.sh -c        # full lint + test suite (ruff, mypy, interrogate, codespell, pytest)
./scripts/run_linters_and_checks.sh -c -- -k test_name   # forward pytest args after `--`

uv run pytest tests/test_transcode.py::test_name  # run a single test directly
ruff format && ruff check                      # ruff is installed globally — invoke directly, not via uv run
mypy                                           # ditto
```

CI lives in `AllenNeuralDynamics/galen-uv-workflows` (reusable workflow `.github/workflows/ci-call.yml`) and runs the same checks against Python 3.10 and 3.13.

## Architecture

This package is the canonical Python source for the [AIND behavior video file standard](https://allenneuraldynamics.github.io/aind-file-standards/file_formats/behavior_videos/). The encoding profiles in `encoding.py` are the source of truth — `SPEC_VERSION` tracks which revision of the file-standards spec they implement and is independent of the package version.

Module layout (everything under `src/aind_video_utils/`):

- `encoding.py` — `EncodingProfile` frozen dataclass + the four canonical preset constants (`OFFLINE_8BIT`, `OFFLINE_10BIT`, `ONLINE_8BIT`, `ONLINE_10BIT`) and `PROFILES` lookup. A profile splits its filters in two: `source_filters` is per-source repair applied before any branch (`setparams` for missing colour metadata via `with_setparams()`, `setpts` for CFR normalization via `prepend_conditioning()`), and `video_filters` is the encoding chain. Callers compose ffmpeg commands via `profile.ffmpeg_input_args()` / `profile.ffmpeg_output_args()`.
- Derivatives — `Derivative` entries on a profile are extra outputs of the same ffmpeg process, built with `-filter_complex`. `with_preview()` adds a frame-decimated preview, `with_poster()` an sRGB JPEG still. `tap` sets the branch point: `"shared"` after the encoding chain, `"source"` after `source_filters` but before the chain (the poster, since JPEG is sRGB and the archive BT.709). Build multi-output commands from `ffmpeg_graph_args()`, `ffmpeg_output_groups()` and `output_paths()`; `ffmpeg_output_args()` raises for them. Frame-dropping derivatives keep `fps_mode="passthrough"`, or any constant-frame-rate stage duplicates their retained frames back up to the source rate.
- `transcode.py` — `transcode_video()` orchestrates a single ffmpeg subprocess using an `EncodingProfile`. By default it probes the source and auto-prepends `setparams` when `color_trc` is missing (`auto_fix_colorspace=True`). `normalize_cfr=True` is opt-in, for legacy h264-in-AVI whose frames ffmpeg otherwise drops, and prepends `setpts`. `fail_on_frame_drop` (on by default) compares the archive's frames encoded with the source's frames decoded, both parsed from ffmpeg's end-of-run summary under `-loglevel level+verbose`; stderr is read on a thread so the pipe never stalls ffmpeg. `preview_fps` and `poster_at_seconds` attach the derivatives above; the source is probed at most once for all of it.
- `probe.py` — thin ffprobe wrapper (`probe()` returns parsed JSON) and named accessors (`get_color_transfer`, `get_frame_dimensions`, `get_nb_frames`, `get_video_range_info`, `get_yuv_format`).
- `frames.py` — `extract_luma_frame()` / `extract_srgb_frame()`. Frame extraction dispatches on pixel format: YUV uses the `colorspace` filter, GBR uses `zscale`.
- `_rawvideo.py` — pixel-format constants (`_SUPPORTED_YUV_FORMATS_8BIT`/`10BIT`, GBR variants) and raw-buffer parsers that the `frames` module uses to decode ffmpeg's stdout. `pix_format_bit_depth()` is the public accessor.
- `color_spaces.py` — Rec.709 transfer-function math and luma-range helpers.
- `video_qc.py` / `plotting.py` — QC plot generation (requires the `plotting` extra: matplotlib + opencv-python-headless).
- `utils.py` — `http_input_flags()` adds `-reconnect` flags when the input path is a URL; called by every ffmpeg/ffprobe invocation that takes an input path.
- `scripts/transcode_cli.py` — `aind-transcode` CLI built on pydantic-settings (`CliApp` + `TomlConfigSettingsSource`); reads `aind-transcode.toml` from CWD, CLI args override TOML. Uses `ThreadPoolExecutor` for `--jobs N`.
- `scripts/video_qc_cli.py` — `aind-video-qc` CLI.

The two CLIs are gated behind optional extras (`[transcode]`, `[plotting]`); the core library only depends on numpy. Anything that imports pydantic-settings, matplotlib, or opencv must live in the relevant optional module so a `pip install aind-video-utils` (no extras) still works.

## Conventions

- Conventional Commits enforced (see commitizen config in `pyproject.toml`); `major_version_zero = true`, so breaking changes bump minor.
- Version bumps are commits with `bump:` prefix — CI explicitly skips runs on those.
- ruff line length 120, target py310, NumPy docstring convention. `benchmarks/`, `notebooks/`, and `scripts/` (the shell scripts dir) are excluded from ruff.
- mypy runs strict on `src/aind_video_utils`; tests are exempt from untyped-def checks.
- Public API is what's re-exported in `src/aind_video_utils/__init__.py` — keep it curated.
