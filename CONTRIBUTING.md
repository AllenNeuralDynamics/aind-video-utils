# Contributing

## Setup

`ffmpeg` and `ffprobe` must be on `PATH`: every code path in this package shells out to them.

```bash
uv sync    # dev dependencies, including every optional extra
```

## Checks

```bash
./scripts/run_linters_and_checks.sh                     # ruff format only
./scripts/run_linters_and_checks.sh -c                  # ruff, mypy, interrogate, codespell, pytest
./scripts/run_linters_and_checks.sh -c -- -k test_name  # forward pytest arguments after --
uv run pytest tests/test_transcode.py::test_name        # one test
```

Run tools through `uv run` or the script, which use the versions `uv.lock` pins; a globally installed ruff or mypy
can disagree with them.

CI runs the reusable workflow in `AllenNeuralDynamics/galen-uv-workflows` on Python 3.10 and 3.13, and also installs the
wheel without extras. The script covers neither, so check them before pushing:

```bash
# mypy on the oldest supported Python
uv run --python 3.10 --isolated --with mypy --with pyarrow --with pydantic-settings --with rich \
  --with matplotlib --with opencv-python-headless mypy

# the tests pass against the wheel without any optional dependency, run outside the
# repository so they import the installed wheel and not src/
rm -rf dist && uv build --wheel -o dist && wheel=$(echo "$PWD"/dist/*.whl) && tmp=$(mktemp -d) && cp -r tests "$tmp" \
  && (cd "$tmp" && uv run --isolated --no-project --with "$wheel" --with pytest pytest -q)
```

## Design

The package is the canonical Python implementation of the
[AIND behavior video file standard](https://allenneuraldynamics.github.io/aind-file-standards/file_formats/behavior_videos/).
`SPEC_VERSION` in `encoding.py` is the revision of that document the code implements, matching its `## Version`
heading, and is independent of the package version. A change to the standard's encoder settings, file names or
derivatives lands here with a `SPEC_VERSION` bump.

Under `src/aind_video_utils/`:

- `encoding.py` holds `EncodingProfile` and the four profiles the standard specifies. A profile's `source_filters`
  repair what the source's own metadata gets wrong (`setparams` from `with_setparams()`, `setpts` from
  `prepend_conditioning()`) and run ahead of every branch; `video_filters` is the encoding chain.
- Derivatives are extra outputs of the same ffmpeg process, built with `-filter_complex`. `with_preview()` keeps every
  Nth frame; `with_poster()` writes an sRGB JPEG of one frame. A derivative's `tap` is `"shared"`, after the encoding
  chain, or `"source"`, after `source_filters` but before the chain, which the poster needs because JPEG is sRGB and the
  archive BT.709. Frame-dropping derivatives keep `fps_mode="passthrough"`, or a constant-frame-rate stage duplicates
  their frames back up to the source rate. Build multi-output commands from `ffmpeg_graph_args()`,
  `ffmpeg_output_groups()` and `output_paths()`.
- `transcode.py` runs one ffmpeg process from a profile. Its frame check compares the primary's frames encoded with the
  source's frames decoded, both parsed from ffmpeg's end-of-run summary at `-loglevel level+verbose`; stderr is read on
  a thread so the pipe never stalls ffmpeg.
- `preview_metadata.py` writes `preview_metadata.parquet` with pyarrow.
- `probe.py` wraps ffprobe. `frames.py` extracts frames, dispatching on pixel format: YUV through the `colorspace`
  filter, GBR through `zscale`. `_rawvideo.py` parses ffmpeg's raw output. `color_spaces.py` has the Rec.709 transfer
  math. `utils.py` adds `-reconnect` flags for URL inputs, and every ffmpeg or ffprobe call that takes a path uses it.
- `video_qc.py` and `plotting.py` draw QC figures.
- `scripts/` holds the `aind-transcode` and `aind-video-qc` CLIs, built on pydantic-settings.

The core depends only on numpy. The CLIs, the QC figures and the Parquet writer sit behind the `transcode`, `plotting`
and `parquet` extras, so a module that needs pydantic-settings, rich, matplotlib, opencv or pyarrow either lives in an
optional module or imports it inside the function that uses it.

The public API is what `src/aind_video_utils/__init__.py` re-exports; add to it deliberately.

## Conventions

- [Conventional Commits](https://www.conventionalcommits.org/), checked by commitizen: `<type>(<scope>): <summary>`
  with type `feat`, `fix`, `docs`, `ci`, `build`, `perf`, `refactor`, `style` or `test`. `major_version_zero` is set,
  so a breaking change bumps the minor version.
- CI bumps the version after a merge to `main`, in a commit starting `bump:`, which CI skips.
- ruff: line length 120, target py310, NumPy docstrings. `benchmarks/`, `notebooks/` and `scripts/` at the repository
  root are excluded.
- mypy is strict on `src/aind_video_utils`; tests may leave functions untyped.
- Internal members branch in this repository; external contributors open a pull request from a fork.
