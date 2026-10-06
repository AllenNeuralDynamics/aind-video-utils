# AGENTS.md

Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup, checks, design and conventions. On top of it:

- Run tools through `uv run` or `./scripts/run_linters_and_checks.sh`, never a global ruff or mypy, whose versions
  differ from the lock file.
- Before every commit, run `./scripts/run_linters_and_checks.sh -c`, the Python 3.10 mypy and the no-extras import
  check from CONTRIBUTING.md, and fold any reformatting into the same commit.
- The encoding profiles mirror `docs/file_formats/behavior_videos.md` in `AllenNeuralDynamics/aind-file-standards`.
  Read that document before changing a profile, a derivative or an output name, and keep `SPEC_VERSION` equal to its
  version.
