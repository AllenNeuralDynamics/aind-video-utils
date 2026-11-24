# AGENTS.md

Instructions for coding agents in this repository. [CONTRIBUTING.md](CONTRIBUTING.md) holds setup,
checks, tests and conventions, and applies to agents as to people: read it before running or
changing anything. [README.md](README.md) documents use.

- Run every command through `uv` or `./scripts/run_linters_and_checks.sh`; never bare `pip` or
  `python`, and never a globally installed ruff or mypy, whose versions differ from `uv.lock`.
- Before calling a change done, run every check CONTRIBUTING.md lists for a commit, and read the
  script's exit code rather than filtering its output.
- Keep this file to what only an agent needs; development practice belongs in CONTRIBUTING.md,
  usage in README.md, and what only Claude Code needs in CLAUDE.md.
