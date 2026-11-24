#!/bin/sh
# Run linters (default) or linters+checks without rebuilding the uv env.

set -u

main() {
  CHECKS=false
  PYTEST_ARGS=""
  SEEN_DASHDASH=false

  # Parse args: -c/--checks anywhere; collect pytest args (after -- or others)
  for arg in "$@"; do
    if [ "$SEEN_DASHDASH" = true ]; then
      PYTEST_ARGS="$PYTEST_ARGS $arg"
      continue
    fi
    case "$arg" in
      -c|--checks) CHECKS=true ;;
      --) SEEN_DASHDASH=true ;;
      *) PYTEST_ARGS="$PYTEST_ARGS $arg" ;;  # pass unknowns to pytest
    esac
  done

  ENV_PATH="${ENV_PATH:-.venv}"

  # Choose runner: active venv, local .venv, or uv (no sync)
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    BIN="$VIRTUAL_ENV/bin"
    run() { "$BIN/$@"; }
    run_on_stdin_files() { xargs -0 "$BIN/$@"; }
  elif [ -x "$ENV_PATH/bin/python" ]; then
    BIN="$ENV_PATH/bin"
    run() { "$BIN/$@"; }
    run_on_stdin_files() { xargs -0 "$BIN/$@"; }
  else
    PYVER="${PYVER:-}"
    if [ -z "$PYVER" ] && [ -f .python-version ]; then
      PYVER="$(tr -d ' \n' < .python-version)"
    fi
    export UV_PROJECT_ENVIRONMENT="$ENV_PATH"
    UV_ARGS="--frozen --no-sync"
    [ -n "$PYVER" ] && UV_ARGS="$UV_ARGS --python $PYVER"
    run() { uv run $UV_ARGS -- "$@"; }
    run_on_stdin_files() { xargs -0 uv run $UV_ARGS -- "$@"; }
  fi

  # CI checks out only tracked files, so untracked work is not spell-checked here either.
  codespell_tracked() {
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      git ls-files -z | run_on_stdin_files codespell --check-filenames
    else
      run codespell --check-filenames
    fi
  }

  # Every step runs even after a failure so one pass reports all problems;
  # the exit status is nonzero if any step failed.
  FAILED=""
  step() {
    name="$1"
    shift
    echo "+ $name"
    "$@" || FAILED="$FAILED
  $name"
  }

  step "ruff format" run ruff format

  if [ "$CHECKS" = true ]; then
    step "ruff check" run ruff check
    step "mypy" run mypy
    step "interrogate -v src" run interrogate -v src
    step "codespell --check-filenames (tracked files)" codespell_tracked
    # shellcheck disable=SC2086
    step "pytest --cov aind_video_utils$([ -n "$PYTEST_ARGS" ] && printf ' -- %s' "$PYTEST_ARGS")" \
      run pytest --cov aind_video_utils $PYTEST_ARGS
  else
    echo "(checks skipped; pass -c or --checks to enable)"
  fi

  if [ -n "$FAILED" ]; then
    echo "FAILED:$FAILED" >&2
    return 1
  fi
  echo "All steps passed."
}

main "$@"
