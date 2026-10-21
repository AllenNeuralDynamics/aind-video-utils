"""CLI to report MP4 timeline notches and restamp a notched file.

``notches`` reads each file's ``moov`` only, so it is cheap over HTTP, and
writes one CSV row per file.  Give it the files to check, from DocDB or a
batch manifest: it never lists a bucket.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote, urlsplit

from aind_video_utils.mp4_index import read_mp4_frame_index
from aind_video_utils.restamp import _check_repairable, restamp_uniform

COLUMNS: tuple[str, ...] = (
    "source",
    "frames",
    "frame_rate",
    "period_ticks",
    "pts_notches",
    "notch_after_frame",
    "notch_excess_frames",
    "dts_notches",
    "edits",
    "repairable",
    "error",
)


def _readable(source: str) -> str | Path:
    """Map ``s3://bucket/key`` to its public HTTPS URL, keep other URLs, and wrap paths."""
    if source.startswith("s3://"):
        parts = urlsplit(source)
        return f"https://{parts.netloc}.s3.amazonaws.com{quote(parts.path)}"
    return source if source.startswith(("http://", "https://")) else Path(source)


def report(source: str) -> dict[str, object]:
    """Return one CSV row describing *source*'s timeline; an unreadable file fills ``error``."""
    row: dict[str, object] = dict.fromkeys(COLUMNS, "")
    row["source"] = source
    try:
        index = read_mp4_frame_index(_readable(source))
        notches = index.timeline_notches()
    except (OSError, ValueError) as err:
        row["error"] = str(err)
        return row
    try:
        _check_repairable(index, notches)
        repairable = "yes"
    except ValueError as err:
        repairable = f"no: {err}"
    rate = Fraction(index.media_timescale, notches.period)
    row.update(
        frames=index.n_samples,
        frame_rate=f"{rate.numerator}/{rate.denominator}",
        period_ticks=notches.period,
        pts_notches=notches.pts_after.size,
        # Positions count frames from 1; the step follows that frame.
        notch_after_frame=";".join(str(int(k) + 1) for k in notches.pts_after),
        notch_excess_frames=";".join(f"{float(Fraction(int(s), notches.period) - 1):g}" for s in notches.pts_steps),
        dts_notches=notches.dts_after.size,
        edits=len(index.edits),
        repairable=repairable,
    )
    return row


def _sources(args: argparse.Namespace) -> Iterator[str]:
    yield from args.sources
    if args.list is not None:
        with args.list.open() as handle:
            yield from (line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#"))


def _notches(args: argparse.Namespace) -> int:
    sources = list(_sources(args))
    if not sources:
        print("no files given: pass paths or URLs, or --list", file=sys.stderr)
        return 2
    out = args.output.open("w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(out, fieldnames=COLUMNS)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for row in pool.map(report, sources):
                writer.writerow(row)
                out.flush()
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


def _restamp(args: argparse.Namespace) -> int:
    restamp_uniform(_readable(args.source), args.destination)
    print(f"Wrote {args.destination}")
    return 0


def main() -> None:
    """Entry point for the aind-mp4-timeline CLI."""
    parser = argparse.ArgumentParser(
        prog="aind-mp4-timeline", description="Report MP4 timeline notches and restamp a notched file."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_notches = subparsers.add_parser(
        "notches",
        help="Report timeline notches per file as CSV, from the moov alone.",
        description=(
            "One row per file. notch_after_frame lists the frames, counted from 1, after which the "
            "presentation timeline steps by more than one frame period; notch_excess_frames is each "
            "step's excess in frame periods. s3:// sources are read anonymously over HTTPS."
        ),
    )
    p_notches.add_argument("sources", nargs="*", help="MP4 paths, http(s):// URLs or s3:// URIs.")
    p_notches.add_argument("--list", type=Path, help="File of sources, one per line; # starts a comment.")
    p_notches.add_argument("--output", "-o", type=Path, help="CSV path; defaults to stdout.")
    p_notches.add_argument("--jobs", "-j", type=int, default=8, help="Files read concurrently (default: 8).")

    p_restamp = subparsers.add_parser(
        "restamp",
        help="Copy an MP4 with its timeline on a uniform grid, without re-encoding.",
    )
    p_restamp.add_argument("source", help="MP4 path, http(s):// URL or s3:// URI.")
    p_restamp.add_argument("destination", type=Path, help="Local output path.")

    args = parser.parse_args()
    sys.exit(_notches(args) if args.command == "notches" else _restamp(args))


if __name__ == "__main__":
    main()
