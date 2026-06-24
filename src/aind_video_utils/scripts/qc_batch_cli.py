"""CLI to compute per-video exposure QC stats across a batch of videos.

Reads an input CSV with either ``url`` or ``bucket,key`` columns. For each
row, samples ``--samples`` frames evenly across the video, computes
per-frame exposure statistics, and writes one row of aggregated metrics
to the output CSV.

The output CSV is resumable: re-running with the same output path skips
rows already present.

Designed for ranking videos by exposure issues (clipping, dynamic range
collapse, color-range mismatch) at scales of thousands of videos.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from aind_video_utils.qc_batch import qc_result_fieldnames, qc_video


def _s3_to_https(bucket: str, key: str, region: str) -> str:
    """Build a public S3 HTTPS URL from bucket/key, with URL-encoded key."""
    safe_key = urllib.parse.quote(key, safe="/")
    return f"https://{bucket}.s3.{region}.amazonaws.com/{safe_key}"


def _row_to_url(row: dict[str, str], region: str) -> tuple[str, str, str]:
    """Resolve a CSV row to (key_id, url, label) tuples.

    ``key_id`` is used for resume bookkeeping. ``label`` is for log lines.
    """
    if "url" in row and row["url"]:
        return row["url"], row["url"], row["url"]
    bucket = row.get("bucket", "")
    key = row.get("key", "")
    if not (bucket and key):
        raise ValueError(f"row missing 'url' or 'bucket,key': {row!r}")
    return f"{bucket}/{key}", _s3_to_https(bucket, key, region), key


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the CLI."""
    p = argparse.ArgumentParser(
        prog="aind-video-qc-batch",
        description="Compute per-video exposure QC stats for a batch of videos.",
    )
    p.add_argument("--input", "-i", type=Path, required=True, help="Input CSV (bucket,key or url columns).")
    p.add_argument("--output", "-o", type=Path, required=True, help="Output CSV (append/resume).")
    p.add_argument("--samples", "-n", type=int, default=10, help="Frames sampled per video (default 10).")
    p.add_argument("--workers", "-w", type=int, default=32, help="Parallel workers (default 32).")
    p.add_argument("--timeout", type=int, default=180, help="Per-video timeout in seconds (default 180).")
    p.add_argument("--region", default="us-west-2", help="AWS region for bucket/key→URL (default us-west-2).")
    p.add_argument("--skip-edge", type=float, default=0.01, help="Fraction of duration to skip at start/end.")
    p.add_argument("--limit", type=int, default=0, help="Process only the first N rows after resume (0=all).")
    p.add_argument("--progress-every", type=int, default=50)
    return p


def _read_jobs(input_csv: Path, region: str) -> list[tuple[str, str, str]]:
    """Read all jobs as (key_id, url, label) from the input CSV."""
    jobs: list[tuple[str, str, str]] = []
    with input_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            jobs.append(_row_to_url(row, region))
    return jobs


def _read_done(output_csv: Path) -> set[str]:
    """Return the set of ``key_id`` values already present in the output CSV."""
    done: set[str] = set()
    if not (output_csv.exists() and output_csv.stat().st_size > 0):
        return done
    with output_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            kid = row.get("_key_id")
            if kid:
                done.add(kid)
    return done


def _probe_with_timeout(url: str, samples: int, skip_edge: float, timeout_s: int) -> dict[str, object]:
    """Run qc_video and return its result as a flat dict, or an error dict."""
    import concurrent.futures as _cf

    out: dict[str, object] = {"ok": False, "error": None}
    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(qc_video, url, n_samples=samples, skip_edge_fraction=skip_edge)
            qc = fut.result(timeout=timeout_s)
        out.update(dataclasses.asdict(qc))
        out["ok"] = True
    except _cf.TimeoutError:
        out["error"] = f"timeout after {timeout_s}s"
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        out["error"] = msg[:500]
    return out


def main() -> int:
    """Entry point for aind-video-qc-batch CLI."""
    args = _build_arg_parser().parse_args()

    all_jobs = _read_jobs(args.input, args.region)
    done = _read_done(args.output)
    todo = [j for j in all_jobs if j[0] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"total={len(all_jobs)} done={len(done)} todo={len(todo)} "
        f"samples={args.samples} workers={args.workers} timeout={args.timeout}s",
        file=sys.stderr,
    )
    if not todo:
        return 0

    fieldnames = ["_key_id", "url", "label", "ok", "error", *qc_result_fieldnames()]
    out_exists = args.output.exists() and args.output.stat().st_size > 0
    out_f = args.output.open("a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames, extrasaction="ignore")
    if not out_exists:
        writer.writeheader()
        out_f.flush()
    write_lock = Lock()

    start = time.time()
    completed = ok = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_probe_with_timeout, url, args.samples, args.skip_edge, args.timeout): (kid, url, label)
            for (kid, url, label) in todo
        }
        for fut in as_completed(futs):
            kid, url, label = futs[fut]
            row = fut.result()
            row.update({"_key_id": kid, "url": url, "label": label})
            with write_lock:
                writer.writerow(row)
                out_f.flush()
            completed += 1
            if row.get("ok"):
                ok += 1
            else:
                err += 1
            if completed % args.progress_every == 0 or completed == len(todo):
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta_min = (len(todo) - completed) / rate / 60 if rate > 0 else float("inf")
                print(
                    f"[{elapsed:6.1f}s] {completed}/{len(todo)} "
                    f"({100 * completed / len(todo):5.1f}%) | "
                    f"{rate:5.1f} v/s | ok={ok} err={err} | "
                    f"ETA {eta_min:5.1f} min",
                    file=sys.stderr,
                )
    out_f.close()
    return 0 if err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
