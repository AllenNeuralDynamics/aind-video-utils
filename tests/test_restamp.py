"""Tests for restamping an MP4 timeline onto a uniform grid."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from aind_video_utils import restamp as restamp_mod
from aind_video_utils.mp4_index import EditListEntry, Mp4FrameIndex, _read_moov, read_mp4_frame_index
from aind_video_utils.probe import probe
from aind_video_utils.restamp import _check_repairable, _setts_expression, _uses_mdta_keys, restamp_uniform
from aind_video_utils.scripts import mp4_timeline_cli

ffmpeg_required = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)

PERIOD = 32  # ticks per frame at 500 fps in a 1/16000 time base


def _index(pts: list[int], dts: list[int], edits: tuple[EditListEntry, ...] = ()) -> Mp4FrameIndex:
    n = len(pts)
    return Mp4FrameIndex(
        media_timescale=16000,
        media_duration=0,
        movie_timescale=1000,
        dts=np.array(dts, dtype=np.int64),
        pts=np.array(pts, dtype=np.int64),
        is_keyframe=np.ones(n, dtype=np.bool_),
        byte_offset=np.zeros(n, dtype=np.int64),
        size=np.ones(n, dtype=np.int64),
        edits=edits,
    )


def test_setts_expression_moves_frames_after_each_notch_back_by_its_excess():
    # Notches of 2 slots after display index 1 and 1 slot after index 3; media_time 64.
    pts = [64, 96, 192, 224, 288]
    index = _index(pts, [0, 32, 64, 96, 128], edits=(EditListEntry(1, 64, 1.0),))
    expr = _setts_expression(index, index.timeline_notches())
    assert expr == "setts=dts=STARTDTS+N*32:pts=PTS-64*gte(PTS\\,128)-32*gte(PTS\\,224)"


def test_setts_expression_without_notches_keeps_pts():
    index = _index([0, 32, 64], [0, 32, 64])
    assert _setts_expression(index, index.timeline_notches()) == "setts=dts=STARTDTS+N*32:pts=PTS"


@pytest.mark.parametrize(
    ("pts", "dts", "edits", "match"),
    [
        ([0, 32, 64], [0, 32, 64], (EditListEntry(1, 0, 1.0), EditListEntry(1, 0, 1.0)), "2 edits"),
        ([0, 32, 64], [0, 32, 64], (EditListEntry(1, 0, 2.0),), "non-unit rate"),
        ([0, 32, 64], [0, 32, 64], (EditListEntry(1, -1, 1.0),), "empty"),
        ([0, 32, 64, 96], [0, 32, 32, 64], (), "DTS step backwards"),
        ([0, 32, 32, 64, 96], [0, 1, 2, 3, 4], (), "share a PTS"),
        ([0, 32, 64, 112, 144], [0, 32, 64, 96, 128], (), "not a whole number"),
    ],
)
def test_check_repairable_refuses(pts, dts, edits, match):
    index = _index(pts, dts, edits)
    with pytest.raises(ValueError, match=match):
        _check_repairable(index, index.timeline_notches())


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


def _encode(path: Path, *extra: str, n_frames: int = 300) -> None:
    """Encode 500 fps B-frame h264 with IDRs every 50 frames, colour-described and tagged."""
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=500", "-frames:v", str(n_frames), *extra,
        "-c:v", "libx264", "-preset", "ultrafast", "-bf", "3", "-g", "50", "-keyint_min", "50",
        "-sc_threshold", "0", "-color_range", "pc", "-colorspace", "bt709",
        "-metadata", "source_s3=s3://bucket/key", "-movflags", "+faststart+write_colr+use_metadata_tags",
        str(path),
    )  # fmt: skip


def _notch(src: Path, dst: Path, at: int) -> None:
    """Stamp a concat seam at decode index *at*, an IDR: PTS two slots late from it, DTS from two packets on."""
    _ffmpeg(
        "-i", str(src), "-c", "copy", "-map_metadata", "0",
        "-bsf:v", f"setts=pts=PTS+64*gte(N\\,{at}):dts=DTS+64*gte(N\\,{at + 2})",
        "-video_track_timescale", "16000", "-movflags", "+faststart+write_colr+use_metadata_tags",
        str(dst),
    )  # fmt: skip


def _framemd5(path: Path) -> list[str]:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "framemd5", "-"], check=True, capture_output=True, text=True
    ).stdout
    return [line for line in out.splitlines() if not line.startswith("#")]


@pytest.fixture
def clean(tmp_path: Path) -> Path:
    path = tmp_path / "clean.mp4"
    _encode(path)
    return path


@pytest.fixture
def seamed(clean: Path, tmp_path: Path) -> Path:
    path = tmp_path / "seamed.mp4"
    _notch(clean, path, 100)
    return path


@ffmpeg_required
def test_seam_fixture_has_the_seam_signature(clean, seamed):
    """The fixture inflates the offsets at the seam IDR as a real concat seam does."""
    notches = read_mp4_frame_index(seamed).timeline_notches()
    assert notches.pts_after.tolist() == [99]
    assert notches.pts_steps.tolist() == [3 * PERIOD]
    assert notches.dts_after.tolist() == [101]
    index = read_mp4_frame_index(seamed)
    clean_index = read_mp4_frame_index(clean)
    assert (index.pts - index.dts)[100] > (clean_index.pts - clean_index.dts)[100]


@ffmpeg_required
def test_restamp_removes_a_seam_from_a_bframe_stream(clean, seamed, tmp_path):
    out = restamp_uniform(seamed, tmp_path / "fixed.mp4")
    fixed, original = read_mp4_frame_index(out), read_mp4_frame_index(clean)
    assert fixed.timeline_notches().is_uniform
    assert np.array_equal(fixed.pts, original.pts)
    assert np.array_equal(fixed.dts, original.dts)
    assert fixed.edits == original.edits
    # framemd5 lines carry each frame's timestamps as well as its hash.
    assert _framemd5(out) == _framemd5(clean)


@ffmpeg_required
def test_restamp_keeps_tags_and_colour_description(seamed, tmp_path):
    out = restamp_uniform(seamed, tmp_path / "fixed.mp4")
    stream = probe(out)["streams"][0]
    assert (stream["color_range"], stream["color_space"]) == ("pc", "bt709")
    assert probe(out)["format"]["tags"]["source_s3"] == "s3://bucket/key"
    assert restamp_mod._has_colr(_read_moov(out))
    assert _uses_mdta_keys(_read_moov(out))


@ffmpeg_required
def test_restamp_of_a_gap_free_stream_keeps_its_timeline(clean, tmp_path):
    out = restamp_uniform(clean, tmp_path / "same.mp4")
    before, after = read_mp4_frame_index(clean), read_mp4_frame_index(out)
    assert np.array_equal(after.pts, before.pts)
    assert np.array_equal(after.dts, before.dts)
    assert after.edits == before.edits


@ffmpeg_required
def test_restamp_closes_a_gap_cut_from_the_source(tmp_path):
    cut = tmp_path / "cut.mp4"
    _encode(cut, "-vf", "select='not(between(n,150,152))'", "-fps_mode", "passthrough", n_frames=297)
    assert not read_mp4_frame_index(cut).timeline_notches().is_uniform
    fixed = read_mp4_frame_index(restamp_uniform(cut, tmp_path / "fixed.mp4"))
    assert fixed.n_samples == 297
    assert fixed.timeline_notches().is_uniform


@ffmpeg_required
def test_restamp_refuses_a_notch_of_part_of_a_period(clean, tmp_path):
    odd = tmp_path / "odd.mp4"
    _ffmpeg("-i", str(clean), "-c", "copy", "-bsf:v", "setts=pts=PTS+16*gte(N\\,100):dts=DTS+16*gte(N\\,100)", str(odd))
    with pytest.raises(ValueError, match="not a whole number"):
        restamp_uniform(odd, tmp_path / "fixed.mp4")


@ffmpeg_required
def test_restamp_refuses_a_file_with_other_streams(clean, tmp_path):
    with_audio = tmp_path / "audio.mp4"
    _ffmpeg("-i", str(clean), "-f", "lavfi", "-i", "sine=duration=0.6", "-c:v", "copy", "-shortest", str(with_audio))
    with pytest.raises(ValueError, match="streams besides its video"):
        restamp_uniform(with_audio, tmp_path / "fixed.mp4")


@ffmpeg_required
def test_a_repair_that_fails_verification_writes_nothing(seamed, tmp_path, monkeypatch):
    """Keeping PTS - DTS leaves a seam notched, since its offsets are inflated; the check must catch it."""
    monkeypatch.setattr(
        restamp_mod,
        "_setts_expression",
        lambda _index, notches: f"setts=dts=STARTDTS+N*{notches.period}:pts=STARTDTS+N*{notches.period}+PTS-DTS",
    )
    dst = tmp_path / "fixed.mp4"
    with pytest.raises(RuntimeError, match="failed verification:.*PTS do not step by 32"):
        restamp_uniform(seamed, dst)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clean.mp4", "seamed.mp4"]


def test_cli_reads_s3_uris_as_public_https():
    assert mp4_timeline_cli._readable("s3://aind-open-data/a b/v.mp4") == (
        "https://aind-open-data.s3.amazonaws.com/a%20b/v.mp4"
    )
    assert mp4_timeline_cli._readable("https://host/v.mp4") == "https://host/v.mp4"
    assert mp4_timeline_cli._readable("v.mp4") == Path("v.mp4")


@ffmpeg_required
def test_cli_reports_notches_as_frame_positions(clean, seamed, tmp_path, monkeypatch):
    listing = tmp_path / "videos.txt"
    listing.write_text(f"# shipped\n{clean}\n\n")
    report = tmp_path / "notches.csv"
    missing = tmp_path / "missing.mp4"
    monkeypatch.setattr(
        "sys.argv",
        ["aind-mp4-timeline", "notches", str(seamed), str(missing), "--list", str(listing), "-o", str(report)],
    )
    with pytest.raises(SystemExit) as exited:
        mp4_timeline_cli.main()
    assert exited.value.code == 0
    import csv

    rows = {row["source"]: row for row in csv.DictReader(report.open())}
    assert rows[str(seamed)]["notch_after_frame"] == "100"
    assert rows[str(seamed)]["notch_excess_frames"] == "2"
    assert rows[str(seamed)]["frame_rate"] == "500/1"
    assert rows[str(seamed)]["repairable"] == "yes"
    assert rows[str(clean)]["pts_notches"] == "0"
    assert "No such file" in rows[str(missing)]["error"]


@ffmpeg_required
def test_cli_restamps(seamed, tmp_path, monkeypatch):
    dst = tmp_path / "fixed.mp4"
    monkeypatch.setattr("sys.argv", ["aind-mp4-timeline", "restamp", str(seamed), str(dst)])
    with pytest.raises(SystemExit):
        mp4_timeline_cli.main()
    assert read_mp4_frame_index(dst).timeline_notches().is_uniform
