"""Tests for restamping an MP4 timeline onto a uniform grid."""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from aind_video_utils import restamp as restamp_mod
from aind_video_utils.mp4_index import EditListEntry, Mp4FrameIndex, _read_moov, read_mp4_frame_index
from aind_video_utils.probe import probe
from aind_video_utils.restamp import (
    _check_repairable,
    _nominal_timescale,
    _setts_expression,
    _uses_mdta_keys,
    restamp_nominal,
    restamp_uniform,
)
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
    # Notches of 2 slots after display index 1 and 1 slot after index 3.
    pts = [64, 96, 192, 224, 288]
    index = _index(pts, [0, 32, 64, 96, 128], edits=(EditListEntry(1, 64, 1.0),))
    expr = _setts_expression(index, index.timeline_notches())
    assert expr == "setts=dts=STARTDTS+N*32:pts=PTS-64*gte(PTS-STARTPTS\\,128)-32*gte(PTS-STARTPTS\\,224)"


def test_check_repairable_accepts_a_leading_empty_edit():
    index = _index([64, 96, 128], [0, 32, 64], edits=(EditListEntry(2, -1, 1.0), EditListEntry(1, 64, 1.0)))
    _check_repairable(index, index.timeline_notches())


def test_setts_expression_without_notches_keeps_pts():
    index = _index([0, 32, 64], [0, 32, 64])
    assert _setts_expression(index, index.timeline_notches()) == "setts=dts=STARTDTS+N*32:pts=PTS"


@pytest.mark.parametrize(
    ("pts", "dts", "edits", "match"),
    [
        ([0, 32, 64], [0, 32, 64], (EditListEntry(1, 0, 1.0), EditListEntry(1, 0, 1.0)), "2 edits"),
        ([0, 32, 64], [0, 32, 64], (EditListEntry(1, 0, 2.0),), "non-unit rate"),
        ([0, 32, 64], [0, 32, 64], (EditListEntry(1, -1, 1.0),), "empty"),
        ([0, 32, 64], [0, 32, 64], (EditListEntry(1, 0, 1.0), EditListEntry(1, -1, 1.0)), "2 edits"),
        (
            [0, 32, 64],
            [0, 32, 64],
            (EditListEntry(1, -1, 1.0), EditListEntry(1, 0, 1.0), EditListEntry(1, 0, 1.0)),
            "3 edits",
        ),
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
def test_restamp_drops_a_leading_empty_edit(clean, seamed, tmp_path):
    late = tmp_path / "late.mp4"
    _ffmpeg(
        "-i", str(seamed), "-c", "copy", "-map_metadata", "0", "-bsf:v", f"setts=pts=PTS+{PERIOD}:dts=DTS+{PERIOD}",
        "-video_track_timescale", "16000", "-movflags", "+faststart+write_colr+use_metadata_tags", str(late),
    )  # fmt: skip
    assert read_mp4_frame_index(late).edits[0].media_time == -1
    fixed, original = read_mp4_frame_index(restamp_uniform(late, tmp_path / "fixed.mp4")), read_mp4_frame_index(clean)
    assert np.array_equal(fixed.pts, original.pts)
    assert np.array_equal(fixed.dts, original.dts)
    assert fixed.edits == original.edits


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


# --- nominal-rate restamp ---


@ffmpeg_required
@pytest.mark.parametrize("rate", ["120", "60", "30", "25", "20", "500", "30000/1001", "24000/1001"])
def test_nominal_timescale_matches_a_fresh_encode(tmp_path, rate):
    fresh = tmp_path / "fresh.mp4"
    _ffmpeg("-f", "lavfi", "-i", f"testsrc2=size=64x48:rate={rate}", "-frames:v", "2", "-c:v", "libx264", str(fresh))
    assert _nominal_timescale(Fraction(rate)) == read_mp4_frame_index(fresh).media_timescale


def _chunked(tmp_path: Path, rate: int, *extra_bsf: str) -> Path:
    """Encode 3 s at *rate* the way the batch does: 1 s Matroska chunks, then ``concat -c copy`` into MP4.

    Matroska rounds every timestamp to 1 ms, and the first chunk's B-frame delay
    leaves a 2-frame notch after it.
    """
    chunks = tmp_path / f"chunks{rate}"
    chunks.mkdir()
    _ffmpeg(
        "-f", "lavfi", "-i", f"testsrc2=size=64x48:rate={rate}", "-frames:v", str(3 * rate),
        "-vf", f"setpts=N/{rate}/TB", "-c:v", "libx264", "-preset", "ultrafast", "-bf", "2",
        "-g", str(rate), "-keyint_min", str(rate), "-sc_threshold", "0",
        "-color_range", "pc", "-colorspace", "bt709", "-metadata", "source_s3=s3://bucket/key",
        "-f", "segment", "-segment_time", "1", "-reset_timestamps", "1", str(chunks / "%02d.mkv"),
    )  # fmt: skip
    listing = chunks / "list.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in sorted(chunks.glob("*.mkv"))))
    out = tmp_path / f"chunked{rate}.mp4"
    _ffmpeg(
        "-f", "concat", "-i", str(listing), "-c", "copy", "-map_metadata", "0", *extra_bsf,
        "-video_track_timescale", "16000", "-movflags", "+faststart+write_colr+use_metadata_tags", str(out),
    )  # fmt: skip
    return out


def _packet_hashes(path: Path) -> list[str]:
    """Per-packet MD5 of the coded bytes, without timestamps."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-c", "copy", "-f", "framemd5", "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line.rsplit(",", 1)[-1].strip() for line in out.splitlines() if not line.startswith("#")]


@ffmpeg_required
@pytest.mark.parametrize("rate", [120, 60, 30])
def test_nominal_restamp_puts_a_chunked_file_on_its_grid(tmp_path, rate):
    src = _chunked(tmp_path, rate)
    before = read_mp4_frame_index(src)
    assert not before.timeline_notches().is_uniform
    result = restamp_nominal(src, tmp_path / "fixed.mp4", frame_rate=rate)
    after = read_mp4_frame_index(result.path)
    period = 15360 // rate
    assert after.media_timescale == result.timescale == 15360
    assert result.source_timescale == 16000
    assert (np.diff(np.sort(after.pts)) == period).all()
    assert (np.diff(after.dts) == period).all()
    assert after.edits == (EditListEntry(3000, 2 * period, 1.0),)
    assert np.array_equal(after.display_order, before.display_order)
    assert _packet_hashes(result.path) == _packet_hashes(src)
    assert result.notch_after_frame == (rate,)
    assert result.notch_excess_frames == (2,)
    assert 0 < result.max_residual < 0.25
    stream = probe(result.path)["streams"][0]
    assert (stream["color_range"], stream["color_space"]) == ("pc", "bt709")
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"chunked{rate}.mp4", f"chunks{rate}", "fixed.mp4"]


@ffmpeg_required
def test_nominal_restamp_can_keep_gaps(tmp_path):
    result = restamp_nominal(_chunked(tmp_path, 120), tmp_path / "fixed.mp4", frame_rate=120, keep_gaps=True)
    steps = np.diff(np.sort(read_mp4_frame_index(result.path).pts))
    assert np.flatnonzero(steps != 128).tolist() == [119]
    assert steps[119] == 3 * 128
    assert result.as_record()["keep_gaps"] is True


@ffmpeg_required
def test_nominal_restamp_refuses_a_wrong_rate(tmp_path):
    with pytest.raises(ValueError, match="periods from its slot at 125 fps"):
        restamp_nominal(_chunked(tmp_path, 120), tmp_path / "fixed.mp4", frame_rate=125)


@ffmpeg_required
def test_nominal_restamp_refuses_a_frame_off_its_slot(tmp_path):
    # 53 ticks is 0.4 of a 120 fps period at 16000.
    src = _chunked(tmp_path, 120, "-bsf:v", "setts=pts=PTS+53*eq(N\\,150)")
    with pytest.raises(ValueError, match="periods from its slot"):
        restamp_nominal(src, tmp_path / "fixed.mp4", frame_rate=120)


@ffmpeg_required
def test_a_nominal_restamp_that_fails_verification_writes_nothing(tmp_path, monkeypatch):
    src = _chunked(tmp_path, 120)
    erase = restamp_mod._nominal_setts_expression
    monkeypatch.setattr(restamp_mod, "_nominal_setts_expression", lambda index, plan, _keep: erase(index, plan, True))
    with pytest.raises(RuntimeError, match="failed verification:.*PTS are not at their slots"):
        restamp_nominal(src, tmp_path / "fixed.mp4", frame_rate=120)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["chunked120.mp4", "chunks120"]


@ffmpeg_required
def test_cli_checks_and_restamps_at_a_nominal_rate(tmp_path, monkeypatch, capsys):
    import csv

    src = _chunked(tmp_path, 120)
    report = tmp_path / "notches.csv"
    monkeypatch.setattr(
        "sys.argv", ["aind-mp4-timeline", "notches", str(src), "--frame-rate", "120", "-o", str(report)]
    )
    with pytest.raises(SystemExit):
        mp4_timeline_cli.main()
    (row,) = csv.DictReader(report.open())
    assert row["repairable"].startswith("no: ")
    assert (row["nominal_rate"], row["nominal_repairable"]) == ("120/1", "yes")

    dst = tmp_path / "fixed.mp4"
    monkeypatch.setattr("sys.argv", ["aind-mp4-timeline", "restamp", "--frame-rate", "120", str(src), str(dst)])
    with pytest.raises(SystemExit) as exited:
        mp4_timeline_cli.main()
    assert exited.value.code == 0
    record = json.loads(capsys.readouterr().out)
    assert (record["frame_rate"], record["timescale"], record["notch_after_frame"]) == ("120/1", 15360, [120])


def test_cli_rejects_keep_gaps_without_a_rate(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["aind-mp4-timeline", "restamp", "--keep-gaps", "a.mp4", str(tmp_path / "b.mp4")])
    with pytest.raises(SystemExit) as exited:
        mp4_timeline_cli.main()
    assert exited.value.code == 2
