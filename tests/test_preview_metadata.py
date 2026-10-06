"""Tests for preview_metadata.parquet writing."""

from __future__ import annotations

from pathlib import Path

import pytest

from aind_video_utils import transcode as transcode_mod
from aind_video_utils.preview_metadata import write_preview_metadata
from aind_video_utils.transcode import transcode_video

pq = pytest.importorskip("pyarrow.parquet")


def _metadata_csv(path: Path, n_frames: int) -> Path:
    rows = "".join(f"{0.002 * i + 1000.0},{i},{i * 2000}\n" for i in range(n_frames))
    path.write_text("ReferenceTime,CameraFrameNumber,CameraFrameTime\n" + rows)
    return path


def test_keeps_every_nth_row_starting_with_the_first(tmp_path):
    out = write_preview_metadata(_metadata_csv(tmp_path / "metadata.csv", 45), tmp_path / "p.parquet", 20)
    table = pq.read_table(out)
    assert table.column_names == ["ReferenceTime"]
    assert table.column("ReferenceTime").to_pylist() == pytest.approx([1000.0, 1000.04, 1000.08])


def test_row_count_is_ceil_of_frames_over_factor(tmp_path):
    out = write_preview_metadata(_metadata_csv(tmp_path / "metadata.csv", 41), tmp_path / "p.parquet", 20)
    assert pq.read_table(out).num_rows == 3


def test_copies_other_columns_unchanged(tmp_path):
    csv = _metadata_csv(tmp_path / "metadata.csv", 45)
    out = write_preview_metadata(csv, tmp_path / "p.parquet", 20, columns=None)
    table = pq.read_table(out)
    assert table.column("CameraFrameNumber").to_pylist() == [0, 20, 40]
    assert table.column("CameraFrameTime").to_pylist() == [0, 40000, 80000]


def test_columns_must_include_reference_time(tmp_path):
    with pytest.raises(ValueError, match="ReferenceTime"):
        write_preview_metadata(
            _metadata_csv(tmp_path / "metadata.csv", 5), tmp_path / "p.parquet", 2, columns=["CameraFrameNumber"]
        )


def test_missing_reference_time_column_is_a_value_error(tmp_path):
    csv = tmp_path / "metadata.csv"
    csv.write_text("CameraFrameNumber\n0\n1\n")
    with pytest.raises(ValueError, match="ReferenceTime"):
        write_preview_metadata(csv, tmp_path / "p.parquet", 2)


def test_rejects_a_nonpositive_factor(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        write_preview_metadata(_metadata_csv(tmp_path / "metadata.csv", 5), tmp_path / "p.parquet", 0)


# ---------------------------------------------------------------------------
# transcode_video(metadata_csv=...)
# ---------------------------------------------------------------------------


def _summary(decoded: int, primary: int, preview: int) -> bytes:
    return (
        f"[info] Input stream #0:0 (video): {decoded} packets read (1 bytes); {decoded} frames decoded\n"
        f"[info] Output stream #0:0 (video): {primary} frames encoded; {primary} packets muxed (1 bytes);\n"
        f"[info] Output stream #1:0 (video): {preview} frames encoded; {preview} packets muxed (1 bytes);\n"
    ).encode()


class _FakePopen:
    def __init__(self, cmd, stderr_bytes: bytes):
        import io

        self.args = cmd
        self.stdout = io.BytesIO(b"frame=100\nprogress=end\n")
        self.stderr = io.BytesIO(stderr_bytes)
        self.returncode = 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch(monkeypatch, stderr_bytes: bytes) -> None:
    def fake_probe(_path, **_kwargs):
        return {"streams": [{"pix_fmt": "gbrp", "color_space": "gbr", "color_range": "pc", "r_frame_rate": "500/1"}]}

    monkeypatch.setattr(transcode_mod, "probe", fake_probe)
    monkeypatch.setattr(transcode_mod.subprocess, "Popen", lambda cmd, **_kw: _FakePopen(cmd, stderr_bytes))


def test_transcode_writes_preview_metadata_beside_the_primary(monkeypatch, tmp_path):
    _patch(monkeypatch, _summary(100, 100, 5))
    csv = _metadata_csv(tmp_path / "metadata.csv", 100)
    transcode_video(tmp_path / "in.avi", tmp_path / "video.mp4", preview_fps=25.0, metadata_csv=csv)
    table = pq.read_table(tmp_path / "preview_metadata.parquet")
    assert table.num_rows == 5
    assert table.column("ReferenceTime").to_pylist() == pytest.approx([1000.0 + 0.04 * k for k in range(5)])


def test_transcode_refuses_metadata_that_does_not_match_the_frames(monkeypatch, tmp_path):
    _patch(monkeypatch, _summary(100, 100, 5))
    csv = _metadata_csv(tmp_path / "metadata.csv", 99)
    with pytest.raises(RuntimeError, match="99 rows"):
        transcode_video(tmp_path / "in.avi", tmp_path / "video.mp4", preview_fps=25.0, metadata_csv=csv)
    assert not (tmp_path / "preview_metadata.parquet").exists()


def test_transcode_needs_a_preview_for_metadata(tmp_path):
    with pytest.raises(ValueError, match="preview_fps"):
        transcode_video(tmp_path / "in.avi", tmp_path / "video.mp4", metadata_csv=tmp_path / "metadata.csv")


def test_transcode_reads_metadata_before_encoding(monkeypatch, tmp_path):
    """A bad metadata.csv must fail before a long encode, not after it."""
    calls: list = []
    monkeypatch.setattr(transcode_mod.subprocess, "Popen", lambda *a, **k: calls.append(a))
    csv = tmp_path / "metadata.csv"
    csv.write_text("CameraFrameNumber\n0\n")
    with pytest.raises(ValueError, match="ReferenceTime"):
        transcode_video(tmp_path / "in.avi", tmp_path / "video.mp4", preview_fps=25.0, metadata_csv=csv)
    assert calls == []


def _poster_run(monkeypatch, tmp_path, *, duration: str | None, **kwargs) -> tuple[list, str]:
    """Transcode a Matroska-like source with a poster; return the probes made and the poster's select."""
    seen: list = []

    def fake_probe(_path, count_packets=False):
        seen.append(count_packets)
        stream = {"pix_fmt": "gbrp", "color_space": "gbr", "color_range": "pc", "r_frame_rate": "500/1"}
        if count_packets:
            stream["nb_read_packets"] = "90"
        return {"streams": [stream], "format": {} if duration is None else {"duration": duration}}

    captured: list = []

    def fake_popen(cmd, **_kw):
        captured.append(cmd)
        return _FakePopen(cmd, _summary(100, 100, 1))

    monkeypatch.setattr(transcode_mod, "probe", fake_probe)
    monkeypatch.setattr(transcode_mod.subprocess, "Popen", fake_popen)
    transcode_video(tmp_path / "in.mkv", tmp_path / "video.mp4", poster=True, fail_on_frame_drop=False, **kwargs)
    assert captured[0][-1] == str(tmp_path / "poster.jpg")
    graph = captured[0][captured[0].index("-filter_complex") + 1]
    return seen, graph


def test_poster_estimates_the_middle_from_the_duration_without_reading_the_file(monkeypatch, tmp_path):
    seen, graph = _poster_run(monkeypatch, tmp_path, duration="0.2")
    assert seen == [False]
    assert "select=eq(n\\,50)" in graph


def test_poster_counts_packets_when_asked(monkeypatch, tmp_path):
    seen, graph = _poster_run(monkeypatch, tmp_path, duration="0.2", count_frames=True)
    assert seen == [False, True]
    assert "select=eq(n\\,45)" in graph


def test_poster_counts_packets_when_the_source_records_no_duration(monkeypatch, tmp_path):
    """A recording killed mid-write has neither a frame count nor a duration."""
    seen, graph = _poster_run(monkeypatch, tmp_path, duration=None)
    assert seen == [False, True]
    assert "select=eq(n\\,45)" in graph
