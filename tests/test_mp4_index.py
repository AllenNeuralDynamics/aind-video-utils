"""Tests for the MP4 moov frame-index parser."""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from aind_video_utils import EditListEntry, Mp4FrameIndex, extract_frame_by_index, read_mp4_frame_index
from aind_video_utils.mp4_index import (
    _chunk_offsets,
    _composition_offsets,
    _decode_timestamps,
    _keyframe_flags,
    _sample_byte_offsets,
    _sample_sizes,
    _samples_per_chunk,
)

ffmpeg_required = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


# ---------------------------------------------------------------------------
# Byte-level box builders (a "full box" is a 1-byte version + 3 flag bytes)
# ---------------------------------------------------------------------------


def _full_box(payload: bytes, version: int = 0) -> bytes:
    return bytes([version, 0, 0, 0]) + payload


def _stts(entries: list[tuple[int, int]]) -> bytes:
    body = struct.pack(">I", len(entries)) + b"".join(struct.pack(">II", c, d) for c, d in entries)
    return _full_box(body)


def _ctts(entries: list[tuple[int, int]], version: int = 0) -> bytes:
    fmt = ">Ii" if version == 1 else ">II"
    body = struct.pack(">I", len(entries)) + b"".join(struct.pack(fmt, c, o) for c, o in entries)
    return _full_box(body, version=version)


def _stss(sample_numbers: list[int]) -> bytes:
    body = struct.pack(">I", len(sample_numbers)) + b"".join(struct.pack(">I", n) for n in sample_numbers)
    return _full_box(body)


def _stsz(sizes: list[int], uniform: int = 0) -> bytes:
    count = len(sizes) if uniform == 0 else uniform  # count is a placeholder when uniform
    if uniform != 0:
        body = struct.pack(">II", uniform, count) + b""
    else:
        body = struct.pack(">II", 0, len(sizes)) + b"".join(struct.pack(">I", s) for s in sizes)
    return _full_box(body)


def _stsz_uniform(size: int, count: int) -> bytes:
    return _full_box(struct.pack(">II", size, count))


def _stco(offsets: list[int]) -> bytes:
    body = struct.pack(">I", len(offsets)) + b"".join(struct.pack(">I", o) for o in offsets)
    return _full_box(body)


def _co64(offsets: list[int]) -> bytes:
    body = struct.pack(">I", len(offsets)) + b"".join(struct.pack(">Q", o) for o in offsets)
    return _full_box(body)


def _stsc(entries: list[tuple[int, int, int]]) -> bytes:
    body = struct.pack(">I", len(entries)) + b"".join(struct.pack(">III", a, b, c) for a, b, c in entries)
    return _full_box(body)


# ---------------------------------------------------------------------------
# stts -> decode timestamps
# ---------------------------------------------------------------------------


def test_decode_timestamps_runlength_expands_and_accumulates():
    dts = _decode_timestamps(_stts([(3, 10)]), 0, 3)
    assert dts.tolist() == [0, 10, 20]


def test_decode_timestamps_multiple_runs():
    dts = _decode_timestamps(_stts([(2, 10), (2, 5)]), 0, 4)
    assert dts.tolist() == [0, 10, 20, 25]


def test_decode_timestamps_sample_count_mismatch_raises():
    with pytest.raises(ValueError, match="stts describes"):
        _decode_timestamps(_stts([(3, 10)]), 0, 4)


# ---------------------------------------------------------------------------
# ctts -> composition offsets (PTS = DTS + offset)
# ---------------------------------------------------------------------------


def test_composition_offsets_none_is_all_zero():
    assert _composition_offsets(b"", None, 3).tolist() == [0, 0, 0]


def test_composition_offsets_v0_unsigned():
    off = _composition_offsets(_ctts([(2, 5), (1, 0)]), 0, 3)
    assert off.tolist() == [5, 5, 0]


def test_composition_offsets_v1_signed_negative():
    off = _composition_offsets(_ctts([(1, -8), (2, 4)], version=1), 0, 3)
    assert off.tolist() == [-8, 4, 4]


def test_composition_offsets_short_table_zero_pads_tail():
    off = _composition_offsets(_ctts([(2, 7)]), 0, 4)
    assert off.tolist() == [7, 7, 0, 0]


# ---------------------------------------------------------------------------
# stss -> keyframe flags
# ---------------------------------------------------------------------------


def test_keyframe_flags_none_is_all_true():
    assert _keyframe_flags(b"", None, 3).tolist() == [True, True, True]


def test_keyframe_flags_from_sample_numbers():
    assert _keyframe_flags(_stss([1, 3]), 0, 4).tolist() == [True, False, True, False]


# ---------------------------------------------------------------------------
# stsz -> sample sizes
# ---------------------------------------------------------------------------


def test_sample_sizes_explicit_table():
    assert _sample_sizes(_stsz([10, 20, 30]), 0).tolist() == [10, 20, 30]


def test_sample_sizes_uniform():
    assert _sample_sizes(_stsz_uniform(7, 3), 0).tolist() == [7, 7, 7]


# ---------------------------------------------------------------------------
# stco / co64 -> chunk offsets
# ---------------------------------------------------------------------------


def test_chunk_offsets_stco_32bit():
    assert _chunk_offsets(_stco([100, 200]), 0, None).tolist() == [100, 200]


def test_chunk_offsets_co64_64bit_beyond_4gb():
    big = 5_000_000_000  # > 2**32, requires 64-bit read
    assert _chunk_offsets(_co64([big, big + 10]), None, 0).tolist() == [big, big + 10]


def test_chunk_offsets_co64_preferred_when_both_present():
    # co64 wins if both are (nonsensically) present; the co64 buffer is used.
    assert _chunk_offsets(_co64([9]), 0, 0).tolist() == [9]


def test_chunk_offsets_none_raises():
    with pytest.raises(ValueError, match="neither stco nor co64"):
        _chunk_offsets(b"", None, None)


# ---------------------------------------------------------------------------
# stsc + stsz + stco -> per-sample byte offsets
# ---------------------------------------------------------------------------


def test_sample_byte_offsets_two_chunks_two_samples_each():
    spc = _samples_per_chunk(_stsc([(1, 2, 1)]), 0, 2)
    assert spc.tolist() == [2, 2]
    offsets = _sample_byte_offsets(
        np.array([1000, 2000], dtype=np.int64),
        spc,
        np.array([10, 20, 30, 40], dtype=np.int64),
    )
    # chunk0 @1000: 1000, 1010 ; chunk1 @2000: 2000, 2030
    assert offsets.tolist() == [1000, 1010, 2000, 2030]


def test_samples_per_chunk_multiple_runs():
    # chunk1 holds 1 sample; chunks 2 and 3 hold 2 each.
    spc = _samples_per_chunk(_stsc([(1, 1, 1), (2, 2, 1)]), 0, 3)
    assert spc.tolist() == [1, 2, 2]


def test_sample_byte_offsets_sample_count_mismatch_raises():
    with pytest.raises(ValueError, match="stsc/stco describe"):
        _sample_byte_offsets(
            np.array([0], dtype=np.int64),
            np.array([2], dtype=np.int64),  # says 2 samples
            np.array([10, 20, 30], dtype=np.int64),  # but 3 sizes
        )


# ---------------------------------------------------------------------------
# Mp4FrameIndex dataclass helpers (constructed directly, no MP4 needed)
# ---------------------------------------------------------------------------


def _index(
    *,
    pts: list[int],
    dts: list[int] | None = None,
    is_keyframe: list[bool] | None = None,
    edits: tuple[EditListEntry, ...] = (),
    media_timescale: int = 16000,
    media_duration: int = 0,
    movie_timescale: int = 1000,
) -> Mp4FrameIndex:
    n = len(pts)
    return Mp4FrameIndex(
        media_timescale=media_timescale,
        media_duration=media_duration,
        movie_timescale=movie_timescale,
        dts=np.array(dts if dts is not None else list(range(n)), dtype=np.int64),
        pts=np.array(pts, dtype=np.int64),
        is_keyframe=np.array(is_keyframe if is_keyframe is not None else [True] * n, dtype=np.bool_),
        byte_offset=np.zeros(n, dtype=np.int64),
        size=np.ones(n, dtype=np.int64),
        edits=edits,
    )


def test_n_samples():
    assert _index(pts=[0, 1, 2]).n_samples == 3


def test_display_order_sorts_by_pts_stably():
    # Reordered B-frame timeline: decode order != display order.
    order = _index(pts=[64, 192, 128, 96]).display_order
    assert order.tolist() == [0, 3, 2, 1]


def test_keyframe_decode_indices():
    idx = _index(pts=[0, 1, 2, 3], is_keyframe=[True, False, True, False])
    assert idx.keyframe_decode_indices.tolist() == [0, 2]


def test_keyframe_at_or_before_returns_preceding_keyframe():
    idx = _index(pts=list(range(6)), is_keyframe=[True, False, False, True, False, False])
    assert idx.keyframe_at_or_before(0) == 0
    assert idx.keyframe_at_or_before(2) == 0
    assert idx.keyframe_at_or_before(3) == 3
    assert idx.keyframe_at_or_before(5) == 3


def test_keyframe_at_or_before_out_of_range_raises():
    idx = _index(pts=[0, 1, 2])
    with pytest.raises(ValueError, match="out of range"):
        idx.keyframe_at_or_before(3)
    with pytest.raises(ValueError, match="out of range"):
        idx.keyframe_at_or_before(-1)


def test_keyframe_at_or_before_no_preceding_keyframe_raises():
    idx = _index(pts=[0, 1, 2], is_keyframe=[False, False, True])
    with pytest.raises(ValueError, match="no keyframe at or before"):
        idx.keyframe_at_or_before(1)


# ---------------------------------------------------------------------------
# Edit-list safety (the elst "ignore-after-assert" contract)
# ---------------------------------------------------------------------------


def test_no_edit_list_is_safe():
    assert _index(pts=[64, 96, 128], dts=[0, 32, 64]).is_frame_addressing_safe()


def test_trivial_reorder_compensation_edit_is_safe():
    # media_time == min(pts) trims only the reorder delay; segment covers the media.
    idx = _index(
        pts=[64, 192, 128, 96],
        dts=[0, 32, 64, 96],
        media_duration=128,
        edits=(EditListEntry(segment_duration=100, media_time=64, media_rate=1.0),),
    )
    assert idx.is_frame_addressing_safe()


def test_front_trim_beyond_first_frame_is_unsafe():
    idx = _index(
        pts=[64, 192, 128, 96],
        dts=[0, 32, 64, 96],
        media_duration=128,
        edits=(EditListEntry(segment_duration=100, media_time=200, media_rate=1.0),),
    )
    assert not idx.is_frame_addressing_safe()


def test_nonunit_rate_is_unsafe():
    idx = _index(
        pts=[64, 96],
        dts=[0, 32],
        media_duration=64,
        edits=(EditListEntry(segment_duration=100, media_time=0, media_rate=0.5),),
    )
    assert not idx.is_frame_addressing_safe()


def test_empty_edit_is_unsafe():
    idx = _index(
        pts=[64, 96],
        dts=[0, 32],
        media_duration=64,
        edits=(EditListEntry(segment_duration=100, media_time=-1, media_rate=1.0),),
    )
    assert not idx.is_frame_addressing_safe()


def test_multiple_edits_are_unsafe():
    idx = _index(
        pts=[64, 96],
        dts=[0, 32],
        media_duration=64,
        edits=(
            EditListEntry(segment_duration=50, media_time=0, media_rate=1.0),
            EditListEntry(segment_duration=50, media_time=100, media_rate=1.0),
        ),
    )
    assert not idx.is_frame_addressing_safe()


def test_tail_trim_is_unsafe():
    # Segment far shorter than the presented span => the tail is cut.
    idx = _index(
        pts=[0, 32, 64, 96],
        dts=[0, 32, 64, 96],
        media_duration=128,
        edits=(EditListEntry(segment_duration=1, media_time=0, media_rate=1.0),),
    )
    assert not idx.is_frame_addressing_safe()


# ---------------------------------------------------------------------------
# seek_plan (the verified frame-exact extraction recipe, computed on the index)
# ---------------------------------------------------------------------------
#
# A reordered 2-GOP timeline: keyframes at decode 0 and 3, B-frame reordering
# within each GOP, and (optionally) a trivial reorder-compensation edit
# (media_time == 64 == min pts). display_order works out to [0, 2, 1, 3, 5, 4].
def _reordered_index(edits: tuple[EditListEntry, ...] = ()) -> Mp4FrameIndex:
    return _index(
        pts=[64, 128, 96, 160, 224, 192],
        dts=[0, 32, 64, 96, 128, 160],
        is_keyframe=[True, False, False, True, False, False],
        media_timescale=16000,
        media_duration=224,
        edits=edits,
    )


def test_seek_plan_display_order_is_reordered():
    assert _reordered_index().display_order.tolist() == [0, 2, 1, 3, 5, 4]


def test_seek_plan_subtracts_media_time_and_counts_from_keyframe():
    idx = _reordered_index(edits=(EditListEntry(1000, 64, 1.0),))
    # display 5 -> decode 4, covering keyframe decode 3 (display rank 3).
    # seek = (pts[3] - media_time)/ts = (160-64)/16000 = 0.006 ; count = 5-3 = 2
    seek, count = idx.seek_plan(5)
    assert seek == pytest.approx(0.006)
    assert count == 2


def test_seek_plan_first_frame_seeks_to_zero():
    idx = _reordered_index(edits=(EditListEntry(1000, 64, 1.0),))
    seek, count = idx.seek_plan(0)
    assert seek == pytest.approx(0.0)
    assert count == 0


def test_seek_plan_without_edit_list_uses_raw_pts():
    idx = _reordered_index()  # no edits
    # display 5 -> decode 4, keyframe decode 3, no media_time subtraction.
    seek, count = idx.seek_plan(5)
    assert seek == pytest.approx(160 / 16000)
    assert count == 2


def test_seek_plan_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        _reordered_index().seek_plan(6)


def test_seek_plan_unsafe_edit_list_raises():
    idx = _reordered_index(edits=(EditListEntry(1000, 500, 1.0),))  # front trim
    with pytest.raises(ValueError, match="frame-addressing-safe"):
        idx.seek_plan(2)


# ---------------------------------------------------------------------------
# End-to-end against a real (small) B-frame MP4 built with ffmpeg
# ---------------------------------------------------------------------------


def _make_bframe_mp4(path: Path) -> None:
    """Encode a short faststart H.264 clip with B-frames (=> ctts + edit list)."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=30:duration=2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-bf",
            "2",
            "-g",
            "10",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
    )


def _ffprobe_keyframe_flags(path: Path) -> list[int]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=key_frame",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    # Frame 0 carries SEI side-data that ffprobe appends after a comma on the
    # same line; the key_frame flag is always the first CSV field.
    return [int(line.split(",")[0]) for line in out.splitlines() if line.strip()]


@ffmpeg_required
def test_read_mp4_frame_index_end_to_end(tmp_path: Path) -> None:
    src = tmp_path / "bframes.mp4"
    _make_bframe_mp4(src)
    idx = read_mp4_frame_index(src)

    # Frame count and per-sample keyframe flags match ffprobe (decode order).
    ff_keyframes = _ffprobe_keyframe_flags(src)
    assert idx.n_samples == len(ff_keyframes)
    assert idx.is_keyframe.astype(int).tolist() == ff_keyframes

    # Decode timestamps are strictly increasing; the presence of B-frames means
    # decode order != display order, but PTS sorts into a monotone timeline.
    assert bool(np.all(np.diff(idx.dts) > 0))
    assert bool(np.all(np.diff(idx.pts[idx.display_order]) >= 0))

    # Byte offsets land inside the file and past the header.
    file_size = src.stat().st_size
    assert int(idx.byte_offset.min()) > 0
    assert int((idx.byte_offset + idx.size).max()) <= file_size

    # ffmpeg writes the benign reorder-compensation edit -> safe to ignore.
    assert idx.is_frame_addressing_safe()

    # Every keyframe helper answer is itself a keyframe at or before the query.
    for query in (0, idx.n_samples // 2, idx.n_samples - 1):
        kf = idx.keyframe_at_or_before(query)
        assert kf <= query
        assert bool(idx.is_keyframe[kf])


@ffmpeg_required
def test_read_mp4_frame_index_has_composition_offsets_with_bframes(tmp_path: Path) -> None:
    src = tmp_path / "bframes.mp4"
    _make_bframe_mp4(src)
    idx = read_mp4_frame_index(src)
    # With B-frames, some frames carry a nonzero composition offset (pts != dts).
    assert bool(np.any(idx.pts != idx.dts))


def _decode_frame_from_zero(path: Path, n: int, w: int = 64, h: int = 64) -> np.ndarray:
    """Ground truth: decode from frame 0 and emit display-order frame *n*."""
    out = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"select=eq(n\\,{n}),format=rgb24",
            "-frames:v",
            "1",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(h, w, 3)


@ffmpeg_required
@pytest.mark.parametrize("n", [0, 1, 37, 59])
def test_extract_frame_by_index_matches_decode_from_zero(tmp_path: Path, n: int) -> None:
    src = tmp_path / "bframes.mp4"
    _make_bframe_mp4(src)
    got = extract_frame_by_index(src, n)
    assert got.shape == (64, 64, 3)
    assert np.array_equal(got, _decode_frame_from_zero(src, n))


@ffmpeg_required
def test_extract_frame_by_index_accepts_prebuilt_index(tmp_path: Path) -> None:
    src = tmp_path / "bframes.mp4"
    _make_bframe_mp4(src)
    idx = read_mp4_frame_index(src)
    got = extract_frame_by_index(src, 20, index=idx)
    assert np.array_equal(got, _decode_frame_from_zero(src, 20))


@ffmpeg_required
def test_extract_frame_by_index_out_of_range_raises(tmp_path: Path) -> None:
    src = tmp_path / "bframes.mp4"
    _make_bframe_mp4(src)
    with pytest.raises(ValueError, match="out of range"):
        extract_frame_by_index(src, 10_000)


def test_read_mp4_frame_index_non_mp4_raises(tmp_path: Path) -> None:
    junk = tmp_path / "not.mp4"
    junk.write_bytes(b"this is definitely not an ISO base media file" * 4)
    with pytest.raises(ValueError, match="moov"):
        read_mp4_frame_index(junk)
