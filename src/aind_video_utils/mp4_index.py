"""Frame-accurate sample index parsed directly from an MP4 ``moov`` atom.

The ISO base media file format keys everything on **sample numbers** (1-based,
decode order): ``stss`` lists sync samples by sample number, ``stsc``/``stsz``/
``stco``/``co64`` map sample number to byte offset, and ``stts``/``ctts`` map
sample number to decode/composition time.  Time is *derived* from that index,
not the other way round.  The general-purpose demux APIs (ffmpeg, PyAV) expose
only timestamp-based seeking, so this module reads the sample tables itself to
recover the native frame index.

This is a pure ``struct`` + numpy parser — no ffmpeg, no PyAV, no extra
dependency (URL support uses stdlib ``urllib``).  It reads only the ``moov``
atom (a few MB), never the ``mdat`` payload, so it is cheap even on multi-GB
files — including over HTTP(S), where it fetches the ``moov`` with ``Range``
requests.  It handles both 32-bit ``stco`` and 64-bit ``co64`` chunk offsets —
the latter is mandatory for files over 4 GB.

The index is immune to the glitchy-PTS problem that afflicts concat-seam
timelines: presentation order is recovered by *sorting* PTS (a stable argsort),
which the ±few-tick seam stretches cannot perturb, and frame-to-byte lookups
never touch a timestamp at all.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from aind_video_utils.utils import is_url

# ISO-BMFF video handler type (mdia/hdlr).
_VIDE_HANDLER = b"vide"

# Seconds before an HTTP range request is abandoned.
_HTTP_TIMEOUT = 30


class _ByteSource(Protocol):
    """Random-access byte source: a local file or an HTTP(S) range reader."""

    @property
    def size(self) -> int:
        """Total size of the source in bytes."""
        ...

    def read_at(self, offset: int, length: int) -> bytes:
        """Return *length* bytes starting at *offset*."""
        ...

    def close(self) -> None:
        """Release any held resources."""
        ...


class _LocalByteSource:
    """Random access over a local seekable file."""

    def __init__(self, path: Path) -> None:
        self._handle = path.open("rb")
        self._handle.seek(0, 2)
        self._size = self._handle.tell()

    @property
    def size(self) -> int:
        return self._size

    def read_at(self, offset: int, length: int) -> bytes:
        self._handle.seek(offset)
        return self._handle.read(length)

    def close(self) -> None:
        self._handle.close()


class _HttpByteSource:
    """Random access over an HTTP(S) URL via ``Range`` requests.

    Fetches only the byte ranges asked for, so the multi-GB ``mdat`` payload is
    never downloaded — only the (small, front-loaded on faststart files)
    ``moov`` atom.  Requires a server that honours range requests (S3 and any
    standard static host do).
    """

    def __init__(self, url: str) -> None:
        self._url = url
        # A 1-byte range probe both confirms range support and reports the
        # total size via the Content-Range header.
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
            if response.status != 206:
                raise ValueError(
                    f"{url}: server does not support HTTP range requests "
                    f"(status {response.status}); cannot read the moov remotely"
                )
            content_range = response.headers.get("Content-Range", "")
            response.read()
        total = content_range.rsplit("/", 1)[-1] if "/" in content_range else ""
        if not total.isdigit():
            raise ValueError(f"{url}: missing or unparseable Content-Range total: {content_range!r}")
        self._size = int(total)

    @property
    def size(self) -> int:
        return self._size

    def read_at(self, offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        request = urllib.request.Request(self._url, headers={"Range": f"bytes={offset}-{offset + length - 1}"})
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
            if response.status != 206:
                raise ValueError(f"{self._url}: expected 206 Partial Content, got {response.status}")
            data: bytes = response.read()
        return data

    def close(self) -> None:  # nothing persistent to release
        pass


def _open_source(path: str | Path) -> _ByteSource:
    """Open *path* as a local file or an HTTP(S) range source."""
    if is_url(path):
        return _HttpByteSource(str(path))
    return _LocalByteSource(Path(path))


@dataclass(frozen=True)
class EditListEntry:
    """A single ``edts/elst`` edit-list entry.

    Attributes
    ----------
    segment_duration : int
        Duration of this edit in the *movie* timescale (``mvhd``).
    media_time : int
        Starting time in this edit in the *media* timescale (``mdhd``);
        ``-1`` denotes an empty edit (blank/delay insertion).
    media_rate : float
        Playback rate for this edit (``1.0`` is normal forward playback).
    """

    segment_duration: int
    media_time: int
    media_rate: float


@dataclass(frozen=True, eq=False)
class Mp4FrameIndex:
    """Per-sample index for the video track of an MP4 file.

    All arrays are in **decode order** and indexed 0-based by sample number.
    Use :attr:`display_order` to map presentation position to decode index.

    Attributes
    ----------
    media_timescale : int
        Ticks per second of the media timeline (``mdhd`` timescale).
    media_duration : int
        Track duration in ``media_timescale`` ticks (``mdhd`` duration).
    movie_timescale : int
        Ticks per second of the movie/presentation timeline (``mvhd``).
    dts : NDArray[np.int64]
        Decode timestamp of each sample, in ``media_timescale`` ticks.
    pts : NDArray[np.int64]
        Presentation timestamp of each sample (``dts`` plus the ``ctts``
        composition offset), in ``media_timescale`` ticks.
    is_keyframe : NDArray[np.bool_]
        ``True`` where the sample is a sync sample (``stss``); all ``True``
        when the file has no ``stss`` box (all-intra).
    byte_offset : NDArray[np.int64]
        Absolute byte offset of each sample within the file.
    size : NDArray[np.int64]
        Size of each sample in bytes.
    edits : tuple[EditListEntry, ...]
        The track's edit list, empty when there is no ``elst`` box.
    """

    media_timescale: int
    media_duration: int
    movie_timescale: int
    dts: npt.NDArray[np.int64]
    pts: npt.NDArray[np.int64]
    is_keyframe: npt.NDArray[np.bool_]
    byte_offset: npt.NDArray[np.int64]
    size: npt.NDArray[np.int64]
    edits: tuple[EditListEntry, ...]

    @property
    def n_samples(self) -> int:
        """Total number of video samples (frames)."""
        return int(self.dts.shape[0])

    @property
    def display_order(self) -> npt.NDArray[np.intp]:
        """Decode indices ordered by presentation time.

        ``display_order[k]`` is the decode-order index of the ``k``-th frame in
        presentation order.  Computed with a *stable* argsort of :attr:`pts`, so
        equal timestamps keep decode order and the seam-glitch stretches (which
        preserve monotonicity) cannot change the ranking.
        """
        return np.argsort(self.pts, kind="stable")

    @property
    def keyframe_decode_indices(self) -> npt.NDArray[np.intp]:
        """Decode-order indices of the sync samples (keyframes), ascending."""
        return np.flatnonzero(self.is_keyframe)

    def keyframe_at_or_before(self, decode_index: int) -> int:
        """Return the decode index of the nearest keyframe at or before *decode_index*.

        Parameters
        ----------
        decode_index : int
            A 0-based decode-order sample index.

        Returns
        -------
        int
            Decode index of the closest preceding (or equal) keyframe — the
            sample a decoder must start from to reconstruct *decode_index*.

        Raises
        ------
        ValueError
            If *decode_index* is out of range, or no keyframe precedes it.
        """
        if not 0 <= decode_index < self.n_samples:
            raise ValueError(f"decode_index {decode_index} out of range [0, {self.n_samples})")
        keyframes = self.keyframe_decode_indices
        pos = int(np.searchsorted(keyframes, decode_index, side="right")) - 1
        if pos < 0:
            raise ValueError(f"no keyframe at or before decode_index {decode_index}")
        return int(keyframes[pos])

    def seek_plan(self, display_index: int) -> tuple[float, int]:
        """Return an ``(seek_seconds, frames_after_keyframe)`` plan for frame *display_index*.

        Encodes the verified frame-exact extraction recipe: seek an MP4 decoder
        to ``seek_seconds`` with input-side ``-ss`` (which addresses the
        *presentation* timeline), then decode ``frames_after_keyframe`` frames
        forward — counting the landed keyframe as 0 — to reach *display_index*.

        The seek targets the keyframe beginning *display_index*'s GOP.  Its
        presentation time is the keyframe's media PTS minus the edit list's
        ``media_time`` (a single trivial ``elst`` maps that media time to
        presentation 0); without the subtraction the seek lands ``media_time``
        ticks late.

        Parameters
        ----------
        display_index : int
            0-based frame index in presentation order.

        Returns
        -------
        seek_seconds : float
            Presentation-timeline time to pass to input-side ``-ss``.  Format
            with enough decimals that it rounds to the exact media tick (``-ss``
            lands on the nearest keyframe at or before the request).
        frames_after_keyframe : int
            Number of frames to decode past the keyframe to reach the target.

        Raises
        ------
        ValueError
            If *display_index* is out of range, or the edit list is not
            frame-addressing-safe (see :meth:`is_frame_addressing_safe`).
        """
        if not 0 <= display_index < self.n_samples:
            raise ValueError(f"display_index {display_index} out of range [0, {self.n_samples})")
        if not self.is_frame_addressing_safe():
            raise ValueError("edit list is not frame-addressing-safe; cannot derive a simple seek time")
        order = self.display_order
        decode_index = int(order[display_index])
        keyframe = self.keyframe_at_or_before(decode_index)
        keyframe_display_rank = int(np.flatnonzero(order == keyframe)[0])
        media_time = self.edits[0].media_time if self.edits else 0
        seek_seconds = (int(self.pts[keyframe]) - media_time) / self.media_timescale
        return seek_seconds, display_index - keyframe_display_rank

    def is_frame_addressing_safe(self) -> bool:
        """Whether the edit list is safe to ignore for frame-index addressing.

        Frame addressing works on media sample order and ignores the edit list.
        That is correct only when the ``elst`` merely shifts the presentation
        clock without dropping or reordering samples — i.e. no edits, or a
        single normal-rate edit whose media_time trims at most the reorder delay
        (so every displayed frame is retained) and whose segment spans the rest
        of the media (so the tail is not cut).

        Returns
        -------
        bool
            ``True`` when every media sample is presented exactly once in PTS
            order.  ``False`` for empty edits, multi-entry lists, non-unit
            rates, front trims that skip displayed frames, or tail trims —
            cases where the caller must honor the edit list instead.
        """
        if not self.edits:
            return True
        if len(self.edits) != 1:
            return False
        edit = self.edits[0]
        if edit.media_rate != 1.0 or edit.media_time < 0:
            return False
        # A front trim beyond the first displayed frame (minimum PTS) drops
        # frames; media_time up to min(pts) only removes reorder delay.
        min_pts = int(self.pts.min())
        if not 0 <= edit.media_time <= min_pts:
            return False
        # The segment (in movie ticks) must cover the media from media_time to
        # its end, else the tail is trimmed. Allow one max-sample slop for
        # rounding between the two timescales.
        if self.movie_timescale <= 0:
            return False
        segment_media_ticks = edit.segment_duration * self.media_timescale / self.movie_timescale
        presented_span = self.media_duration - edit.media_time
        slop = int(np.diff(self.dts).max()) if self.n_samples > 1 else 0
        return segment_media_ticks >= presented_span - slop


def _iter_child_boxes(buf: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """Yield ``(box_type, body_start, box_end)`` for each box in ``buf[start:end]``."""
    off = start
    while off + 8 <= end:
        size = int.from_bytes(buf[off : off + 4], "big")
        box_type = buf[off + 4 : off + 8]
        header_len = 8
        if size == 1:
            size = int.from_bytes(buf[off + 8 : off + 16], "big")
            header_len = 16
        elif size == 0:
            size = end - off
        if size < header_len or off + size > end:
            break
        yield box_type, off + header_len, off + size
        off += size


def _find_child(buf: bytes, start: int, end: int, box_type: bytes) -> tuple[int, int] | None:
    """Return ``(body_start, box_end)`` of the first child box of *box_type*, or ``None``."""
    for found_type, body_start, box_end in _iter_child_boxes(buf, start, end):
        if found_type == box_type:
            return body_start, box_end
    return None


def _locate_and_read_moov(source: _ByteSource) -> bytes:
    """Return the ``moov`` box body, scanning top-level boxes without reading ``mdat``.

    Reads only box headers (16 bytes each) until ``moov`` is found, then fetches
    just its body — so over HTTP this is a handful of small range requests plus
    one for the moov itself, never the ``mdat`` payload.
    """
    file_size = source.size
    offset = 0
    while True:
        header = source.read_at(offset, 16)
        if len(header) < 8:
            raise ValueError("reached end of source without finding a moov box")
        size = int.from_bytes(header[0:4], "big")
        box_type = header[4:8]
        header_len = 8
        if size == 1:
            if len(header) < 16:
                raise ValueError("truncated 64-bit box header")
            size = int.from_bytes(header[8:16], "big")
            header_len = 16
        elif size == 0:
            size = file_size - offset
        if size < header_len:
            raise ValueError(f"invalid box size {size} at offset {offset}")
        if box_type == b"moov":
            return source.read_at(offset + header_len, size - header_len)
        offset += size


def _box_version(buf: bytes, body_start: int) -> int:
    """Return the version byte of a full box."""
    return buf[body_start]


def _parse_timescale_duration(buf: bytes, body_start: int) -> tuple[int, int]:
    """Return ``(timescale, duration)`` from an ``mvhd``/``mdhd`` full box."""
    version = _box_version(buf, body_start)
    if version == 1:
        timescale = int.from_bytes(buf[body_start + 20 : body_start + 24], "big")
        duration = int.from_bytes(buf[body_start + 24 : body_start + 32], "big")
    else:
        timescale = int.from_bytes(buf[body_start + 12 : body_start + 16], "big")
        duration = int.from_bytes(buf[body_start + 16 : body_start + 20], "big")
    return timescale, duration


def _parse_elst(buf: bytes, body_start: int) -> tuple[EditListEntry, ...]:
    """Parse an ``elst`` full box into edit-list entries."""
    version = _box_version(buf, body_start)
    count = int.from_bytes(buf[body_start + 4 : body_start + 8], "big")
    pos = body_start + 8
    entries: list[EditListEntry] = []
    for _ in range(count):
        if version == 1:
            segment = int.from_bytes(buf[pos : pos + 8], "big")
            media_time = int.from_bytes(buf[pos + 8 : pos + 16], "big", signed=True)
            pos += 16
        else:
            segment = int.from_bytes(buf[pos : pos + 4], "big")
            media_time = int.from_bytes(buf[pos + 4 : pos + 8], "big", signed=True)
            pos += 8
        rate_fixed = int.from_bytes(buf[pos : pos + 4], "big", signed=True)
        pos += 4
        entries.append(EditListEntry(segment, media_time, rate_fixed / 65536.0))
    return tuple(entries)


def _entry_count(buf: bytes, body_start: int) -> int:
    """Return the ``entry_count`` field of a table full box."""
    return int.from_bytes(buf[body_start + 4 : body_start + 8], "big")


def _decode_timestamps(buf: bytes, stts_body: int, n_samples: int) -> npt.NDArray[np.int64]:
    """Expand an ``stts`` table into a per-sample decode-timestamp array."""
    count = _entry_count(buf, stts_body)
    table = np.frombuffer(
        buf,
        dtype=np.dtype([("count", ">u4"), ("delta", ">u4")]),
        count=count,
        offset=stts_body + 8,
    )
    deltas = np.repeat(table["delta"].astype(np.int64), table["count"].astype(np.int64))
    if deltas.shape[0] != n_samples:
        raise ValueError(f"stts describes {deltas.shape[0]} samples but stsz has {n_samples}")
    dts = np.empty(n_samples, dtype=np.int64)
    dts[0] = 0
    if n_samples > 1:
        np.cumsum(deltas[:-1], out=dts[1:])
    return dts


def _composition_offsets(buf: bytes, ctts_body: int | None, n_samples: int) -> npt.NDArray[np.int64]:
    """Expand a ``ctts`` table into a per-sample composition-offset array.

    Returns all-zero offsets when there is no ``ctts`` box, and zero-pads the
    tail when the table covers fewer samples than the track holds.
    """
    if ctts_body is None:
        return np.zeros(n_samples, dtype=np.int64)
    version = _box_version(buf, ctts_body)
    count = _entry_count(buf, ctts_body)
    offset_dtype = ">i4" if version == 1 else ">u4"
    table = np.frombuffer(
        buf,
        dtype=np.dtype([("count", ">u4"), ("offset", offset_dtype)]),
        count=count,
        offset=ctts_body + 8,
    )
    offsets = np.repeat(table["offset"].astype(np.int64), table["count"].astype(np.int64))
    if offsets.shape[0] < n_samples:
        offsets = np.concatenate([offsets, np.zeros(n_samples - offsets.shape[0], dtype=np.int64)])
    return offsets[:n_samples]


def _keyframe_flags(buf: bytes, stss_body: int | None, n_samples: int) -> npt.NDArray[np.bool_]:
    """Build a per-sample keyframe mask from an ``stss`` table (all-True if absent)."""
    if stss_body is None:
        return np.ones(n_samples, dtype=np.bool_)
    count = _entry_count(buf, stss_body)
    sample_numbers = np.frombuffer(buf, dtype=">u4", count=count, offset=stss_body + 8).astype(np.int64)
    flags = np.zeros(n_samples, dtype=np.bool_)
    flags[sample_numbers - 1] = True
    return flags


def _sample_sizes(buf: bytes, stsz_body: int) -> npt.NDArray[np.int64]:
    """Return the per-sample size array from an ``stsz`` table."""
    uniform_size = int.from_bytes(buf[stsz_body + 4 : stsz_body + 8], "big")
    sample_count = int.from_bytes(buf[stsz_body + 8 : stsz_body + 12], "big")
    if uniform_size != 0:
        return np.full(sample_count, uniform_size, dtype=np.int64)
    return np.frombuffer(buf, dtype=">u4", count=sample_count, offset=stsz_body + 12).astype(np.int64)


def _chunk_offsets(buf: bytes, stco_body: int | None, co64_body: int | None) -> npt.NDArray[np.int64]:
    """Return chunk byte offsets from ``stco`` (32-bit) or ``co64`` (64-bit)."""
    if co64_body is not None:
        count = _entry_count(buf, co64_body)
        return np.frombuffer(buf, dtype=">u8", count=count, offset=co64_body + 8).astype(np.int64)
    if stco_body is not None:
        count = _entry_count(buf, stco_body)
        return np.frombuffer(buf, dtype=">u4", count=count, offset=stco_body + 8).astype(np.int64)
    raise ValueError("stbl has neither stco nor co64 chunk-offset box")


def _samples_per_chunk(buf: bytes, stsc_body: int, n_chunks: int) -> npt.NDArray[np.int64]:
    """Expand an ``stsc`` run-length table into a per-chunk sample count."""
    count = _entry_count(buf, stsc_body)
    table = np.frombuffer(
        buf,
        dtype=np.dtype([("first_chunk", ">u4"), ("spc", ">u4"), ("sdi", ">u4")]),
        count=count,
        offset=stsc_body + 8,
    )
    first_chunks = table["first_chunk"].astype(np.int64)  # 1-based
    per_chunk_values = table["spc"].astype(np.int64)
    spc = np.empty(n_chunks, dtype=np.int64)
    for entry in range(count):
        start = int(first_chunks[entry]) - 1
        stop = int(first_chunks[entry + 1]) - 1 if entry + 1 < count else n_chunks
        spc[start:stop] = per_chunk_values[entry]
    return spc


def _sample_byte_offsets(
    chunk_offsets: npt.NDArray[np.int64],
    spc: npt.NDArray[np.int64],
    sizes: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Compute the absolute byte offset of every sample from the chunk layout."""
    n_samples = int(sizes.shape[0])
    if int(spc.sum()) != n_samples:
        raise ValueError(f"stsc/stco describe {int(spc.sum())} samples but stsz has {n_samples}")
    sample_chunk = np.repeat(np.arange(chunk_offsets.shape[0]), spc)
    cumulative = np.empty(n_samples + 1, dtype=np.int64)
    cumulative[0] = 0
    np.cumsum(sizes, out=cumulative[1:])
    chunk_first_sample = np.zeros(chunk_offsets.shape[0], dtype=np.int64)
    chunk_first_sample[1:] = np.cumsum(spc)[:-1]
    within_chunk = cumulative[:n_samples] - cumulative[chunk_first_sample][sample_chunk]
    return chunk_offsets[sample_chunk] + within_chunk


def _find_video_stbl(moov: bytes) -> tuple[int, int, int, int, tuple[EditListEntry, ...]]:
    """Locate the video track and return ``(stbl_start, stbl_end, timescale, duration, edits)``."""
    for box_type, trak_start, trak_end in _iter_child_boxes(moov, 0, len(moov)):
        if box_type != b"trak":
            continue
        mdia = _find_child(moov, trak_start, trak_end, b"mdia")
        if mdia is None:
            continue
        hdlr = _find_child(moov, mdia[0], mdia[1], b"hdlr")
        if hdlr is None or moov[hdlr[0] + 8 : hdlr[0] + 12] != _VIDE_HANDLER:
            continue
        mdhd = _find_child(moov, mdia[0], mdia[1], b"mdhd")
        if mdhd is None:
            raise ValueError("video mdia has no mdhd box")
        timescale, duration = _parse_timescale_duration(moov, mdhd[0])
        edits: tuple[EditListEntry, ...] = ()
        edts = _find_child(moov, trak_start, trak_end, b"edts")
        if edts is not None:
            elst = _find_child(moov, edts[0], edts[1], b"elst")
            if elst is not None:
                edits = _parse_elst(moov, elst[0])
        minf = _find_child(moov, mdia[0], mdia[1], b"minf")
        if minf is None:
            raise ValueError("video mdia has no minf box")
        stbl = _find_child(moov, minf[0], minf[1], b"stbl")
        if stbl is None:
            raise ValueError("video minf has no stbl box")
        return stbl[0], stbl[1], timescale, duration, edits
    raise ValueError("no video track (mdia/hdlr == 'vide') found in moov")


def read_mp4_frame_index(path: str | Path) -> Mp4FrameIndex:
    """Parse the video sample tables of an MP4 file into a frame index.

    Reads only the ``moov`` atom (not the ``mdat`` payload), so it is fast even
    on multi-GB files.  Accepts a local, seekable file or an HTTP(S) URL: URLs
    are read with ``Range`` requests (the server must honour them, as S3 and
    standard static hosts do), fetching only the ``moov`` — which sits at the
    front of a faststart file — never the payload.

    Parameters
    ----------
    path : str | Path
        Path to, or ``http(s)://`` URL of, a non-fragmented MP4/MOV file.

    Returns
    -------
    Mp4FrameIndex
        Per-sample decode/composition timestamps, keyframe flags, and byte
        offsets for the file's video track.

    Raises
    ------
    ValueError
        If the source has no ``moov``, no video track, a malformed sample table
        (e.g. ``stts``/``stsc`` sample counts disagree with ``stsz``), or is a
        URL whose server does not support range requests.
    """
    source = _open_source(path)
    try:
        moov = _locate_and_read_moov(source)
    finally:
        source.close()
    stbl_start, stbl_end, timescale, duration, edits = _find_video_stbl(moov)

    mvhd = _find_child(moov, 0, len(moov), b"mvhd")
    movie_timescale = _parse_timescale_duration(moov, mvhd[0])[0] if mvhd is not None else 0

    stsz = _find_child(moov, stbl_start, stbl_end, b"stsz")
    if stsz is None:
        raise ValueError("stbl has no stsz sample-size box")
    sizes = _sample_sizes(moov, stsz[0])
    n_samples = int(sizes.shape[0])

    stts = _find_child(moov, stbl_start, stbl_end, b"stts")
    if stts is None:
        raise ValueError("stbl has no stts time-to-sample box")
    dts = _decode_timestamps(moov, stts[0], n_samples)

    ctts = _find_child(moov, stbl_start, stbl_end, b"ctts")
    composition = _composition_offsets(moov, ctts[0] if ctts else None, n_samples)
    pts = dts + composition

    stss = _find_child(moov, stbl_start, stbl_end, b"stss")
    is_keyframe = _keyframe_flags(moov, stss[0] if stss else None, n_samples)

    stsc = _find_child(moov, stbl_start, stbl_end, b"stsc")
    if stsc is None:
        raise ValueError("stbl has no stsc sample-to-chunk box")
    stco = _find_child(moov, stbl_start, stbl_end, b"stco")
    co64 = _find_child(moov, stbl_start, stbl_end, b"co64")
    chunk_offsets = _chunk_offsets(moov, stco[0] if stco else None, co64[0] if co64 else None)
    spc = _samples_per_chunk(moov, stsc[0], int(chunk_offsets.shape[0]))
    byte_offset = _sample_byte_offsets(chunk_offsets, spc, sizes)

    return Mp4FrameIndex(
        media_timescale=timescale,
        media_duration=duration,
        movie_timescale=movie_timescale,
        dts=dts,
        pts=pts,
        is_keyframe=is_keyframe,
        byte_offset=byte_offset,
        size=sizes,
        edits=edits,
    )
