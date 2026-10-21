"""Write ``preview_metadata.parquet``, the per-frame metadata of a preview.

The spec requires a preview to sit beside a Parquet file holding rows
``0, N, 2N, ...`` of ``metadata.csv``, so its row *k* describes preview frame
*k*.  ``ReferenceTime`` is required; any other column is optional and copied
unchanged.  Needs the ``parquet`` extra (pyarrow).

The values are copied as recorded, flaws included: the preview's metadata is a
proxy for ``metadata.csv``, not a corrected version of it.  Where frames were
dropped, the recorded times may need repair
(:func:`aind_video_utils.acquisition_timing.repair_frame_times`), and the rows
between the kept ones are what a repair needs, so repair from ``metadata.csv``
itself.  :class:`PreviewTimingWarning` says when that applies.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

PREVIEW_METADATA_FILENAME = "preview_metadata.parquet"
"""The spec's name for a preview's metadata, beside the primary video."""

REQUIRED_COLUMNS: tuple[str, ...] = ("ReferenceTime",)
"""Columns of ``metadata.csv`` the spec requires the preview metadata to hold."""


FRAME_COUNTER_COLUMN = "CameraFrameNumber"


class PreviewTimingWarning(UserWarning):
    """``metadata.csv`` records dropped frames, so its times may need a repair the preview's rows cannot support."""


def _pyarrow() -> Any:
    """Import pyarrow's CSV and Parquet modules, naming the extra that provides them."""
    try:
        import pyarrow as pa
        import pyarrow.csv
        import pyarrow.parquet
    except ImportError as err:
        raise ImportError("writing preview_metadata.parquet needs pyarrow: install aind-video-utils[parquet]") from err
    return pa


def read_frame_metadata(metadata_csv: Path, columns: Sequence[str] | None = REQUIRED_COLUMNS) -> Any:
    """Read *columns* of a camera's ``metadata.csv`` into a pyarrow table, one row per frame.

    Parameters
    ----------
    metadata_csv : Path
        The camera's ``metadata.csv``.
    columns : Sequence[str] | None
        Columns to keep; ``None`` keeps every column.  Types are inferred from
        the file rather than imposed, since the spec fixes none.

    Returns
    -------
    pyarrow.Table
        The selected columns, in file order.

    Raises
    ------
    ValueError
        If ``columns`` omits or the file lacks a column the spec requires.
    """
    pa = _pyarrow()
    if columns is not None and (missing := [c for c in REQUIRED_COLUMNS if c not in columns]):
        raise ValueError(f"columns must include {missing}, which preview_metadata.parquet requires")
    convert = pa.csv.ConvertOptions(include_columns=None if columns is None else list(columns))
    try:
        table = pa.csv.read_csv(metadata_csv, convert_options=convert)
    except KeyError as err:
        raise ValueError(f"{metadata_csv}: {err.args[0] if err.args else err}") from err
    if absent := [c for c in REQUIRED_COLUMNS if c not in table.column_names]:
        raise ValueError(f"{metadata_csv} has no column {absent}")
    return table


def warn_if_frames_dropped(metadata_csv: Path) -> None:
    """Warn with :class:`PreviewTimingWarning` if the frame counter of *metadata_csv* skips or steps backwards.

    A file without the counter column raises no warning, since it shows
    nothing either way.
    """
    pa = _pyarrow()
    convert = pa.csv.ConvertOptions(include_columns=[FRAME_COUNTER_COLUMN], include_missing_columns=True)
    column = pa.csv.read_csv(metadata_csv, convert_options=convert).column(FRAME_COUNTER_COLUMN)
    if column.null_count == len(column):
        return
    steps = np.diff(column.to_numpy(zero_copy_only=False).astype(np.int64))
    skips, backwards = int((steps > 1).sum()), int((steps < 0).sum())
    if not (skips or backwards):
        return
    found = f"skips {skips} time(s)" if skips else f"steps backwards {backwards} time(s)"
    warnings.warn(
        f"{metadata_csv}: {FRAME_COUNTER_COLUMN} {found}, so its recorded times may need repair; the preview's "
        "metadata copies them as recorded and keeps too few rows to repair them, so repair from "
        f"{metadata_csv.name} itself (aind_video_utils.repair_frame_times).",
        PreviewTimingWarning,
        stacklevel=3,
    )


def preview_metadata_path(preview_path: Path) -> Path:
    """Return the metadata path paired with *preview_path*, the spec's name for ``preview.mp4``."""
    return preview_path.with_name(f"{preview_path.stem}_metadata.parquet")


def write_decimated(table: Any, output_path: Path, factor: int) -> Path:
    """Write rows ``0, factor, 2 * factor, ...`` of *table* to *output_path* as Parquet.

    pyarrow's default encodings are kept so any Parquet reader can open the file.
    """
    if factor < 1:
        raise ValueError(f"factor must be at least 1, got {factor}")
    pa = _pyarrow()
    pa.parquet.write_table(table.take(pa.array(range(0, table.num_rows, factor))), output_path)
    return output_path


def write_preview_metadata(
    metadata_csv: Path,
    output_path: Path,
    factor: int,
    *,
    columns: Sequence[str] | None = REQUIRED_COLUMNS,
) -> Path:
    """Write a preview's ``preview_metadata.parquet`` from the camera's ``metadata.csv``.

    Parameters
    ----------
    metadata_csv : Path
        The camera's ``metadata.csv``, one row per frame of the primary video.
    output_path : Path
        Destination, normally ``preview_metadata.parquet`` beside the preview.
    factor : int
        The preview's decimation factor *N*, as
        :func:`aind_video_utils.encoding.preview_decimation` returns it.
    columns : Sequence[str] | None
        Columns to copy; must include ``ReferenceTime``.  ``None`` copies all.

    Returns
    -------
    Path
        *output_path*.

    Warns
    -----
    PreviewTimingWarning
        If ``CameraFrameNumber`` in *metadata_csv* skips or steps backwards.
    """
    warn_if_frames_dropped(metadata_csv)
    return write_decimated(read_frame_metadata(metadata_csv, columns), output_path, factor)
