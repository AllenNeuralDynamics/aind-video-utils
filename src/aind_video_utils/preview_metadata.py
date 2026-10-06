"""Write ``preview_metadata.parquet``, the per-frame metadata of a preview.

The spec requires a preview to sit beside a Parquet file holding rows
``0, N, 2N, ...`` of ``metadata.csv``, so its row *k* describes preview frame
*k*.  ``ReferenceTime`` is required; any other column is optional and copied
unchanged.  Needs the ``parquet`` extra (pyarrow).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

PREVIEW_METADATA_FILENAME = "preview_metadata.parquet"
"""The spec's name for a preview's metadata, beside the primary video."""

REQUIRED_COLUMNS: tuple[str, ...] = ("ReferenceTime",)
"""Columns of ``metadata.csv`` the spec requires the preview metadata to hold."""


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
    """
    return write_decimated(read_frame_metadata(metadata_csv, columns), output_path, factor)
