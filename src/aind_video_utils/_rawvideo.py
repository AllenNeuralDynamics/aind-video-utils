"""Internal helpers for pixel format constants and raw video buffer parsing."""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt

_SUPPORTED_YUV_FORMATS_8BIT = {
    "".join(["yuv", r, chroma, "p"]) for r, chroma in itertools.product(["j", ""], ["420", "422", "444"])
}

# yuvj* formats are deprecated 8-bit-only aliases; 10-bit has no j-prefix variants
_SUPPORTED_YUV_FORMATS_10BIT = {f"yuv{chroma}p10le" for chroma in ["420", "422", "444"]}

_SUPPORTED_GBR_FORMATS_8BIT = {"gbrp"}
_SUPPORTED_GBR_FORMATS_10BIT = {"gbrp10le"}

_ALL_SUPPORTED_FORMATS_8BIT = _SUPPORTED_YUV_FORMATS_8BIT | _SUPPORTED_GBR_FORMATS_8BIT
_ALL_SUPPORTED_FORMATS_10BIT = _SUPPORTED_YUV_FORMATS_10BIT | _SUPPORTED_GBR_FORMATS_10BIT


def _is_gbr_format(pix_fmt: str) -> bool:
    """Check whether a pixel format string is a GBR planar format."""
    return pix_fmt in _SUPPORTED_GBR_FORMATS_8BIT | _SUPPORTED_GBR_FORMATS_10BIT


def pix_format_bit_depth(pix_fmt: str) -> int:
    """Return the bit depth for a supported pixel format.

    Parameters
    ----------
    pix_fmt : str
        FFmpeg pixel format string (e.g. ``"yuv420p"``, ``"gbrp10le"``).

    Returns
    -------
    int
        Bits per component (8 or 10).

    Raises
    ------
    ValueError
        If *pix_fmt* is not a supported format.
    """
    if pix_fmt in _ALL_SUPPORTED_FORMATS_10BIT:
        return 10
    elif pix_fmt in _ALL_SUPPORTED_FORMATS_8BIT:
        return 8
    raise ValueError(f"Unsupported pixel format: {pix_fmt}")


def luma_from_yuv420p_buff_eltype(
    pxdata: bytes, w: int, h: int, eltype: type[np.uint8 | np.uint16]
) -> npt.NDArray[np.uint8 | np.uint16]:
    """Extract the luma (Y) plane from a raw YUV planar buffer.

    Reads the first ``w * h`` elements (the Y plane) and ignores any
    chroma data that follows.

    Parameters
    ----------
    pxdata : bytes
        Raw pixel data from ffmpeg rawvideo output.
    w, h : int
        Frame width and height in pixels.
    eltype : type
        NumPy element type (``np.uint8`` for 8-bit, ``np.uint16`` for 10-bit).

    Returns
    -------
    NDArray
        Luma plane with shape ``(h, w)``.
    """
    y_len = w * h
    yarr = np.frombuffer(pxdata, dtype=eltype, count=y_len).reshape(h, w)
    return yarr  # type: ignore[return-value]


def rgb_from_rawvideo_rgb24_buff(pxdata: bytes, w: int, h: int) -> npt.NDArray[np.uint8]:
    """Parse a raw rgb24 buffer into an ``(h, w, 3)`` uint8 array.

    Parameters
    ----------
    pxdata : bytes
        Raw pixel data in packed RGB24 format.
    w, h : int
        Frame width and height in pixels.

    Returns
    -------
    NDArray[np.uint8]
        RGB image with shape ``(h, w, 3)``.
    """
    rgb_len = w * h * 3
    rgbarr = np.frombuffer(pxdata, dtype=np.uint8, count=rgb_len).reshape(h, w, 3)
    return rgbarr


def luma_from_rawvideo_yuvp420_buff(pxdata: bytes, w: int, h: int) -> npt.NDArray[np.uint8]:
    """Extract the 8-bit luma plane from a raw YUV420P buffer.

    Parameters
    ----------
    pxdata : bytes
        Raw pixel data from ffmpeg rawvideo output.
    w, h : int
        Frame width and height in pixels.

    Returns
    -------
    NDArray[np.uint8]
        Luma plane with shape ``(h, w)``.
    """
    return luma_from_yuv420p_buff_eltype(pxdata, w, h, np.uint8)  # type: ignore[return-value]


def luma_from_rawvideo_yuv420p10le_buff(pxdata: bytes, w: int, h: int) -> npt.NDArray[np.uint16]:
    """Extract the 10-bit luma plane from a raw YUV420P10LE buffer.

    Parameters
    ----------
    pxdata : bytes
        Raw pixel data from ffmpeg rawvideo output.
    w, h : int
        Frame width and height in pixels.

    Returns
    -------
    NDArray[np.uint16]
        Luma plane with shape ``(h, w)``.
    """
    return luma_from_yuv420p_buff_eltype(pxdata, w, h, np.uint16)  # type: ignore[return-value]
