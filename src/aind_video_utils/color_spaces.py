"""Rec. 709 transfer characteristic and luma range helpers."""

from __future__ import annotations


def rec_709_trc_to_linear(value: float) -> float:
    """Convert a Rec. 709 transfer characteristic value to linear light.

    Parameters
    ----------
    value : float
        The input value in the range [0.0, 1.0].

    Returns
    -------
    float
        The linear light value in the range [0.0, 1.0].
    """
    if value < 0.081:
        return value / 4.5
    else:
        return float(((value + 0.099) / 1.099) ** (1 / 0.45))


def linear_to_rec_709_trc(value: float) -> float:
    """Convert a linear light value to Rec. 709 transfer characteristic.

    Parameters
    ----------
    value : float
        The input linear light value in the range [0.0, 1.0].

    Returns
    -------
    float
        The Rec. 709 transfer characteristic value in the range [0.0, 1.0].
    """
    if value < 0.018:
        return value * 4.5
    else:
        return float(1.099 * (value**0.45) - 0.099)


def luma_range(bit_depth: int, is_full_range: bool) -> tuple[int, int]:
    """Get the luma value range for a given bit depth and color range.

    Parameters
    ----------
    bit_depth : int
        The bit depth (e.g., 8 or 10).
    is_full_range : bool
        Whether the color range is full (True) or limited (False).

    Returns
    -------
    tuple[int, int]
        The (minimum, maximum) luma values.
    """
    if is_full_range:
        min_luma = 0
        max_luma = (1 << bit_depth) - 1
    else:
        if bit_depth == 8:
            min_luma = 16
            max_luma = 235
        elif bit_depth == 10:
            min_luma = 64
            max_luma = 940
        else:
            raise ValueError(f"Unsupported bit depth for limited range: {bit_depth}")
    return min_luma, max_luma
