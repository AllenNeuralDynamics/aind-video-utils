from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.typing import ArrayLike, NDArray


def get_hist_bins(
    frame: npt.NDArray[np.uint8] | npt.NDArray[np.uint16],
) -> range:
    if frame.dtype == "uint8":
        bins = range(2**8 + 1)
    elif frame.dtype == "uint16":
        bins = range(2**10 + 1)
    else:
        raise ValueError(f"Unsupported frame dtype: {frame.dtype}")
    return bins


def plot_hist(ax: Axes, frame: npt.NDArray[np.uint8] | npt.NDArray[np.uint16]) -> Any:
    bins = get_hist_bins(frame)
    return ax.hist(frame.ravel(), bins=bins)


def plot_frame_and_hist(
    frame: npt.NDArray[np.uint8] | npt.NDArray[np.uint16],
) -> tuple[Figure, npt.NDArray[Any], AxesImage, Any]:
    f, axs = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw={"height_ratios": [3, 1]},
        figsize=(10, 8),
    )
    imreturn = axs[0].imshow(frame, cmap="gray")
    histreturn = plot_hist(axs[1], frame)
    return f, axs, imreturn, histreturn


def grayscale_with_clipping_highlights(
    image: ArrayLike,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | Colormap = "gray",
    underexposed_color: tuple[float, float, float] = (0.0, 0.0, 1.0),
    overexposed_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> NDArray[np.floating]:
    """
    Convert a grayscale image to RGB, highlighting clipped pixels.

    Parameters
    ----------
    image
        2D grayscale image.
    vmin
        Minimum display value. Pixels <= vmin are marked underexposed.
        Default: image minimum (so nothing is underexposed).
    vmax
        Maximum display value. Pixels >= vmax are marked overexposed.
        Default: image maximum (so nothing is overexposed).
    cmap
        Colormap for non-clipped pixels.
    underexposed_color
        RGB values (0-1) for underexposed pixels.
    overexposed_color
        RGB values (0-1) for overexposed pixels.

    Returns
    -------
    NDArray[np.floating]
        RGB image with shape (H, W, 3) and values in [0, 1].
    """
    img = np.asarray(image, dtype=np.float64)

    if vmin is None:
        vmin = float(np.nanmin(img))
    if vmax is None:
        vmax = float(np.nanmax(img))

    colormap: Colormap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = colormap(norm(img))
    rgb = rgba[..., :3].copy()

    rgb[img <= vmin] = underexposed_color
    rgb[img >= vmax] = overexposed_color

    return rgb


def imshow_clipping(
    image: ArrayLike,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | Colormap = "gray",
    ax: Axes | None = None,
    **imshow_kwargs,  # noqa: ANN003
) -> AxesImage:
    """Display grayscale image with blue/red clipping highlights."""
    rgb = grayscale_with_clipping_highlights(image, vmin=vmin, vmax=vmax, cmap=cmap)
    if ax is None:
        ax = plt.gca()
    return ax.imshow(rgb, **imshow_kwargs)


def intensity_histogram(
    image: ArrayLike,
    clip_vmin: float | None = None,
    clip_vmax: float | None = None,
    log_scale: bool = False,
    ax: Axes | None = None,
    auto_dual_scale: bool = True,
    dual_scale_threshold: float = 3.0,
    force_dual_scale: bool = False,
    extreme_values: tuple[int, int] = (0, 255),
    intensity_range: tuple[int, int] = (0, 255),
    underexposed_color: str = "#2166ac",
    overexposed_color: str = "#b2182b",
    main_color: str = "0.25",
    stem_linewidth: float = 0.8,
    marker_size: float = 1.5,
) -> Axes:
    """
    Plot intensity histogram with Tufte-style stems and optional dual-scale for extremes.

    Parameters
    ----------
    image
        Input image (will be flattened).
    clip_vmin, clip_vmax
        Clipping thresholds to display as vertical lines.
    log_scale
        If True, use log scale for counts.
    ax
        Axes to plot on. Created if None.
    auto_dual_scale
        Automatically use dual scale if extreme counts dominate.
    dual_scale_threshold
        Use dual scale if extreme count > threshold * median(interior counts).
    force_dual_scale
        Always use dual scale regardless of data.
    extreme_values
        Intensity values considered "extreme" (typically sensor min/max).
    intensity_range
        Range of x-axis (min, max intensity to display).
    underexposed_color, overexposed_color
        Colors for extreme value indicators and clip lines.
    main_color
        Color for main histogram stems.
    stem_linewidth
        Line width for stems.
    marker_size
        Marker size for stem tops.

    Returns
    -------
    plt.Axes
        The primary axes (secondary axes accessible via ax.right_ax if dual scale used).
    """
    img = np.asarray(image).ravel()
    lo, hi = intensity_range
    ext_lo, ext_hi = extreme_values

    # Compute histogram over integer bins
    bins = np.arange(lo, hi + 2) - 0.5
    counts, edges = np.histogram(img, bins=bins)
    intensities = np.arange(lo, hi + 1)

    # Separate extreme and interior counts
    ext_lo_idx = ext_lo - lo
    ext_hi_idx = ext_hi - lo
    count_lo = counts[ext_lo_idx] if 0 <= ext_lo_idx < len(counts) else 0
    count_hi = counts[ext_hi_idx] if 0 <= ext_hi_idx < len(counts) else 0

    interior_mask = np.ones(len(counts), dtype=bool)
    if 0 <= ext_lo_idx < len(counts):
        interior_mask[ext_lo_idx] = False
    if 0 <= ext_hi_idx < len(counts):
        interior_mask[ext_hi_idx] = False

    interior_counts = counts[interior_mask]
    interior_intensities = intensities[interior_mask]

    # Decide whether to use dual scale
    use_dual = force_dual_scale
    if auto_dual_scale and not use_dual:
        if len(interior_counts) > 0 and np.median(interior_counts) > 0:
            median_interior = np.median(interior_counts[interior_counts > 0])
            max_extreme = max(count_lo, count_hi)
            use_dual = max_extreme > dual_scale_threshold * median_interior

    # Set up axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.figure

    _apply_tufte_style(ax)

    # Plot main histogram as stems
    y_data = interior_counts.astype(float)
    if log_scale and not use_dual:
        y_data = np.where(y_data > 0, y_data, np.nan)

    ax.vlines(
        interior_intensities, 0, y_data, colors=main_color, linewidth=stem_linewidth
    )
    ax.plot(interior_intensities, y_data, "o", color=main_color, markersize=marker_size)

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
    else:
        ax.set_ylim(bottom=0)

    ax.set_xlim(lo - 1, hi + 1)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count")

    # Handle extreme values
    if use_dual:
        ax2 = ax.twinx()
        _apply_tufte_style(ax2, right_spine=True)
        ax2.set_ylabel("Extreme counts", color="0.4")

        # Plot extreme counts as larger markers on secondary axis
        extreme_x = []
        extreme_y = []
        extreme_c = []
        if count_lo > 0:
            extreme_x.append(ext_lo)
            extreme_y.append(count_lo)
            extreme_c.append(underexposed_color)
        if count_hi > 0:
            extreme_x.append(ext_hi)
            extreme_y.append(count_hi)
            extreme_c.append(overexposed_color)

        for x, y, c in zip(extreme_x, extreme_y, extreme_c):
            ax2.vlines(x, 0, y, colors=c, linewidth=stem_linewidth * 1.5)
            ax2.plot(x, y, "o", color=c, markersize=marker_size * 3)

        if log_scale:
            ax2.set_yscale("log")
            ax2.set_ylim(bottom=0.5)
        else:
            ax2.set_ylim(bottom=0)

        # Attach for external access
        ax.right_ax = ax2  # type: ignore[attr-defined]

    else:
        # Plot extremes on same scale, just colored differently
        if count_lo > 0:
            y_lo = count_lo if not log_scale else max(count_lo, 0.5)
            ax.vlines(
                ext_lo, 0, y_lo, colors=underexposed_color, linewidth=stem_linewidth
            )
            ax.plot(
                ext_lo, y_lo, "o", color=underexposed_color, markersize=marker_size * 2
            )
        if count_hi > 0:
            y_hi = count_hi if not log_scale else max(count_hi, 0.5)
            ax.vlines(
                ext_hi, 0, y_hi, colors=overexposed_color, linewidth=stem_linewidth
            )
            ax.plot(
                ext_hi, y_hi, "o", color=overexposed_color, markersize=marker_size * 2
            )

    # Add clip range indicators
    if clip_vmin is not None:
        ax.axvline(
            clip_vmin, color=underexposed_color, linestyle="--", linewidth=1, alpha=0.7
        )
    if clip_vmax is not None:
        ax.axvline(
            clip_vmax, color=overexposed_color, linestyle="--", linewidth=1, alpha=0.7
        )

    plt.tight_layout()
    return ax


def _apply_tufte_style(ax: Axes, right_spine: bool = False) -> None:
    """Apply minimalist Tufte-inspired styling to axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(right_spine)
    if not right_spine:
        ax.spines["left"].set_position(("outward", 5))
    else:
        ax.spines["right"].set_position(("outward", 5))
        ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 5))

    ax.tick_params(direction="out", length=3, width=0.8)
    ax.tick_params(axis="x", which="both", top=False)
    ax.tick_params(axis="y", which="both", right=right_spine, left=not right_spine)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
