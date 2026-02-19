"""Matplotlib-based visualization for intensity histograms and frame comparisons."""

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.colors import Colormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.image import AxesImage
from matplotlib.ticker import Formatter
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore[import-untyped]
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "SparseFormatter",
    "apply_sparse_ticklabels",
    "bivariate_intensity_histogram",
    "bivariate_with_marginals",
    "get_hist_bins",
    "grayscale_with_clipping_highlights",
    "imshow_clipping",
    "intensity_histogram",
    "luma_comparison_figure",
    "plot_frame_and_hist",
    "plot_hist",
]


def get_hist_bins(
    frame: npt.NDArray[np.uint8] | npt.NDArray[np.uint16],
) -> range:
    """Return histogram bin edges appropriate for the frame's dtype.

    Parameters
    ----------
    frame : NDArray[np.uint8] | NDArray[np.uint16]
        Image array. uint8 yields 256 bins; uint16 yields 1024 (10-bit).

    Returns
    -------
    range
        Bin edges for use with ``np.histogram`` or ``ax.hist``.
    """
    if frame.dtype == "uint8":
        bins = range(2**8 + 1)
    elif frame.dtype == "uint16":
        bins = range(2**10 + 1)
    else:
        raise ValueError(f"Unsupported frame dtype: {frame.dtype}")
    return bins


def plot_hist(ax: Axes, frame: npt.NDArray[np.uint8] | npt.NDArray[np.uint16]) -> Any:
    """Plot a pixel-intensity histogram on the given axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    frame : NDArray[np.uint8] | NDArray[np.uint16]
        Image array whose pixel values are histogrammed.

    Returns
    -------
    Any
        The return value of ``ax.hist``.
    """
    bins = get_hist_bins(frame)
    return ax.hist(frame.ravel(), bins=bins)


def plot_frame_and_hist(
    frame: npt.NDArray[np.uint8] | npt.NDArray[np.uint16],
) -> tuple[Figure, npt.NDArray[Any], AxesImage, Any]:
    """Display a grayscale image above its intensity histogram.

    Parameters
    ----------
    frame : NDArray[np.uint8] | NDArray[np.uint16]
        Grayscale image to visualize.

    Returns
    -------
    fig : Figure
    axs : NDArray
        Array of two axes (image, histogram).
    imreturn : AxesImage
    histreturn : Any
    """
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

    return rgb  # type: ignore[no-any-return]


def imshow_clipping(
    image: ArrayLike,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | Colormap = "gray",
    ax: Axes | None = None,
    **imshow_kwargs: Any,
) -> AxesImage:
    """Display grayscale image with blue/red clipping highlights."""
    rgb = grayscale_with_clipping_highlights(image, vmin=vmin, vmax=vmax, cmap=cmap)
    if ax is None:
        ax = plt.gca()
    return ax.imshow(rgb, **imshow_kwargs)


# =============================================================================
# Tufte-style intensity histogram
# =============================================================================


class SparseFormatter(Formatter):
    """Tick label formatter that shows every *n*-th label and hides the rest.

    Parameters
    ----------
    base : Formatter
        The underlying formatter to delegate label rendering to.
    every : int
        Show a label every *every* ticks (default 2 = every other tick).
    """

    def __init__(self, base: Formatter, every: int = 2) -> None:
        self.base = base
        self.every = every

    def set_locs(self, locs: Any) -> None:
        # Let the base formatter see the tick locations
        if hasattr(self.base, "set_locs"):
            self.base.set_locs(locs)
        # Keep Formatter's own bookkeeping
        super().set_locs(locs)

    def __call__(self, x: float, pos: int | None = None) -> str:
        if pos is not None and (pos - 1) % self.every == 0:
            return self.base(x, pos)
        return ""


def apply_sparse_ticklabels(axis: Axis, every: int = 2) -> None:
    """Replace the major tick formatter on *axis* with a sparse version.

    Parameters
    ----------
    axis : Axis
        A matplotlib ``XAxis`` or ``YAxis``.
    every : int
        Show a label every *every* ticks.
    """
    base = axis.get_major_formatter()
    axis.set_major_formatter(SparseFormatter(base, every=every))


def _apply_tufte_style(
    ax: Axes,
    right_spine: bool = False,
    top_spine: bool = False,
    sparse_tick_labels: bool = True,
) -> None:
    """Apply minimalist Tufte-inspired styling to axes."""
    ax.spines["top"].set_visible(top_spine)
    ax.spines["right"].set_visible(right_spine)
    ax.spines["left"].set_visible(not right_spine)
    ax.spines["bottom"].set_visible(not top_spine)

    offset = 5
    if not right_spine:
        ax.spines["left"].set_position(("outward", offset))
    else:
        ax.spines["right"].set_position(("outward", offset))
    if not top_spine:
        ax.spines["bottom"].set_position(("outward", offset))
    else:
        ax.spines["top"].set_position(("outward", offset))

    ax.tick_params(direction="out", length=3, width=0.8)
    ax.tick_params(axis="x", which="both", top=top_spine, bottom=not top_spine)
    ax.tick_params(axis="y", which="both", right=right_spine, left=not right_spine)

    # label every other tick with major formatter

    if sparse_tick_labels:
        apply_sparse_ticklabels(ax.xaxis, every=2)
        apply_sparse_ticklabels(ax.yaxis, every=2)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def intensity_histogram(  # noqa: C901
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
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_stems: bool = False,
) -> Axes:
    """
    Plot intensity histogram with Tufte-style stems and optional dual-scale.

    Parameters
    ----------
    image
        Input image (will be flattened).
    clip_vmin, clip_vmax
        Clipping thresholds to display as reference lines.
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
        Range of intensity axis (min, max to display).
    underexposed_color, overexposed_color
        Colors for extreme value indicators and clip lines.
    main_color
        Color for main histogram stems/dots.
    stem_linewidth
        Line width for stems (if show_stems=True).
    marker_size
        Marker size for dots.
    orientation
        'horizontal' (intensity on x-axis) or 'vertical' (intensity on y-axis).
    show_xlabel, show_ylabel
        Whether to show axis labels.
    show_stems
        If True, draw stem lines from zero to each dot. If False, show dots only.

    Returns
    -------
    Axes
        The primary axes (secondary axes accessible via ax.secondary_ax if dual scale).
    """
    img = np.asarray(image).ravel()
    lo, hi = intensity_range
    ext_lo, ext_hi = extreme_values

    # For discrete integer data, bin edges at half-integers ensure each
    # integer value falls into exactly one bin: value v -> bin [v-0.5, v+0.5)
    bin_edges = np.arange(lo - 0.5, hi + 1.5, 1)  # -0.5, 0.5, ..., 255.5
    counts, _ = np.histogram(img, bins=bin_edges)
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
        positive_interior = interior_counts[interior_counts > 0]
        if len(positive_interior) > 0:
            median_interior = np.median(positive_interior)
            max_extreme = max(count_lo, count_hi)
            use_dual = max_extreme > dual_scale_threshold * median_interior

    # Set up axes
    if ax is None:
        figsize = (8, 3) if orientation == "horizontal" else (3, 8)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        _ = ax.figure

    is_vertical = orientation == "vertical"
    _apply_tufte_style(ax, right_spine=is_vertical, top_spine=False)

    # Prepare data
    y_data = interior_counts.astype(float)
    if log_scale:
        y_data = np.where(y_data > 0, y_data, np.nan)

    # Plot main histogram as stems/dots
    if is_vertical:
        if show_stems:
            ax.hlines(
                interior_intensities,
                0,
                y_data,
                colors=main_color,
                linewidth=stem_linewidth,
            )
        ax.plot(y_data, interior_intensities, "o", color=main_color, markersize=marker_size)
        ax.set_ylim(lo - 1, hi + 1)
        if log_scale:
            ax.set_xscale("log")
            ax.set_xlim(left=0.5)
        else:
            ax.set_xlim(left=0)
        if show_xlabel:
            ax.set_xlabel("Count")
        if show_ylabel:
            ax.set_ylabel("Intensity")
    else:
        if show_stems:
            ax.vlines(
                interior_intensities,
                0,
                y_data,
                colors=main_color,
                linewidth=stem_linewidth,
            )
        ax.plot(interior_intensities, y_data, "o", color=main_color, markersize=marker_size)
        ax.set_xlim(lo - 1, hi + 1)
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.5)
        else:
            ax.set_ylim(bottom=0)
        if show_xlabel:
            ax.set_xlabel("Intensity")
        if show_ylabel:
            ax.set_ylabel("Count")

    # Handle extreme values
    if use_dual:
        if is_vertical:
            ax2 = ax.twiny()
            _apply_tufte_style(ax2, right_spine=True, top_spine=True)
        else:
            ax2 = ax.twinx()
            _apply_tufte_style(ax2, right_spine=True, top_spine=False)

        extreme_data = []
        if count_lo > 0:
            extreme_data.append((ext_lo, count_lo, underexposed_color))
        if count_hi > 0:
            extreme_data.append((ext_hi, count_hi, overexposed_color))

        for intensity, count, color in extreme_data:
            if is_vertical:
                if show_stems:
                    ax2.hlines(
                        intensity,
                        0,
                        count,
                        colors=color,
                        linewidth=stem_linewidth * 1.5,
                    )
                ax2.plot(count, intensity, "o", color=color, markersize=marker_size * 3)
            else:
                if show_stems:
                    ax2.vlines(
                        intensity,
                        0,
                        count,
                        colors=color,
                        linewidth=stem_linewidth * 1.5,
                    )
                ax2.plot(intensity, count, "o", color=color, markersize=marker_size * 3)

        if log_scale:
            if is_vertical:
                ax2.set_xscale("log")
                ax2.set_xlim(left=0.5)
            else:
                ax2.set_yscale("log")
                ax2.set_ylim(bottom=0.5)
        else:
            if is_vertical:
                ax2.set_xlim(left=0)
            else:
                ax2.set_ylim(bottom=0)

        ax.secondary_ax = ax2  # type: ignore[attr-defined]

    else:
        # Plot extremes on same scale with different colors
        for intensity, count, color in [
            (ext_lo, count_lo, underexposed_color),
            (ext_hi, count_hi, overexposed_color),
        ]:
            if count > 0:
                y_val = count if not log_scale else max(count, 0.5)
                if is_vertical:
                    if show_stems:
                        ax.hlines(intensity, 0, y_val, colors=color, linewidth=stem_linewidth)
                    ax.plot(y_val, intensity, "o", color=color, markersize=marker_size * 2)
                else:
                    if show_stems:
                        ax.vlines(intensity, 0, y_val, colors=color, linewidth=stem_linewidth)
                    ax.plot(intensity, y_val, "o", color=color, markersize=marker_size * 2)

    # Add clip range indicators
    if clip_vmin is not None:
        if is_vertical:
            ax.axhline(
                clip_vmin,
                color=underexposed_color,
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )
        else:
            ax.axvline(
                clip_vmin,
                color=underexposed_color,
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )
    if clip_vmax is not None:
        if is_vertical:
            ax.axhline(
                clip_vmax,
                color=overexposed_color,
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )
        else:
            ax.axvline(
                clip_vmax,
                color=overexposed_color,
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )

    return ax


def bivariate_intensity_histogram(
    input_image: ArrayLike,
    output_image: ArrayLike,
    ax: Axes | None = None,
    intensity_range: tuple[int, int] = (0, 255),
    output_limits: tuple[int, int] | None = (16, 235),
    input_limits: tuple[int, int] | None = None,
    show_identity: bool = True,
    cmap: str = "gray_r",
    log_scale: bool = True,
) -> Axes:
    """
    Create a 2D histogram comparing input vs output pixel intensities.

    Parameters
    ----------
    input_image, output_image
        Images to compare (will be flattened). Expected to contain discrete
        integer values within intensity_range.
    ax
        Axes to plot on. Created if None.
    intensity_range
        Range for both axes (typically 0-255).
    output_limits
        Reference lines for output range (e.g., bt709 16-235). None to hide.
    input_limits
        Reference lines for input range. None to hide.
    show_identity
        Show diagonal identity line.
    cmap
        Colormap for density visualization.
    log_scale
        Use log scale for the histogram counts.

    Returns
    -------
    Axes
    """
    input_flat = np.asarray(input_image).ravel()
    output_flat = np.asarray(output_image).ravel()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    _apply_tufte_style(ax)

    lo, hi = intensity_range
    for limits in (output_limits, input_limits):
        if limits is not None:
            lo = min(lo, limits[0])
            hi = max(hi, limits[1])

    # For discrete integer data, bin edges at half-integers ensure each
    # integer value falls into exactly one bin: value v -> bin [v-0.5, v+0.5)
    bin_edges = np.arange(lo - 0.5, hi + 1.5, 1)  # -0.5, 0.5, ..., 255.5

    hist, _, _ = np.histogram2d(
        input_flat,
        output_flat,
        bins=[bin_edges, bin_edges],
    )

    # Mask zeros for better visualization
    hist_masked = np.ma.masked_where(hist == 0, hist)  # type: ignore[no-untyped-call]

    # Plot with imshow (transpose so x=input, y=output)
    if log_scale:
        norm = LogNorm(vmin=1, vmax=hist.max()) if hist.max() > 0 else None
    else:
        norm = None

    ax.imshow(
        hist_masked.T,
        extent=(lo - 0.5, hi + 0.5, lo - 0.5, hi + 0.5),
        origin="lower",
        aspect="equal",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    # Reference lines for output range limits
    if output_limits is not None:
        ax.axhline(output_limits[0], color="#2166ac", linestyle="--", linewidth=1, alpha=0.8)
        ax.axhline(output_limits[1], color="#b2182b", linestyle="--", linewidth=1, alpha=0.8)

    # Reference lines for input range limits
    if input_limits is not None:
        ax.axvline(input_limits[0], color="#2166ac", linestyle="--", linewidth=1, alpha=0.8)
        ax.axvline(input_limits[1], color="#b2182b", linestyle="--", linewidth=1, alpha=0.8)

    if show_identity:
        ax.plot([lo, hi], [lo, hi], color="0.5", linestyle=":", linewidth=1, alpha=0.6)

    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(lo - 0.5, hi + 0.5)
    ax.set_xlabel("Input intensity")
    ax.set_ylabel("Output intensity")

    return ax


def bivariate_with_marginals(
    x_data: ArrayLike,
    y_data: ArrayLike,
    intensity_range: tuple[int, int] = (0, 255),
    x_clip: tuple[int, int] | None = None,
    y_clip: tuple[int, int] | None = None,
    x_limits: tuple[int, int] | None = None,
    y_limits: tuple[int, int] | None = (16, 235),
    show_identity: bool = True,
    cmap: str = "gray_r",
    log_scale: bool = True,
    log_histograms: bool = False,
    show_stems: bool = False,
    marginal_size: str = "15%",
    marginal_pad: float = 0.1,
    xlabel: str = "Input intensity",
    ylabel: str = "Output intensity",
    figsize: tuple[float, float] = (6, 6),
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Axes, Axes]:
    """
    Create a 2D histogram with marginal distributions on top and right.

    Uses make_axes_locatable to ensure marginals match the main axes size
    even with equal aspect ratio.

    Parameters
    ----------
    x_data, y_data
        Data arrays to compare (will be flattened). Expected to contain
        discrete integer values within intensity_range.
    intensity_range
        Range for both axes (typically 0-255).
    x_clip, y_clip
        Clipping thresholds for marginal histogram highlights.
        Defaults to x_limits/y_limits if set, otherwise intensity_range.
    x_limits
        Reference lines for x-axis limits. None to hide.
    y_limits
        Reference lines for y-axis limits (e.g., bt709 16-235). None to hide.
    show_identity
        Show diagonal identity line on scatter.
    cmap
        Colormap for density visualization.
    log_scale
        Use log scale for the 2D histogram counts.
    log_histograms
        Use log scale for the marginal histogram counts.
    show_stems
        If True, draw stem lines in marginal histograms.
    marginal_size
        Size of marginal axes as percentage of main axes.
    marginal_pad
        Padding between main axes and marginals.
    xlabel, ylabel
        Axis labels.
    figsize
        Figure size (only used if ax is None).
    ax
        Axes to plot on. If None, creates new figure and axes.

    Returns
    -------
    fig : Figure
        The figure.
    ax_main : Axes
        The main 2D histogram axes.
    ax_top : Axes
        The top marginal axes.
    ax_right : Axes
        The right marginal axes.
    """
    x_arr = np.asarray(x_data)
    y_arr = np.asarray(y_data)

    lo, hi = intensity_range
    if x_clip is None:
        x_clip = x_limits if x_limits is not None else intensity_range
    if y_clip is None:
        y_clip = y_limits if y_limits is not None else intensity_range

    # Create figure and main axes
    if ax is None:
        fig, ax_main = plt.subplots(figsize=figsize)
    else:
        ax_main = ax
        fig = ax.get_figure()  # type: ignore[assignment]
        assert isinstance(fig, Figure)

    # Plot 2D histogram
    bivariate_intensity_histogram(
        x_arr,
        y_arr,
        ax=ax_main,
        intensity_range=intensity_range,
        output_limits=y_limits,
        input_limits=x_limits,
        show_identity=show_identity,
        cmap=cmap,
        log_scale=log_scale,
    )
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)

    # Create marginals anchored to the main axes
    divider = make_axes_locatable(ax_main)

    # Top marginal (x distribution)
    ax_top = divider.append_axes("top", size=marginal_size, pad=marginal_pad, sharex=ax_main)
    intensity_histogram(
        x_arr,
        clip_vmin=x_clip[0],
        clip_vmax=x_clip[1],
        intensity_range=intensity_range,
        extreme_values=intensity_range,
        log_scale=log_histograms,
        ax=ax_top,
        orientation="horizontal",
        show_xlabel=False,
        show_ylabel=False,
        show_stems=show_stems,
    )
    ax_top.tick_params(labelbottom=False, labelleft=True)
    _apply_tufte_style(ax_top)

    # Right marginal (y distribution)
    ax_right = divider.append_axes("right", size=marginal_size, pad=marginal_pad, sharey=ax_main)
    intensity_histogram(
        y_arr,
        clip_vmin=y_clip[0],
        clip_vmax=y_clip[1],
        intensity_range=intensity_range,
        extreme_values=intensity_range,
        log_scale=log_histograms,
        ax=ax_right,
        orientation="vertical",
        show_xlabel=False,
        show_ylabel=False,
        show_stems=show_stems,
    )
    ax_right.tick_params(labelleft=False, labelbottom=True)
    _apply_tufte_style(ax_right)

    return fig, ax_main, ax_top, ax_right


def luma_comparison_figure(
    input_luma: ArrayLike,
    output_luma: ArrayLike,
    input_srgb: ArrayLike | None = None,
    output_srgb: ArrayLike | None = None,
    intensity_range: tuple[int, int] = (0, 255),
    input_clip: tuple[int, int] | None = None,
    output_clip: tuple[int, int] = (16, 235),
    input_limits: tuple[int, int] | None = None,
    input_column_title: str = "Input",
    output_column_title: str = "Output",
    input_srgb_title: str | None = None,
    output_srgb_title: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (7, 10),
    log_histograms: bool = False,
    show_stems: bool = False,
) -> tuple[Figure, Axes, Axes, Axes, Axes, Axes, Axes, Axes, GridSpec]:
    """
    Create a comprehensive comparison figure for luma compression analysis.

    Layout:
                    Figure Title
        ┌─────────────────┬─────────────────┐
        │   Input         │   Output        │  <- column titles
        ├─────────────────┼─────────────────┤
        │   sRGB image    │   sRGB image    │  <- with optional subtitles
        ├─────────────────┼─────────────────┤
        │   luma w/       │   luma w/       │
        │   highlights    │   highlights    │
        ├─────────────────┼─────────────────┤
        │  [marginal]                       │
        ├─────────────────┬─────────────────┤
        │                 │                 │
        │   bivariate     │ [marginal]      │
        │   histogram     │                 │
        │                 │                 │
        └─────────────────┴─────────────────┘

    Parameters
    ----------
    input_luma, output_luma
        Grayscale luma planes to compare.
    input_srgb, output_srgb
        Optional sRGB images for display (if None, luma shown as grayscale).
    intensity_range
        Display range for all axes (typically 0-255).
    input_clip
        Clipping thresholds for input highlights. Defaults to intensity_range.
    output_clip
        Clipping thresholds for output highlights (16-235 for bt709).
    input_limits
        Reference lines for input range on bivariate plot. Defaults to input_clip.
    input_column_title, output_column_title
        Column titles displayed above each column.
    input_srgb_title, output_srgb_title
        Optional subtitles for the sRGB image panels.
    title
        Optional figure title.
    figsize
        Figure size.
    log_histograms
        Use log scale for histogram counts.
    show_stems
        If True, draw stem lines in histograms. If False, show dots only.

    Returns
    -------
    fig : Figure
    ax_input_srgb, ax_output_srgb : Axes
    ax_input_luma, ax_output_luma : Axes
    ax_bivariate : Axes
    ax_top_marginal, ax_right_marginal : Axes
    gs : GridSpec
    """
    input_luma_arr = np.asarray(input_luma)
    output_luma_arr = np.asarray(output_luma)

    if input_clip is None:
        input_clip = intensity_range
    if input_limits is None:
        input_limits = input_clip

    lo, hi = intensity_range

    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize)

    # Layout ratios
    title_height = 0.08
    img_height = 1.0
    bivariate_height = 1.2

    gs = GridSpec(
        4,
        2,
        figure=fig,
        width_ratios=[1, 1],
        height_ratios=[
            title_height,
            img_height,
            img_height,
            bivariate_height,
        ],
        hspace=0.12,
        wspace=0.08,
    )

    # === Row 0: Column titles ===
    ax_input_title = fig.add_subplot(gs[0, 0])
    ax_output_title = fig.add_subplot(gs[0, 1])

    for ax_title, col_title in [
        (ax_input_title, input_column_title),
        (ax_output_title, output_column_title),
    ]:
        ax_title.text(
            0.5,
            0.5,
            col_title,
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            transform=ax_title.transAxes,
        )
        ax_title.axis("off")

    # === Row 1: sRGB images ===
    ax_input_srgb = fig.add_subplot(gs[1, 0])
    ax_output_srgb = fig.add_subplot(gs[1, 1])

    if input_srgb is not None:
        ax_input_srgb.imshow(np.asarray(input_srgb))
    else:
        ax_input_srgb.imshow(input_luma_arr, cmap="gray", vmin=lo, vmax=hi)
    if input_srgb_title:
        ax_input_srgb.set_title(input_srgb_title, fontsize=9)
    ax_input_srgb.axis("off")

    if output_srgb is not None:
        ax_output_srgb.imshow(np.asarray(output_srgb))
    else:
        ax_output_srgb.imshow(output_luma_arr, cmap="gray", vmin=lo, vmax=hi)
    if output_srgb_title:
        ax_output_srgb.set_title(output_srgb_title, fontsize=9)
    ax_output_srgb.axis("off")

    # === Row 2: Luma with clipping highlights ===
    ax_input_luma = fig.add_subplot(gs[2, 0])
    ax_output_luma = fig.add_subplot(gs[2, 1])

    rgb_input = grayscale_with_clipping_highlights(input_luma_arr, vmin=input_clip[0], vmax=input_clip[1])
    ax_input_luma.imshow(rgb_input)
    ax_input_luma.set_title(f"Luma (clip: {input_clip[0]}–{input_clip[1]})", fontsize=9)
    ax_input_luma.axis("off")

    rgb_output = grayscale_with_clipping_highlights(output_luma_arr, vmin=output_clip[0], vmax=output_clip[1])
    ax_output_luma.imshow(rgb_output)
    ax_output_luma.set_title(f"Luma (clip: {output_clip[0]}–{output_clip[1]})", fontsize=9)
    ax_output_luma.axis("off")

    # === Row 4: Bivariate histogram with marginals ===
    ax_bivariate = fig.add_subplot(gs[3, :])

    _, _, ax_top_marginal, ax_right_marginal = bivariate_with_marginals(
        input_luma_arr,
        output_luma_arr,
        intensity_range=intensity_range,
        x_clip=input_clip,
        y_clip=output_clip,
        x_limits=input_limits,
        y_limits=output_clip,
        log_scale=True,
        log_histograms=log_histograms,
        show_stems=show_stems,
        ax=ax_bivariate,
    )

    # Figure title
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.93)

    return (
        fig,
        ax_input_srgb,
        ax_output_srgb,
        ax_input_luma,
        ax_output_luma,
        ax_bivariate,
        ax_top_marginal,
        ax_right_marginal,
        gs,
    )
