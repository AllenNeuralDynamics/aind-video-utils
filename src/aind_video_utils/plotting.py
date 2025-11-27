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
