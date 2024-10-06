import matplotlib.pyplot as plt


def get_hist_bins(frame):
    if frame.dtype == "uint8":
        bins = range(2**8 + 1)
    elif frame.dtype == "uint16":
        bins = range(2**10 + 1)
    return bins


def plot_hist(ax, frame):
    bins = get_hist_bins(frame)
    return ax.hist(frame.ravel(), bins=bins)


def plot_frame_and_hist(frame):
    f, axs = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw={"height_ratios": [3, 1]},
        figsize=(10, 8),
    )
    imreturn = axs[0].imshow(frame, cmap="gray")
    histreturn = plot_hist(axs[1], frame)
    return f, axs, imreturn, histreturn
