# %%
from matplotlib import pyplot as plt
import matplotlib.ticker as tck
import pandas as pd
import numpy as np
import seaborn as sns

# %matplotlib ipympl
# %%
# Data location
benchmarks_file = "/home/galen.lynch/encode-testing/benchmarks.csv"

# %%
# Load the data
benchmarks = pd.read_csv(benchmarks_file, sep="\t")

# add a column for codec type (h264 or h265) where it is h264 if the codec is h264_nvenc or libx264
benchmarks["codec_type"] = np.where(benchmarks["codec"].str.contains("264"), "h264", "h265")
# %% Set the index to the nickname
benchmarks.set_index("nickname", inplace=True)


# %%
# Plotting functions
def tufte_style_lineplot(ax, x, y, color="black", dotsize=20, label=None, **kwargs):
    ax.plot(x, y, linestyle="-", color=color, linewidth=1, zorder=1, **kwargs)
    ax.scatter(x, y, color="white", s=100, zorder=2)
    ax.scatter(x, y, color=color, s=dotsize, zorder=3, label=label)


def tufte_style_scatter(ax, x, y, color="black", dotsize=20, **kwargs):
    ax.scatter(x, y, color="white", s=100, zorder=2)
    ax.scatter(x, y, color=color, s=dotsize, zorder=3, **kwargs)


def plot_data_convenience_fun(xs, ys, clip_lt=None, clip_gt=None):
    xsperm = np.argsort(xs)
    xs_out = xs[xsperm]
    ys_out = ys[xsperm]
    if clip_lt is not None:
        clipmask = xs_out > clip_lt
        xs_out = xs_out[clipmask]
        ys_out = ys_out[clipmask]
    if clip_gt is not None:
        clipmask = xs_out < clip_gt
        xs_out = xs_out[clipmask]
        ys_out = ys_out[clipmask]

    return xs_out, ys_out


# %%
f, ax = plt.subplots()
seldata = benchmarks[benchmarks.index.str.startswith("h264") & ~benchmarks.index.str.endswith("_constrained")]
x_var = "fps"
y_var = "vmaf"
ax = sns.scatterplot(
    seldata,
    x=x_var,
    y=y_var,
    hue="preset",
    style="compute_type",
    ax=ax,
)
for i, nickname in enumerate(["pipeline_encode", "lili_encode", "twostage_encode"]):
    ax.scatter(
        benchmarks.loc[nickname, x_var],
        benchmarks.loc[nickname, y_var],
        marker="o",
        color=f"C{i + 3}",
        label=nickname,
    )
ax.legend()

# %%
# Select data where nickname starts with "h264" but is not "h264_nvenc_constrained"
f, ax = plt.subplots()
seldata = benchmarks[benchmarks.index.str.startswith("h264") & ~benchmarks.index.str.endswith("_constrained")]
ax = sns.scatterplot(
    seldata,
    x="compression_ratio",
    y="vmaf",
    hue="preset",
    style="compute_type",
    ax=ax,
)
for i, nickname in enumerate(["pipeline_encode", "lili_encode", "twostage_encode"]):
    ax.scatter(
        benchmarks.loc[nickname, "compression_ratio"],
        benchmarks.loc[nickname, "vmaf"],
        marker="o",
        color=f"C{i + 3}",
        label=nickname,
    )
ax.legend()

# %%
# Select data where nickname starts with "h265" but is not "h265_nvenc_constrained"
seldata = benchmarks[benchmarks.index.str.startswith("h265") & ~benchmarks.index.str.endswith("_constrained")]
sns.relplot(
    seldata,
    x="compression_ratio",
    y="vmaf",
    hue="preset",
    style="compute_type",
)

# %%
seldata = benchmarks[benchmarks.index.str.startswith("h264_slow") | benchmarks.index.str.startswith("h265_slow")]
sns.relplot(
    seldata,
    x="compression_ratio",
    y="vmaf",
    hue="codec_type",
)
# %%
# Select data where the nickname contains "nvenc" and does not end with "constrained"
seldata = benchmarks[benchmarks.index.str.contains("nvenc") & ~benchmarks.index.str.endswith("_constrained")]
sns.relplot(
    seldata,
    x="compression_ratio",
    y="vmaf",
    hue="codec_type",
)
sns.relplot(
    seldata,
    x="vmaf",
    y="fps",
    hue="codec_type",
)

# %%
f, ax = plt.subplots()
ax = sns.scatterplot(seldata, x="compression_ratio", y="vmaf", hue="codec_type", ax=ax)
for i, nickname in enumerate(["pipeline_encode", "lili_encode", "twostage_encode"]):
    ax.scatter(
        benchmarks.loc[nickname, "compression_ratio"],
        benchmarks.loc[nickname, "vmaf"],
        marker="x",
        color=f"C{i + 2}",
        label=nickname,
    )
ax.legend()
# %%
f, ax = plt.subplots()
x_var = "compression_ratio"
y_var = "fps"
ax = sns.scatterplot(
    seldata,
    x=x_var,
    y=y_var,
    hue="codec_type",
    ax=ax,
)
ax.scatter(
    benchmarks.loc["pipeline_encode", x_var],
    benchmarks.loc["pipeline_encode", y_var],
    marker="x",
    color="red",
    label="pipeline_encode",
)
ax.scatter(
    benchmarks.loc["lili_encode", x_var],
    benchmarks.loc["lili_encode", y_var],
    marker="x",
    color="blue",
    label="lili_encode",
)
ax.legend()
# %%

seldata = benchmarks[(benchmarks["compute_type"] == "GPU") & ~benchmarks.index.str.endswith("_constrained")]
sns.relplot(seldata, x="compression_ratio", y="vmaf", hue="codec")
# %%
w = 540
h = 720
fps = 500
bpp = 24
data_rate = w * h * fps * bpp / 8
bytes_per_hour = data_rate * 3600

max_write_speed = 2 * 1700 * 1e9  # two 1700 MB/s drives
max_storage = 2 * 4e12  # two 4 TB bytes
network_data_transfer_rate = 1e9 / 8  # 1 Gb/s
network_data_transfer_per_hour = network_data_transfer_rate * 3600

cameras_per_drive = max_write_speed / (data_rate)
network_transfer_ratio = data_rate / network_data_transfer_rate
storage_hours = max_storage / data_rate / 3600 / 2

storage_fn = lambda x: max_storage / x
network_fn = lambda x: (network_data_transfer_per_hour * 24) / (x + network_data_transfer_per_hour)
xs = np.linspace(0.01, 10, 100)
storage_limit = [storage_fn(x * 1e12) for x in xs]
network_limit = [network_fn(x * 1e12) for x in xs]
# %%
f, ax = plt.subplots()
# ax.axvline(2 * max_write_speed * 3600 / 1e12, color="red", label="Max Write Speed")

ax.scatter(2 * bytes_per_hour / 1e12, 3, color="C3", label="Two cameras")
ax.fill_between(xs, storage_limit, color="C0", label="Storage", alpha=0.5)
ax.fill_between(xs, network_limit, color="C4", label="Network", alpha=0.5)
ax.set_xlabel("Total data rate (TB/hour)")
ax.set_ylabel("Hours of recording")
ax.set_ylim(0, 10)
ax.set_xlim(0, 10)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
f.savefig("/home/galen.lynch/encode-testing/storage_network_tradeoff.png", dpi=300)
# %%
x_var = "compression_ratio"
y_var = "vmaf"

gpu_encodes = benchmarks[benchmarks.index.str.startswith("h264_nvenc") & ~benchmarks.index.str.endswith("_constrained")]

fast_encodes = benchmarks[benchmarks.index.str.startswith("h264_fast")]
slow_encodes = benchmarks[benchmarks.index.str.startswith("h264_slow")]
f, ax = plt.subplots()
labels = ["GPU", "CPU Fast", "CPU Slow"]
for i, data in enumerate([gpu_encodes, fast_encodes, slow_encodes]):
    xs, ys = plot_data_convenience_fun(data[x_var].to_numpy(), data[y_var].to_numpy(), clip_gt=800)
    tufte_style_lineplot(ax, xs, ys, color=f"C{i}", label=labels[i])

labels = ["Dynamic foraging", "Lili's"]
for i, nickname in enumerate(["pipeline_encode", "lili_encode"]):
    tufte_style_scatter(
        ax,
        benchmarks.loc[nickname, x_var],
        benchmarks.loc[nickname, y_var],
        marker="x",
        color=f"C{i + 3}",
        label=labels[i],
        dotsize=40,
    )

first_stage = "h264_nvenc_cq_12"
second_stage = "twostage_encode"
tufte_style_scatter(
    ax,
    benchmarks.loc[first_stage, x_var],
    benchmarks.loc[first_stage, y_var],
    marker="^",
    color=f"C5",
    dotsize=40,
    label="First stage",
)
tufte_style_scatter(
    ax,
    benchmarks.loc[second_stage, x_var],
    benchmarks.loc[second_stage, y_var],
    marker="D",
    color=f"C7",
    dotsize=40,
    label="1st + 2nd",
)

ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(81, 100)
# ax.yaxis.set_major_locator(tck.MultipleLocator(5))
ax.set_yticks([81, 85, 90, 95, 100])
ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
ax.spines["bottom"].set_bounds(0, 600)
ax.set_xlabel("Compression ratio")
ax.set_ylabel("Perceptual quality (VMAF)")
ax.set_title("Visual loss and compression")
ax.legend()
f.savefig("/home/galen.lynch/encode-testing/vmaf_vs_compression.png", dpi=300)

# %%
x_var = "compression_ratio"
y_var = "fps"
f, ax = plt.subplots()
labels = ["GPU", "CPU Fast", "CPU Slow"]
for i, data in enumerate([gpu_encodes, fast_encodes, slow_encodes]):
    xs, ys = plot_data_convenience_fun(data[x_var].to_numpy(), data[y_var].to_numpy(), clip_gt=800)
    tufte_style_lineplot(ax, xs, ys, color=f"C{i}", label=labels[i])

labels = ["Dynamic foraging", "Lili's"]
for i, nickname in enumerate(["pipeline_encode", "lili_encode"]):
    tufte_style_scatter(
        ax,
        benchmarks.loc[nickname, x_var],
        benchmarks.loc[nickname, y_var],
        marker="x",
        color=f"C{i + 3}",
        label=labels[i],
        dotsize=40,
    )
labels = ["First stage", "Second stage"]
first_stage = "h264_nvenc_cq_12"
for i, nickname in enumerate([first_stage, "twostage_encode"]):
    tufte_style_scatter(
        ax,
        benchmarks.loc[nickname, x_var],
        benchmarks.loc[nickname, y_var],
        marker="^",
        color=f"C{i + 5}",
        dotsize=40,
        label=labels[i],
    )
tufte_style_scatter(
    ax,
    benchmarks.loc["twostage_encode", x_var],
    benchmarks.loc[first_stage, y_var],
    marker="D",
    color="C7",
    dotsize=40,
    label="1st + 2nd",
)
ax.spines[["top", "right"]].set_visible(False)
# ax.set_ylim(84, 100)
ax.spines["left"].set_bounds(0, 3000)
ax.yaxis.set_major_locator(tck.MultipleLocator(500))
ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
ax.spines["bottom"].set_bounds(0, 600)
ax.set_xlabel("Compression ratio")
ax.set_ylabel("Frames per second")
ax.set_title("Online throughput and compression")
ax.legend()
f.savefig("/home/galen.lynch/encode-testing/fps_vs_compression.png", dpi=300)


# %%
