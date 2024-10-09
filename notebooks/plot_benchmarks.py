# %%
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import seaborn as sns

# %matplotlib ipympl
# %%
# Data location
benchmarks_file = "/home/glynch/encoding-tests/benchmarks.csv"

# %%
# Load the data
benchmarks = pd.read_csv(benchmarks_file, sep="\t")

# add a column for codec type (h264 or h265) where it is h264 if the codec is h264_nvenc or libx264
benchmarks["codec_type"] = np.where(
    benchmarks["codec"].str.contains("264"), "h264", "h265"
)
# %% Set the index to the nickname
benchmarks.set_index("nickname", inplace=True)


# %%
# Plotting functions
def tufte_style_lineplot(
    ax, x, y, color="black", dotsize=20, legend=None, **kwargs
):
    ax.plot(x, y, linestyle="-", color=color, linewidth=1, zorder=1, **kwargs)
    ax.scatter(x, y, color="white", s=100, zorder=2)
    ax.scatter(x, y, color=color, s=dotsize, zorder=3)


def tufte_style_scatter(
    ax, x, y, color="black", dotsize=20, legend=None, **kwargs
):
    ax.scatter(x, y, color="white", s=100, zorder=2)
    ax.scatter(x, y, color=color, s=dotsize, zorder=3, **kwargs)


# %%
# Select data where nickname starts with "h264" but is not "h264_nvenc_constrained"
seldata = benchmarks[
    benchmarks.index.str.startswith("h264")
    & ~benchmarks.index.str.endswith("_constrained")
]
ax = sns.scatterplot(
    seldata,
    x="compression_ratio",
    y="vmaf",
    hue="preset",
    style="compute_type",
)
for i, nickname in enumerate(
    ["pipeline_encode", "lili_encode", "twostage_encode"]
):
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
seldata = benchmarks[
    benchmarks.index.str.startswith("h265")
    & ~benchmarks.index.str.endswith("_constrained")
]
sns.relplot(
    seldata,
    x="compression_ratio",
    y="vmaf",
    hue="preset",
    style="compute_type",
)

# %%
# Select data where the nickname contains "nvenc" and does not end with "constrained"
seldata = benchmarks[
    benchmarks.index.str.contains("nvenc")
    & ~benchmarks.index.str.endswith("_constrained")
]
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

ax = sns.scatterplot(
    seldata,
    x="compression_ratio",
    y="vmaf",
    hue="codec_type",
)
for i, nickname in enumerate(
    ["pipeline_encode", "lili_encode", "twostage_encode"]
):
    ax.scatter(
        benchmarks.loc[nickname, "compression_ratio"],
        benchmarks.loc[nickname, "vmaf"],
        marker="x",
        color=f"C{i + 2}",
        label=nickname,
    )
ax.legend()
# %%
x_var = "compression_ratio"
y_var = "fps"
ax = sns.scatterplot(
    seldata,
    x=x_var,
    y=y_var,
    hue="codec_type",
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

seldata = benchmarks[
    (benchmarks["compute_type"] == "GPU")
    & ~benchmarks.index.str.endswith("_constrained")
]
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
network_fn = lambda x: (network_data_transfer_per_hour * 24) / (
    x + network_data_transfer_per_hour
)
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
# %%
x_var = "compression_ratio"
y_var = "vmaf"

gpu_encodes = benchmarks[
    benchmarks.index.str.startswith("h264_nvenc")
    & ~benchmarks.index.str.endswith("_constrained")
]

fast_encodes = benchmarks[benchmarks.index.str.startswith("h264_fast")]
slow_encodes = benchmarks[benchmarks.index.str.startswith("h264_slow")]
f, ax = plt.subplots()
for i, data in enumerate([gpu_encodes, fast_encodes, slow_encodes]):
    tufte_style_lineplot(ax, data[x_var], data[y_var], color=f"C{i}")

for i, nickname in enumerate(
    ["pipeline_encode", "lili_encode", "twostage_encode"]
):

    tufte_style_scatter(
        ax,
        benchmarks.loc[nickname, x_var],
        benchmarks.loc[nickname, y_var],
        marker="x",
        color=f"C{i + 3}",
        label=nickname,
        dotsize=40,
    )

ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(84, 100)
ax.spines["left"].set_bounds(85, 100)
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.spines["bottom"].set_bounds(0, 500)
ax.set_xlabel("Compression ratio")
ax.set_ylabel("Perceptual quality (VMAF)")
ax.set_title("Visual loss and compression")


# %%
x_var = "compression_ratio"
y_var = "fps"
f, ax = plt.subplots()
for i, data in enumerate([gpu_encodes, fast_encodes, slow_encodes]):
    tufte_style_lineplot(ax, data[x_var], data[y_var], color=f"C{i}")

for i, nickname in enumerate(["pipeline_encode", "lili_encode"]):
    tufte_style_scatter(
        ax,
        benchmarks.loc[nickname, x_var],
        benchmarks.loc[nickname, y_var],
        marker="x",
        color=f"C{i + 3}",
        label=nickname,
        dotsize=40,
    )

for i, nickname in enumerate(["twostage_encode", "h265_nvenc_constrained"]):
    tufte_style_scatter(
        ax,
        benchmarks.loc[nickname, x_var],
        benchmarks.loc[nickname, y_var],
        marker="^",
        color=f"C{i + 5}",
        dotsize=40,
    )
ax.spines[["top", "right"]].set_visible(False)
# ax.set_ylim(84, 100)
ax.spines["left"].set_bounds(0, 3000)
ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
ax.spines["bottom"].set_bounds(0, 500)
ax.set_xlabel("Compression ratio")
ax.set_ylabel("Frames per second (fps)")
ax.set_title("Performance and compression")


# %%
