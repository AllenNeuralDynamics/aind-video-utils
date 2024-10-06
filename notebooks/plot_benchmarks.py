# %%
from matplotlib import pyplot as plt
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
benchmarks["codec_type"] = np.where(
    benchmarks["codec"].str.contains("264"), "h264", "h265"
)
# %%
# Select data where nickname starts with "h264" but is not "h264_nvenc_constrained"
seldata = benchmarks[
    benchmarks["nickname"].str.startswith("h264")
    & ~benchmarks["nickname"].str.endswith("_constrained")
]
sns.relplot(
    seldata,
    x="compression_ratio",
    y="vmaf",
    hue="preset",
    style="compute_type",
)
# %%

seldata = benchmarks[
    (benchmarks["compute_type"] == "GPU")
    & ~benchmarks["nickname"].str.endswith("_constrained")
]
sns.relplot(seldata, x="compression_ratio", y="vmaf", hue="codec")
# %%
