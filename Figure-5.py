import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.offsetbox import AnchoredText
from matplotlib.legend_handler import HandlerTuple

# ==== CONFIG ====
FIGSIZE = (8, 5)
PANEL_FONT = 14
AXIS_FONT = 6
LINE_WIDTH = 0.8
SCENARIOS = {
    "Historical": {"color": "black", "alpha": 0.2},
    "SSP5-8.5": {"color": "red", "alpha": 0.2},
    "SSP3-7.0": {"color": "blue", "alpha": 0.2},
    "SSP2-4.5": {"color": "goldenrod", "alpha": 0.2},
}
TICKS_X = [0, 50, 100, 150]
TICKS_XLABELS = ['1950', '2000', '2050', '2100']

# ==== HELPERS ====
def load_csv_series(path):
    """Load a CSV and return its 'mean', 'min', 'max' if available."""
    df = pd.read_csv(path)
    if {"mean", "min", "max"}.issubset(df.columns):
        return df["mean"], df["min"], df["max"]
    return df["mean"], None, None

def plot_time_series_with_ci(ax, file_paths, ylabel, panel_label):
    """Generic time series plot with shaded CI for multiple scenarios."""
    for label, path in file_paths.items():
        mean, min_, max_ = load_csv_series(path)
        x_hist = np.arange(1950, 2015)
        x_fut = np.arange(2015, 2101)
        x = np.arange(len(x_hist)) + 1 if label == "Historical" else np.arange(len(x_fut)) + 65
        ax.plot(x, mean, color=SCENARIOS[label]["color"], lw=LINE_WIDTH)
        if min_ is not None and max_ is not None:
            ax.fill_between(x, min_, max_, color=SCENARIOS[label]["color"],
                            alpha=SCENARIOS[label]["alpha"], lw=0.0)

    legend_handles = [
        ((mpatches.Patch(facecolor=SCENARIOS[label]["color"], alpha=SCENARIOS[label]["alpha"]),
          mlines.Line2D([], [], color=SCENARIOS[label]["color"], lw=LINE_WIDTH)),
         f"{label} (mean ± 1s.d.)")
        for label in file_paths.keys()
    ]
    ax.legend([h[0] for h in legend_handles], [h[1] for h in legend_handles],
              handler_map={tuple: HandlerTuple(ndivide=None)},
              fontsize=AXIS_FONT, frameon=False,
              loc='upper left' if "Burned" in ylabel else 'upper right')

    ax.set_xticks(TICKS_X)
    ax.set_xticklabels(TICKS_XLABELS, fontsize=AXIS_FONT)
    ax.tick_params(direction='in', pad=1, length=1, labelsize=AXIS_FONT)
    ax.set_ylabel(ylabel, fontsize=AXIS_FONT, labelpad=2.5)
    ax.add_artist(AnchoredText(panel_label, loc='lower right', frameon=False,
                               prop=dict(fontsize=PANEL_FONT)))

def plot_comparison_boxplots(ax, hist, futures, ylabel, panel_label):
    """Box + scatter comparison plots."""
    groups = ["Historical"] + list(futures.keys())
    all_data = [hist] + list(futures.values())
    positions = np.arange(len(groups)) * 2.0
    meanprops = dict(marker='^', markerfacecolor='green', markeredgecolor='green',
                     markersize=3, markeredgewidth=0.5)

    for i, group in enumerate(groups):
        bp = ax.boxplot([all_data[i]], positions=[positions[i]], sym='',
                        showmeans=True, meanprops=meanprops, widths=0.3, whis=[5, 95])
        plt.setp(bp['boxes'], color=SCENARIOS[group]["color"])
        plt.setp(bp['whiskers'], color=SCENARIOS[group]["color"])
        plt.setp(bp['caps'], color=SCENARIOS[group]["color"])
        plt.setp(bp['medians'], color=SCENARIOS[group]["color"])
        jitter = np.random.normal(0, 0.05, size=len(all_data[i]))
        ax.scatter(np.full_like(all_data[i], positions[i]) + jitter, all_data[i],
                   alpha=0.3, color=SCENARIOS[group]["color"], s=2, edgecolors='none')

    for i, group in enumerate(groups[1:]):
        ax.plot([], c=SCENARIOS[group]["color"], label=f"{group} (N={len(all_data[i+1])})")
    ax.legend(frameon=False, fontsize=AXIS_FONT,
              loc="upper left" if "Burned" in ylabel else "lower left")
    ax.set_xticks(positions)
    ax.set_xticklabels(groups, fontsize=AXIS_FONT)
    ax.tick_params(direction='in', pad=1, length=1, labelsize=AXIS_FONT)
    ax.set_ylabel(ylabel, fontsize=AXIS_FONT, labelpad=2.5)
    ax.add_artist(AnchoredText(panel_label, loc='lower right', frameon=False,
                               prop=dict(fontsize=PANEL_FONT)))

# ==== MAIN ====
fig = plt.figure(figsize=FIGSIZE)
gs = plt.GridSpec(2, 3, figure=fig)

# === File path templates ===
BA_PATHS = {
    "Historical": "BA-from-cmip6-10-models-past.csv",
    "SSP5-8.5": "BA-from-cmip6-10-models-ssp585.csv",
    "SSP3-7.0": "BA-from-cmip6-10-models-ssp370.csv",
    "SSP2-4.5": "BA-from-cmip6-10-models-ssp245.csv"
}
SNOW_PATHS = {
    "Historical": "Projection-snow-cover-duration-from-cmip6-10-models-past.csv",
    "SSP5-8.5": "Projection-snow-cover-duration-from-cmip6-10-models-ssp585.csv",
    "SSP3-7.0": "Projection-snow-cover-duration-from-cmip6-10-models-ssp370.csv",
    "SSP2-4.5": "Projection-snow-cover-duration-from-cmip6-10-models-ssp245.csv"
}
BASE_DIR = "Figure-5-data/"

# === Panel a (Burned Area TS) ===
plot_time_series_with_ci(fig.add_subplot(gs[0, 0:2]),
                         {k: BASE_DIR + v for k, v in BA_PATHS.items()},
                         ylabel="Burned area (Mha)", panel_label="a")

# === Panel b (Burned Area Boxplots) ===
past_df, *_ = load_csv_series(BASE_DIR + BA_PATHS["Historical"])
ssp585_df, *_ = load_csv_series(BASE_DIR + BA_PATHS["SSP5-8.5"])
ssp370_df, *_ = load_csv_series(BASE_DIR + BA_PATHS["SSP3-7.0"])
ssp245_df, *_ = load_csv_series(BASE_DIR + BA_PATHS["SSP2-4.5"])
plot_comparison_boxplots(fig.add_subplot(gs[0, 2]), past_df,
                         {"SSP5-8.5": ssp585_df, "SSP3-7.0": ssp370_df, "SSP2-4.5": ssp245_df},
                         ylabel="Burned area (Mha)", panel_label="b")

# === Panel c (Snow Cover TS) ===
plot_time_series_with_ci(fig.add_subplot(gs[1, 0:2]),
                         {k: BASE_DIR + v for k, v in SNOW_PATHS.items()},
                         ylabel="Snow cover duration (days)", panel_label="c")

# === Panel d (Snow Cover Boxplots) ===
past_snow, *_ = load_csv_series(BASE_DIR + SNOW_PATHS["Historical"])
ssp585_snow, *_ = load_csv_series(BASE_DIR + SNOW_PATHS["SSP5-8.5"])
ssp370_snow, *_ = load_csv_series(BASE_DIR + SNOW_PATHS["SSP3-7.0"])
ssp245_snow, *_ = load_csv_series(BASE_DIR + SNOW_PATHS["SSP2-4.5"])
plot_comparison_boxplots(fig.add_subplot(gs[1, 2]), past_snow,
                         {"SSP5-8.5": ssp585_snow, "SSP3-7.0": ssp370_snow, "SSP2-4.5": ssp245_snow},
                         ylabel="Snow cover duration (days)", panel_label="d")

plt.savefig('Figure-5.png',
            dpi=900, bbox_inches="tight")
plt.show()
