import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.optimize import curve_fit
from sklearn.utils import resample
import scipy.stats as stats
from sklearn.metrics import r2_score
from mpl_toolkits.basemap import Basemap
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# ---------------- CONFIG ----------------
FIGSIZE = (8, 6)
OUTPUT_FIG = "Figure-2.png"

# File paths
PATH_CSV = "Figure-2-data/1982-2019-BA-DU-SD-ED.csv"
PATH_NETCDF = "Figure-2-data/PLS_results_standardized.nc"

# Year labeling for x-axis
YEARS_LABELS = ['1982', '1990', '2000', '2010', '2018']

# ---------------- FUNCTIONS ----------------
def fit_linear_with_bootstrap(x, y, n_bootstrap=1000):
    """Fit linear model + bootstrap CI."""
    def linear(x, a, b): return a * x + b
    popt, _ = curve_fit(linear, x, y)
    y_fit = linear(x, *popt)
    slope, intercept, _, p_value, _ = stats.linregress(x, y)
    r2 = r2_score(y, y_fit)

    preds = np.zeros((n_bootstrap, len(x)))
    for i in range(n_bootstrap):
        xb, yb = resample(x, y)
        try:
            poptb, _ = curve_fit(linear, xb, yb)
            preds[i, :] = linear(x, *poptb)
        except:
            preds[i, :] = np.nan

    lower = np.nanpercentile(preds, 2.5, axis=0)
    upper = np.nanpercentile(preds, 97.5, axis=0)
    return y_fit, lower, upper, slope, p_value, r2

def plot_trend(ax, x, y, ylabel, panel_label, color='black'):
    """Trend line + CI scatter plot."""
    y_fit, lower, upper, slope, pval, r2 = fit_linear_with_bootstrap(x, y)
    ax.scatter(x, y, s=4, color=color)
    ax.plot(x, y_fit, color='red')
    ax.fill_between(x, lower, upper, color=color, alpha=0.1, linewidth=0)

    ax.legend(handles=[
        mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=2, label='Observed'),
        mlines.Line2D([], [], color='red', label=f'Linear Fit (P={pval:.3f})'),
        mpatches.Patch(color=color, alpha=0.1, label='95% CI')],
        fontsize=5, frameon=False, loc='upper left')

    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_xticks([0, 8, 18, 28, 36])
    ax.set_xticklabels(YEARS_LABELS, fontsize=6)
    ax.tick_params(direction='out', length=1, labelsize=6)

    ax.add_artist(AnchoredText(panel_label, loc='lower right',
                               frameon=False, pad=0, prop=dict(fontsize=16)))

def plot_polar_map(ax, data, lons, lats, cmap, vmin, vmax, cbar_label, panel_label):
    """Plot a polar stereographic map."""
    # Ensure shape is (nlat, nlon)
    if data.shape != (len(lats), len(lons)):
        data = data.T

    lon2d, lat2d = np.meshgrid(lons, lats)

    m = Basemap(projection='npstere', boundinglat=60, lon_0=0,
                resolution='l', ax=ax)
    x, y = m(lon2d, lat2d)

    cs = m.pcolormesh(x, y, data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    m.drawcoastlines(linewidth=0.5)

    cbar = m.colorbar(cs, location='bottom', pad="5%")
    cbar.set_label(cbar_label, fontsize=7)
    cbar.ax.tick_params(length=1, labelsize=6, pad=0.2)

    ax.add_artist(AnchoredText(panel_label, loc='upper right',
                               frameon=False, pad=0, prop=dict(fontsize=16)))

# ---------------- MAIN FIGURE ----------------
fig = plt.figure(figsize=FIGSIZE)
gs = gridspec.GridSpec(3, 6, wspace=1.2, hspace=0)

# Load time series data from CSV
df = pd.read_csv(PATH_CSV)

# Panel a — Burned area trend
plot_trend(fig.add_subplot(gs[0, 0:2]),
           np.arange(len(df['BA'])), df['BA'],
           "Burned area (Mha)", "a")

# Panel b — Snow cover duration
plot_trend(fig.add_subplot(gs[0, 2:4]),
           np.arange(len(df['DU-under-fire'])), df['DU-under-fire'],
           "Snow cover duration", "b")

# Panel c — Start day & End day (dual axis)
ax_c = fig.add_subplot(gs[0, 4:6])
plot_trend(ax_c, np.arange(len(df['SD-under-fire'])), df['SD-under-fire'],
           "Start day of snow cover", "c", color='black')

ax_c2 = ax_c.twinx()
y_fit, lower, upper, slope, pval, r2 = fit_linear_with_bootstrap(np.arange(len(df['ED-under-fire'])), df['ED-under-fire'])
ax_c2.scatter(np.arange(len(df['ED-under-fire'])), df['ED-under-fire'], s=4, color='red')
ax_c2.plot(np.arange(len(df['ED-under-fire'])), y_fit, color='red')
ax_c2.fill_between(np.arange(len(df['ED-under-fire'])), lower, upper, color='red', alpha=0.1)
ax_c2.set_ylabel("End day of snow cover", fontsize=7, color='red')
ax_c2.tick_params(axis='y', labelcolor='red', length=1, labelsize=6)

# Panels d & e — Read precomputed coefficients from NetCDF
with Dataset(PATH_NETCDF) as nc:
    coef_sd = nc.variables['Coef_SD'][:]  # shape probably (lon, lat)
    coef_ed = nc.variables['Coef_ED'][:]
    lons = np.arange(-180, 180, 0.25)
    lats = np.arange(60.125, 89.875 + 0.25, 0.25)

# Panel d — SD sensitivity
plot_polar_map(fig.add_subplot(gs[1:3, 1:3]), coef_sd, lons, lats,
               'RdBu_r', -1, 0, "Sensitivity (SD)", "d")

# Panel e — ED sensitivity
plot_polar_map(fig.add_subplot(gs[1:3, 3:5]), coef_ed, lons, lats,
               'RdBu_r', 0, 1, "Sensitivity (ED)", "e")

# Save & show
plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved figure to {OUTPUT_FIG}")