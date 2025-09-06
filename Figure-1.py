import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import gridspec
from matplotlib.offsetbox import AnchoredText
from cartopy.util import add_cyclic_point
from netCDF4 import Dataset as NetCDFFile

np.bool = np.bool_  # fix deprecated warning

# --------------------------
# SETTINGS
# --------------------------
FIGSIZE = (7, 7)
FONTSIZE_LABEL = 8
FONTSIZE_LEGEND = 6
FONTSIZE_PANEL = 18

# File paths
PATH_SNOW = 'Figure-1-data/Snow-cover-duration-60-pentad-yearmean.nc'
PATH_FIRE = 'Figure-1-data/FireCCILT11-1982-2019-BA-yearmean.nc'
###  multiple annual BA data is processed from Google Earth Engine
PATH_DATA_CSV = 'Figure-1-data/1982-2019-muti-BA-and-DU.csv'
OUTPUT_FIG = 'Figure-1.png'

# --------------------------
# HELPER FUNCTIONS
# --------------------------
def plot_polar_map(ax, data, lons, lats, levels, cmap, colorbar_label, panel_label, extend="both"):
    data, lons = add_cyclic_point(data, coord=lons)
    lon2d, lat2d = np.meshgrid(lons, lats)

    m = Basemap(projection='npstere', boundinglat=60, lon_0=0,
                resolution='l', round=True, ax=ax)
    x, y = m(lon2d, lat2d)
    cs = m.contourf(x, y, data, levels=levels, cmap=cmap, extend=extend)

    m.plot(*m(np.linspace(-180, 180), np.linspace(66.5, 66.5)),
           linewidth=1, color='black', linestyle='--')
    m.drawcoastlines(linewidth=0.5)
    m.drawmeridians(np.arange(0, 360, 90), linewidth=0.2,
                    fontsize=FONTSIZE_LABEL, labels=[1, 1, 1, 0], color='grey')

    cb = plt.colorbar(cs, ax=ax, orientation="horizontal", pad=0.05)
    cb.ax.tick_params(labelsize=6, pad=1, length=0.1)
    cb.set_label(colorbar_label, fontsize=FONTSIZE_LABEL, labelpad=2, y=0.45)

    ax.add_artist(AnchoredText(panel_label, loc='lower right',
                               borderpad=0., frameon=False, pad=0, prop=dict(fontsize=FONTSIZE_PANEL)))

def plot_bar(ax, data_df, ylabel, ylim=None, legend=False, colors=None, panel_label=None):
    data_df.plot.bar(ax=ax, color=colors, rot=0)

    if not legend:
        ax.legend().set_visible(False)
    else:
        ax.legend(frameon=False, fontsize=FONTSIZE_LEGEND, loc="upper left", ncol=3)

    ax.tick_params(direction='in', pad=1, length=1)
    plt.xticks(fontsize=6, rotation=45)
    plt.yticks(fontsize=6)

    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL, labelpad=2)
    if ylim:
        ax.set_ylim(ylim)

    if panel_label:
        ax.add_artist(AnchoredText(panel_label, loc='upper right',
                                   borderpad=0., frameon=False, pad=0,
                                   prop=dict(fontsize=FONTSIZE_PANEL)))

# --------------------------
# MAIN FIGURE
# --------------------------
fig = plt.figure(figsize=FIGSIZE)
gs = gridspec.GridSpec(4, 2, wspace=0.1, hspace=0.2)

# Panel a: Snow cover duration map
nc = NetCDFFile(PATH_SNOW)
lons = nc.variables['lon'][:]
lats = nc.variables['lat'][:]
snow_duration = nc.variables['DU'][:]
plot_polar_map(plt.subplot(gs[0:2, 0]), snow_duration[0, :, :], lons, lats,
               levels=np.arange(100, 250, 10), cmap="rainbow_r",
               colorbar_label="Snow cover duration (days)", panel_label="a")

# Panel b: Burned area map
nc = NetCDFFile(PATH_FIRE)
lons = nc.variables['lon'][:]
lats = nc.variables['lat'][:]
burned_area = nc.variables['burned_area'][:] / 1_000_000
burned_area[burned_area == 0] = np.nan
plot_polar_map(plt.subplot(gs[0:2, 1]), burned_area[0, :, :], lons, lats,
               levels=np.arange(0, 3, 0.1), cmap="OrRd",
               colorbar_label=r"Burned area (Mha)", panel_label="b", extend="max")

# Read CSV data for Panels c & d
df_csv = pd.read_csv(PATH_DATA_CSV)

# Panel c: Snow duration time series
df_snow = df_csv[['Year', 'DU-all']].copy()
df_snow = df_snow.set_index('Year')
plot_bar(plt.subplot(gs[2, :]), df_snow,
         ylabel="Snow cover duration (day)",
         ylim=(190, 240), legend=False, panel_label="c")

# Panel d: Burned area time series
df_fire = df_csv[['Year', 'FireCCI51', 'MCD64A1', 'FireCCILT11']].set_index('Year')
plot_bar(plt.subplot(gs[3, :]), df_fire,
         ylabel="Burned area (Mha)",
         legend=True,
         colors={"FireCCI51": "green", "MCD64A1": "red", "FireCCILT11": "black"},
         panel_label="d")

# Save & show
plt.savefig(OUTPUT_FIG, dpi=900, bbox_inches="tight")
plt.show()
