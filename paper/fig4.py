import sys
import os

cmez_repository_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(cmez_repository_path)


import pandas as pd
import os
import glob
import re
import string
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from SPHighRes.core.database.database import load_from_sqlite
from SPHighRes.plot.sp import prepare_sp_analysis, plot_times_by_station
from SPHighRes.plot.vpvs import plot_vij_histogram_station

data_path = os.path.join(cmez_repository_path, "data")

# Paths for Figure 1
catalog_path = os.path.join(data_path,"ts_tp","catalog_cmez.db")
picks_path = os.path.join(data_path,"ts_tp","picks_cmez.db")
stations_path = os.path.join(data_path,"stations","delaware_onlystations_160824.csv")
# Paths for Figure 2
vpvs_path = os.path.join(data_path,"vpvs")
# Output path for combined figure
output_path = os.path.join(os.path.dirname(__file__),"fig4.png")

# Custom palette for Figure 1
custom_palette = {
    "PB35": "#26fafa",
    "PB36": "#2dfa26",
    "PB28": "#ad16db",
    "PB37": "#1a3be3",
    "WB03": "#ffffff",
    "SA02": "#f1840f",
    "PB24": "#0ea024",
}

# Load data for Figure 1
picks = load_from_sqlite(picks_path)
catalog = load_from_sqlite(catalog_path)
stations = pd.read_csv(stations_path)
stations = stations[["network", "station", "latitude", "longitude", "elevation"]]
stations = stations[stations["station"].isin(list(custom_palette.keys()))]

stations_with_picks = list(set(picks["station"].to_list()))
order = stations[stations["station"].isin(stations_with_picks)]
order = order.sort_values("longitude", ignore_index=True, ascending=True)
order = order.drop_duplicates(subset="station")
order = order["station"].to_list()

# Prepare picks for Figure 1
catalog, picks = prepare_sp_analysis(catalog, picks, cat_columns_level=0)
picks["ts-tp"] = picks["tt_S"] - picks["tt_P"]
picks["ts-tp"] = picks["ts-tp"].astype(float)

# Create main figure with two subplots (1 for Figure 1, 1 for Figure 2 right column)
# Create main figure with GridSpec
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1], wspace=0.1)
ax1 = fig.add_subplot(gs[0, :]) 

text_loc = [0.02, 0.92]
box = dict(boxstyle='round', 
                    facecolor='white', 
                    alpha=1)

# --- Plot Figure 1 (Top, Spanning All Columns) ---
plot_times_by_station(picks, order=order, palette=custom_palette, 
                      ylim=(0, 2), show=False, ax=ax1)
# ax1.text(text_loc[0], text_loc[1], 
#                 f"{string.ascii_lowercase[0]})", 
#                 horizontalalignment='left', 
#                 verticalalignment="top", 
#                 transform=ax1.transAxes, 
#                 fontsize="large", 
#                 fontweight="normal",
#                 bbox=box)

ax1.annotate(f"({string.ascii_lowercase[0]})",
                xy=(-0.1, 1.05),  # Slightly outside top-left
                xycoords='axes fraction',
                ha='left',
                va='bottom',
                fontsize="large",
                fontweight="normal",
                # bbox=box
                )

# Arrow annotation for Figure 1
fig.text(0.15, 0.4, "W", ha="center", va="center", fontsize=12, fontweight="bold")
fig.text(0.87, 0.4, "E", ha="center", va="center", fontsize=12, fontweight="bold")
arrow = mpatches.FancyArrow(0.15, 0.37, 0.7, 0, width=0.001, transform=fig.transFigure, color="black")
fig.patches.append(arrow)



# # --- Load and Plot Figure 2 (Three Columns Below) ---
r_color = {"5": "yellow", "10": "orange", "15": "green", "20": "blue", "25": "purple", "30": "black"}
custom_palette_fig2 = {"PB28": "#ad16db", "SA02": "#f1840f", "WB03": "#ffffff"}

# Create shared y-axis
ax_shared = None
y_label = True
axes_list = []
for n, (key, val) in enumerate(custom_palette_fig2.items()):
    query = glob.glob(os.path.join(vpvs_path,  f"{key}*.csv"))
    sorted_files = sorted(query, key=lambda x: int(re.search(r"_(\d+)\.csv$", x).group(1)))

    if ax_shared is None:
        ax = fig.add_subplot(gs[1, n])  # First subplot
        ax_shared = ax  # Save this for sharing y-axis
    else:
        ax = fig.add_subplot(gs[1, n], 
                             sharey=ax_shared)  # Share y-axis with first plot
        y_label = False

    

    # ax = fig.add_subplot(gs[1, n])  # Place in correct column
    for path in sorted_files:
        basename = os.path.basename(path).split(".")[0]
        station, r = basename.split("_")

        if r not in r_color.keys():
            continue

        data = pd.read_csv(path)
        Q1 = data["v_ij"].quantile(0.10)
        Q3 = data["v_ij"].quantile(0.90)
        iqr_data = data[(data["v_ij"] >= Q1) & (data["v_ij"] <= Q3)]

        max_color = r_color[r] if r in ["20", "25", "30"] else None

        
        plot_vij_histogram_station(iqr_data, color=r_color[r], 
                                   ax=ax, max=max_color,
                                   y_label=y_label)
        text_loc = [0.05, 0.2]
        ax.text(
            text_loc[0],
            text_loc[1],
            f"{station}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            fontsize="medium",
            fontweight="normal",
            bbox=dict(boxstyle="round", facecolor="white", alpha=1),
        )
        axes_list.append(ax)
    # print(y_label)
    if not y_label:
        ax.tick_params(labelleft=False)  # Hide labels but keep grid

    text_loc = [0.05, 0.92]
    # ax.text(text_loc[0], text_loc[1], 
    #             f"{string.ascii_lowercase[n+1]})", 
    #             horizontalalignment='left', 
    #             verticalalignment="top", 
    #             transform=ax.transAxes, 
    #             fontsize="large", 
    #             fontweight="normal",
    #             bbox=box)
    ax.annotate(f"({string.ascii_lowercase[n+1]})",
                xy=(-0.1, 1.05),  # Slightly outside top-left
                xycoords='axes fraction',
                ha='left',
                va='bottom',
                fontsize="large",
                fontweight="normal",
                # bbox=box
                )

# Add legends
legend_elements = [Line2D([0], [0], color=color, lw=2, label=f"{key} km") for key, color in r_color.items()]
legend_max = [Line2D([0], [0], color="black", lw=2, linestyle="--", label="Max. Value")]

fig.legend(handles=legend_elements,
           loc="lower left", 
           ncol=6, frameon=True, 
           title="Radius", 
           bbox_to_anchor=(0.1, -0.04))
fig.legend(handles=legend_max,
           loc="lower center", 
           frameon=True, 
           bbox_to_anchor=(0.8, -0.04))

# Adjust layout and save
plt.subplots_adjust(hspace=0.55)
plt.savefig(output_path, dpi=300, 
            bbox_inches="tight")
plt.show()