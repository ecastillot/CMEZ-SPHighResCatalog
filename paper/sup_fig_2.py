import matplotlib.pyplot as plt
import os
import string
import numpy as np
import pandas as pd
from SPHighRes.plot.vpvs import plot_vij_histogram_station
from matplotlib.lines import Line2D

# --- NEW GLOBAL FILE CONTAINING ALL STATIONS ---
global_file = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/vpvs/vpvs_all_stations.h5"
sheng_file = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/vpvs_sheng/vpvs_all_stations.h5"
# df = pd.read_csv(global_file)

# df_sheng = pd.read_csv(sheng_file)
# df = pd.concat([df, df_sheng], ignore_index=True)

output_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/paper/sup_fig_2.png"

# --- COLOR MAPS ---
custom_palette = {
    "PB37": "#1a3be3",
    "PB28": "#ad16db",
    "PB35": "#26fafa",
    "PB36": "#2dfa26",
    "SA02": "#f1840f",
    "PB26": "#f1840f",
    "PB31": "#f1840f",
    "PB24": "#0ea024",
    "WB03": "#ffffff",
    # "PB16": "red",
    # "PB04": "red",
}

r_color = {
    "5": "yellow", "10": "orange",
    "15": "green", "20": "blue",
    "25": "purple", "30": "black"
}

# --- FIGURE ---
fig, axes = plt.subplots(3, 3, figsize=(10, 6), sharey=True)
rows, cols = axes.shape

# --- Add (a), (b), (c)... labels to subplots ---
n = 0
for col in range(cols):
    for row in range(rows):
        box = dict(boxstyle='round', facecolor='white', alpha=1)
        axes[row, col].text(
            0.05, 0.92,
            f"{string.ascii_lowercase[n]})",
            ha='left', va='top',
            transform=axes[row, col].transAxes,
            fontsize="large", fontweight="normal",
            bbox=box
        )
        n += 1

# ---------------------------------------------------------------------
#                    MAIN LOOP — NEW DATA STRUCTURE
# ---------------------------------------------------------------------
for n, (station, color_station) in enumerate(custom_palette.items()):

    # dataframe ONLY for this station
    if station in ["PB04", "PB16"]:
        hdf_file = sheng_file
    else:
        # continue
        hdf_file = global_file
    
    try:
        df_sta = pd.read_hdf(hdf_file, key=f"station_{station}")
    except Exception as e:
        print(f"Station {station} error: {e}. Skipping...")
        continue

    print(len(df_sta), f"total pairs for station {station}")

    # exit()

    # get subplot location
    row, col = divmod(n, 3)
    ax = axes[col][row]

    # loop radius categories (5,10,15...)
    for r_str, color in r_color.items():
        R = float(r_str)


        # replicate the old behavior:
        # station radius R = r_i < R AND r_j < R
        df_r = df_sta[(df_sta["r_i"] <= R) & (df_sta["r_j"] <= R)]
        if df_r.empty:
            continue

        # IQR filter 10–90%
        Q1 = df_r["v_ij"].quantile(0.10)
        Q3 = df_r["v_ij"].quantile(0.90)
        iqr_data = df_r[(df_r["v_ij"] >= Q1) & (df_r["v_ij"] <= Q3)]

        # use special "max" highlight for high radii
        max_color = color if r_str in ["20", "25", "30"] else None

        # plot histogram
        plot_vij_histogram_station(
            iqr_data, color=color, ax=ax, max=max_color
        )

    # --- add station text ---
    ax.text(
        0.05, 0.2, f"{station}",
        ha="left", va="top",
        transform=ax.transAxes,
        fontsize="medium", fontweight="normal",
        bbox=dict(boxstyle="round", facecolor="white", alpha=1)
    )


# remove y-labels on other columns
for ncol in range(cols):
    for nrow in range(rows):
        if ncol != 0:
            axes[nrow, ncol].set_ylabel("")

fig.subplots_adjust(wspace=0.15)

# --- LEGENDS ---
legend_elements = [
    Line2D([0], [0], color=color, lw=2, label=f"{key} km")
    for key, color in r_color.items()
]

legend_max = [
    Line2D([0], [0], color="black", lw=2, linestyle="--", label="Max. Value")
]

fig.legend(
    handles=legend_elements,
    ncol=int(len(r_color)/2),
    loc=(0.53, 1 - 0.12),
    frameon=False,
    title="Radius"
)

fig.legend(
    handles=legend_max,
    ncol=1,
    loc=(0.25, 1 - 0.10),
    frameon=False
)

plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
