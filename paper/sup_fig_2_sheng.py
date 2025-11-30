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

output_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/paper/sup_fig_2_sheng.png"

# --- COLOR MAPS ---
custom_palette = {
    # "PB37": "#1a3be3",
    # "PB28": "#ad16db",
    # "PB35": "#26fafa",
    # "PB36": "#2dfa26",
    # "SA02": "#f1840f",
    # "PB26": "#f1840f",
    # "PB31": "#f1840f",
    # "PB24": "#0ea024",
    # "WB03": "#ffffff",
    "PB16": "red",
    "PB04": "red",
}

r_color = {
    "5": "yellow", "10": "orange",
    "15": "green", "20": "blue",
    "25": "purple", "30": "black"
}

# --- FIGURE ---
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
rows, cols = 1, 2     # layout definition

# --- Add subplot labels (a), (b) ---
box = dict(boxstyle='round', facecolor='white', alpha=1)
for idx in range(cols):
    axes[idx].text(
        0.05, 0.92,
        f"{string.ascii_lowercase[idx]})",
        ha='left', va='top',
        transform=axes[idx].transAxes,
        fontsize="large", fontweight="normal",
        bbox=box
    )

# ---------------------------------------------------------------------
#                    MAIN LOOP — MEMORY-SAFE FILTERING
# ---------------------------------------------------------------------
chunk_size = 1_000_000   # safe for your node

for n, (station, color_station) in enumerate(custom_palette.items()):

    # pick correct HDF
    hdf_file = sheng_file if station in ["PB04", "PB16"] else global_file

    try:
        df_sta = pd.read_hdf(hdf_file, key=f"station_{station}")
    except Exception as e:
        print(f"Station {station} error: {e}. Skipping...")
        continue

    print(len(df_sta), f"total pairs for station {station}")

    # determine subplot (your layout is 1x2 → only column index matters)
    col = n % 2
    ax = axes[col]

    # loop radius categories (5,10,15...)
    for r_str, color in r_color.items():
        R = float(r_str)

        # --------- Memory-safe filtering in chunks ----------
        filtered_chunks = []
        for start in range(0, len(df_sta), chunk_size):
            chunk = df_sta.iloc[start:start + chunk_size]
            mask = (chunk["r_i"] <= R) & (chunk["r_j"] <= R)
            part = chunk[mask]
            if not part.empty:
                filtered_chunks.append(part)

        if not filtered_chunks:
            continue

        df_r = pd.concat(filtered_chunks, ignore_index=True)

        # --------- IQR 10–90% ----------
        Q1 = df_r["v_ij"].quantile(0.10)
        Q3 = df_r["v_ij"].quantile(0.90)
        iqr_data = df_r[(df_r["v_ij"] >= Q1) & (df_r["v_ij"] <= Q3)]

        # special highlight for R ≥ 20
        max_color = color if r_str in ["20", "25", "30"] else None

        # plot histogram
        plot_vij_histogram_station(
            iqr_data, color=color, ax=ax, max=max_color
        )

    # --- station label ---
    ax.text(
        0.05, 0.2, f"{station}",
        ha="left", va="top",
        transform=ax.transAxes,
        fontsize="medium", fontweight="normal",
        bbox=dict(boxstyle="round", facecolor="white", alpha=1)
    )

# Remove y-labels on second subplot
axes[1].set_ylabel("")

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
    ncol=int(len(r_color) / 2),
    loc=(0.38, 0.85),
    frameon=False,
    title="Radius"
)

fig.legend(
    handles=legend_max,
    ncol=1,
    loc=(0.1, 0.85),
    frameon=False
)

plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()