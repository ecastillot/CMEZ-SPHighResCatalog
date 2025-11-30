import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def extract_station_radius(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts station and radius from the 'key' column.
    Example key: /sta_WB03/vpvs_5km/ev_xxx
    """
    pattern = r"/sta_(\w+)/vpvs_(\d+)km"
    df[["station", "radius"]] = df["key"].str.extract(pattern)
    df["radius"] = df["radius"].astype(int)
    return df

def plot_median_std_combined_by_station(
    df: pd.DataFrame,
    save_folder: str,
    dpi: int = 300,
    depth_range=(0, 20, 0.5),     # (min, max, bin_size)
    vp_range=(5, 6, 0.05),       # (min, max, bin_size)
    vpvs_range=(1.5, 1.8, 0.01) # (min, max, bin_size)
):
    """
    Creates two figures per station:
        1) median_<station>.png — 3 subplots: Z, Vp, Vp/Vs (medians)
        2) std_<station>.png    — 3 subplots: Z, Vp, Vp/Vs (std)

    Uses fixed bin ranges across radii to ensure consistency.
    Bin ranges can be changed via function parameters.
    """

    os.makedirs(save_folder, exist_ok=True)

    df = df[df["radius"] != 1]

    # Define bins based on requested ranges
    depth_bins = np.arange(depth_range[0], depth_range[1] + depth_range[2], depth_range[2])
    vp_bins = np.arange(vp_range[0], vp_range[1] + vp_range[2], vp_range[2])
    vpvs_bins = np.arange(vpvs_range[0], vpvs_range[1] + vpvs_range[2], vpvs_range[2])

    params = {
        "Depth (km)": ("median_depth", "std_depth", depth_bins),
        "Vp (km/s)": ("median_vp", "std_vp", vp_bins),
        "Vp/Vs Ratio": ("median_vpvs", "std_vpvs", vpvs_bins),
    }

    for station, df_sta in df.groupby("station"):

        # ---------- FIGURE 1: MEDIANS ----------
        fig_med, axes_med = plt.subplots(3, 1, figsize=(7, 9))
        
        for ax, (xlabel, (median_col, _, bins)) in zip(axes_med, params.items()):
            for radius, df_rad in df_sta.groupby("radius"):
                ax.hist(df_rad[median_col], bins=bins, histtype="step", linewidth=1,
                        label=f"{radius} km")
            ax.set_title(f"Median {xlabel} — Station {station}")
            ax.set_ylabel("Count")
            ax.legend(title="Radius")

        plt.tight_layout()
        median_path = os.path.join(save_folder, f"median_{station}.png")
        fig_med.savefig(median_path, dpi=dpi)
        plt.close(fig_med)
        print(f"Saved: {median_path}")

        # ---------- FIGURE 2: STDs ----------
        fig_std, axes_std = plt.subplots(3, 1, figsize=(7, 9))

        for ax, (xlabel, (_, std_col, bins)) in zip(axes_std, params.items()):
            # df_sta[std_col] = df_sta[std_col].clip(lower=0)  # Ensure no negative std values
            # df_sta["sigma"] = df_sta[median_col] - df_sta[std_col]
            for radius, df_rad in df_sta.groupby("radius"):
                ax.hist(df_rad[std_col], histtype="step", linewidth=1.8,
                        label=f"{radius} km")
            ax.set_title(f"STD {xlabel} — Station {station}")
            ax.set_ylabel("Count")
            ax.legend(title="Radius")

        plt.tight_layout()
        std_path = os.path.join(save_folder, f"std_{station}.png")
        fig_std.savefig(std_path, dpi=dpi)
        plt.close(fig_std)
        print(f"Saved: {std_path}")

def plot_median_std_by_station(df: pd.DataFrame, save_folder: str,
                                dpi: int = 300):
    """
    For each station, creates ONE FIGURE PER PARAMETER with:
      - Ax1: histogram of median values (line only, grouped by radius)
      - Ax2: histogram of std values  (line only, grouped by radius)

    Saves files as z_<station>.png, vp_<station>.png, vpvs_<station>.png
    """
    os.makedirs(save_folder, exist_ok=True)

    params = {
        "z": ("median_depth", "std_depth", "Depth (km)"),
        "vp": ("median_vp", "std_vp", "Vp (km/s)"),
        "vpvs": ("median_vpvs", "std_vpvs", "Vp/Vs ratio"),
    }

    df = df[df["radius"] != 1]

    for station, df_sta in df.groupby("station"):
        for prefix, (median_col, std_col, xlabel) in params.items():

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=False)
            
            # ---- Top: median/median histogram ----
            for radius, df_rad in df_sta.groupby("radius"):
                ax1.hist(
                    df_rad[median_col], bins=15, histtype="step", linewidth=1.8,
                    label=f"{radius} km"
                )
            ax1.set_title(f"{xlabel} Median Distribution — Station {station}")
            ax1.set_ylabel("Count")
            ax1.legend(title="Radius")

            # ---- Bottom: std histogram ----
            for radius, df_rad in df_sta.groupby("radius"):
                ax2.hist(
                    df_rad[std_col], bins=15, histtype="step", linewidth=1.8,
                    label=f"{radius} km"
                )
            ax2.set_title(f"{xlabel} STD Distribution — Station {station}")
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel("Count")

            plt.tight_layout()

            save_path = os.path.join(save_folder, f"{prefix}_{station}.png")
            fig.savefig(save_path, dpi=dpi)
            plt.close(fig)

            print(f"Saved: {save_path}")

if __name__ == "__main__":
    # ---------- USAGE EXAMPLE ----------

    df = pd.read_csv("/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/z/summary.csv")  # Load your CSV
    df = extract_station_radius(df)
    out = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/SPHighRes/plot/bts_analysis3"
    plot_median_std_combined_by_station(df, out)
    # plot_median_std_by_station(df, out)
