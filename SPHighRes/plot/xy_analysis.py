import pandas as pd
from obspy.geodetics import gps2dist_azimuth
import numpy as np
import matplotlib.pyplot as plt
import math
import string
def merge_catalogs(df_high, df_nlloc, highres_id_col="ev_id", nlloc_id_col="ev_id"):
    """
    Merge both catalogs on event id.
    """
    return df_high.merge(
        df_nlloc, 
        left_on=highres_id_col, 
        right_on=nlloc_id_col, 
        how="left",
        suffixes=("_high", "_nlloc")
    )

def compute_location_differences_latlon(df, 
                                        lat_high="latitude_high", lon_high="longitude_high",
                                        lat_nlloc="latitude_nlloc", lon_nlloc="longitude_nlloc"):
    """
    Compute ΔLat, ΔLon in km and distance using Obspy geodetics.
    """
    # Compute differences in km
    df["dlat_km"] = (df[lat_nlloc] - df[lat_high]) * 111.32  # approx. for reference
    df["dlon_km"] = df.apply(
        lambda r: (r[lon_nlloc] - r[lon_high]) * 111.32 * np.cos(np.deg2rad((r[lat_high] + r[lat_nlloc])/2)),
        axis=1
    )
    
    # Compute geodetic distance using Obspy (meters → km)
    df["dist_km"] = df.apply(
        lambda r: gps2dist_azimuth(r[lat_high], r[lon_high],
                                   r[lat_nlloc], r[lon_nlloc])[0] / 1000.0,
        axis=1
    )
    
    return df

def plot_location_comparison(df, save_path="location_comparison.png"):
    """
    Create figure: (1) dLon_km vs dLat_km and (2) histogram of distance misfit (km).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Axis 1: dx vs dy
    axes[0].scatter(df["dlon_km"], df["dlat_km"],color='blue', alpha=0.5, edgecolor='k')
    axes[0].axhline(0, linewidth=1, color='k')
    axes[0].axvline(0, linewidth=1, color='k')
    axes[0].set_xlabel("ΔLongitude (km)")
    axes[0].set_ylabel("ΔLatitude (km)")
    axes[0].set_title("Location Misfit: HighRes vs NLLoc")

    # Axis 2: Histogram of distance misfit
    axes[1].hist(df["dist_km"], bins=np.arange(0, int(df["dist_km"].max()) + 1, 1),
                 color='blue', alpha=0.5, edgecolor='k')
    axes[1].set_xlabel("Distance Misfit (km)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Histogram of Distance Misfit")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

def compare_highres_vs_nlloc(df_high, df_nlloc,
                             id_high="ev_id", id_nlloc="ev_id",
                             lat_high="latitude", lon_high="longitude",
                             lat_nlloc="latitude", lon_nlloc="longitude",
                             fig_path="comparison.png"):
    """
    Wrapper function to compare HighRes vs NLLoc.
    """
    df = merge_catalogs(df_high, df_nlloc, id_high, id_nlloc)

    # rename lat/lon columns after merge
    df = df.rename(columns={
        f"{lat_high}_high": "latitude_high",
        f"{lon_high}_high": "longitude_high",
        f"{lat_nlloc}_nlloc": "latitude_nlloc",
        f"{lon_nlloc}_nlloc": "longitude_nlloc"
    })

    df = compute_location_differences_latlon(df)
    return plot_location_comparison(df, save_path=fig_path)

def compare_highres_vs_nlloc_by_region(df_high, df_nlloc, df_sp,
                             id_high="ev_id", id_nlloc="ev_id",
                             lat_high="latitude", lon_high="longitude",
                             lat_nlloc="latitude", lon_nlloc="longitude",
                             max_xy=None,
                             max_dist=None,
                             fig_path="comparison.png"):
    """
    Wrapper function to compare HighRes vs NLLoc with fixed axis limits across regions.
    Adds a dashed circle of radius 3 km in top row subplots.
    """
    df = merge_catalogs(df_high, df_nlloc, id_high, id_nlloc)


    # rename lat/lon columns after merge
    df = df.rename(columns={
        f"{lat_high}_high": "latitude_high",
        f"{lon_high}_high": "longitude_high",
        f"{lat_nlloc}_nlloc": "latitude_nlloc",
        f"{lon_nlloc}_nlloc": "longitude_nlloc"
    })


    # print("HEEEEEEEERE")
    station_counts = df_sp[["station"]].value_counts()
    # print("Station counts:", station_counts)


    df = compute_location_differences_latlon(df)
    df = df.merge(df_sp, how="right", on="ev_id", suffixes=("", "_sp"))
    
    station_counts = df[["station"]].value_counts()
    # print("Station counts:", station_counts)

    df = df.dropna(subset=["region"])
    # exit()
    # print(df.info())

    #count rows by station
    station_counts = df[["station"]].value_counts()
    # print("Station counts:", station_counts)

    # Compute global limits
    if max_xy is None:
        max_xy = np.ceil(max(abs(df["dlon_km"]).max(), abs(df["dlat_km"]).max()))

    if max_dist is None:
        max_dist = np.ceil(df["dist_km"].max())

    nbins = np.arange(0, max_dist + 1, 0.5)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))  # slightly taller for balance

    # Generate circle coordinates for radius 3 km
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x = 3 * np.cos(theta)
    circle_y = 3 * np.sin(theta)
    
    rows, cols = 2,axes.shape[1]  # Get the number of rows and columns
    print(rows, cols)
    n = 0
    for row in range(rows):  # Then iterate over rows
        for col in range(cols):  # Iterate over columns first
            box = dict(boxstyle='round', 
                        facecolor='white', 
                        alpha=1)
            text_loc = [0.05, 0.99]
            axes[row][col].annotate(f"({string.ascii_lowercase[n]})",
                            xy=(-0.1, 1.05),  # Outside top-left in axes coordinates
                            xycoords='axes fraction',
                            ha='left',
                            va='bottom',
                            fontsize="large",
                            fontweight="normal"
            )
            n += 1
    

    for i, (region, df_by_r) in enumerate(df.groupby("region")):
        print(f"Region {region}: N={len(df_by_r)} picks")
        # ---------- Row 1: dx vs dy ----------
        ax = axes[0][i]
        ax.scatter(df_by_r["dlon_km"], df_by_r["dlat_km"],
                   color='blue', alpha=0.5, edgecolor='k',
                   label=f"N={len(df_by_r)}")

        # Horizontal & vertical reference lines
        ax.axhline(0, linewidth=1, color='k')
        ax.axvline(0, linewidth=1, color='k')

        # Add dashed 3 km radius circle
        ax.plot(circle_x, circle_y, '--', color='black', linewidth=1)

        ax.set_xlabel("ΔLongitude (km)")
        ax.set_ylabel("ΔLatitude (km)")
        ax.set_title(f"Region {region}")
        ax.set_aspect('equal', adjustable='box')

        ax.legend(loc="upper left")

        # Fixed axis limits
        ax.set_xlim(-max_xy, max_xy)
        ax.set_ylim(-max_xy, max_xy)

        # ---------- Row 2: Histogram ----------
        ax2 = axes[1][i]
        ax2.hist(df_by_r["dist_km"],
                 bins=nbins,
                 color='blue', alpha=0.5, edgecolor='k')
        ax2.set_xlabel("Distance Misfit (km)")
        ax2.set_ylabel("Count")
        ax2.set_xlim(0, max_dist)
        ax2.set_title(f"Region {region}")
        ax2.grid(True, linestyle='--', alpha=0.7)


        # Make hist axes same aspect width as row 1 for consistency
        ax2.set_aspect('auto')

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return fig_path

    # print(df.info())

    

    # df = compute_location_differences_latlon(df)
    # return plot_location_comparison(df, save_path=fig_path)

if __name__ == "__main__":

    cmez_df = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/sp/3_km/sp_data.csv"
    cmez_df = pd.read_csv(cmez_df)
    print(len(cmez_df), "total picks in CMEZ")

    cmez_df = cmez_df[cmez_df["station"].isin(["PB36","PB35",
                                            "PB28","PB37","SA02",
                                            "WB03","PB24"])]
    cmez_df = cmez_df[cmez_df["preferred"]]
    # print(len(cmez_df), "total picks in CMEZ")
    cmez_df = cmez_df.drop_duplicates(subset=["ev_id","station"])
    print(len(cmez_df), "total picks in CMEZ after dropping duplicates")
    station_counts = cmez_df["station"].value_counts()
    print("Station counts:", station_counts)
    # print(cmez_df.columns)
    
    # exit()

    highres_df = pd.read_csv("/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/HighRes/origin.csv")
    nlloc_df = pd.read_csv("/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/Nlloc/origin.csv")

    #drop duplciates
    highres_df = highres_df.drop_duplicates(subset=["ev_id"])
    nlloc_df = nlloc_df.drop_duplicates(subset=["ev_id"])

    # Example usage
    compare_highres_vs_nlloc_by_region(
        df_high=highres_df,
        df_nlloc=nlloc_df,
        df_sp = cmez_df,
        max_xy=3,
        max_dist=4,
        fig_path="/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/SPHighRes/others/HighRes_vs_NLLoc_region.png"
    )