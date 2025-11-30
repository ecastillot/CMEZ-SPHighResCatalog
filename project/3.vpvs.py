from SPHighRes.core.event.stations import Stations
from SPHighRes.core.event.events import get_texnet_original_usgs_catalog, get_texnet_high_resolution_catalog
import pandas as pd
import os
from SPHighRes.core.vel.vpvs import build_vpvs_dataframe

rmax = [30]
max_interevent_d_km=30
# rmax = [30]
# station_list = ["PB36"]
max_events = 1e3

# CMEZ HighRes
station_list = ["PB35","PB36","PB28","PB37","SA02","PB24","WB03",
                "PB26","PB31"]

# station_list = ["PB28"]
stations_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/stations/delaware_onlystations_160824.csv"
output_folder = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/vpvs"
picks_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/HighRes/picks.db"
highres_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/HighRes/origin_all.csv"
author = "HighRes_CMEZ"
catalog = get_texnet_high_resolution_catalog(highres_path,xy_epsg=4326,
                    author=author,
                    # region_lims=region
                    )

#sheng highres
# station_list = ["PB04","PB16"]
# stations_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/stations/delaware_onlystations_160824.csv"
# output_folder = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/vpvs_sheng"
# picks_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/growclust_and_sheng/picks.db"
# highres_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/growclust_and_sheng/origin.csv"
# author = "HighRes_Sheng"
# catalog = get_texnet_high_resolution_catalog(highres_path,xy_epsg=4326,
#                     author=author,
#                     # region_lims=region
#                     )


cat_df = catalog.data


stations = pd.read_csv(stations_path)
stations = stations[stations["station"].isin(station_list)]
stations_df = stations.copy()
stations_df = stations_df[["network","station","station_latitude",
                "station_longitude","station_elevation"]]

stations["sta_id"] = stations.index
stations = stations.drop_duplicates(subset=["station"],ignore_index=True)
stations = Stations(stations,xy_epsg=4326,author="Delaware_Stations")


# ---- Output CSV ----

hdf_file = os.path.join(output_folder, "vpvs_all_stations.h5")
if os.path.exists(hdf_file):
    ## ask first to remove existing file
    msg = "HDF file already exists. Remove it and continue? (y/n): "
    answer = input(msg)
    if answer.lower() == "y":
        os.remove(hdf_file)
    else:
        print("Using exiting file...")


# ---- For each rmax ----
for r in rmax:
    print(f"\n===== Processing rmax={r} km =====")

    sp_data_folder_path = os.path.join(output_folder, f"{r}_km", "sp_data")
    sp_data_path = os.path.join(sp_data_folder_path, "sp_data.csv")

    # Load or generate SP data
    if not os.path.exists(sp_data_path):
        os.makedirs(sp_data_folder_path, exist_ok=True)
        sp_data = stations.get_events_by_sp(
            catalog, rmax=r,
            picks_path=picks_path,
            output_folder=sp_data_folder_path
        )["sp_data"]
    else:
        sp_data = pd.read_csv(sp_data_path)

    sp_data = sp_data[sp_data["r"] <= r]

    # ---- Station-by-station loop ----
    for sta in station_list:
        sta_df = sp_data[sp_data["station"] == sta]

        if sta_df.empty:
            print(f"  Skipping {sta} → no data")
            continue

        # Filter and merge metadata
        # sta_df = sta_df[sta_df["preferred"]]

        sta_df = sta_df[(sta_df["tt_P"] >= 0) & (sta_df["tt_S"] >= 0)]
        sta_df = sta_df.drop_duplicates(subset=["ev_id"],ignore_index=True)

        sta_df = pd.merge(
            sta_df, cat_df[["ev_id","latitude","longitude"]],
            on="ev_id", how="left"
        )
        sta_df = sta_df.drop_duplicates(subset=["ev_id"],ignore_index=True)

        print(sta_df)

        if len(sta_df) == 0:
            print(f"  Skipping {sta} → no picks")
            continue

        print(f"  Computing Vp/Vs for station {sta} ({len(sta_df)} picks)")

        others = {"rmax": r, "station": sta}

        build_vpvs_dataframe(
        sta_df,
        station_name=sta,
        max_interevent_d_km=max_interevent_d_km,
        max_events=max_events,
        remove_outliers=False,
        others=others,
        hdf_path=hdf_file
    )