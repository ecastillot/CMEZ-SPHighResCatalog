import pandas as pd
import os
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations
from typing import Tuple, List, Dict
from scipy.stats import mode
from SPHighRes.core.event.stations import Stations
from SPHighRes.core.event.events import get_texnet_high_resolution_catalog
import pandas as pd
import os
import pandas as pd
from itertools import combinations
from obspy.geodetics import locations2degrees, kilometers2degrees
from obspy.geodetics.base import gps2dist_azimuth


def compute_event_distance(ev1_lat, ev1_lon, ev2_lat, ev2_lon):
    """
    Compute distance in kilometers between two earthquake locations.
    """
    dist_m, _, _ = gps2dist_azimuth(ev1_lat, ev1_lon, ev2_lat, ev2_lon)
    return dist_m / 1000.0


def compute_vpvs(tt_p_i, tt_s_i, tt_p_j, tt_s_j):
    """
    Compute Vp/Vs ratio using (S1 - S2)/(P1 - P2).
    Returns None if value is negative or invalid.
    """
    numerator = tt_s_i - tt_s_j
    denominator = tt_p_i - tt_p_j

    if denominator == 0:
        return None

    vpvs = numerator / denominator
    return vpvs if vpvs > 0 else None

def remove_outliers_iqr(df, column="v_ij"):
    """
    Remove outliers from a DataFrame column using the IQR method.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[column] >= lower) & (df[column] <= upper)]


def build_vpvs_dataframe(df,
                         station_name,
                         max_interevent_d_km=3.0,
                         remove_outliers=True,
                         max_events=None,
                         others=None,
                         hdf_path=None,
                         chunk_size=1000000):
    """
    Build Vp/Vs dataset for ONE station.

    Saves in memory if hdf_path=None, otherwise writes in chunks to HDF.
    """
    from tqdm import tqdm
    import pandas as pd
    from itertools import combinations

    if max_events is not None and len(df) > max_events:
        df = df.sample(n=int(max_events), random_state=42).reset_index(drop=True)

    in_memory = hdf_path is None
    buffer = []  # holds rows until chunk_size
    all_records = [] if in_memory else None

    good_logs = 0
    bad_logs = 0

    n_events = len(df)
    n_pairs = n_events * (n_events - 1) // 2


    print(f"\nProcessing station {station_name}: {n_events} events → {n_pairs} combinations")

    comb = combinations(df.itertuples(), 2)

    for ev1, ev2 in tqdm(comb, desc=f"Station {station_name}", total=n_pairs):

        # Distance
        r_ij = compute_event_distance(
            ev1.latitude, ev1.longitude,
            ev2.latitude, ev2.longitude
        )
        if r_ij > max_interevent_d_km:
            continue

        # Vp/Vs
        vpvs_val = compute_vpvs(ev1.tt_P, ev1.tt_S, ev2.tt_P, ev2.tt_S)
        if vpvs_val is None or vpvs_val <= 0:
            bad_logs += 1
            continue

        good_logs += 1

        row = {
            "station": station_name,
            "ev_i": ev1.ev_id,
            "ev_j": ev2.ev_id,
            "r_i": ev1.r,
            "r_j": ev2.r,
            "r_ij": r_ij,
            "v_ij": vpvs_val,
            "tt_Pi": ev1.tt_P,
            "tt_Pj": ev2.tt_P,
            "tt_Si": ev1.tt_S,
            "tt_Sj": ev2.tt_S,
            "tt_SPi": ev1.tt_S - ev1.tt_P,
            "tt_SPj": ev2.tt_S - ev2.tt_P,
        }

        if others is not None:
            row.update(others)

        if in_memory:
            all_records.append(row)
        else:
            buffer.append(row)
            if len(buffer) >= chunk_size:
                pd.DataFrame(buffer).to_hdf(
                    hdf_path,
                    key=f"station_{station_name}",
                    mode="a",
                    format="table",
                    append=True
                )
                buffer = []

    # Write any remaining rows in buffer
    if not in_memory and buffer:
        pd.DataFrame(buffer).to_hdf(
            hdf_path,
            key=f"station_{station_name}",
            mode="a",
            format="table",
            append=True
        )

    print(f"Station {station_name} → Good: {good_logs} | Bad: {bad_logs}")

    if in_memory:
        df_station = pd.DataFrame(all_records)
        if remove_outliers and not df_station.empty:
            df_station = remove_outliers_iqr(df_station, "v_ij")
        return df_station
    else:
        return None

if __name__ == "__main__":
    rmax = [5,10,15,20,25,30]
    station_list = ["PB35","PB36","PB28","PB37","SA02","PB24","WB03"]
    # station_list = ["WB03"]
    stations_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/stations/delaware_onlystations_160824.csv"
    output_folder = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/vpvs"
    picks_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/HighRes/picks.db"
    highres_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/HighRes/origin.csv"



    catalog = get_texnet_high_resolution_catalog(highres_path,xy_epsg=4326,
                        author="HighRes_CMEZ",
                        # region_lims=region
                        )
    cat_df = catalog.data


    stations = pd.read_csv(stations_path)
    stations = stations[stations["station"].isin(station_list)]
    stations_df = stations.copy()
    stations_df = stations_df[["network","station","station_latitude",
                    "station_longitude","station_elevation"]]

    stations["sta_id"] = stations.index
    stations = Stations(stations,xy_epsg=4326,author="Delaware_Stations")


    for r in rmax:
        sp_data_path = os.path.join(output_folder,f"{r}_km","sp_data.csv")
        # sp_data = pd.read_csv(sp_data_path)

        if not os.path.exists(sp_data_path):

            sp_data = stations.get_events_by_sp(catalog,rmax=r ,
                                picks_path= picks_path,
                                output_folder=sp_data_path
                                )
            sp_data = sp_data["sp_data"]
        else:
            sp_data = pd.read_csv(sp_data_path)
            
        # exit()
        if station_list is not None:
            sp_data = sp_data[sp_data["station"].isin(station_list)]

        sp_data = sp_data[sp_data["preferred"]]
        sp_data = pd.merge(sp_data,stations_df,on=["station"],how="left")
        sp_data = pd.merge(sp_data,cat_df[["ev_id","latitude","longitude"]],
                           left_on=["ev_id"],right_on=["ev_id"],how="left")
        # get rif of negative tt_P or tt_S values
        sp_data = sp_data[(sp_data["tt_P"] >=0) & (sp_data["tt_S"] >=0)]
        sp_data = sp_data.drop_duplicates(ignore_index=True)

        # print(sp_data.info())
        # exit()
        result_df = build_vpvs_dataframe(sp_data, max_distance_km=3,
                                         remove_outliers=False)
        result_df.reset_index(drop=True,inplace=True)

        result_path = os.path.join(output_folder,f"vpvs_rmax_{r}km.csv")
        result_df.to_csv(result_path,index=False)

        print("========================================")
        print(f"Vp/Vs results for rmax={r} km:")
        # describe results by station
        for station, station_df in result_df.groupby("station"):
            print(f"Station: {station}")
            print(station_df.describe())
