import pandas as pd
import os
import numpy as np
import glob
import concurrent.futures as cf
import numpy as np
from SPHighRes.vel.vel import VelModel
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

def get_vpvs_files(vpvs_folder, radii_km=None, station_col="station"):
    """
    Returns a dictionary mapping (station, radius_km) → file_path
    for files named like: vpvs_rmax_5km.csv
    
    Each file contains multiple stations; this function reads the stations
    and assigns the same file path to each (station, radius).
    """

    vpvs_files = {}
    all_files = glob.glob(os.path.join(vpvs_folder, "*.csv"))

    for file in all_files:
        filename = os.path.basename(file)
        name, _ = os.path.splitext(filename)

        # Expected format: vpvs_rmax_5km
        parts = name.split("_")
        if len(parts) != 3 or not parts[-1].endswith("km"):
            print(f"Skipping file with unexpected name format: {filename}")
            continue

        # Extract radius
        radius_str = parts[-1].replace("km", "")
        try:
            radius = int(radius_str)
        except ValueError:
            print(f"Could not parse radius from: {filename}")
            continue

        # Filter by radii if needed
        if radii_km is not None and radius not in radii_km:
            continue

        # Read file to find station names
        df = pd.read_csv(file)

        # Check station column exists
        if station_col not in df.columns:
            raise ValueError(
                f"Column '{station_col}' not found in '{filename}'. "
                f"Columns available: {list(df.columns)}"
            )

        # Assign this file to each station
        for station in df[station_col].unique():
            vpvs_files[(station, str(radius))] = file

    return vpvs_files

def get_radii_paths(vpvs_files,station_list=None):
    """
    Groups by radius.
    Returns:
        dict: { radius: [(station, path), ...] }
    """
    radii_dict = {}
    for (station, radius), path in vpvs_files.items():
        if station_list is not None and station not in station_list:
            continue
        radii_dict.setdefault(radius, []).append((station, path))
    return radii_dict

def get_station_paths(vpvs_files):
    """
    Groups by station.
    Returns:
        dict: { station: [(radius, path), ...] }
    """
    station_dict = {}
    for (station, radius), path in vpvs_files.items():
        station_dict.setdefault(station, []).append((radius, path))
    return station_dict


def get_vp_models(vp_folder,
                average=False,
                depths = np.linspace(-1.2,20,200)):

    vp_files = {}
    all_files = glob.glob(os.path.join(vp_folder, "*.csv"))
    for file in all_files:
        filename = os.path.basename(file)
        name,fmt = os.path.splitext(filename)
        region = str(name.split("R")[-1])

        df = pd.read_csv(file)
        df = df.rename(columns={
            "Depth[km]":"Depth (km)",
            "Vp_mean[km/s]":"VP (km/s)"})
        df["VS (km/s)"] = np.nan
        vel = VelModel(df,name=f"R{region}")

        if average:
            avg = {"Depth (km)":[], "AVG_VP (km/s)":[]}
            for depth in depths:
                avg_vel = vel.get_average_velocity(phase_hint="P", 
                                                    zmax=depth)
                avg["Depth (km)"].append(depth)
                avg["AVG_VP (km/s)"].append(avg_vel)

            vel = pd.DataFrame(avg)
            vel = vel.dropna().reset_index(drop=True)

        vp_files[region] = vel
    return vp_files



def solve_depth_bootstrap(alpha_array, vp_func, depths):
    # alpha_array: (n_boot,)
    vp_values = vp_func(depths)  # shape (n_depth,)
    depths_bs = np.zeros_like(alpha_array)
    vp_bs = np.zeros_like(alpha_array)

    for i, alpha in enumerate(alpha_array):
        f = abs(depths - alpha * vp_values)
        f_masked = np.where(f >= 0, f, np.inf)
        best_idx = np.argmin(f_masked)
        depths_bs[i] = depths[best_idx]
        vp_bs[i] = vp_values[best_idx]

    return depths_bs, vp_bs



def eq_depths(
    s_p_df,
    vpvs_folder,
    vp_folder,
    save_folder,
    depths=np.linspace(-1.2, 20, 200),
    n_boot: int = 500,
    trend_sigma: float = 1,
    n_sample_use: int = 50,
    station_list=None,
    radii_km=None,
    vpvs_bin=(1.4, 2.4, 0.025),  # min, max, bin width
    depth_bin=(0, 15, 1),  # min, max, bin width
    epsilon=0.01,
    random_state=None
):
    """
    Estimate earthquake depths using VP/VS bootstrapping (stable version).

    Parameters
    ----------
    s_p_df : pd.DataFrame
        DataFrame with station picks (columns: station, region, ts-tp, ev_id, ...)
    vpvs_folder : str
        Folder containing VP/VS CSV files.
    vp_folder : str
        Folder containing VP models per region.
    save_folder : str
        Output folder for results and summary.
    depths : np.ndarray
        Depths at which to evaluate VP.
    n_boot : int
        Number of bootstrap iterations.
    trend_sigma : float
        Keep only VP/VS values within mean ± trend_sigma * std to follow trend.
    n_sample_use : int
        Number of VP/VS values per bootstrap to compute median alpha.
    radii_km : list[int] or None
        Filter VPVS files by radius if given.
    random_state : int or None
        Random seed for reproducibility.
    """
    import numpy as np
    import pandas as pd
    import os
    from sklearn.neighbors import KernelDensity

    from scipy.interpolate import interp1d

    os.makedirs(save_folder, exist_ok=True)
    results_path = os.path.join(save_folder, "results.h5")
    summary_file = os.path.join(save_folder, "summary.csv")

    for f in [summary_file, results_path]:
        if os.path.exists(f):
            os.remove(f)

    store = pd.HDFStore(results_path, mode="w")
    with open(summary_file, "w") as f:
        f.write("key,ev_id,station,region,tstp,median_depth,mode_depth,median_vp,median_vpvs,"
                "iqr_depth,std_depth,std_vp,std_vpvs\n")

    # Load VP models
    vp_models = get_vp_models(vp_folder, average=True, depths=depths)
    vp_interp = {k: interp1d(x['Depth (km)'], x['AVG_VP (km/s)'],
                              kind='linear', fill_value='extrapolate')
                 for k, x in vp_models.items()}

    # Load VPVS files
    vpvs_files = get_vpvs_files(vpvs_folder, radii_km=radii_km)
    radii_dict = get_radii_paths(vpvs_files,station_list=station_list)
    
    print(radii_dict)
    exit()

    print("Radii found:", list(radii_dict.keys()))

    rng = np.random.default_rng(random_state)
    vpvs_min, vpvs_max, bin_width = vpvs_bin
    bins = np.arange(vpvs_min, vpvs_max + bin_width, bin_width)

    for radius, station_paths in radii_dict.items():
        print(f"Processing radius: {radius} km with {len(station_paths)} stations")
        for station, path in station_paths:
            print(f" Processing station: {station} from file: {path}")


            ### VP/VS DATA PROCESSING ###
            vpvs_df = pd.read_csv(path)
            vpvs_df = vpvs_df[vpvs_df["station"]==station] #super important filter
            vpvs_data = vpvs_df["v_ij"].values

            n_data = len(vpvs_data)
            if n_data == 0:
                continue

            # --- (1) Remove outliers using IQR (robust) ---
            q1, q3 = np.percentile(vpvs_data, [25, 75])
            iqr = q3 - q1
            vpvs_clean = vpvs_data[(vpvs_data >= q1 - 1.5*iqr) & (vpvs_data <= q3 + 1.5*iqr)]

            if len(vpvs_clean) < 5:  # too few to estimate mode
                print(f"Not enough VP/VS values after IQR filtering for station {station}, skipping.")
                continue

            # --- (2) Estimate mode using KDE ---
            kde = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(vpvs_clean.reshape(-1, 1))
            x_vals = np.linspace(vpvs_clean.min(), vpvs_clean.max(), 400).reshape(-1, 1)
            log_dens = kde.score_samples(x_vals)
            mode = x_vals[np.argmax(log_dens)][0]

            # --- (3) Local sigma relative to mode ---
            sigma = np.std(vpvs_clean - mode)

            # --- (4) Keep only values within ± trend_sigma * sigma around the mode ---
            lower = mode - trend_sigma * sigma
            upper = mode + trend_sigma * sigma
            vpvs_filtered = vpvs_clean[(vpvs_clean >= lower) & (vpvs_clean <= upper)]

            if len(vpvs_filtered) < 5:
                # fallback: use vpvs_clean if mode filter removed too much
                vpvs_filtered = vpvs_clean

            ### END VP/VS DATA PROCESSING ###

            sta_s_p_df = s_p_df[s_p_df["station"] == station]
            print(f"  Station: {station} with {len(sta_s_p_df)} events")
            for region, region_eq in sta_s_p_df.groupby("region"):

                # Get VP model for region
                vp_func = vp_interp.get(str(region), None)
                if vp_func is None:
                    continue

                for _, eq in region_eq.iterrows():

                    # Bootstrap sampling
                    indices = rng.integers(0, len(vpvs_filtered), size=(n_boot, n_sample_use))
                    vpvs_samples = vpvs_filtered[indices]

                            
                    ts_tp = eq["ts-tp"]

                    # --- Compute VP/VS mode per bootstrap ---
                    vpvs_mode_array = np.empty(n_boot)

                    for i in range(n_boot):


                        # Select values within the bin range
                        in_range = (vpvs_samples[i] >= vpvs_min) & (vpvs_samples[i] < vpvs_max)
                        valid_values = vpvs_samples[i, in_range]
                        
                        if len(valid_values) == 0:
                            # fallback: use median of the original bootstrap
                            vpvs_mode_array[i] = np.median(vpvs_samples[i])
                            continue
                        
                        # Histogram of values in range
                        counts, _ = np.histogram(valid_values, bins=bins)
                        mode_idx = np.argmax(counts)
                        
                        # Take median inside the modal bin
                        in_modal_bin = (valid_values >= bins[mode_idx]) & (valid_values < bins[mode_idx + 1])
                        vpvs_mode_array[i] = np.median(valid_values[in_modal_bin])


                    # --- Compute alpha per bootstrap using VP/VS mode ---
                    alpha_array = ts_tp / np.clip(vpvs_mode_array - 1, epsilon, None)

                    # --- Solve depth per bootstrap ---
                    depths_bs, vp_bs = solve_depth_bootstrap(alpha_array, vp_func, depths)

                    # --- Create result DataFrame ---
                    df_result = pd.DataFrame({
                        "vpvs": vpvs_mode_array,
                        "vp": vp_bs,
                        "depth": depths_bs
                    })

                    # print(df_result.describe())


                    # --- Depth mode ---
                    depth_min, depth_max, depth_width = depth_bin
                    depth_bins = np.arange(depth_min, depth_max + depth_width, depth_width)
                    counts, _ = np.histogram(depths_bs, bins=depth_bins)
                    mode_depth_idx = np.argmax(counts)
                    in_modal_bin = (depths_bs >= depth_bins[mode_depth_idx]) & (depths_bs < depth_bins[mode_depth_idx + 1])
                    depth_mode = np.median(depths_bs[in_modal_bin])



                    # Save HDF5
                    key = f"/sta_{station}/vpvs_{radius}km/ev_{eq['ev_id']}"
                    print(
                    key,
                    "depth:",round(np.median(df_result["depth"]),2), 
                    "depth_mode:", round(depth_mode, 2),
                    "vpvs:", round(np.median(df_result["vpvs"]),2),
                    "std_depth:", round(np.std(df_result["depth"]),2),
                    "vpvs_std:", round(np.std(df_result["vpvs"]),2)
                    )

                    continue

                    store.put(key, df_result, format="table")

                    # Append summary
                    with open(summary_file, "a", encoding="utf-8") as f:
                        row = [
                            key,
                            str(eq["ev_id"]),
                            station,
                            str(region),
                            f"{ts_tp:.6f}",
                            f"{np.median(df_result['depth']):.6f}",
                            f"{depth_mode:.6f}", # depth mode
                            f"{np.median(df_result['vp']):.6f}",
                            f"{np.median(df_result['vpvs']):.6f}",
                            f"{df_result['depth'].quantile(0.75) - df_result['depth'].quantile(0.25):.6f}",
                            f"{np.std(df_result['depth']):.6f}",
                            f"{np.std(df_result['vp']):.6f}",
                            f"{np.std(df_result['vpvs']):.6f}",
                        ]
                        f.write(",".join(row) + "\n")

    store.close()


if __name__ == "__main__":
    vpvs_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/vpvs"
    s_p_df_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/sp/3_km/sp_data.csv"
    vp_folder = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/vp/regions"
    save_folder = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/z"
    s_p_df = pd.read_csv(s_p_df_path)
    eq_depths(s_p_df,vpvs_path,vp_folder,save_folder)