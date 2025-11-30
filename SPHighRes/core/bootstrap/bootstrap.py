import pandas as pd
import os
import numpy as np
import glob
import concurrent.futures as cf
import numpy as np
from SPHighRes.vel.vel import VelModel
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

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
        # f = abs(depths - alpha * vp_values)
        f = depths - alpha * vp_values
        f_masked = np.where(f >= 0, f, np.inf)
        best_idx = np.argmin(f_masked)
        depths_bs[i] = depths[best_idx]
        vp_bs[i] = vp_values[best_idx]

    return depths_bs, vp_bs



def eq_depths(
    s_p_df,
    vpvs_df,
    vp_folder,
    save_folder,
    depths=np.linspace(-1.2, 20, 200),
    n_boot: int = 500,
    trend_sigma: float = 1,
    n_sample_use: int = 50,
    station_list=None,
    radii_km=None,
    radii_interevent_km=None,
    vpvs_bin=(1.4, 2.4, 0.025),  # min, max, bin width
    depth_bin=(0, 15, 1),  # min, max, bin width
    epsilon=0.01,
    bootstrap_details_folder=None,
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
    import string
    from scipy import stats
    from sklearn.neighbors import KernelDensity
    import matplotlib.pyplot as plt

    from scipy.interpolate import interp1d

    os.makedirs(save_folder, exist_ok=True)
    results_path = os.path.join(save_folder, "results.h5")
    summary_file = os.path.join(save_folder, "summary.csv")

    for f in [summary_file, results_path]:
        if os.path.exists(f):
            os.remove(f)

    store = pd.HDFStore(results_path, mode="w")
    with open(summary_file, "w") as f:
        f.write("key,ev_id,station,preferred,region,tstp,median_depth,mode_depth,median_vp,median_vpvs,"
                "iqr_depth,std_depth,std_vp,std_vpvs\n")

    # Load VP models
    vp_models = get_vp_models(vp_folder, average=True, depths=depths)
    vp_interp = {k: interp1d(x['Depth (km)'], x['AVG_VP (km/s)'],
                              kind='linear', fill_value='extrapolate')
                 for k, x in vp_models.items()}

    if station_list is not None:
        vpvs_df = vpvs_df[vpvs_df["station"].isin(station_list)]
        s_p_df = s_p_df[s_p_df["station"].isin(station_list)]


    if radii_interevent_km is not None:
        vpvs_df = vpvs_df[vpvs_df["r_ij"] <= radii_interevent_km]

    if radii_km is not None:
        radii_dict = { str(r): vpvs_df[(vpvs_df["r_i"] <= r) & (vpvs_df["r_j"] <=r)] for r in radii_km }
    else:
        max_radius = int(vpvs_df[["r_i","r_j"]].max().max())
        radii_dict = { str(max_radius): vpvs_df }

    
    rng = np.random.default_rng(random_state)
    vpvs_min, vpvs_max, bin_width = vpvs_bin
    bins = np.arange(vpvs_min, vpvs_max + bin_width, bin_width)
    global_bins = bins

    s_p_df = s_p_df[(s_p_df["tt_P"] >=0) & (s_p_df["tt_S"] >=0)]
    s_p_df = s_p_df.drop_duplicates(ignore_index=True)

    for radius, df in radii_dict.items():
        print(f"Processing radius: {radius} km")
        for station, vpvs_df in df.groupby("station"):

            sta_bs_details = {"station":station,
                              "radius_km":radius,
                              }
            if bootstrap_details_folder is not None:
                os.makedirs(bootstrap_details_folder, exist_ok=True)
                

            print(f" Processing station: {station} with {len(vpvs_df)} VP/VS values")

            ### VP/VS DATA PROCESSING ###
            vpvs_data = vpvs_df["v_ij"].values
            sta_bs_details["vpvs_raw"] = vpvs_data

            n_data = len(vpvs_data)
            if n_data == 0:
                continue
            
            # --- (1) Remove outliers using IQR (robust) ---
            q1,q2, q3 = np.percentile(vpvs_data, [25,50, 75])
            iqr = q3 - q1
            vpvs_clean = vpvs_data[(vpvs_data >= q1 - 1.5*iqr) & (vpvs_data <= q3 + 1.5*iqr)]
            sta_bs_details["vpvs_raw_q1"] = q1
            sta_bs_details["vpvs_raw_q2"] = q2
            sta_bs_details["vpvs_raw_q3"] = q3
            sta_bs_details["vpvs_raw_iqr_l"] = q1 - 1.5*iqr
            sta_bs_details["vpvs_raw_iqr_u"] = q3 + 1.5*iqr


            if len(vpvs_clean) < 5:  # too few to estimate mode
                print(f"Not enough VP/VS values after IQR filtering for station {station}, skipping.")
                continue

            # --- (2) Estimate mode using KDE ---
            kde = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(vpvs_clean.reshape(-1, 1))
            x_vals = np.linspace(vpvs_clean.min(), vpvs_clean.max(), 400).reshape(-1, 1)
            log_dens = kde.score_samples(x_vals)
            mode = x_vals[np.argmax(log_dens)][0]
            median = np.median(vpvs_clean)


            # --- (3) Local sigma relative to mode ---
            sigma = np.std(vpvs_clean - mode)

            # --- (4) Keep only values within ± trend_sigma * sigma around the mode ---
            lower = mode - trend_sigma * sigma
            upper = mode + trend_sigma * sigma
        
            sta_bs_details["vpvs_iqr"] = vpvs_clean
            sta_bs_details["vpvs_iqr_mode"] = mode
            sta_bs_details["vpvs_iqr_median"] = median
            sta_bs_details["vpvs_iqr_sigma_l"] = lower
            sta_bs_details["vpvs_iqr_sigma_u"] = upper


            vpvs_filtered = vpvs_clean[(vpvs_clean >= lower) & (vpvs_clean <= upper)]


            sta_bs_details["vpvs_sigma"] = vpvs_filtered

            if len(vpvs_filtered) < 5:
                # fallback: use vpvs_clean if mode filter removed too much
                vpvs_filtered = vpvs_clean

            # sta_bs_details = {"vpvs_sigma": vpvs_filtered,"vpvs_sigma_mode":mode}
            # plt.hist(vpvs_filtered,bins=30)
            # plt.savefig("/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/SPHighRes/core/bootstrap/bootstrap.png")
            # exit()

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
                    
                    ev_id = eq["ev_id"]
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
                        counts, edges  = np.histogram(valid_values, bins=bins)
                        mode_idx = np.argmax(counts)
                        
                        # Take median inside the modal bin
                        in_modal_bin = (valid_values >= bins[mode_idx]) & (valid_values < bins[mode_idx + 1])
                        median_in_modal_bin = np.median(valid_values[in_modal_bin])
                        vpvs_mode_array[i] = median_in_modal_bin

                        sta_bs_details["vpvs_sample"] = valid_values
                        sta_bs_details["vpvs_sample_median"] = median_in_modal_bin


                    # --- Compute alpha per bootstrap using VP/VS mode ---
                    alpha_array = ts_tp / np.clip(vpvs_mode_array - 1, epsilon, None)

                    # --- Solve depth per bootstrap ---
                    depths_bs, vp_bs = solve_depth_bootstrap(alpha_array, vp_func, depths)

                    if bootstrap_details_folder is not None:
                        sta_bs_path = os.path.join(bootstrap_details_folder, f"bs_details_{station}_{radius}km_{ev_id}.png")
                        # Plot bootstrap details once
                        if "ev_id" not in list(sta_bs_details.keys()):
                            sta_bs_details["ev_id"] = ev_id

                            fig,axes = plt.subplots(2,3,figsize=(12,6)) 

                            # --- Add (a), (b), (c)... labels to subplots ---
                            rows, cols = axes.shape
                            n = 0
                            for row in range(rows):
                                for col in range(cols):
                                    box = dict(boxstyle='round', facecolor='white', alpha=1)
                                    axes[row, col].annotate(f"({string.ascii_lowercase[n]})",
                                                            xy=(-0.1, 1.05),  # outside top-left
                                                            xycoords='axes fraction',
                                                            ha='left',
                                                            va='bottom',
                                                            fontsize="large",
                                                            fontweight="normal",
                                                            # bbox=box
                                                            )
                                    n += 1



                            raw_len = len(sta_bs_details["vpvs_raw"])

                            axes[0,0].hist(sta_bs_details["vpvs_raw"], bins=global_bins, 
                                           label=f"{raw_len:.2e} values")
                            axes[0,0].set_title(f"Station {station} (r <= {radius} km)")
                            axes[0,0].set_xlabel("Vp/Vs (raw)")
                            axes[0,0].set_ylabel("Counts")
                            axes[0,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                            l=axes[0,0].legend()
                            # axes[0,0].text(0.6, 0.6,
                            #                 f"r <= {radius} km",
                            #                 ha='center', va='top', 
                            #                 transform=axes[0,0].transAxes)

                            # Get current x-axis limits
                            xmin, xmax = axes[0,0].get_xlim()

                            perc = 100 * len(sta_bs_details['vpvs_iqr']) / raw_len  # percentage of original
                            axes[0,1].hist(sta_bs_details["vpvs_iqr"], bins=global_bins,
                                            #  label=f"{len(sta_bs_details['vpvs_iqr']):.2e} values")
                                             label=f"{perc:.1f}% of raw")
                            
                            lower = sta_bs_details["vpvs_raw_iqr_l"]
                            upper = sta_bs_details["vpvs_raw_iqr_u"]

                            # Shade regions outside ±σ
                            axes[0,1].axvspan(xmin, lower, color="gray", alpha=0.25, label="_nolegend_")
                            axes[0,1].axvspan(upper, xmax, color="gray", alpha=0.25, label="_nolegend_")

                            # axes[0,1].axvline(sta_bs_details["vpvs_iqr_mode"],color='r',label=' Mode')
                            # q1 - 1.5*iqr
                            axes[0,1].axvline(sta_bs_details["vpvs_raw_q1"],
                                              color='g',linestyle='--',
                                            #   label=f'Q1'
                                              )
                            axes[0,1].axvline(sta_bs_details["vpvs_raw_q2"],color='r',linestyle='--',
                                              label=f'Q2')
                            axes[0,1].axvline(sta_bs_details["vpvs_raw_q3"],color='g',linestyle='--',
                                              label=f'Qk (k=1,3)')
                            axes[0,1].axvline(upper,color='k',linestyle='--',
                                              label=rf'Qk ± 1.5*IQR')
                                            #   label=rf'Qk ± IQR k=1,3')
                            # axes[0,1].axvline(upper,color='g',linestyle='--',
                            #                   label=rf'± Qk ± IQR k=1,3')
                            axes[0,1].set_title(f"Station {station} (No outliers)")
                            axes[0,1].set_xlabel("Vp/Vs (IQR-cleaned)")
                            axes[0,1].set_ylabel("Counts")
                            axes[0,1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                            axes[0,1].set_xlim(xmin, xmax)
                            axes[0,1].legend()

                            perc = 100 * len(sta_bs_details['vpvs_sigma']) / raw_len  # percentage of original
                            axes[0,2].hist(sta_bs_details["vpvs_sigma"], bins=global_bins,
                                                # label=f"{len(sta_bs_details['vpvs_sigma']):.2e} values")
                                                label=f"{perc:.1f}% of raw")
                            axes[0,2].axvline(sta_bs_details["vpvs_iqr_mode"],color='orange',label=' Mode')
                            axes[0,2].axvline(sta_bs_details["vpvs_iqr_median"],color='red',linestyle='--',
                                              label=' Median')
                            axes[0,2].axvline(sta_bs_details["vpvs_iqr_sigma_l"],color='g',linestyle='--')
                            axes[0,2].axvline(sta_bs_details["vpvs_iqr_sigma_u"],color='g',linestyle='--',
                                              label=rf'±{trend_sigma}σ')
                            

                            lower = sta_bs_details["vpvs_iqr_sigma_l"]
                            upper = sta_bs_details["vpvs_iqr_sigma_u"]

                            # Shade regions outside ±σ
                            axes[0,2].axvspan(xmin, lower, color="gray", alpha=0.25, label="_nolegend_")
                            axes[0,2].axvspan(upper, xmax, color="gray", alpha=0.25, label="_nolegend_")
                            

                            axes[0,2].set_title(f"Station {station} (σ filtered)")
                            axes[0,2].set_xlabel("Vp/Vs (IQR-cleaned, mode ± σ filtered)")
                            axes[0,2].set_ylabel("Counts")
                            axes[0,2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                            axes[0,2].set_xlim(xmin, xmax)
                            axes[0,2].legend()

                            sampling = len(sta_bs_details["vpvs_sample"])
                            formatted = f"{sampling /1e3:.0f}k"
                            axes[1,0].hist(sta_bs_details["vpvs_sample"], bins=global_bins,
                                           label=f"{formatted} samples")
                            axes[1,0].axvline(sta_bs_details["vpvs_sample_median"],color='r',
                                              linestyle='--',
                                              label=' Median')
                            axes[1,0].set_title(f"Event {ev_id} \nVp/Vs Bootstrap $\\it{{i}}$ / {sampling}")
                            axes[1,0].set_xlabel("Vp/Vs (Bootstrap Sample Values)")
                            axes[1,0].set_ylabel("Counts")
                            axes[1,0].legend()


                            # axes[1,1] → boxplot for Vp/Vs bootstrap medians
                            box1 = axes[1,1].boxplot(vpvs_mode_array, vert=True, patch_artist=True,
                                                    boxprops=dict(facecolor='lightblue', color='blue'),
                                                    medianprops=dict(color='none', linewidth=2),
                                                    whiskerprops=dict(color='green', linestyle='--'),
                                                    capprops=dict(color='green', linestyle='--'),
                                                    labels=[f"Event {ev_id}\nStation {station}"])  # x-axis label
                            
                            axes[1,1].set_title("Vp/Vs Bootstrap Medians")
                            axes[1,1].set_ylabel("Vp/Vs")

                            # add custom legend showing number of bootstraps
                            formatted = f"{vpvs_mode_array.size/1e3:.0f}k"
                            axes[1,1].legend([box1["boxes"][0]], [f"{formatted} bootstraps"],
                                             loc='upper right')

                            # axes[1,2] → boxplot for depth bootstraps
                            box2 = axes[1,2].boxplot(depths_bs, vert=True, patch_artist=True,
                                                    boxprops=dict(facecolor='lightgray', color='black'),
                                                    medianprops=dict(color='none', linewidth=2),
                                                    whiskerprops=dict(color='green', linestyle='--'),
                                                    capprops=dict(color='green', linestyle='--'),
                                                    labels=[f"Event {ev_id}\nStation {station}"])  # x-axis label
                            
                            # median = np.median(depths_bs)
                            # sigma = np.std(depths_bs)
                            # # Add ±σ lines
                            # axes[1,2].axhline(median + sigma, color='orange', linestyle='--', label='Median ± σ')
                            # axes[1,2].axhline(median - sigma, color='orange', linestyle='--')
                            # axes[1,2].axhline(median, color='red', linestyle='-', label='Median')

                            # # Add legend (avoid duplicate labels)
                            # handles, labels = axes[1,2].get_legend_handles_labels()
                            # by_label = dict(zip(labels, handles))
                            # # axes[1,2].legend(by_label.values(), by_label.keys())
                            
                            axes[1,2].set_title("Depth Uncertainty from Bootstrap")
                            axes[1,2].set_ylabel("Depth from surface (km)")


                            # add custom legend showing number of bootstraps
                            formatted = f"{depths_bs.size/1e3:.0f}k"
                            axes[1,2].legend([box2["boxes"][0]], [f"{formatted} bootstraps"],
                                             loc='upper right')

                                           
                            plt.tight_layout()
                            fig.savefig(sta_bs_path,dpi=300)
                            plt.close(fig)
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


                    store.put(key, df_result, format="table")

                    # Append summary
                    with open(summary_file, "a", encoding="utf-8") as f:
                        row = [
                            key,
                            str(eq["ev_id"]),
                            station,
                            str(eq["preferred"]),
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