import pandas as pd
import os
import numpy as np
import glob
import concurrent.futures as cf
import numpy as np
from SPHighRes.vel.vel import VelModel
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

def get_vpvs_files(vpvs_folder, radii_km=None):
    vpvs_files = {}
    
    # if radii_km is None:
    #     radii_km =  []
    
    all_files = glob.glob(os.path.join(vpvs_folder, "*.csv"))
    vpvs_files = {}
    for file in all_files:
        filename = os.path.basename(file)
        name,fmt = os.path.splitext(filename)
        name_split = name.split("_")
        if name_split.__len__() !=2:
            print(f"Skipping file with unexpected name format: {filename}")
            continue

        station, radius_str = name_split

        if radii_km is not None:
            if radius_str not in str(int(radii_km)):
                continue
        vpvs_files[(station, radius_str)] = file

    return vpvs_files

def get_radii_paths(vpvs_files):
    """
    Groups by radius.
    Returns:
        dict: { radius: [(station, path), ...] }
    """
    radii_dict = {}
    for (station, radius), path in vpvs_files.items():
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

def get_vpvs_samples(vpvs_path,sample:int=100):

    data = pd.read_csv(vpvs_path)
    if len(data) < sample:
        sample = len(data)
    
    sampled_data = data.sample(n=sample, random_state=42)

    return sampled_data

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

def solve_depth(alpha, vp_interp, z_min=-1, z_max=20):
    """Solve z - alpha*Vp(z) = 0 for a given alpha"""
    f = lambda z: z - alpha * vp_interp(z)
    
    # Solve in depth range
    sol = root_scalar(f, bracket=[z_min, z_max], method='brentq')
    
    if sol.converged:
        return sol.root
    return np.nan

def vpvs_analysis(s_p_df,vpvs_folder,
                    vp_folder,
                    depths = np.linspace(-1.2,20,200),
                    iterations:int=100,
                    radii_km=None,
                    save_folder:str=None,
                    ):
    
    vp_models = get_vp_models(vp_folder,average=True,depths=depths)

    vp_interp = {k:interp1d(x['Depth (km)'], x['AVG_VP (km/s)'],
                     kind='linear', fill_value='extrapolate') for k, x in vp_models.items()}

    print(vp_interp)

    vpvs_files = get_vpvs_files(vpvs_folder,radii_km=radii_km)

    radii_dict = get_radii_paths(vpvs_files)

    for radius, station_paths in radii_dict.items():
        print(f"Radius: {radius} km")

        for station, path in station_paths:
            print(f"  Station: {station}, Path: {path}")
            sta_s_p_df = s_p_df[s_p_df["station"]==station]

            for i,region_eq in sta_s_p_df.groupby("region"):

                vp = vp_models.get(str(i),None)
                # print(i,vp)
                # exit()
                for i,eq in region_eq.iterrows():
                    print(eq)
                    vpvs_df = get_vpvs_samples(path,sample=iterations)

                    eq_df = pd.DataFrame()
                    eq_df["vpvs"] = vpvs_df["v_ij"]
                    eq_df["ts-tp"] = eq["ts-tp"]
                    eq_df["alpha"] = eq["ts-tp"] / eq_df["vpvs"]
                    eq_df["region"] = eq["region"]

                    eq_df.reset_index(drop=True,inplace=True)
                    eq_df["ev_id"] = eq["ev_id"]

                    avg_vp = vp_interp.get(str(i),None)

                    print(avg_vp(2))
                    # eq_df['z_teo'] = eq_df['alpha'].apply(solve_depth, args=(avg_vp,))
                    # print(avg_vp)
                    # print(eq_df)
                    exit()

                    # for depth in depths:
                    #     avg_vp = vp.get_average_velocity(phase_hint="P", 
                    #                                         zmax=depth)
                        
                    #     eq_df["teo_z"] = eq_df["alpha"] * avg_vp


                    # print(len(sta_s_p_df),sta_s_p_df.head())

    print(radii_dict)
    # print(vpvs_files.keys())

if __name__ == "__main__":
    vpvs_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/vpvs"
    s_p_df_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/sp/3_km/sp_data.csv"
    vp_folder = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/vp/regions"
    s_p_df = pd.read_csv(s_p_df_path)
    vpvs_analysis(s_p_df,vpvs_path,vp_folder)