import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from SPHighRes.core.event.utils import latlon2yx_in_km
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

radii_km = 30
key_col = "z_new_from_sea_level"

path = f"/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/z/summary_standard_{radii_km}km.csv"
outpath = f"/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/z/summary_reloc_standard_{radii_km}km.csv"
df = pd.read_csv(path,parse_dates=['origin_time'])
df = latlon2yx_in_km(df,epsg=4326)
df = df.rename(columns={"x[km]":"x","y[km]":"y"})
df["Author_new"] = df.apply(lambda x: "S-P Relative Reloc" if pd.isna(x[key_col]) else "S-P Method",axis=1)


ids_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/HighRes/origin.csv"
ids_df = pd.read_csv(ids_path)
ids = ids_df["ev_id"].unique()

df = df[df["ev_id"].isin(ids)].copy()
df=df.reset_index(drop=True)

df = df.sort_values(by=["origin_time"],ignore_index=True)
print(df)

# Load DataFrame (assuming df is already loaded with the given structure)
print("Relocation procedure started...")
while df[key_col].isna().any():
    # Identify reference events (those with both z_ori and z_new)
    ref_events = df.dropna(subset=[key_col])
    target_events = df[df[key_col].isna()].copy()
    
    
    # Build a KDTree for efficient spatial searching
    ref_tree = cKDTree(ref_events[['x', 'y']].values)
    
    # Power parameter for IDW
    p = 2  # Adjust this if needed
    max_distance = 500 # 1 km radius limit
    epsilon = 1e6
    
    # Function to estimate z_new
    for idx, row in target_events.iterrows():
        dists, idxs = ref_tree.query([row['x'], row['y']], k=5, distance_upper_bound=max_distance)
        valid = dists < max_distance  # Filter out large distances
        
        if np.any(valid):
            dists, idxs = dists[valid], idxs[valid]
            weights = 1 / ((dists ** p) + epsilon)  # Inverse distance weighting
            delta_z = ref_events.iloc[idxs][key_col].values - ref_events.iloc[idxs]['z_ori_from_sea_level'].values
            target_events.at[idx, key_col] = row['z_ori_from_sea_level'] + np.sum(weights * delta_z) / np.sum(weights)
            
        else:
            target_events.at[idx, key_col] = row['z_ori_from_sea_level']  # If no neighbors, keep original depth
    
    # Update the original DataFrame
    df.update(target_events[['ev_id', key_col, 'Author_new']])


df.to_csv(outpath,index=False)
print(df)