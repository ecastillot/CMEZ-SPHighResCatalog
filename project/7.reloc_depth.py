import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from SPHighRes.core.event.utils import latlon2yx_in_km
import seaborn as sns
import matplotlib.pyplot as plt

radii_km = 30

path = f"/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/z/summary_standard_{radii_km}km.csv"
outpath = f"/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/z/summary_reloc_standard_{radii_km}km.csv"

df = pd.read_csv(path, parse_dates=['origin_time'])

df = latlon2yx_in_km(df, epsg=4326)
df = df.rename(columns={"x[km]": "x", "y[km]": "y"})
df["Author_new"] = df.apply(lambda x: "S-P Relative Reloc" if pd.isna(x["z_new_from_sea_level"]) else "S-P Method", axis=1)

# Load list of event IDs
ids_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/HighRes/origin.csv"
ids_df = pd.read_csv(ids_path)
ids = ids_df["ev_id"].unique()

df = df[df["ev_id"].isin(ids)].copy()
df = df.reset_index(drop=True)

df = df.sort_values(by=["origin_time"], ignore_index=True)

# Ensure uncertainty column exists and preserve original ones
if "z_new_std" not in df.columns:
    df["z_new_std"] = np.nan

print(df)
print("Relocation procedure started...")

# Iterative update until all missing depths are filled
while df["z_new_from_sea_level"].isna().any():

    ref_events = df.dropna(subset=["z_new_from_sea_level"]).copy()
    ref_events = ref_events[ref_events["preferred"] == True]
    target_events = df[df["z_new_from_sea_level"].isna()].copy()

    # KD-tree from reference events
    ref_tree = cKDTree(ref_events[['x', 'y']].values)

    p = 2
    max_distance = 500
    epsilon = 1e-6

    for idx, row in target_events.iterrows():

        dists, idxs = ref_tree.query(
            [row['x'], row['y']],
            k=5,
            distance_upper_bound=max_distance
        )
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)

        valid = dists < max_distance

        if not np.any(valid):
            # No neighbors: keep original depth and assign reasonable uncertainty
            target_events.at[idx, "z_new_from_sea_level"] = row["z_ori_from_sea_level"]
            target_events.at[idx, "z_new_std"] = 1.0
            target_events.at[idx, "Author_new"] = "S-P Relative Reloc"
            continue

        # Filter valid neighbors
        dists = dists[valid]
        idxs = idxs[valid]

        # IDW weights
        weights = 1 / (dists**p + epsilon)
        W = np.sum(weights)

        # Reference subset
        ref_subset = ref_events.iloc[idxs]

        # Compute delta_z of neighbors
        delta_z = ref_subset["z_new_from_sea_level"].values - ref_subset["z_ori_from_sea_level"].values

        # sea level difference
        sea_level = abs(row["z_ori_from_surface"] - row["z_ori_from_sea_level"])

        # New relocated depth
        delta_mean = np.sum(weights * delta_z) / W
        new_z_from_sea_level = row["z_ori_from_sea_level"] + delta_mean
        new_z_from_surface = new_z_from_sea_level + sea_level


        target_events.at[idx, "z_new_from_sea_level"] = new_z_from_sea_level
        target_events.at[idx, "z_new_from_surface"] = new_z_from_surface


        # -------- UNCERTAINTY PROPAGATION --------

        # # 1. measurement uncertainty from neighbors
        # sigma_i = ref_subset["z_new_std"].values

        # # handle missing uncertainties
        # if np.all(np.isnan(sigma_i)):
        #     sigma_i = np.full_like(sigma_i, 1.0)  # fallback
        # else:
        #     med = np.nanmedian(sigma_i)
        #     sigma_i = np.where(np.isnan(sigma_i), med, sigma_i)

        # sigma_meas = np.sqrt(np.sum((weights**2) * (sigma_i**2)) / (W**2))

        # # 2. scatter among neighbors
        # diff = delta_z - delta_mean
        # scatter = np.sqrt(np.sum(weights * diff**2) / W)

        # # effective sample size
        # neff = (W**2) / np.sum(weights**2)
        # sigma_scatter = scatter / np.sqrt(neff)

        # # Final uncertainty
        # sigma_total = np.sqrt(sigma_meas**2 + sigma_scatter**2)
        # sigma_total = max(sigma_total, 0.05)  # avoid zero

        # target_events.at[idx, "z_new_std"] = sigma_total
        # target_events.at[idx, "Author_new"] = "S-P Relative Reloc"

    # Update dataframe
    df.update(target_events[["ev_id", "z_new_from_sea_level", "z_new_from_surface", "z_new_std", "Author_new"]])

# Save results
df.to_csv(outpath, index=False)
print(df)
