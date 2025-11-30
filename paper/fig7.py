import sys
import os

cmez_repository_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(cmez_repository_path)
import string
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import numpy as np
from scipy.interpolate import griddata
from SPHighRes.plot.depthvstime import *

data_path = os.path.join(cmez_repository_path, "data")

# sphighres_path = os.path.join(data_path,"earthquakes","SPHighRes","SPHighRes.csv")
sphighres_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/z/summary_reloc_standard_30km.csv"
outpath = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/z/summary_reloc_standard_30km_with_region.csv"
stations_path = os.path.join(data_path,"stations","delaware_onlystations_160824.csv")
basement_path =  os.path.join(data_path,"basement","TEXAS_basement_depth_km_AOI.tif")
output_fig = os.path.join(os.path.dirname(__file__),"fig_7.png")

start_lat, start_lon = 31.7, -104.8  # Replace with actual lat/lon
end_lat, end_lon = 31.7, -103.8      # Replace with actual lat/lon
width_deg = 0.2
start_point = (start_lat, start_lon)
end_point = (end_lat, end_lon)
basement_xplot = get_basement_cross_plot_data(basement_path,start_point,
                                              end_point,width_deg) 



df_events = pd.read_csv(sphighres_path) 
df_events = df_events[~(df_events["preferred"]==False)]
# df_events = df_events[df_events["Author_new"]=="S-P Method"]
print(f"Total events: {len(df_events)}")
elev_coords = basement_xplot[["Latitude", "Longitude"]].values
elev_tree = cKDTree(elev_coords)
# Get event coordinates
event_coords = df_events[["latitude", "longitude"]].values
# Query nearest elevation point for each event
distances, indices = elev_tree.query(event_coords, k=1)
# Assign elevation to events
df_events["basement_elevation_from_sea_level"] = basement_xplot.iloc[indices]["Elevation"].values 



def get_region(row):
  if row["basement_elevation_from_sea_level"] <=2 and row["basement_elevation_from_sea_level"] > 1:
    region = "R0"
  elif row["basement_elevation_from_sea_level"] <=3 and row["basement_elevation_from_sea_level"] > 2:
    region = "R1"
  elif row["basement_elevation_from_sea_level"] <=4 and row["basement_elevation_from_sea_level"] > 3:
    region = "R2"
  elif row["basement_elevation_from_sea_level"] <=5 and row["basement_elevation_from_sea_level"] > 4:
    region = "R3"
  elif row["basement_elevation_from_sea_level"] <=6 and row["basement_elevation_from_sea_level"] > 5:
    region = "R4"
  else:
    region = None
  return region
df_events["region"] = df_events.apply(get_region,axis=1)
sp_events = df_events.copy()

df_events.to_csv(outpath,index=False)

fig, axes = plt.subplots(2, 1, figsize=(5, 7))

# First plot: depth vs time with IQR
plot_depth_vs_time_by_region_with_iqr(df_events,
                                      xlim=["2018-01-01", "2024-06-01"],
                                      ylim=[2, 13],
                                      depth_col='z_new_from_sea_level',
                                basement_min=2.0,
                                 basement_max=6.0,
                                      colors=["magenta","blue","green"],
                                      legend=False,
                                      ax=axes[0])
plot_depth_vs_time_by_region_with_iqr(sp_events,
                                      xlim=["2018-01-01", "2024-06-01"],
                                      ylim=[2, 13],
                                      depth_col='z_ori_from_sea_level',
                                basement_min=2.0,
                                 basement_max=6.0,
                                      colors=["magenta","blue","green"],
                                      legend=False,
                                      ax=axes[1])

axes[0].set_xlabel(None,fontsize=9)
axes[0].tick_params(axis='both', labelsize=9)
axes[0].set_ylabel("Depth from sea level (km)",fontsize=9)
axes[0].set_title(None)
axes[0].legend( loc='lower right',ncol=3)
# vertical dashed line in 
axes[0].axvline(pd.to_datetime("2020-03-01"), color="red", 
                linestyle="--", linewidth=2)

axes[1].set_xlabel("Time",fontsize=9)
axes[1].tick_params(axis='both', labelsize=9)
axes[1].set_ylabel("Depth from sea level (km)",fontsize=9)
axes[1].set_title(None)
axes[1].legend( loc='lower right',ncol=3)
axes[1].axvline(pd.to_datetime("2020-03-01"), color="red", 
                linestyle="--", linewidth=2)

# Auto-labeling with letters (a), (b), (c), ...
for n, ax in enumerate(axes):
    ax.annotate(f"({string.ascii_lowercase[n]})",
                xy=(-0.1, 1.05),  # Slightly outside top-left
                xycoords='axes fraction',
                ha='left',
                va='bottom',
                fontsize="large",
                fontweight="normal",
                # bbox=box
                )


fig.tight_layout()


fig.tight_layout()
fig.savefig(output_fig, dpi=300, bbox_inches='tight')
plt.show()