from SPHighRes.core.event.stations import Stations
from SPHighRes.core.event.events import get_texnet_high_resolution_catalog
import pandas as pd
import os
import numpy as np
from SPHighRes.core.vel.vpvs import build_vpvs_dataframe
# from project.reloc_depth.utils import latlon2yx_in_km

radii_km = 30

path = f"/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/z/summary_standard_{radii_km}km.csv"


cross_elv_data = pd.read_csv("/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/terrain/cross_elevation_plot_data_31.7_-104.8_31.7_-103.8.csv")  # Update with actual file path
cross_elv_data["Elevation"] = cross_elv_data["Elevation"]*-1/1e3

highres_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/HighRes/origin_all.csv"
sp_catalog_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/z/summary.csv"

sp_catalog = pd.read_csv(sp_catalog_path)
sp_catalog['radii'] = sp_catalog['key'].str.extract(r'vpvs_(\d+)km').astype(int)
sp_catalog = sp_catalog[sp_catalog['radii'] == radii_km]


sp_columns = {"ev_id":"ev_id","station":"station",
              "preferred":"preferred",
                "median_depth":"z_new_from_surface",
              "std_depth":"z_new_std",
              "iqr_depth":"z_new_iqr",
              "region":"region",
              "median_vpvs":"median_vpvs",
              "std_vpvs":"std_vpvs",
              "median_vp":"median_vp",
              "std_vp":"std_vp",
              }
sp_catalog.rename(columns=sp_columns,inplace=True)
sp_catalog = sp_catalog[list(sp_columns.values())]

# preferred column is a boolean type
sp_catalog["preferred"] = sp_catalog["preferred"].astype(bool)

sp_catalog["Author_new"] = "S-P Method"
sp_catalog["S-P"] = True



# columns = ['ev_id','station',"origin_time","latitude","longitude","depth_km",]

print(sp_catalog.head())

catalog = get_texnet_high_resolution_catalog(highres_path,xy_epsg=4326,
                    author="HighRes_CMEZ",
                    # region_lims=region
                    )
highres_catalog = catalog.data

highres_cols = {"ev_id":"ev_id","origin_time":"origin_time",
                "latitude":"latitude","longitude":"longitude",
                "depth":"z_ori_from_sea_level"}
highres_catalog.rename(columns=highres_cols,inplace=True)
highres_catalog = highres_catalog[list(highres_cols.values())]

highres_catalog["Author_ori"] = "TexNet HighRes"

highres_catalog["z_ori_from_surface"] = highres_catalog["z_ori_from_sea_level"] - np.interp(highres_catalog["longitude"], 
                                            cross_elv_data["Longitude"], 
                                            cross_elv_data["Elevation"])



cat_df = pd.merge(highres_catalog,sp_catalog,on=["ev_id"],how="left")
cat_df["z_new_from_sea_level"] = cat_df["z_new_from_surface"] + np.interp(cat_df["longitude"], 
                                                   cross_elv_data["Longitude"], 
                                                   cross_elv_data["Elevation"])


cat_df.to_csv(path,index=False)
print(cat_df)