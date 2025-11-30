from SPHighRes.core.event.stations import Stations
from SPHighRes.core.event.events import get_texnet_high_resolution_catalog
import pandas as pd
import os

rmax = [1,2,3,4,5]
# region = [-104.84329,-103.79942,31.39610,31.91505]
# highres_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/HighRes/origin.csv"
highres_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/HighRes/origin_all.csv"
stations_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/stations/delaware_onlystations_160824.csv"
picks_path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/HighRes/picks.db"
output_folder = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/sp"



catalog = get_texnet_high_resolution_catalog(highres_path,xy_epsg=4326,
                        author="HighRes_CMEZ",
                        # region_lims=region
                        )
stations = pd.read_csv(stations_path)
stations["sta_id"] = stations.index

# stations = stations[stations["station"].isin(["PB35","PB36","PB28","PB37",
#                                               "WB03","SA02","PB24","PB04",
#                                               "PB16"
#                                               ])]

stations = Stations(stations,xy_epsg=4326,author="Delaware_Stations")

for r in rmax:
    r_folder_path = os.path.join(output_folder,f"{r}_km")
    stations.get_events_by_sp(catalog,rmax=r ,
                            picks_path= picks_path,
                            output_folder=r_folder_path)