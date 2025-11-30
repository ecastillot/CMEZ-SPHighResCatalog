
import sys
import os
import pandas as pd
from SPHighRes.plot.xy_analysis import compare_highres_vs_nlloc_by_region
cmez_repository_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(cmez_repository_path)

data_path = os.path.join(cmez_repository_path, "data")
cmez_df = os.path.join(data_path,"sp","3_km","sp_data.csv")
cmez_df = pd.read_csv(cmez_df)

high_res_path = os.path.join(data_path,"HighRes","origin.csv")
nlloc_path = os.path.join(data_path,"Nlloc","origin.csv")

print(len(cmez_df), "total picks in CMEZ")

cmez_df = cmez_df[cmez_df["station"].isin(["PB36","PB35",
                                        "PB28","PB37","SA02","PB26",
                                        "PB31","WB03","PB24"])]
cmez_df = cmez_df[cmez_df["preferred"]]
# print(len(cmez_df), "total picks in CMEZ")
cmez_df = cmez_df.drop_duplicates(subset=["ev_id","station"])
station_counts = cmez_df["station"].value_counts()
print("Station counts:", station_counts)

highres_df = pd.read_csv(high_res_path)
nlloc_df = pd.read_csv(nlloc_path)

#drop duplciates
highres_df = highres_df.drop_duplicates(subset=["ev_id"])
nlloc_df = nlloc_df.drop_duplicates(subset=["ev_id"])

# Example usage
fig_path = os.path.join(os.path.dirname(__file__), "xy_analysis.png")
print("Saving figure to:", fig_path)
compare_highres_vs_nlloc_by_region(
    df_high=highres_df,
    df_nlloc=nlloc_df,
    df_sp = cmez_df,
    max_xy=3,
    max_dist=4,
    fig_path=fig_path
)