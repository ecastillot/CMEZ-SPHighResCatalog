from SPHighRes.core.bootstrap.bootstrap import eq_depths
import pandas as pd

vpvs_path = '/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/vpvs/vpvs_all_stations.h5'
s_p_df_path = '/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/sp/3_km/sp_data.csv'
vp_folder = '/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/vp/regions'
save_folder = '/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/z'
s_p_df = pd.read_csv(s_p_df_path)
# vpvs_df = pd.read_csv(vpvs_path)

with pd.HDFStore(vpvs_path, mode="r") as store:
  keys = [k for k in store.keys()] 
# load and concatenate
dfs = []
for key in keys:
    try:
        df = pd.read_hdf(vpvs_path, key=key)
        dfs.append(df)
    except Exception as e:
        print(f"Error with {key}: {e}")

# final dataframe
vpvs_df = pd.concat(dfs, ignore_index=True)


print(vpvs_df)
print('#######')
# exit()
eq_depths(s_p_df,vpvs_df,vp_folder,save_folder,
            n_boot = 1000,
            n_sample_use= 1000,
            # n_boot = 1000,
            # n_sample_use= 1000,
            trend_sigma=1,
            radii_km=[30],
            # station_list=['PB36'],
            station_list=['PB37','PB28','PB35','PB36',
                        'SA02','PB24','WB03',
                          'PB26','PB31',
                        # 'PB04', 'PB16'
            ],
            radii_interevent_km=30,
            bootstrap_details_folder='/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/paper/bootstrap_details/'
            )

# def eq_depths(
#     s_p_df,
#     vpvs_df,
#     vp_folder,
#     save_folder,
#     depths=np.linspace(-1.2, 20, 200),
#     n_boot: int = 500,
#     trend_sigma: float = 1,
#     n_sample_use: int = 50,
#     station_list=None,
#     radii_km=None,
#     vpvs_bin=(1.4, 2.4, 0.025),  # min, max, bin width
#     depth_bin=(0, 15, 1),  # min, max, bin width
#     epsilon=0.01,
#     random_state=None
# ):
