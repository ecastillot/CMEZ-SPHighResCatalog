import sys
import os

cmez_repository_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(cmez_repository_path)

import pandas as pd
import os
import string
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from SPHighRes.plot.sp import sup_fig_1


data_path = os.path.join(cmez_repository_path, "data")
stations_path = os.path.join(data_path,"stations","delaware_onlystations_160824.csv")
output_path = os.path.join(os.path.dirname(__file__),"sup_fig_t.png")
sp_path = os.path.join(data_path,"sp")
sp_sheng_path = os.path.join(data_path,"sp_sheng")
radii = [1,2,3,4,5]

# custom_palette = {"PB35": "#26fafa", 
#                   "PB36": "#2dfa26", 
#                   "PB28": "#ad16db", 
#                   "PB37": "#1a3be3", 
#                   "WB03": "#ffffff", 
#                   "SA02": "#f1840f", 
#                   "PB24": "#0ea024", 
#                   "PB26": "#f1840f", 
#                   "PB31": "#0ea024", 
#                   "PB04": "red", 
#                   "PB16": "red", 
#                   }

custom_palette = {"PB35": "magenta", 
                  "PB36": "magenta", 
                  "PB28": "magenta", 
                  "PB37": "magenta", 
                  "SA02": "blue", 
                  "PB26": "blue", 
                  "PB31": "green", 
                  "PB24": "green", 
                  "WB03": "green", 
                  "PB04": "red", 
                  "PB16": "red", 
                  }

all_stations = pd.read_csv(stations_path)

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(6, 6, figure=fig)  # 2 rows, 3 columns
gs.update(wspace = 0.3, hspace = 2)

# Define axes with correct positioning
axes = []
axes.append(fig.add_subplot(gs[1:3, 0:2]))  
axes.append(fig.add_subplot(gs[3:5, 0:2]))  
axes.append(fig.add_subplot(gs[0:2, 2:6])) 
axes.append(fig.add_subplot(gs[2:4, 2:6]))  
axes.append(fig.add_subplot(gs[4:6, 2:6]))  

# plt.show()
# exit()

for n,r in enumerate(radii):
  print(r)
  #   catalog_path = f"{sp_path}/{r}_km/catalog_sp_method.db"
  #   picks_path = f"{sp_path}/{r}_km/picks_sp_method.db"


  #   picks = load_from_sqlite(picks_path)
  #   catalog = load_from_sqlite(catalog_path)

  picks_path = os.path.join(sp_path,F"{r}_km", "sp_data.csv")
  picks_path_sheng = os.path.join(sp_sheng_path,F"{r}_km", "sp_data.csv")

  picks = pd.read_csv(picks_path)
  if os.path.exists(picks_path_sheng):
    picks_sheng = pd.read_csv(picks_path_sheng)

    if not picks_sheng.empty:
      picks = pd.concat([picks,picks_sheng],ignore_index=True)

  picks = picks[picks["preferred"]]

  stations_columns = ["network","station","latitude","longitude","elevation"]
  stations = all_stations[stations_columns]

  stations_with_picks = list(set(picks["station"].to_list()))
  order = stations.copy()
  order = order[order["station"].isin(stations_with_picks)]
  order = order.sort_values("longitude",ignore_index=True,ascending=True)
  order = order.drop_duplicates(subset="station")
  order = order["station"].to_list()




  #   catalog,picks = prepare_sp_analysis(catalog,picks,cat_columns_level=0)

  #   picks["ts-tp"] = picks["tt_S"] - picks["tt_P"]
  #   picks['ts-tp'] = picks['ts-tp'].astype(float)
  stats_by_station = picks.groupby('station')['ts-tp'].describe()

  general_palette = dict([(x,"lightgray") for x in order])

  for key,value in custom_palette.items():
    general_palette[key] = value


  ax = sup_fig_1(picks,order=order,
                        palette=general_palette,
                        ylim=(0,2),
                        show=False,
                        ax=axes[n]  # Assign the correct subplot
                        )
  ax.set_xlabel("")
  ax.set_title(f"{r} km",
                  fontdict={"size":10,
                            "weight":"bold"})

  if n>1:
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    text_loc = [0.025, 0.92]
  else:
    text_loc = [0.05, 0.92]
    

  box = dict(boxstyle='round', 
            facecolor='white', 
            alpha=1)
  ax.text(text_loc[0], text_loc[1], 
          f"{string.ascii_lowercase[n]})", 
          horizontalalignment='left', 
          verticalalignment="top", 
        transform=ax.transAxes, 
        fontsize="large", 
        fontweight="normal",
        bbox=box)


# fig.subplots_adjust(bottom=0.2)
# arrow = mpatches.FancyArrow(0.15, 0.05, 0.8, 0, width=0.001, transform=fig.transFigure, color="black")
# fig.patches.append(arrow)

# fig.text(0.15, 0.07, "W", ha="center", va="center", fontsize=12, fontweight="bold")
# fig.text(0.95, 0.07, "E", ha="center", va="center", fontsize=12, fontweight="bold")

plt.tight_layout()  # Improve spacing
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()