import pandas as pd

path = "/groups/igonin/ecastillo/CMEZ-SPHighResCatalog/data/sp/3_km/sp_data.csv"
data = pd.read_csv(path)

info = {"PB36": {"color":"#26fafa","region":1,"vp/vs":1.72}, 
                  "PB35": {"color":"#2dfa26","region":1,"vp/vs":1.69}, 
                  "PB28": {"color":"#ad16db","region":1,"vp/vs":1.70}, 
                  "PB37": {"color":"#1a3be3","region":1,"vp/vs":1.70}, 
                  "SA02": {"color":"#f1840f","region":2,"vp/vs":1.72}, 
                  "PB24": {"color":"#0ea024","region":3,"vp/vs":1.65}, 
                  "WB03": {"color":"gray","region":3,"vp/vs":1.65}, 

                  "PB26": {"color":"#f50ef1","region":2,"vp/vs":1.70},
                    "PB31": {"color":"#f50e4c","region":3,"vp/vs":1.70},
                    # "PB04": {"color":"red","region":2,"vp/vs":1.72},
                    # "PB16": {"color":"red","region":2,"vp/vs":1.72}
                  }

data["region"] = data["station"].map(lambda x: info[x]["region"] if x in info else -1)
print(data)
data.to_csv(path,index=False)