#!/bin/bash
#SBATCH -N 1                
#SBATCH -n 32             
#SBATCH -p normal            
#SBATCH -J sp_bts_z      
#SBATCH -o /groups/igonin/ecastillo/CMEZ-SPHighResCatalog/project/4.bts_z.out

source /groups/igonin/ecastillo/anaconda3/etc/profile.d/conda.sh
conda activate utd

python /groups/igonin/ecastillo/CMEZ-SPHighResCatalog/project/4.bts_z.py
