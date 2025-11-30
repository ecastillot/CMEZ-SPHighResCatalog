#!/bin/bash
#SBATCH -N 1                
#SBATCH -n 64              
#SBATCH -p normal            
#SBATCH -J vpvs     
#SBATCH -o /groups/igonin/ecastillo/CMEZ-SPHighResCatalog/project/3.vpvs.out

source /groups/igonin/ecastillo/anaconda3/etc/profile.d/conda.sh
conda activate utd

python /groups/igonin/ecastillo/CMEZ-SPHighResCatalog/project/3.vpvs.py
