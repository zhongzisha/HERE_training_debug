#!/bin/bash

#SBATCH --mail-type=FAIL

module load gcc/9.2.0
module load CUDA/11.8
module load cuDNN/8.9.2/CUDA-11
source /data/zhongz2/venv_py38_hf2/bin/activate


cd /data/zhongz2/temp29/debug

srun python extract_heatmap_CPTAC.py

exit;

sbatch --job-name CPTACheat \
--nodes=64 --ntasks-per-node=1 \
--cpus-per-task=1 --partition=multinode \
--mem=100gb --time=108:00:00 \
extract_heatmap_CPTAC.sh











