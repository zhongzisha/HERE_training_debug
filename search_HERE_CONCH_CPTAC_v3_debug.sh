#!/bin/bash

#SBATCH --mail-type=FAIL

module load gcc/9.2.0
module load CUDA/11.3.0
# module load cuDNN/8.2.1/CUDA-11.3
export NCCL_ROOT=${DATA_ROOT}/nccl_2.15.5-1+cuda11.0_x86_64
export NCCL_DIR=${DATA_ROOT}/nccl_2.15.5-1+cuda11.0_x86_64
export NCCL_PATH=${DATA_ROOT}/nccl_2.15.5-1+cuda11.0_x86_64
export NCCL_HOME=$NCCL_ROOT
export CUDA_HOME=/usr/local/CUDA/11.3.0
# export CUDNN_ROOT=${DATA_ROOT}/cudnn-11.3-linux-x64-v8.2.0.53
# export CUDNN_PATH=${CUDNN_ROOT}
export LD_LIBRARY_PATH=${NCCL_ROOT}/lib:$LD_LIBRARY_PATH
source ${DATA_ROOT}/venv_py38_hf2/bin/activate

cd /data/zhongz2/temp29/debug



srun python search_HERE_CONCH_CPTAC_v3_debug.py




exit;

sbatch --job-name CPTAC \
--nodes=16 --ntasks-per-node=1 \
--gres=gpu:v100x:1 \
--cpus-per-task=4 --partition=gpu \
--mem=32gb --time=108:00:00 \
search_HERE_CONCH_CPTAC_v3_debug.sh









