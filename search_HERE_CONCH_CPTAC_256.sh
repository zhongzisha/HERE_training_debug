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

 

python search_HERE_CONCH_CPTAC_256.py




exit;

sbatch --job-name CPTAC \
--nodes=1 --ntasks-per-node=1 \
--cpus-per-task=16 --partition=gpu \
--gres=gpu:v100x:1 --mem=100gb --time=108:00:00 \
search_HERE_CONCH_CPTAC_256.sh









