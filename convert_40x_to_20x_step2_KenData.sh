#!/bin/bash

#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-core=1

module load gcc/9.2.0
module load CUDA/11.3.0
# module load cuDNN/8.2.1/CUDA-11.3
export NCCL_ROOT=/data/zhongz2/nccl_2.15.5-1+cuda11.0_x86_64
export NCCL_DIR=/data/zhongz2/nccl_2.15.5-1+cuda11.0_x86_64
export NCCL_PATH=/data/zhongz2/nccl_2.15.5-1+cuda11.0_x86_64
export NCCL_HOME=$NCCL_ROOT
export CUDA_HOME=/usr/local/CUDA/11.3.0
# export CUDNN_ROOT=/data/zhongz2/cudnn-11.3-linux-x64-v8.2.0.53
# export CUDNN_PATH=${CUDNN_ROOT}
export LD_LIBRARY_PATH=${NCCL_ROOT}/lib:$LD_LIBRARY_PATH
source /data/zhongz2/venv_py38_hf2/bin/activate

export OMP_NUM_THREADS=8

PROJ_NAME=KenData_20240814
CSV_FILENAME="/data/zhongz2/${PROJ_NAME}/40x_items_to_be_converted_1.csv"
CSV_FILENAME="/data/zhongz2/${PROJ_NAME}/40x_items_to_be_converted_2.csv"
CSV_FILENAME="/data/zhongz2/${PROJ_NAME}/40x_items_to_be_converted_3.csv"
CSV_FILENAME="/data/zhongz2/${PROJ_NAME}/40x_items_to_be_converted_4.csv"
SAVE_ROOT="/data/zhongz2/${PROJ_NAME}/images_20x/"
IMAGE_EXT=".ndpi"

if [ ! -d ${SAVE_ROOT} ]; then
  mkdir -p ${SAVE_ROOT}
fi

srun python convert_40x_to_20x_step2.py ${CSV_FILENAME} ${SAVE_ROOT} ${IMAGE_EXT}

exit;

sbatch --job-name TCGA --partition=multinode --ntasks=32 --time=108:00:00 --gres=lscratch:100 --mem=64gb convert_40x_to_20x_step2.sh
sbatch --job-name TCGA --partition=quick --ntasks=64 --time=04:00:00 --gres=lscratch:100 --mem=64gb convert_40x_to_20x_step2.sh

sbatch --job-name Ken --partition=quick --ntasks=4 --time=04:00:00 --gres=lscratch:100 --mem=64gb \
convert_40x_to_20x_step2_KenData.sh

sbatch --job-name Ken --partition=multinode --ntasks=64 --time=108:00:00 --gres=lscratch:100 --mem=64gb \
convert_40x_to_20x_step2_KenData.sh



