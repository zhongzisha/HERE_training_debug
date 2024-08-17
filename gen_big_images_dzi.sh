#!/bin/bash

#SBATCH --mail-type=FAIL


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

# export OMP_NUM_THREADS=8

srun python gen_big_images_dzi.py "${1}" "${2}"

exit; 

sbatch --partition=multinode --mem=64G --time=108:00:00 --gres=lscratch:20 \
    --cpus-per-task=1 --nodes=8 --ntasks-per-node=1 gen_big_images_dzi.sh \
    /data/zhongz2/tcga/TCGA-ALL2_256/svs \
    /data/zhongz2/tcga/TCGA-ALL2_256/dzi_images

sbatch --partition=multinode --mem=64G --time=108:00:00 --gres=lscratch:20 \
    --cpus-per-task=1 --nodes=8 --ntasks-per-node=1 gen_big_images_dzi.sh \
    /data/zhongz2/KenData_20240814_256/svs \
    /data/Jiang_Lab/Data/Zisha_Zhong/KenData_20240814_dzi_images

