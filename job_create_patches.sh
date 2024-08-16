#!/bin/bash

#SBATCH --mail-type=FAIL

if [ "1" == "1" ]; then
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
fi

cd /data/zhongz2/CLAM

PATCH_SIZE=256
DATA_DIRECTORY=${1}  # svs dir
PRESET=${2}
RESULTS_DIRECTORY="${DATA_DIRECTORY}/../preset_${PRESET}_${PATCH_SIZE}_orignalcode"

srun python create_patches_fp_parallel.py \
--source ${DATA_DIRECTORY} \
--save_dir ${RESULTS_DIRECTORY} \
--patch_size ${PATCH_SIZE} \
--preset ${PRESET}.csv --seg --patch --stitch

exit;


DATA_DIRECTORY=/data/zhongz2/KenData_20240814/svs
PRESET=KenData
sbatch --nodes=32 --ntasks-per-node=1 \
  --cpus-per-task=2 --partition=quick \
  --gres=lscratch:64 --mem=32gb --time=04:00:00 \
  job_create_patches.sh ${DATA_DIRECTORY} ${PRESET}




