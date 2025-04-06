#!/bin/bash


#SBATCH --mail-type=FAIL


if [ "1" == "1" ]; then
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
  # export LD_LIBRARY_PATH=/data/zhongz2/venv_py38_hf2/lib/python3.8/site-packages/nvidia/cudnn::$LD_LIBRARY_PATH
  # CUDNN_PATH1=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
  # export LD_LIBRARY_PATH=/data/zhongz2/venv_py38_hf2/lib/python3.8/site-packages/nvidia/cudnn::$LD_LIBRARY_PATH
  #python3 -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))"
  if [ "${1}" == "Yottixel" ]; then 
    module load cuDNN/8.9.2/CUDA-11
  fi
fi

cd /data/zhongz2/temp29/debug/

python generate_faiss_index_v2_CPTAC_intersection_HEREYottixel256.py ${1} ${2} ${3}


exit;

sbatch --time=108:00:00 --cpus-per-task=4 --ntasks=1 --partition=norm --ntasks-per-node=1 --mem=100G \
    generate_faiss_index_v2_CPTAC_intersection_HEREYottixel256.sh CPTAC_HEREYottixel CONCH 0








