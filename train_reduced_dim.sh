#!/bin/bash

#SBATCH --job-name wait
#SBATCH --partition=gpu
#SBATCH --mail-type=FAIL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64gb
#SBATCH --ntasks-per-core=1
#SBATCH --time=108:00:00

if [ "0" == "1" ]; then
  module load CUDA/11.3.0
  module load cuDNN/8.2.1/CUDA-11.3
  export NCCL_ROOT=/data/zhongz2/nccl_2.15.5-1+cuda11.0_x86_64
  export NCCL_DIR=/data/zhongz2/nccl_2.15.5-1+cuda11.0_x86_64
  export NCCL_PATH=/data/zhongz2/nccl_2.15.5-1+cuda11.0_x86_64
  export NCCL_HOME=$NCCL_ROOT
  export CUDA_HOME=/usr/local/CUDA/11.3.0
  export LD_LIBRARY_PATH=${NCCL_ROOT}/lib:$LD_LIBRARY_PATH
  source /data/zhongz2/venv_py38_hf2/bin/activate
fi 
if [ "1" == "1" ]; then
    module load gcc
    source /data/zhongz2/anaconda3/bin/activate th24
fi
export OMP_NUM_THREADS=4

OUTSIDE_TEST_FILENAMES="/data/zhongz2/TransNEO_256/testgenerated7.csv|/data/zhongz2/METABRIC_256/testgenerated7.csv"

NUM_GPUS=${1}
SPLIT_NUM=${2}
ACCUM_ITER=${3}
FINAL_SAVE_ROOT=${4}
BACKBONE=${5}
DROPOUT=${6}
REDUCED_DIM=${7}
TCGA_ROOT_DIR=${8}

SAVE_ROOT=/lscratch/$SLURM_JOB_ID/results/ngpus${NUM_GPUS}_accum${ACCUM_ITER}_backbone${BACKBONE}_dropout${DROPOUT}_rd${REDUCED_DIM}/split_${SPLIT_NUM}
if [ ! -d ${SAVE_ROOT} ]; then
    mkdir -p ${SAVE_ROOT}
fi

# copy data
FEATS_DIR=/lscratch/${SLURM_JOB_ID}/${BACKBONE}/all
if [ ! -d ${FEATS_DIR} ]; then 
    mkdir -p /lscratch/${SLURM_JOB_ID}/${BACKBONE}
    if [ -d ${TCGA_ROOT_DIR}/TCGA-ALL2_256/featsHF/${BACKBONE}/pt_files_train ]; then
        time cp -RL ${TCGA_ROOT_DIR}/TCGA-ALL2_256/featsHF/${BACKBONE}/pt_files_train ${FEATS_DIR}
    else
        time cp -RL ${TCGA_ROOT_DIR}/TCGA-ALL2_256/featsHF/${BACKBONE}/pt_files ${FEATS_DIR}
    fi
    # copy outside data, make sure their filenames are different)
    time cp -RL ${TCGA_ROOT_DIR}/TransNEO_256/featsHF/${BACKBONE}/pt_files/* ${FEATS_DIR}
    time cp -RL ${TCGA_ROOT_DIR}/METABRIC_256/featsHF/${BACKBONE}/pt_files/* ${FEATS_DIR}
    cd ${FEATS_DIR}
    du -sh
fi

while
  PORT=$(shuf -n 1 -i 20080-60080)
  netstat -atun | grep -q "${PORT}"
do
  continue
done

cd /data/zhongz2/temp29/debug/

torchrun \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:${PORT} \
    train_reduced_dim.py \
    --split_num ${SPLIT_NUM} \
    --accum_iter ${ACCUM_ITER} \
    --feats_dir ${FEATS_DIR} \
    --save_root ${SAVE_ROOT} \
    --outside_test_filenames ${OUTSIDE_TEST_FILENAMES} \
    --backbone ${BACKBONE} \
    --dropout ${DROPOUT} \
    --reduced_dim ${REDUCED_DIM} \
    --max_epochs 100

rsync -avh /lscratch/$SLURM_JOB_ID/results/ngpus${NUM_GPUS}_accum${ACCUM_ITER}_backbone${BACKBONE}_dropout${DROPOUT}_rd${REDUCED_DIM} ${FINAL_SAVE_ROOT}

exit;


if [ ! -d "splits" ]; then
    ln -sf /data/zhongz2/TCGA-ALL2_256/generated7/splits2/ splits
fi
NUM_GPUS=2
GRES_STR="--gres=gpu:v100x:${NUM_GPUS},lscratch:700"
FINAL_SAVE_ROOT=/data/zhongz2/temp29/debug/results_20240724_e100
FINAL_SAVE_ROOT=/data/zhongz2/temp29/debug/results_20240724_e100_reducedDim
if [ ! -d ${FINAL_SAVE_ROOT} ]; then
    mkdir -p ${FINAL_SAVE_ROOT}
fi

# "ProvGigaPath" # 521G
# "mobilenetv3" # 535G
# "CLIP" # 174G
# "PLIP" # 174G
# "CONCH" # 174G
# BACKBONES=("CONCH")
# BACKBONES=("mobilenetv3" "CLIP")

BACKBONES=("PLIP")
BACKBONES=("mobilenetv3" "CLIP" "PLIP" "ProvGigaPath" "CONCH")
BACKBONES=("UNI")
TCGA_ROOT_DIR=/data/zhongz2/tcga
DROPOUT=0.25

for SPLIT_NUM in 0 1 2 3 4; do     #  0 1 2 3 4
    for ACCUM_ITER in 4; do # 4 8
        for i in ${!BACKBONES[@]}; do
            for REDUCED_DIM in 128 256 512; do
                BACKBONE=${BACKBONES[${i}]}
                sbatch ${GRES_STR} train_reduced_dim.sh ${NUM_GPUS} ${SPLIT_NUM} ${ACCUM_ITER} ${FINAL_SAVE_ROOT} ${BACKBONE} ${DROPOUT} ${REDUCED_DIM} ${TCGA_ROOT_DIR}
                sleep 2
            done
        done
    done
done

exit;


# example
NUM_GPUS=2
SPLIT_NUM=0
ACCUM_ITER=4
FINAL_SAVE_ROOT=/data/zhongz2/temp29/debug/results_reduced_dim
BACKBONE=UNI
DROPOUT=0.25
REDUCED_DIM=128
TCGA_ROOT_DIR=/data/zhongz2/tcga



