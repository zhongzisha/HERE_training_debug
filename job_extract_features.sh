#!/bin/bash

##SBATCH --job-name fe
#SBATCH --partition=gpu
#SBATCH --mail-type=FAIL
#SBATCH --cpus-per-task=8
#SBATCH --mem=100gb
##SBATCH --ntasks-per-core=1
#SBATCH --time=108:00:00


if [ "0" == "1" ]; then
if [ ${CLUSTER_NAME} == "Biowulf" ]; then
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
fi
fi

if [ "$CLUSTER_NAME" == "FRCE" ]; then
    source $FRCE_DATA_ROOT/anaconda3/bin/activate th21_ds
    module load cuda/11.8
    module load cudnn/8.8.3-cuda11 
else
    source /data/zhongz2/anaconda3/bin/activate th21_ds
    module load CUDA/12.1
    module load cuDNN/8.9.2/CUDA-12
    module load gcc/11.3.0   
fi
export OMP_NUM_THREADS=8


# 20221220
PROJ_NAME=${1}
DATA_VERSION=${2}
PATCH_SIZE=${3}
MODEL_NAME=${4}
TCGA_ROOT_DIR=${5}
START_IDX=${6}
END_IDX=${7}
BATCH_SIZE=${8}

IMAGE_EXT=".svs"
DIR_TO_COORDS=${TCGA_ROOT_DIR}/${PROJ_NAME}_${PATCH_SIZE}
DATA_DIRECTORY=${TCGA_ROOT_DIR}/${PROJ_NAME}_${PATCH_SIZE}/svs
CSV_FILE_NAME=${TCGA_ROOT_DIR}/${PROJ_NAME}_${PATCH_SIZE}/${DATA_VERSION}/all_with_fpkm_withTIDECytoSig_withMPP_withGene_withCBIO_withCLAM.csv
if [ ! -e ${CSV_FILE_NAME} ]; then
  CSV_FILE_NAME=None
fi
CSV_FILE_NAME=None
FEATURES_DIRECTORY=${DIR_TO_COORDS}/feats/${MODEL_NAME}   ###!!! take care of this

srun python extract_features.py \
--data_h5_dir ${DIR_TO_COORDS} \
--data_slide_dir ${DATA_DIRECTORY} \
--csv_path ${CSV_FILE_NAME} \
--feat_dir ${FEATURES_DIRECTORY} \
--batch_size ${BATCH_SIZE} \
--slide_ext ${IMAGE_EXT} \
--model_name ${MODEL_NAME} \
--start_idx ${START_IDX} \
--end_idx ${END_IDX}


exit;

PROJ_NAME="TCGA-ALL2"
DATA_VERSION=generated7
PATCH_SIZE=256
# MODEL_NAME=CONCH
# MODEL_NAME=ProvGigaPath
MODEL_NAME=PLIP
TCGA_ROOT_DIR=/data/zhongz2/tcga

sbatch --gres=gpu:v100x:1,lscratch:32 \
  --nodes=8 \
    job_extract_features.sh \
    ${PROJ_NAME} ${DATA_VERSION} ${PATCH_SIZE} ${MODEL_NAME} ${TCGA_ROOT_DIR}




PROJ_NAME="KenData"
DATA_VERSION=generated7
PATCH_SIZE=256
MODEL_NAME=ProvGigaPath
TCGA_ROOT_DIR=/data/zhongz2

sbatch --gres=gpu:v100x:1,lscratch:32 \
  --nodes=8 \
    job_extract_features.sh \
    ${PROJ_NAME} ${DATA_VERSION} ${PATCH_SIZE} ${MODEL_NAME} ${TCGA_ROOT_DIR}




PROJ_NAME="ST"
DATA_VERSION=generated7
PATCH_SIZE=256
MODEL_NAME=ProvGigaPath
MODEL_NAME=CONCH
MODEL_NAME=PLIP
TCGA_ROOT_DIR=/data/zhongz2

sbatch --gres=gpu:v100x:1,lscratch:32 \
  --nodes=1 \
    job_extract_features.sh \
    ${PROJ_NAME} ${DATA_VERSION} ${PATCH_SIZE} ${MODEL_NAME} ${TCGA_ROOT_DIR}



PROJ_NAME="KenData"
DATA_VERSION=generated7
PATCH_SIZE=256
MODEL_NAME=CONCH
MODEL_NAME=PLIP
TCGA_ROOT_DIR=/data/zhongz2

sbatch --gres=gpu:v100x:1,lscratch:32 \
  --nodes=1 \
    job_extract_features.sh \
    ${PROJ_NAME} ${DATA_VERSION} ${PATCH_SIZE} ${MODEL_NAME} ${TCGA_ROOT_DIR}






PROJ_NAME="TNBC"
DATA_VERSION=generated7
PATCH_SIZE=256
TCGA_ROOT_DIR=/data/zhongz2

for MODEL_NAME in "CONCH"; do
sbatch --job-name=$MODEL_NAME --gres=gpu:v100x:1,lscratch:32 \
  --nodes=16 \
    job_extract_features.sh \
    ${PROJ_NAME} ${DATA_VERSION} ${PATCH_SIZE} ${MODEL_NAME} ${TCGA_ROOT_DIR}
done



PROJ_NAME="KenData_20240814"
DATA_VERSION=generated7
PATCH_SIZE=256
TCGA_ROOT_DIR=/data/zhongz2

# for MODEL_NAME in "CONCH"; do
# sbatch --job-name=$MODEL_NAME --gres=gpu:v100x:1,lscratch:32 \
#   --ntasks=16 --ntasks-per-node=1 \
#     job_extract_features.sh \
#     ${PROJ_NAME} ${DATA_VERSION} ${PATCH_SIZE} ${MODEL_NAME} ${TCGA_ROOT_DIR}
# done
MODEL_NAME="ProvGigaPath"
sbatch --job-name=p1 --gres=gpu:v100x:1,lscratch:32 \
  --nodes=8 --ntasks-per-node=1 \
    job_extract_features.sh \
    ${PROJ_NAME} ${DATA_VERSION} ${PATCH_SIZE} ${MODEL_NAME} ${TCGA_ROOT_DIR} 0 100 768

sbatch --job-name=p2 --gres=gpu:v100x:1,lscratch:32 \
  --ntasks=4 --ntasks-per-node=1 \
    job_extract_features.sh \
    ${PROJ_NAME} ${DATA_VERSION} ${PATCH_SIZE} ${MODEL_NAME} ${TCGA_ROOT_DIR} 30 60 768 

sbatch --job-name=p3 --gres=gpu:v100x:1,lscratch:32 \
  --ntasks=4 --ntasks-per-node=1 \
    job_extract_features.sh \
    ${PROJ_NAME} ${DATA_VERSION} ${PATCH_SIZE} ${MODEL_NAME} ${TCGA_ROOT_DIR} 60 100 768 

PROJ_NAME="ST_SPOT"
DATA_VERSION=generated7
PATCH_SIZE=256
TCGA_ROOT_DIR=/data/zhongz2

for MODEL_NAME in "ProvGigaPath"; do
sbatch --job-name=$MODEL_NAME --gres=gpu:v100x:1,lscratch:32 \
  --nodes=16 \
    job_extract_features.sh \
    ${PROJ_NAME} ${DATA_VERSION} ${PATCH_SIZE} ${MODEL_NAME} ${TCGA_ROOT_DIR}
done





