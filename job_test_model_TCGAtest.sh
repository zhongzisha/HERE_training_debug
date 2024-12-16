#!/bin/bash

#SBATCH --mail-type=FAIL

if [ "$CLUSTER_NAME" == "FRCE" ]; then
    source $FRCE_DATA_ROOT/anaconda3/bin/activate th23
    module load cuda/11.8
    module load cudnn/8.8.3-cuda11 
else
    source /data/zhongz2/anaconda3/bin/activate th24
    module load CUDA/12.1
    module load cuDNN/8.9.2/CUDA-12
    module load gcc/11.3.0   
fi

cd /data/zhongz2/temp29/debug

MODEL_NAME=${1}  # "UNI"
CSV_PATH=${2}  # "./splits/test-0.csv"
RESULT_DIR=${3}  # 
HIDARE_CHECKPOINT=${4}

IMAGE_EXT=".svs"
DATA_H5_DIR="/data/zhongz2/tcga/TCGA-ALL2_256/patches"
DATA_SLIDE_DIR="/data/zhongz2/tcga/TCGA-ALL2_256/svs"
hidare_method_postfix=${MODEL_NAME}
FEATS_DIR=${RESULT_DIR}  # /lscratch/$SLURM_JOB_ID/HERE_${MODEL_NAME}
# FEATS_DIR=/data/zhongz2/download/${SAVE_PREFIX}/${MODEL_NAME}
# mkdir -p $FEATS_DIR
# ln -sf /data/zhongz2/tcga/TCGA-ALL2_256/feats/${MODEL_NAME}/pt_files ${FEATS_DIR}

# if [ ${hidare_method_postfix} == "mobilenetv3" ]; then
#     BEST_SPLIT=3
#     BEST_EPOCH=32
# fi
# if [ ${hidare_method_postfix} == "CLIP" ]; then
#     BEST_SPLIT=1
#     BEST_EPOCH=97
# fi
# if [ ${hidare_method_postfix} == "PLIP" ]; then
#     BEST_SPLIT=3
#     BEST_EPOCH=66
# fi
# if [ ${hidare_method_postfix} == "ProvGigaPath" ]; then
#     BEST_SPLIT=1
#     BEST_EPOCH=39
# fi
# if [ ${hidare_method_postfix} == "CONCH" ]; then
#     BEST_SPLIT=3
#     BEST_EPOCH=53
# fi
# if [ ${hidare_method_postfix} == "UNI" ]; then
#     BEST_SPLIT=3
#     BEST_EPOCH=58
# fi
# HIDARE_CHECKPOINT=/data/zhongz2/temp29/debug/results_20240724_e100/ngpus2_accum4_backbone${hidare_method_postfix}_dropout0.25/split_${BEST_SPLIT}/snapshot_${BEST_EPOCH}.pt



python test_model.py \
--data_h5_dir ${DATA_H5_DIR} \
--data_slide_dir ${DATA_SLIDE_DIR} \
--csv_path ${CSV_PATH} \
--feat_dir ${FEATS_DIR} \
--batch_size 256 \
--slide_ext ${IMAGE_EXT} \
--model_name ${MODEL_NAME} \
--hidare_checkpoint ${HIDARE_CHECKPOINT}



exit;


for MODEL_NAME in "CONCH"; do  # "UNI" "CONCH" "ProvGigaPath"
for SPLIT in 3; do  # 0 1 2 3 4


CSV_PATH="./splits/test-${SPLIT}.csv"

for HIDARE_CHECKPOINT in `ls /data/zhongz2/temp29/debug/results_20241215_e100_debug/ngpus2_accum4_backbone${MODEL_NAME}_dropout0.25/split_${SPLIT}/snapshot_*.pt`; do

RESULT_DIR=${HIDARE_CHECKPOINT::-3}
echo $RESULT_DIR
mkdir -p $RESULT_DIR
if [ -d "${RESULT_DIR}/pred_files" ]; then
    continue
fi

if [ ! -d "${RESULT_DIR}/pt_files" ]; then
    ln -sf /data/zhongz2/tcga/TCGA-ALL2_256/feats/${MODEL_NAME}/pt_files ${RESULT_DIR}
fi

sbatch --job-name TEST --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --partition=gpu \
--gres=gpu:v100x:1,lscratch:100 --mem=64gb --time=24:00:00 \
job_test_model_TCGAtest.sh ${MODEL_NAME} ${CSV_PATH} ${RESULT_DIR} ${HIDARE_CHECKPOINT}
sleep 1

done
done
done

exit;

MODEL_NAME="CONCH"
SPLIT=3
CSV_PATH="./splits/trainval-${SPLIT}.csv"
SAVE_PREFIX="TCGA_trainval${SPLIT}"



