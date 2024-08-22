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
# export OMP_NUM_THREADS=16

cd /data/zhongz2/temp29/debug/

EXP_NAME=${1}
METHOD=${2}
EPOCH_NUM=${3}
HIDARE_CHECKPOINT=${4}
BACKBONE=${5}

echo $*
echo "Ready?! Go!"

if [ $EXP_NAME == "BCSS" ]; then

  for PATCH_SIZE in 512 256; do
    for RATIO in 0.8 0.5; do
      EXP_NAME1=bcss_${PATCH_SIZE}_${RATIO}
      DATA_ROOT=/data/zhongz2/temp11/bcss_${PATCH_SIZE}_256_${RATIO}_50_False
      SAVE_ROOT=/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801
      python extract_features_patch_retrieval_eval.py \
        --exp_name "${EXP_NAME1}" \
        --method_name "${METHOD}" \
        --patch_label_file ${DATA_ROOT}/patch_label_file.csv \
        --patch_data_path ${DATA_ROOT}/ALL \
        --codebook_semantic ../search/SISH/checkpoints/codebook_semantic.pt \
        --checkpoint ../search/SISH/checkpoints/model_9.pt \
        --save_filename ${SAVE_ROOT}/${EXP_NAME1}_${METHOD}_${EPOCH_NUM}_feats.pkl \
        --hidare_checkpoint ${HIDARE_CHECKPOINT} \
        --backbone ${BACKBONE}

    done
  done
elif [ $EXP_NAME == "faiss_bins_count_and_size" ]; then

python extract_features_patch_retrieval_eval.py --action "faiss_bins_count_and_size"

else

  if [ $EXP_NAME == "PanNuke" ]; then
    DATA_ROOT=/data/zhongz2/temp_PanNuke/
    SAVE_ROOT=/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801
  elif [ $EXP_NAME == "NuCLS" ]; then
    DATA_ROOT=/data/zhongz2/temp_NuCLS/
    SAVE_ROOT=/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801
  elif [ $EXP_NAME == "kather100k" ]; then
    DATA_ROOT=/data/zhongz2/temp10
    SAVE_ROOT=/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801
  fi
  echo ${EXP_NAME} ${DATA_ROOT} ${SAVE_ROOT}

  python extract_features_patch_retrieval_eval.py \
    --exp_name "${EXP_NAME}" \
    --method_name "${METHOD}" \
    --patch_label_file ${DATA_ROOT}/patch_label_file.csv \
    --patch_data_path ${DATA_ROOT}/ALL \
    --codebook_semantic ../search/SISH/checkpoints/codebook_semantic.pt \
    --checkpoint ../search/SISH/checkpoints/model_9.pt \
    --save_filename ${SAVE_ROOT}/${EXP_NAME}_${METHOD}_${EPOCH_NUM}_feats.pkl \
    --hidare_checkpoint ${HIDARE_CHECKPOINT} \
    --backbone ${BACKBONE}
fi

exit

for EXP_NAME in "BCSS" "NuCLS" "PanNuke" "kather100k"; do 
  for hidare_method_postfix in "ProvGigaPath"; do
    if [ ${hidare_method_postfix} == "mobilenetv3" ]; then
        BEST_SPLIT=1
        BEST_EPOCH=57
    fi
    if [ ${hidare_method_postfix} == "CLIP" ]; then
        BEST_SPLIT=3
        BEST_EPOCH=99
    fi
    if [ ${hidare_method_postfix} == "PLIP" ]; then
        BEST_SPLIT=1
        BEST_EPOCH=95
    fi
    if [ ${hidare_method_postfix} == "PLIP_RetrainedV14" ]; then
        BEST_SPLIT=1
        BEST_EPOCH=82
    fi
    if [ ${hidare_method_postfix} == "ProvGigaPath" ]; then
        BEST_SPLIT=1
        BEST_EPOCH=39
    fi
    HIDARE_CHECKPOINT=/data/zhongz2/results_histo256_generated7fp_hf_TCGA-ALL2_32_2gpus/adam_RegNormNone_Encoderimagenet${hidare_method_postfix}_CLSLOSSweighted_ce_accum4_wd1e-4_reguNone1e-4/split_${BEST_SPLIT}/snapshot_${BEST_EPOCH}.pt
    HIDARE_CHECKPOINT=/data/zhongz2/temp29/debug/results/ngpus2_accum4_backbone${hidare_method_postfix}_dropout0.25/split_${BEST_SPLIT}/snapshot_${BEST_EPOCH}.pt
    METHOD="ProvGigaPath"
    METHOD="CONCH"
    METHOD="HiDARE_ProvGigaPath"
    EPOCH_NUM=0  # this is for check PLIP_Retrained*
    echo "EXP_NAME=${EXP_NAME}"
    echo "METHOD=${METHOD}"
    echo "EPOCH_NUM=${EPOCH_NUM}"
    echo "HIDARE_CHECKPOINT=${HIDARE_CHECKPOINT}"

    sbatch --job-name ${METHOD} --time=108:00:00 --mem=32G --cpus-per-task=4 --partition=gpu --gres=gpu:v100x:1 \
      job_BCSS_PLIP_Retrained.sh ${EXP_NAME} ${METHOD} ${EPOCH_NUM} ${HIDARE_CHECKPOINT}

  done
done






# for debug
EXP_NAME="BCSS"
METHOD="HiDARE_mobilenetv3"
EPOCH_NUM=0
hidare_method_postfix="mobilenetv3"
BEST_SPLIT=1
BEST_EPOCH=39
HIDARE_CHECKPOINT=/data/zhongz2/temp29/debug/results_20240724_e100/ngpus2_accum4_backbone${hidare_method_postfix}_dropout0.25/split_${BEST_SPLIT}/snapshot_${BEST_EPOCH}.pt
PATCH_SIZE=512
RATIO=0.8
EXP_NAME1=bcss_${PATCH_SIZE}_${RATIO}
DATA_ROOT=/data/zhongz2/temp11/bcss_${PATCH_SIZE}_256_${RATIO}_50_False
SAVE_ROOT=/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801
python extract_features_patch_retrieval_eval.py \
  --exp_name "${EXP_NAME1}" \
  --method_name "${METHOD}" \
  --patch_label_file ${DATA_ROOT}/patch_label_file.csv \
  --patch_data_path ${DATA_ROOT}/ALL \
  --codebook_semantic ../search/SISH/checkpoints/codebook_semantic.pt \
  --checkpoint ../search/SISH/checkpoints/model_9.pt \
  --save_filename ${SAVE_ROOT}/${EXP_NAME1}_${METHOD}_${EPOCH_NUM}_feats.pkl \
  --hidare_checkpoint ${HIDARE_CHECKPOINT}


EXP_NAME="kather100k"
METHOD="HiDARE_mobilenetv3"
EPOCH_NUM=0
hidare_method_postfix="mobilenetv3"
      BEST_SPLIT=3
      BEST_EPOCH=32
HIDARE_CHECKPOINT=/data/zhongz2/temp29/debug/results/ngpus2_accum4_backbone${hidare_method_postfix}_dropout0.25/split_${BEST_SPLIT}/snapshot_${BEST_EPOCH}.pt
PATCH_SIZE=512
RATIO=0.8
EXP_NAME1=bcss_${PATCH_SIZE}_${RATIO}
    DATA_ROOT=/data/zhongz2/temp10
    SAVE_ROOT=/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801
  python extract_features_patch_retrieval_eval.py \
    --exp_name "${EXP_NAME}" \
    --method_name "${METHOD}" \
    --patch_label_file ${DATA_ROOT}/patch_label_file.csv \
    --patch_data_path ${DATA_ROOT}/ALL \
    --codebook_semantic ../search/SISH/checkpoints/codebook_semantic.pt \
    --checkpoint ../search/SISH/checkpoints/model_9.pt \
    --save_filename ${SAVE_ROOT}/${EXP_NAME}_${METHOD}_${EPOCH_NUM}_feats.pkl \
    --hidare_checkpoint ${HIDARE_CHECKPOINT}

# submit jobs
for EXP_NAME in "BCSS" "NuCLS" "PanNuke" "kather100k"; do 
for METHOD in "Yottixel" "RetCCL" "MobileNetV3" "DenseNet121" "CLIP" "PLIP" "HIPT" "ProvGigaPath" "CONCH"; do
EPOCH_NUM=0
hidare_method_postfix="ProvGigaPath"
BEST_SPLIT=1
BEST_EPOCH=39
BACKBONE="ProvGigaPath"
HIDARE_CHECKPOINT=/data/zhongz2/temp29/debug/results/ngpus2_accum4_backbone${hidare_method_postfix}_dropout0.25/split_${BEST_SPLIT}/snapshot_${BEST_EPOCH}.pt
sbatch --job-name ${METHOD} --time=108:00:00 --mem=100G --cpus-per-task=4 --partition=gpu --gres=gpu:v100x:1 \
      job_BCSS_PLIP_Retrained.sh ${EXP_NAME} ${METHOD} ${EPOCH_NUM} ${HIDARE_CHECKPOINT} ${BACKBONE}
done
done

# submit HERE jobs
for EXP_NAME in "BCSS" "NuCLS" "PanNuke" "kather100k"; do  # "BCSS" "NuCLS" "PanNuke" 
for hidare_method_postfix in "CONCH"; do  # "mobilenetv3" "CLIP" "PLIP" "ProvGigaPath" "CONCH"
  if [ ${hidare_method_postfix} == "mobilenetv3" ]; then
      BEST_SPLIT=3
      BEST_EPOCH=32
  fi
  if [ ${hidare_method_postfix} == "CLIP" ]; then
      BEST_SPLIT=1
      BEST_EPOCH=97
  fi
  if [ ${hidare_method_postfix} == "PLIP" ]; then
      BEST_SPLIT=3
      BEST_EPOCH=66
  fi
  if [ ${hidare_method_postfix} == "ProvGigaPath" ]; then
      BEST_SPLIT=1
      BEST_EPOCH=39
  fi
  if [ ${hidare_method_postfix} == "CONCH" ]; then
      BEST_SPLIT=3
      BEST_EPOCH=53
  fi
  HIDARE_CHECKPOINT=/data/zhongz2/temp29/debug/results_20240724_e100/ngpus2_accum4_backbone${hidare_method_postfix}_dropout0.25/split_${BEST_SPLIT}/snapshot_${BEST_EPOCH}.pt
  
  METHOD="HiDARE_${hidare_method_postfix}"
  EPOCH_NUM=0  # this is for check PLIP_Retrained*
sbatch --job-name ${METHOD} --time=108:00:00 --mem=100G --cpus-per-task=4 --partition=gpu --gres=gpu:v100x:1 \
      job_BCSS_PLIP_Retrained.sh ${EXP_NAME} ${METHOD} ${EPOCH_NUM} ${HIDARE_CHECKPOINT} ${hidare_method_postfix}


done
done





sbatch --job-name stats --time=108:00:00 --mem=800G --cpus-per-task=8 --partition=largemem \
      job_BCSS_PLIP_Retrained.sh "faiss_bins_count_and_size" "None" 0 "None"


