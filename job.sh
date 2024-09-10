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

ONLY_STEP1=${1}
GROUP_NAME=${2}

# SAVE_ROOT="/data/zhongz2/temp29/debug/debug_results/ngpus2_accum4_backboneProvGigaPath_dropout0.25/analysis/ST"

# srun python test_deployment_shared_attention_two_images_comparison_v42.py \
#     --save_root ${SAVE_ROOT} \
#     --svs_dir "/data/zhongz2/ST_256/svs" \
#     --patches_dir "/data/zhongz2/ST_256/patches" \
#     --image_ext ".svs" \
#     --backbone "ProvGigaPath" \
#     --ckpt_path "/data/zhongz2/temp29/debug/results/ngpus2_accum4_backboneProvGigaPath_dropout0.25/split_1/snapshot_39.pt" \
#     --cluster_feat_name "feat_before_attention_feat" \
#     --csv_filename "/data/zhongz2/ST_256/all_20231117.xlsx" \
#     --cluster_task_name ${GROUP_NAME} \
#     --cluster_task_index 0 \
#     --num_patches 128 \
#     --only_step1 ${ONLY_STEP1}

SAVE_ROOT="/data/zhongz2/temp29/debug/results_20240724_e100/ngpus2_accum4_backboneCONCH_dropout0.25/analysis/ST_v1"
srun python test_deployment_shared_attention_two_images_comparison_v42.py \
    --save_root ${SAVE_ROOT} \
    --svs_dir "/data/zhongz2/ST_256/svs" \
    --patches_dir "/data/zhongz2/ST_256/patches" \
    --image_ext ".svs" \
    --backbone "CONCH" \
    --ckpt_path "/data/zhongz2/temp29/debug/results_20240724_e100/ngpus2_accum4_backboneCONCH_dropout0.25/split_3/snapshot_53.pt" \
    --cluster_feat_name "feat_before_attention_feat" \
    --csv_filename "/data/zhongz2/ST_256/all_20231117.xlsx" \
    --cluster_task_name ${GROUP_NAME} \
    --cluster_task_index 0 \
    --num_patches 128 \
    --only_step1 ${ONLY_STEP1}

exit;

sbatch --partition=gpu --mem=100G --time=108:00:00 --gres=gpu:v100x:1,lscratch:32 --cpus-per-task=8 --nodes=8 --ntasks-per-node=1 \
    job.sh "yes" "one_patient"

sbatch --partition=gpu --mem=100G --time=108:00:00 --gres=gpu:v100x:1,lscratch:32 --cpus-per-task=8 --nodes=8 --ntasks-per-node=1 \
    job.sh "no" "one_patient"
sbatch --partition=gpu --mem=100G --time=108:00:00 --gres=gpu:k80:1,lscratch:32 --cpus-per-task=8 --nodes=1 --ntasks-per-node=1 \
    job.sh "no" "response_groups"
sbatch --partition=gpu --mem=32G --time=108:00:00 --gres=gpu:k80:1,lscratch:32 --cpus-per-task=2 --nodes=1 --ntasks-per-node=1 \
    job.sh "gen_patches" "one_patient"
sbatch --partition=gpu --mem=32G --time=108:00:00 --gres=gpu:k80:1,lscratch:32 --cpus-per-task=2 --nodes=1 --ntasks-per-node=1 \
    job.sh "gen_patches" "response_groups"











