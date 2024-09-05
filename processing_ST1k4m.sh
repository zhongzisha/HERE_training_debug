#!/bin/bash

#SBATCH --mail-type=FAIL



if [ "$CLUSTER_NAME" == "FRCE" ]; then
    source $FRCE_DATA_ROOT/anaconda3/bin/activate th21_ds
    module load cuda/11.8
    module load cudnn/8.8.3-cuda11
    MYTMP_DIR=/tmp/zhongz2
else
    source /data/zhongz2/anaconda3/bin/activate th24
    module load CUDA/12.1
    module load cuDNN/8.9.2/CUDA-12
    module load gcc/11.3.0
    MYTMP_DIR=/lscratch/$SLURM_JOB_ID
fi
export PYTHONPATH=`pwd`:$PYTHONPATH



cd /data/zhongz2/temp29/debug

srun python processing_ST1k4m.py

exit;

sbatch --time=24:00:00 --ntasks=32 --ntasks-per-node=1 --partition=multinode --mem=100G --cpus-per-task=2 processing_ST1k4m.sh

