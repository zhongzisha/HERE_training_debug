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
export OMP_NUM_THREADS=8

DATA_VERSION=v0

srun python ST_prediction_exps.py $DATA_VERSION

exit;

sbatch --ntasks=25 --ntask-per-node=1 --partition=gpu --gres=gpu:v100x:1,lscratch:10 --cpus-per-task=4 --time=108:00:00 --mem=100G \
ST_prediction_exps.sh 

sbatch --ntasks=25 --tasks-per-node=1 --partition=multinode --gres=lscratch:10 --cpus-per-task=4 --time=108:00:00 --mem=100G \
ST_prediction_exps.sh 













