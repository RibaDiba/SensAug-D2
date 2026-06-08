#!/bin/bash

#SBATCH --job-name=RGB-TEST-9
#SBATCH --partition=pli
#SBATCH --account=puchalla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

#SBATCH --output=./logs/RGB-TEST-9/%x.%j.out   # STDOUT → ./logs/RGB-TEST-9/<jobid>.o3t
#SBATCH --error=./logs/RGB-TEST-9/%x.%j.err    # STDERR → ./logs/RGB-TEST-9/<jobid>.e3r

# config:
NAME="RGB-TEST-9"
MODE="sensaug"                 # sensaug | baseline
ITER=1000
DATASET="../datasets/rgb"      # COCO root with train/ val/ test/
CONFIG="../config/config.yml"

# create a logs dir (must match the --output/--error path above)
mkdir -p ./logs/${NAME}



python ../training_scripts/train.py \
    --config ${CONFIG} \
    --mode ${MODE} \
    --dataset ${DATASET} \
    --jobname ${NAME} \
    --num_iterations ${ITER}
