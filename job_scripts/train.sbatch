#!/bin/bash

#SBATCH --job-name=no-augs-run-1

#SBATCH --account=puchalla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

#SBATCH --output=./logs/no-augs-run-1/%x.%j.out   # STDOUT → ./logs/no-augs-run-1/<jobid>.o3t
#SBATCH --error=./logs/no-augs-run-1/%x.%j.err    # STDERR → ./logs/no-augs-run-1/<jobid>.e3r

# config:
NAME="no-augs-run-1"
ITER=10000
REPO_ROOT="/projects/PUCHALLA/LLP2024/SensAug-D2"
DATASET="${REPO_ROOT}/data/coco_test"      # COCO root with train/ val/ test/

# create a logs dir (must match the --output/--error path above)
mkdir -p ./logs/${NAME}



python ${REPO_ROOT}/training_scripts/train.py \
    --dataset ${DATASET} \
    --jobname ${NAME} \
    --num_iterations ${ITER}
