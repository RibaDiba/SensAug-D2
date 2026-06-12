#!/bin/bash

#SBATCH --job-name=sensaug-run-4

#SBATCH --account=puchalla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

#SBATCH --output=./logs/sensaug-run-4/%x.%j.out   # STDOUT → ./logs/sensaug-run-2/<jobid>.o3t
#SBATCH --error=./logs/sensaug-run-4/%x.%j.err    # STDERR → ./logs/sensaug-run-2/<jobid>.e3r

# config:
NAME="sensaug-run-4"
ITER=10000
REPO_ROOT="/projects/PUCHALLA/LLP2024/SensAug-D2"
DATASET="${REPO_ROOT}/data/coco_test"      # COCO root with train/ val/ test/

# Compute nodes have no internet: torch.hub must find the pre-downloaded
# InceptionV3 weights (for KID) in this project-local cache.
export TORCH_HOME="${REPO_ROOT}/.torch_cache"

# create a logs dir (must match the --output/--error path above)
mkdir -p ./logs/${NAME}



python ${REPO_ROOT}/training_scripts/train.py \
    --dataset ${DATASET} \
    --jobname ${NAME} \
    --num_iterations ${ITER}
