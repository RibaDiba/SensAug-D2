#!/bin/bash

#SBATCH --job-name=cityscapes-testing-4

#SBATCH --account=puchalla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

#SBATCH --output=./logs/cityscapes-testing-4/%x.%j.out   # STDOUT → ./logs/cityscapes-testing-4/<jobid>.o3t
#SBATCH --error=./logs/cityscapes-testing-4/%x.%j.err    # STDERR → ./logs/cityscapes-testing-4/<jobid>.e3r

# config:
NAME="cityscapes-testing-4"
ITER=1000
MODE="sensaug"    # sensaug | baseline | no-augs — set per condition. Baked into
                  # this script so sbatch snapshots it at submit time; do NOT
                  # rely on editing the shared config .yml between submissions
                  # (jobs read the yml at START time, so the last edit wins for
                  # every queued job).
REPO_ROOT="/projects/PUCHALLA/LLP2024/SensAug-D2"
DATASET="${REPO_ROOT}/data/cityscapes/coco"      # COCO root with train/ val/ test/

case "${MODE}" in
    sensaug|baseline) MODE_FLAG="--mode ${MODE}" ;;
    no-augs)          MODE_FLAG="--no-augs" ;;
    *) echo "Unknown MODE '${MODE}' (expected sensaug | baseline | no-augs)" >&2; exit 1 ;;
esac

# Compute nodes have no internet: torch.hub must find the pre-downloaded
# InceptionV3 weights (for KID) in this project-local cache.
export TORCH_HOME="${REPO_ROOT}/.torch_cache"

# create a logs dir (must match the --output/--error path above)
mkdir -p ./logs/${NAME}



python ${REPO_ROOT}/training_scripts/train.py \
    --config ${REPO_ROOT}/config/cityscapes_config.yml \
    --dataset ${DATASET} \
    --jobname ${NAME} \
    --num_iterations ${ITER} \
