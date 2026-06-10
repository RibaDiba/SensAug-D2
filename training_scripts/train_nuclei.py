"""Training entry point for the BBBC038 / DSB2018 nuclei comparison.

This is a thin wrapper that reuses the shared `train()` from training_scripts/train.py
but swaps the dataset registration: instead of the tumor pipeline's
`register_instances()` (generic my_dataset_* names + train/val/test subdir layout), it
calls `register_nuclei()`, which registers the isolated `nuclei_dsb2018_*` names from the
COCO JSON produced by scripts/bbbc038_to_coco.py.

Nothing in the tumor path imports this file, and this file never registers the tumor
dataset — the two pipelines stay disjoint.

Usage (run from the repo root):
    python -m training_scripts.train_nuclei \
        --dataset_root data/coco_test \
        --mode baseline
"""

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_scripts.train import train
from register_nuclei import register_nuclei


def parse_args():
    parser = argparse.ArgumentParser(description="SensAug Detectron2 Training — BBBC038 nuclei")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "config" / "config_nuclei.yml"),
        help="Path to the YAML config file (defaults to config/config_nuclei.yml).",
    )
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="COCO root holding {train,val,test}/<split>.json + images. "
                            "In this workspace, data/coco_test matches that layout.")
    parser.add_argument("--num_iterations", type=int, default=None, help="Override SOLVER.MAX_ITER")
    parser.add_argument("--base_model", type=str, default=None, help="Override BASE_MODEL")
    parser.add_argument("--jobname", type=str, default=None, help="Override JOBNAME (used in hook outputs)")
    parser.add_argument("--eval_period", type=int, default=None, help="Override EVAL_PERIOD (IoU eval frequency)")
    parser.add_argument(
        "--mode",
        choices=["sensaug", "baseline"],
        required=True,
        help="sensaug = sensitivity-informed adaptive aug (SensAugHook); "
             "baseline = static augmentations, no sensitivity analysis",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.num_iterations is not None:
        config["SOLVER"]["MAX_ITER"] = args.num_iterations
    if args.base_model is not None:
        config["BASE_MODEL"] = args.base_model
    if args.jobname is not None:
        config["JOBNAME"] = args.jobname
    if args.eval_period is not None:
        config["EVAL_PERIOD"] = args.eval_period
    register_nuclei(args.dataset_root)
    train(config, args.mode)
