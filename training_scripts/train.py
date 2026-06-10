import argparse
import sys
from pathlib import Path
import yaml
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

import numpy as np
import os, json, cv2, random, importlib, sys, argparse
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# custom trainers
from trainer import Trainer

def flatten_cfg(d, prefix=""):
    """
    Flatten nested config dict to a list of [KEY.SUBKEY, value, ...] for merge_from_list.
    """
    opts = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            opts.extend(flatten_cfg(v, key))
        else:
            opts.extend([key, v])
    return opts

def register_instances(dataset_path: str) -> None:
    """
    Register train/val/test COCO splits from dataset_path.

    Expected layout:
        dataset_path/
            train/  train.json + images
            val/    val.json   + images
            test/   test.json  + images
    """
    splits = {
        "my_dataset_train": ("train", "train.json"),
        "my_dataset_val":   ("val",   "val.json"),
        "my_dataset_test":  ("test",  "test.json"),
    }
    for name, (split_dir, ann_file) in splits.items():
        split_path = os.path.join(dataset_path, split_dir)
        register_coco_instances(
            name,
            {},
            os.path.join(split_path, ann_file),
            split_path,
        )

def parse_args():
    parser = argparse.ArgumentParser(description="SensAug Detectron2 Training")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "config" / "config.yml"),
        help="Path to the YAML config file"
    )
    parser.add_argument("--num_iterations", type=int, default=None, help="Override SOLVER.MAX_ITER")
    parser.add_argument("--base_model", type=str, default=None, help="Override BASE_MODEL")
    parser.add_argument("--no-augs", action="store_true", help="Disable all live augmentations during training")
    parser.add_argument("--jobname", type=str, default=None, help="Override JOBNAME (used in hook outputs)")
    parser.add_argument("--eval_period", type=int, default=None, help="Override EVAL_PERIOD (IoU eval frequency)")
    parser.add_argument(
        "--mode",
        choices=["sensaug", "baseline"],
        default="sensaug",
        help="sensaug = sensitivity-informed adaptive aug (SensAugHook); "
             "baseline = static augmentations, no sensitivity analysis",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the COCO dataset root containing train/, val/, test/ subdirs "
             "(each with <split>.json + images). Splits are registered as "
             "my_dataset_{train,val,test}.",
    )
    return parser.parse_args()


def train(config, mode):

    train_metadata = MetadataCatalog.get(config["DATASETS"]["TRAIN"][0])
    train_dataset_dicts = DatasetCatalog.get(config["DATASETS"]["TRAIN"][0])

    for ds in config["DATASETS"]["TEST"]:
        MetadataCatalog.get(ds)
        DatasetCatalog.get(ds)

    project_root = Path(__file__).parent.parent

    cfg = get_cfg()
    base_model = config.pop("BASE_MODEL")
    cfg.merge_from_file(model_zoo.get_config_file(base_model))

    job_name = config.pop("JOBNAME", "")
    eval_period = config.pop("EVAL_PERIOD", 100)
    # MODELTYPE is metadata for our pipeline, not a detectron2 cfg key; drop it
    # so it isn't passed to the strict merge_from_list below.
    config.pop("MODELTYPE", None)
    # DATASETS.AUGMENTATION is not a native Detectron2 cfg key; pop it before
    # the strict merge_from_list, then set it as a plain attribute afterward.
    augmentation_enabled = bool(config.get("DATASETS", {}).pop("AUGMENTATION", True))

    # ── Declare SENSAUG namespace before merge_from_list ──────────────────
    # Detectron2's CfgNode is strict: keys must be pre-declared or the merge
    # will raise AssertionError.  We pop the SENSAUG sub-dict, register the
    # namespace with default values, then overwrite with whatever the caller
    # passed in.  This keeps the rest of the config dict clean for merge_from_list.
    from detectron2.config import CfgNode as CN
    sensaug_cfg = config.pop("SENSAUG", {})
    cfg.SENSAUG = CN()
    cfg.SENSAUG.ENABLED         = bool(sensaug_cfg.get("ENABLED",          True))
    cfg.SENSAUG.WARMUP_ITERS    = int(sensaug_cfg.get("WARMUP_ITERS",      1000))
    cfg.SENSAUG.SA_PERIOD       = int(sensaug_cfg.get("SA_PERIOD",         1500))
    cfg.SENSAUG.NUM_LEVELS      = int(sensaug_cfg.get("NUM_LEVELS",        5))
    cfg.SENSAUG.ALPHA_GRID_SIZE = int(sensaug_cfg.get("ALPHA_GRID_SIZE",   11))
    cfg.SENSAUG.LAMBDA          = float(sensaug_cfg.get("LAMBDA",          0.05))
    cfg.SENSAUG.NONE_PROB       = float(sensaug_cfg.get("NONE_PROB",       0.2))
    cfg.SENSAUG.VAL_DATASET     = str(sensaug_cfg.get("VAL_DATASET",       ""))
    cfg.SENSAUG.MAX_SA_IMAGES   = int(sensaug_cfg.get("MAX_SA_IMAGES",     200))
    cfg.SENSAUG.KID_SUBSET_SIZE = int(sensaug_cfg.get("KID_SUBSET_SIZE",   50))
    # ──────────────────────────────────────────────────────────────────────

    cfg.merge_from_list(flatten_cfg(config))

    cfg.JOBNAME = job_name
    cfg.EVAL_PERIOD = eval_period
    cfg.DATASETS.AUGMENTATION = augmentation_enabled
    dir_name = job_name if job_name else Path(base_model).stem
    if cfg.OUTPUT_DIR:
        cfg.OUTPUT_DIR = str(Path(cfg.OUTPUT_DIR) / dir_name)
    else:
        cfg.OUTPUT_DIR = str(project_root / "models" / dir_name)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


    trainer_cls = Trainer
    trainer = trainer_cls(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


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
    if args.no_augs:
        config.setdefault("DATASETS", {})["AUGMENTATION"] = False
    register_instances(args.dataset)
    train(config, args.mode)
