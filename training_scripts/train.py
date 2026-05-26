import argparse
import yaml

# imports
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
from pathlib import Path
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

# custom trainer with hooks 
from hooks.trainer import Trainer

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
    parser.add_argument("--model_name", type=str, default=None, help="Override BASE_MODEL")
    return parser.parse_args()


def train(config):

    train_metadata = MetadataCatalog.get(config["DATASETS"]["TRAIN"][0])
    train_dataset_dicts = DatasetCatalog.get(config["DATASETS"]["TRAIN"][0])

    for ds in config["DATASETS"]["TEST"]:
        MetadataCatalog.get(ds)
        DatasetCatalog.get(ds)

    project_root = Path(__file__).parent.parent

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config.pop("BASE_MODEL")))

    model_name = config.pop("MODELNAME")
    cfg.merge_from_list(flatten_cfg(config))

    cfg.MODELNAME = model_name
    cfg.OUTPUT_DIR = str(project_root / "models" / cfg.MODELNAME)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.num_iterations is not None:
        config["SOLVER"]["MAX_ITER"] = args.num_iterations
    if args.model_name is not None:
        config["BASE_MODEL"] = args.model_name
    train(config)
