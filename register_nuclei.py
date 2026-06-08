"""Opt-in registration for the BBBC038 / DSB2018 nuclei dataset.

This module is imported ONLY by the nuclei training entry point (training_scripts/
train_nuclei.py); the tumor path never references it. Registration happens solely when
`register_nuclei()` is called — nothing fires at import time — so this code cannot affect
any other dataset.

The dataset names are fixed constants (not configurable), keeping the nuclei registration
namespace walled off from the generic `my_dataset_*` names used by the tumor pipeline.
"""

from detectron2.data.datasets import register_coco_instances

NUCLEI_TRAIN = "nuclei_dsb2018_train"
NUCLEI_VAL = "nuclei_dsb2018_val"
NUCLEI_TEST = "nuclei_dsb2018_test"

# Maps each registered name to its split subdir (the toy dataset tree:
# <root>/<split>/<split>.json with images sitting flat beside the JSON).
_SPLITS = {
    NUCLEI_TRAIN: "train",
    NUCLEI_VAL: "val",
    NUCLEI_TEST: "test",
}


def register_nuclei(dataset_root: str) -> None:
    """Register the three nuclei splits as COCO instances.

    Args:
        dataset_root: output of scripts/bbbc038_to_coco.py, holding
                      {train,val,test}/<split>.json with images flat in each split dir.
    """
    import os

    for name, split in _SPLITS.items():
        split_dir = os.path.join(dataset_root, split)
        register_coco_instances(name, {}, os.path.join(split_dir, f"{split}.json"), split_dir)
