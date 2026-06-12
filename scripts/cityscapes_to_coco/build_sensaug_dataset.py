#!/usr/bin/env python3
"""Build a SensAug-D2-ready COCO dataset from Cityscapes.

Two stages, one command:

  1. Convert  - runs the Cityscapes -> COCO instance-seg conversion (the in-repo
     ``main.py``, which reads gtFine/{train,val} ``*_instanceIds.png`` +
     ``*_polygons.json``) to produce two raw COCO files:
         <raw>/instancesonly_filtered_gtFine_train.json
         <raw>/instancesonly_filtered_gtFine_val.json

  2. Reshape  - rewrites those into the exact split-folder layout that
     ``training_scripts/train.py``'s ``register_instances()`` expects:

         <out>/
             train/  train.json + <flat images>
             val/    val.json   + <flat images>
             test/   test.json  + <flat images>

     Each ``<split>.json``'s ``file_name`` is a bare filename relative to its own
     split folder (so Detectron2's ``image_root = <out>/<split>/``), matching the
     repo's bbbc038/toy convention. Cityscapes leftImg8bit names are globally
     unique (``<city>_<seq>_<frame>_leftImg8bit.png``), so flattening is
     collision-free.

Cityscapes withholds test-set labels, so there is no native test split. The 500
official ``val`` images are partitioned (seeded, deterministic) into val + test;
``--val-frac 0.5`` => 250/250.

The 8 instance classes (person, rider, car, truck, bus, train, motorcycle,
bicycle) come straight from the converter.

Usage (from repo root):
    python scripts/cityscapes_to_coco/build_sensaug_dataset.py \
        --datadir data/cityscapes \
        --outdir  data/cityscapes/coco

    # then train:
    python training_scripts/train.py --dataset data/cityscapes/coco --mode sensaug
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

# Make the in-repo converter importable regardless of cwd: when this file is run
# as a script, Python puts its own dir on sys.path[0], so ``import main`` and
# main.py's ``from utils import *`` both resolve to this folder.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def place_image(src: Path, dst: Path, link: bool) -> None:
    """Copy (default) or symlink ``src`` to ``dst``, overwriting any prior file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def index_annotations(coco: dict) -> dict:
    """Group a COCO file's annotations by image id."""
    idx: dict = {}
    for ann in coco["annotations"]:
        idx.setdefault(ann["image_id"], []).append(ann)
    return idx


def write_split(name, images, ann_by_img, categories, datadir, out_split, link):
    """Place each image flat into ``out_split/`` and write ``<name>.json`` with
    bare file_names. Annotations are pulled from ``ann_by_img`` keyed on image id.
    Images whose source PNG is missing on disk are skipped (and reported)."""
    out_split.mkdir(parents=True, exist_ok=True)
    out_images, out_anns, missing = [], [], 0
    for img in images:
        src = Path(datadir) / img["file_name"]          # leftImg8bit/<split>/<city>/x.png
        base = os.path.basename(img["file_name"])
        if not src.exists():
            missing += 1
            continue
        place_image(src, out_split / base, link)
        rec = dict(img)
        rec["file_name"] = base                          # relative to image_root = out_split
        out_images.append(rec)
        out_anns.extend(ann_by_img.get(img["id"], []))
    with open(out_split / f"{name}.json", "w") as f:
        json.dump({"images": out_images, "annotations": out_anns, "categories": categories}, f)
    msg = f"  {name:5s}: {len(out_images):4d} images, {len(out_anns):6d} annotations"
    if missing:
        msg += f"  ({missing} source images missing, skipped)"
    print(msg)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default="data/cityscapes",
                    help="Cityscapes root containing gtFine/ and leftImg8bit/.")
    ap.add_argument("--outdir", default="data/cityscapes/coco",
                    help="Output dataset root; train/ val/ test/ are created inside.")
    ap.add_argument("--raw-dir", default=None,
                    help="Where the intermediate converter JSONs live "
                         "(default: <datadir>/annotations).")
    ap.add_argument("--val-frac", type=float, default=0.5,
                    help="Fraction of the official val split kept as val; the rest "
                         "becomes test. 0.5 => 250/250.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Shuffle seed for the deterministic val/test split.")
    ap.add_argument("--link", action="store_true",
                    help="Symlink images instead of copying (fast, no duplication, "
                         "but symlinks do not transfer cleanly when uploading).")
    ap.add_argument("--skip-convert", action="store_true",
                    help="Skip stage 1; reuse existing raw JSONs in --raw-dir.")
    args = ap.parse_args()

    datadir = args.datadir
    raw_dir = args.raw_dir or os.path.join(datadir, "annotations")
    train_raw = os.path.join(raw_dir, "instancesonly_filtered_gtFine_train.json")
    val_raw = os.path.join(raw_dir, "instancesonly_filtered_gtFine_val.json")

    # ---- Stage 1: convert ----
    if args.skip_convert:
        print(f"[1/2] Skipping conversion; reusing JSONs in {raw_dir}")
    else:
        print(f"[1/2] Converting Cityscapes -> COCO "
              f"(reading {datadir}/gtFine, writing {raw_dir}) ...")
        os.makedirs(raw_dir, exist_ok=True)
        from main import convert_cityscapes_instance_only
        convert_cityscapes_instance_only(datadir, raw_dir)

    for p in (train_raw, val_raw):
        if not os.path.exists(p):
            sys.exit(f"ERROR: expected converter output not found: {p}\n"
                     f"Run without --skip-convert, or fix --raw-dir.")

    # ---- Stage 2: reshape + split ----
    print(f"[2/2] Reshaping into {args.outdir}/{{train,val,test}} ...")
    train_coco = json.load(open(train_raw))
    val_coco = json.load(open(val_raw))

    # Unify categories across both raw files (ids are assigned globally by the
    # converter, so they already agree; this just guards against a class that
    # happens to appear in only one split).
    cat_by_id = {c["id"]: c for c in train_coco["categories"]}
    for c in val_coco["categories"]:
        cat_by_id.setdefault(c["id"], c)
    categories = sorted(cat_by_id.values(), key=lambda c: c["id"])

    out = Path(args.outdir)

    # train: everything
    write_split("train", train_coco["images"], index_annotations(train_coco),
                categories, datadir, out / "train", args.link)

    # val -> val + test (deterministic: sort for stable order, then seeded shuffle)
    val_images = sorted(val_coco["images"], key=lambda im: im["file_name"])
    random.Random(args.seed).shuffle(val_images)
    k = round(len(val_images) * args.val_frac)
    val_idx = index_annotations(val_coco)
    write_split("val", val_images[:k], val_idx, categories, datadir, out / "val", args.link)
    write_split("test", val_images[k:], val_idx, categories, datadir, out / "test", args.link)

    print(f"\nDone -> {args.outdir}/{{train,val,test}}/<split>.json + images")
    print("Train with:")
    print(f"  python training_scripts/train.py --dataset {args.outdir} --mode sensaug")


if __name__ == "__main__":
    main()
