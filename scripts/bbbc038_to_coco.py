"""Convert the BBBC038 / DSB2018 (stage1_train) nuclei dataset to COCO polygon JSON.

This is a standalone, offline preprocessing script. It is run once, writes JSON, and
exits. It imports nothing from the training code and is never imported by it, so its
BBBC038-specific logic cannot leak into the tumor path.

Source layout (BBBC038 stage1_train), one folder per sample:

    <input>/
        <sample_id>/
            images/<sample_id>.png      # the RGB(A) microscopy image
            masks/<nucleus_id>.png      # one binary PNG per nucleus instance

Output (COCO instance-segmentation, polygon segmentation). The directory tree mirrors the
repo's toy dataset (scripts/generate_toy_dataset.py): one subdir per split holding its
images (copied in, flat) plus a <split>.json whose `file_name` is the bare image name.

    <output>/
        train/  <id>.png …  train.json
        val/    <id>.png …  val.json
        test/   <id>.png …  test.json
        viz_*.png                       # optional overlays (--viz N)

Each nucleus PNG becomes exactly one annotation; disconnected contours within a single
PNG are kept together as a multi-polygon segmentation. BBBC038 image filenames are
globally-unique hashes, so flattening them into one dir per split is collision-free.

Usage:
    python scripts/bbbc038_to_coco.py --input data/bbbc038/stage1_train \
        --output data/bbbc038/coco --viz 5
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# Fixed dataset identity. Matches the existing toy/tumor convention (category_id starts
# at 1, single foreground class). These are NOT CLI args on purpose: a comparison run
# cannot accidentally alias or overwrite the tumor dataset's name.
CATEGORY_ID = 1
CATEGORY_NAME = "nucleus"
MIN_CONTOUR_POINTS = 3  # fewer than 3 points is a degenerate (non-area) polygon


def assert_bbbc038_layout(input_dir: Path) -> list[str]:
    """Verify `input_dir` matches the BBBC038 stage1 layout and return sorted sample ids.

    Raises ValueError immediately if the structure does not match, so pointing this
    script at non-BBBC038 data (e.g. the tumor set, which has no per-instance masks/)
    hard-fails instead of silently producing garbage.
    """
    if not input_dir.is_dir():
        raise ValueError(f"--input is not a directory: {input_dir}")

    sample_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    if not sample_dirs:
        raise ValueError(f"No sample sub-folders found under {input_dir}")

    sample_ids = []
    for sample in sample_dirs:
        images_dir = sample / "images"
        masks_dir = sample / "masks"
        if not images_dir.is_dir() or not masks_dir.is_dir():
            raise ValueError(
                f"{sample} does not match BBBC038 layout: expected 'images/' and "
                f"'masks/' subfolders. This script only converts BBBC038 stage1 data."
            )
        image_pngs = sorted(images_dir.glob("*.png"))
        mask_pngs = sorted(masks_dir.glob("*.png"))
        if len(image_pngs) != 1:
            raise ValueError(
                f"{images_dir} should contain exactly one image PNG, found "
                f"{len(image_pngs)}."
            )
        if len(mask_pngs) == 0:
            raise ValueError(f"{masks_dir} contains no per-nucleus mask PNGs.")
        sample_ids.append(sample.name)

    return sample_ids


def load_binary_mask(mask_path: Path) -> np.ndarray:
    """Load a per-nucleus mask PNG as a single-channel uint8 binary mask.

    BBBC038 masks appear as 8-bit, 16-bit, and occasionally RGBA PNGs; collapse all of
    them to a single channel and threshold to {0, 1} before contouring.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        # RGBA / RGB: any non-zero channel marks the nucleus.
        mask = mask.max(axis=2)
    return (mask > 0).astype(np.uint8)


def mask_to_annotation(mask: np.ndarray, ann_id: int, image_id: int):
    """Turn one binary nucleus mask into a single COCO annotation (multi-polygon).

    Returns None if the mask yields no valid polygon (empty / degenerate).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    area = 0.0
    xs, ys = [], []
    for contour in contours:
        if len(contour) < MIN_CONTOUR_POINTS:
            continue  # drop degenerate contours
        poly = contour.reshape(-1, 2)
        segmentation.append([float(v) for v in poly.flatten()])
        area += float(cv2.contourArea(contour))
        xs.extend(poly[:, 0].tolist())
        ys.extend(poly[:, 1].tolist())

    if not segmentation:
        return None

    x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
    bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]

    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": CATEGORY_ID,
        "segmentation": segmentation,  # list of polygons (multi-polygon) -> one instance
        "area": area,
        "bbox": bbox,
        "iscrowd": 0,
    }


def convert_samples(input_dir: Path, sample_ids: list[str]):
    """Convert every sample folder to COCO images/annotations entries.

    Returns (images, annotations, stats) where images/annotations are keyed so they can
    later be partitioned by split. `image_id` is the index into the sorted sample list.
    """
    images = {}        # sample_id -> image entry (file_name is the bare image name)
    src_paths = {}     # sample_id -> source image Path (to copy into the split dir)
    annotations = {}   # sample_id -> list of annotation entries
    ann_id = 1
    skipped_masks = 0

    for image_id, sample_id in enumerate(sample_ids):
        sample = input_dir / sample_id
        image_path = next((sample / "images").glob("*.png"))
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image {image_path}")
        height, width = img.shape[:2]

        src_paths[sample_id] = image_path
        images[sample_id] = {
            "id": image_id,
            "file_name": image_path.name,  # bare name; image is copied into the split dir
            "width": int(width),
            "height": int(height),
        }

        sample_anns = []
        for mask_path in sorted((sample / "masks").glob("*.png")):
            mask = load_binary_mask(mask_path)
            if mask is None or not mask.any():
                skipped_masks += 1
                continue
            ann = mask_to_annotation(mask, ann_id, image_id)
            if ann is None:
                skipped_masks += 1
                continue
            sample_anns.append(ann)
            ann_id += 1
        annotations[sample_id] = sample_anns

    return images, src_paths, annotations, {"skipped_masks": skipped_masks}


def split_ids(sample_ids: list[str], seed: int):
    """Deterministic 80/10/10 split: sort, seeded shuffle, partition."""
    ids = sorted(sample_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(round(0.8 * n))
    n_val = int(round(0.1 * n))
    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def write_split(output_dir: Path, split_name: str, ids, images, src_paths, annotations) -> Path:
    """Write one split as <output>/<split>/<split>.json with images copied in flat.

    Mirrors the toy dataset tree so it plugs straight into register_coco_instances with
    image_root = the split dir and bare `file_name`s.
    """
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    for i in ids:
        shutil.copy2(src_paths[i], split_dir / images[i]["file_name"])

    coco = {
        "images": [images[i] for i in ids],
        "annotations": [a for i in ids for a in annotations[i]],
        "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME}],
    }
    out_path = split_dir / f"{split_name}.json"
    with open(out_path, "w") as f:
        json.dump(coco, f)
    return out_path


def validate(json_paths: dict[str, Path]):
    """Reload each JSON with pycocotools, assert it parses, and print counts."""
    from pycocotools.coco import COCO

    print("\n── Validation ──────────────────────────────────────────")
    for split_name, path in json_paths.items():
        coco = COCO(str(path))  # raises if the JSON is malformed
        img_ids = coco.getImgIds()
        ann_ids = coco.getAnnIds()
        n_imgs, n_anns = len(img_ids), len(ann_ids)
        mean_inst = (n_anns / n_imgs) if n_imgs else 0.0

        # Every segmentation must be a polygon list, never an RLE dict.
        for ann in coco.loadAnns(ann_ids):
            assert isinstance(ann["segmentation"], list), (
                f"{path.name}: annotation {ann['id']} has non-polygon segmentation "
                f"({type(ann['segmentation'])})"
            )

        print(
            f"  {split_name:5s}: {n_imgs:4d} images, {n_anns:6d} instances, "
            f"{mean_inst:6.1f} mean instances/image  -> {path.name}"
        )
    print("  All segmentations are polygon lists (no RLE).")


def write_overlays(output_dir: Path, src_paths: dict, n: int, images, annotations):
    """Dump N overlay PNGs so polygons can be eyeballed against the source nuclei."""
    sample_ids = list(images.keys())[:n]
    for sample_id in sample_ids:
        img = cv2.imread(str(src_paths[sample_id]))
        if img is None:
            continue
        for ann in annotations[sample_id]:
            for poly in ann["segmentation"]:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
        out_path = output_dir / f"viz_{sample_id}.png"
        cv2.imwrite(str(out_path), img)
    print(f"  Wrote {len(sample_ids)} overlay(s) to {output_dir}/viz_*.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert BBBC038 (DSB2018 stage1_train) nuclei masks to COCO polygon JSON."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to BBBC038 stage1_train dir (one folder per sample).")
    parser.add_argument("--output", type=str, required=True,
                        help="Output root; writes <output>/{train,val,test}/<split>.json + images.")
    parser.add_argument("--seed", type=int, default=42, help="Split shuffle seed.")
    parser.add_argument("--viz", type=int, default=0,
                        help="Dump this many polygon-overlay PNGs for inspection.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    print(f"Verifying BBBC038 layout at {input_dir} ...")
    sample_ids = assert_bbbc038_layout(input_dir)
    print(f"  OK: {len(sample_ids)} samples.")

    print("Converting masks to COCO annotations ...")
    images, src_paths, annotations, stats = convert_samples(input_dir, sample_ids)
    total_anns = sum(len(v) for v in annotations.values())
    print(f"  {total_anns} instances across {len(images)} images "
          f"({stats['skipped_masks']} empty/degenerate masks skipped).")

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = split_ids(sample_ids, args.seed)
    json_paths = {
        name: write_split(output_dir, name, ids, images, src_paths, annotations)
        for name, ids in splits.items()
    }

    validate(json_paths)

    if args.viz > 0:
        write_overlays(output_dir, src_paths, args.viz, images, annotations)

    print("\nDone.")


if __name__ == "__main__":
    main()
