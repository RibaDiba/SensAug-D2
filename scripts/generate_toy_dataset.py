"""Generate a toy COCO-format instance segmentation dataset.

Creates random images with colored circles as object instances.
Output layout matches what train.py expects:

    <output_dir>/
        train/  train.json + images
        val/    val.json   + images
        test/   test.json  + images

Usage:
    python scripts/generate_toy_dataset.py --output data/toy
    python scripts/generate_toy_dataset.py --output data/toy --num_train 50 --num_val 10 --num_test 10
"""

import argparse
import json
import os
import random

import cv2
import numpy as np

IMG_W, IMG_H = 256, 256
MIN_RADIUS, MAX_RADIUS = 15, 50
MAX_OBJECTS = 5
CATEGORY = {"id": 1, "name": "circle", "supercategory": "shape"}


def random_color():
    return [random.randint(60, 255) for _ in range(3)]


def generate_image_and_annotations(image_id, ann_id_start):
    img = np.random.randint(20, 80, (IMG_H, IMG_W, 3), dtype=np.uint8)
    annotations = []
    ann_id = ann_id_start
    num_objects = random.randint(1, MAX_OBJECTS)

    for _ in range(num_objects):
        r = random.randint(MIN_RADIUS, MAX_RADIUS)
        cx = random.randint(r + 1, IMG_W - r - 1)
        cy = random.randint(r + 1, IMG_H - r - 1)
        color = random_color()

        cv2.circle(img, (cx, cy), r, color, -1)

        # Build polygon segmentation (circle approximated as polygon)
        angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        poly_x = (cx + r * np.cos(angles)).tolist()
        poly_y = (cy + r * np.sin(angles)).tolist()
        seg = []
        for px, py in zip(poly_x, poly_y):
            seg.extend([round(px, 1), round(py, 1)])

        x_min = max(0, cx - r)
        y_min = max(0, cy - r)
        w = min(IMG_W, cx + r) - x_min
        h = min(IMG_H, cy + r) - y_min

        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": [seg],
            "area": float(np.pi * r * r),
            "bbox": [x_min, y_min, w, h],
            "iscrowd": 0,
        })
        ann_id += 1

    return img, annotations, ann_id


def generate_split(split_dir, num_images):
    os.makedirs(split_dir, exist_ok=True)
    images_info = []
    all_annotations = []
    ann_id = 1

    for i in range(num_images):
        filename = f"{i:04d}.png"
        img, anns, ann_id = generate_image_and_annotations(i, ann_id)
        cv2.imwrite(os.path.join(split_dir, filename), img)

        images_info.append({
            "id": i,
            "file_name": filename,
            "width": IMG_W,
            "height": IMG_H,
        })
        all_annotations.extend(anns)

    coco = {
        "images": images_info,
        "annotations": all_annotations,
        "categories": [CATEGORY],
    }

    split_name = os.path.basename(split_dir)
    json_path = os.path.join(split_dir, f"{split_name}.json")
    with open(json_path, "w") as f:
        json.dump(coco, f)

    print(f"  {split_name}: {num_images} images, {len(all_annotations)} instances -> {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate toy COCO dataset")
    parser.add_argument("--output", type=str, default="data/toy", help="Output directory")
    parser.add_argument("--num_train", type=int, default=30, help="Number of training images")
    parser.add_argument("--num_val", type=int, default=8, help="Number of val images")
    parser.add_argument("--num_test", type=int, default=8, help="Number of test images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Generating toy dataset in {args.output}/")
    generate_split(os.path.join(args.output, "train"), args.num_train)
    generate_split(os.path.join(args.output, "val"), args.num_val)
    generate_split(os.path.join(args.output, "test"), args.num_test)
    print("Done.")


if __name__ == "__main__":
    main()
