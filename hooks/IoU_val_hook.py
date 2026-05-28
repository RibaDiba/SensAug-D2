import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
from hooks.plot_style import apply_style
import torch
from detectron2.engine.hooks import HookBase
from detectron2.data import build_detection_test_loader, DatasetCatalog
from detectron2.utils.events import get_event_storage


def _masks_iou(mask_a, mask_b):
    """Compute IoU between two binary masks.

    Args:
        mask_a: Binary mask tensor of shape (H, W).
        mask_b: Binary mask tensor of shape (H, W).

    Returns:
        IoU score as a float.
    """
    intersection = (mask_a * mask_b).sum()
    union = mask_a.sum() + mask_b.sum() - intersection
    if union == 0:
        return 0.0
    return (intersection / union).item()


def _greedy_match_iou(gt_masks, pred_masks):
    """Compute mean IoU between GT and predicted masks using greedy best-match.

    For each GT mask, the predicted mask with the highest IoU is selected
    without replacement. Unmatched GT masks receive an IoU of 0.

    Args:
        gt_masks: Tensor of shape (N_gt, H, W), binary ground truth masks.
        pred_masks: Tensor of shape (N_pred, H, W), binary predicted masks.

    Returns:
        Mean IoU across all GT masks (float).
    """
    n_gt = gt_masks.shape[0]
    n_pred = pred_masks.shape[0]

    if n_gt == 0:
        return 0.0
    if n_pred == 0:
        return 0.0

    # Build pairwise IoU matrix (N_gt, N_pred)
    iou_matrix = torch.zeros(n_gt, n_pred)
    for i in range(n_gt):
        for j in range(n_pred):
            iou_matrix[i, j] = _masks_iou(gt_masks[i], pred_masks[j])

    matched_ious = []
    used_preds = set()

    for _ in range(min(n_gt, n_pred)):
        # Mask out already-used predictions
        mask = iou_matrix.clone()
        for j in used_preds:
            mask[:, j] = -1

        # Find the best remaining (gt, pred) pair
        best_idx = mask.argmax()
        gi = best_idx // n_pred
        pj = best_idx % n_pred
        best_val = mask[gi, pj].item()

        if best_val <= 0:
            break

        matched_ious.append(best_val)
        used_preds.add(pj.item() if isinstance(pj, torch.Tensor) else pj)
        iou_matrix[gi, :] = -1  # mark GT as used

    # Unmatched GTs contribute IoU = 0
    total_iou = sum(matched_ious)
    return total_iou / n_gt


def _polygons_to_binary_masks(polygon_segmentations, height, width):
    """Convert COCO polygon segmentations to binary mask tensors.

    Args:
        polygon_segmentations: List of polygon annotations, where each is a
            list of coordinate arrays (COCO segmentation format).
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        Float tensor of shape (N, H, W) with binary masks.
    """
    masks = []
    for polys in polygon_segmentations:
        rles = mask_util.frPyObjects(polys, height, width)
        rle = mask_util.merge(rles)
        mask = mask_util.decode(rle)
        masks.append(mask)
    return torch.tensor(np.stack(masks), dtype=torch.float32)


class IoUHook(HookBase):
    """Detectron2 hook that computes per-image mask IoU on the validation set.

    At every ``eval_period`` iterations, this hook runs inference on the
    validation dataset, computes per-image IoU using greedy mask matching,
    and buckets images by IoU thresholds (>=90, >=75, >=50, <50).

    After training completes, it saves a line plot of the IoU bucket counts
    over iterations and optionally exports the final results as JSON.
    """

    def __init__(
        self,
        output_dir=None,
        save_json: bool = False,
        eval_period: int = 100,
        job_name: str = "",
    ):
        """Initialize the IoU validation hook.

        Args:
            output_dir: Directory to save the IoU plot and optional JSON results.
                Created automatically if it does not exist.
            save_json: If True, export full IoU results as JSON after training.
            eval_period: Run IoU evaluation every this many iterations.
            job_name: Job name used in plot titles and filenames.
        """
        self.eval_period = eval_period
        self.output_dir = output_dir
        self.save_json = save_json
        self.job_name = job_name

        self.training_data = {}

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def after_step(self):
        """Evaluate IoU on the validation set at regular intervals.

        Runs every ``eval_period`` iterations (skipping iteration 0) and
        stores the IoU bucket counts for later visualization.
        """
        storage = get_event_storage()
        iteration = storage.iter

        if iteration % self.eval_period == 0 and iteration != 0:
            results = self._get_IoU()
            self.training_data[iteration] = {
                "count_50": results["dataset_metrics"]["count_50"],
                "count_75": results["dataset_metrics"]["count_75"],
                "count_90": results["dataset_metrics"]["count_90"],
                "count_failed": results["dataset_metrics"]["count_failed"],
            }

    def after_train(self):
        """Save the IoU plot after training finishes.

        If ``save_json`` was set to True, also runs a final IoU evaluation
        and writes the results to a JSON file in ``output_dir/json/``.
        """
        self._save_graph()

        if self.save_json:
            results = self._get_IoU()
            out_path = os.path.join(self.output_dir, "json")
            os.makedirs(out_path, exist_ok=True)
            out_path = os.path.join(out_path, f"{self.job_name}_json_results.json")
            with open(out_path, "w") as f:
                json.dump({"results": results}, f, indent=4)

    def _get_IoU(self):
        """Compute per-image mask IoU on the validation set.

        Runs the model in eval mode over the validation dataloader. For each
        image, ground truth masks are loaded from the dataset catalog and
        converted from polygon format to binary masks. Predicted masks are
        matched to GT masks using greedy best-IoU matching. Each image is
        then bucketed by its mean IoU score.

        Returns:
            Dict with key ``"dataset_metrics"`` containing counts:
                - ``count_50``: images with mean IoU >= 50%
                - ``count_75``: images with mean IoU >= 75%
                - ``count_90``: images with mean IoU >= 90%
                - ``count_failed``: images with mean IoU < 50%
        """
        cfg = self.trainer.cfg
        model = self.trainer.model
        model.eval()

        val_dataset_name = cfg.DATASETS.TEST[1]
        val_loader = build_detection_test_loader(cfg, val_dataset_name)

        # Load GT annotations and index by image_id
        dataset_dicts = DatasetCatalog.get(val_dataset_name)
        gt_by_image_id = {}
        gt_by_file_name = {}
        for d in dataset_dicts:
            gt_by_image_id[d["image_id"]] = d
            gt_by_file_name[d["file_name"]] = d

        count_50 = count_75 = count_90 = count_failed = 0

        with torch.no_grad():
            for inputs in val_loader:
                outputs = model(inputs)
                for inp, out in zip(inputs, outputs):
                    instances = out["instances"].to("cpu")

                    # Look up GT for this image
                    image_id = inp.get("image_id")
                    file_name = inp.get("file_name")
                    gt_dict = gt_by_image_id.get(image_id) or gt_by_file_name.get(file_name)

                    if gt_dict is None or len(gt_dict.get("annotations", [])) == 0:
                        count_failed += 1
                        continue

                    if len(instances) == 0:
                        count_failed += 1
                        continue

                    pred_masks = instances.pred_masks.float()  # (N_pred, H, W)

                    # Convert GT polygon annotations to binary masks
                    # Use original image dims since GT polygons are in that space.
                    orig_h, orig_w = gt_dict["height"], gt_dict["width"]
                    gt_segmentations = [
                        ann["segmentation"] for ann in gt_dict["annotations"]
                    ]
                    gt_masks = _polygons_to_binary_masks(gt_segmentations, orig_h, orig_w)

                    # Resize GT masks to match pred_masks if dimensions differ
                    pred_h, pred_w = pred_masks.shape[1], pred_masks.shape[2]
                    if gt_masks.shape[1] != pred_h or gt_masks.shape[2] != pred_w:
                        gt_masks = torch.nn.functional.interpolate(
                            gt_masks.unsqueeze(0),
                            size=(pred_h, pred_w),
                            mode="nearest",
                        ).squeeze(0)

                    mean_iou = _greedy_match_iou(gt_masks, pred_masks)

                    if mean_iou >= 0.90:
                        count_90 += 1
                        count_75 += 1
                        count_50 += 1
                    elif mean_iou >= 0.75:
                        count_75 += 1
                        count_50 += 1
                    elif mean_iou >= 0.50:
                        count_50 += 1
                    else:
                        count_failed += 1

        model.train()
        return {
            "dataset_metrics": {
                "count_50": count_50,
                "count_75": count_75,
                "count_90": count_90,
                "count_failed": count_failed,
            }
        }

    def _save_graph(self):
        """Save a line plot of IoU bucket counts over training iterations.

        Plots four lines (IoU >= 90, >= 75, >= 50, and failed < 50) showing
        how many validation images fall into each bucket at each evaluation
        checkpoint. The plot is saved as a PNG in ``output_dir``.
        """
        if not self.training_data:
            return

        apply_style()

        iterations = list(self.training_data.keys())
        count_50 = [v["count_50"] for v in self.training_data.values()]
        count_75 = [v["count_75"] for v in self.training_data.values()]
        count_90 = [v["count_90"] for v in self.training_data.values()]
        count_failed = [v["count_failed"] for v in self.training_data.values()]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(iterations, count_50, marker="o", label="IoU >= 50")
        ax.plot(iterations, count_75, marker="s", label="IoU >= 75")
        ax.plot(iterations, count_90, marker="^", label="IoU >= 90")
        ax.plot(iterations, count_failed, marker="x", label="IoU < 50 (failed)")

        ax.set_title(f"IoU During Training \u2014 Validation Set \u2014 {self.job_name}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Image Count")
        ax.legend()

        save_path = os.path.join(self.output_dir, f"{self.job_name}_IoU.png")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
