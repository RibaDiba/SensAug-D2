import numpy as np
import torch
from detectron2.evaluation import DatasetEvaluator
from detectron2.data import DatasetCatalog

from hooks.IoU_val_hook import _polygons_to_binary_masks, _greedy_match_iou
from sensitivity.metrics import _foreground_iou

class PerImageIoUEvaluator(DatasetEvaluator):
    """
    Evaluator that computes per-image mask IoU (using greedy matching)
    and foreground IoU on a dataset.
    """
    def __init__(self, dataset_name: str):
        self._dataset_name = dataset_name
        # Pre-load dataset dicts for ground truth lookup
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self._gt_by_image_id = {}
        self._gt_by_file_name = {}
        for d in dataset_dicts:
            self._gt_by_image_id[d["image_id"]] = d
            self._gt_by_file_name[d["file_name"]] = d
        self.reset()

    def reset(self):
        self._greedy_ious = []
        self._fg_ious = []
        self._count_50 = 0
        self._count_75 = 0
        self._count_90 = 0
        self._count_failed = 0
        self._total_images = 0

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            instances = out["instances"].to("cpu")
            image_id = inp.get("image_id")
            file_name = inp.get("file_name")
            gt_dict = self._gt_by_image_id.get(image_id) or self._gt_by_file_name.get(file_name)

            if gt_dict is None or len(gt_dict.get("annotations", [])) == 0:
                # No GT annotations for this image, skip evaluation
                continue

            self._total_images += 1

            # 1. Compute foreground IoU
            fg_iou = _foreground_iou(instances, gt_dict)
            self._fg_ious.append(fg_iou)

            # 2. Compute greedy match IoU
            if len(instances) == 0:
                mean_iou = 0.0
            else:
                pred_masks = instances.pred_masks.float()  # (N_pred, H, W)
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

            self._greedy_ious.append(mean_iou)

            # Bucketing (following the same logic as IoUHook)
            if mean_iou >= 0.90:
                self._count_90 += 1
                self._count_75 += 1
                self._count_50 += 1
            elif mean_iou >= 0.75:
                self._count_75 += 1
                self._count_50 += 1
            elif mean_iou >= 0.50:
                self._count_50 += 1
            else:
                self._count_failed += 1

    def evaluate(self):
        mean_greedy_iou = float(np.mean(self._greedy_ious)) if self._greedy_ious else 0.0
        mean_fg_iou = float(np.mean(self._fg_ious)) if self._fg_ious else 0.0
        
        results = {
            "mean_iou": mean_greedy_iou,
            "mean_foreground_iou": mean_fg_iou,
            "bucket_counts": {
                "count_90": self._count_90,
                "count_75": self._count_75,
                "count_50": self._count_50,
                "count_failed": self._count_failed,
            },
            "total_evaluated_images": self._total_images
        }
        return results
