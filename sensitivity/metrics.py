"""Accuracy measurement for the sensitivity-analysis sweep.

``MA(alpha)`` in SensAug is the model's accuracy on the validation set with a
given perturbation applied at magnitude ``alpha``.  For instance segmentation we
use a **per-image foreground mask mIoU**: every predicted instance mask is merged
into a single binary foreground map, compared (IoU) against the merged
ground-truth foreground, and averaged across images.

The same forward pass also yields the *perturbed images* used as the "fake" set
for the KID degradation term, so callers get both from one evaluation.

The GT-polygon decoding and binary-mask IoU helpers are reused from
:mod:`hooks.IoU_val_hook` to avoid duplicating that logic.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from hooks.IoU_val_hook import _masks_iou, _polygons_to_binary_masks


def _foreground_iou(instances, gt_dict: dict) -> float:
    """IoU between the merged predicted foreground and merged GT foreground.

    Both masks are evaluated at the original image resolution
    (``gt_dict["height"] x gt_dict["width"]``); Detectron2 already rescales
    predicted masks to that size via the ``height``/``width`` inputs.
    """
    h, w = gt_dict["height"], gt_dict["width"]

    # ---- Predicted foreground ----
    if len(instances) > 0 and instances.has("pred_masks"):
        pred_fg = instances.pred_masks.any(dim=0).to(torch.float32)
    else:
        pred_fg = torch.zeros((h, w), dtype=torch.float32)

    # ---- Ground-truth foreground ----
    annos = gt_dict.get("annotations", [])
    segs = [a["segmentation"] for a in annos if "segmentation" in a]
    if segs:
        gt_masks = _polygons_to_binary_masks(segs, h, w)
        gt_fg = (gt_masks.sum(dim=0) > 0).to(torch.float32)
    else:
        gt_fg = torch.zeros((h, w), dtype=torch.float32)

    if pred_fg.shape != gt_fg.shape:
        # Safety net: align predicted foreground to GT resolution.
        pred_fg = torch.nn.functional.interpolate(
            pred_fg[None, None], size=(h, w), mode="nearest"
        )[0, 0]

    if gt_fg.sum() == 0 and pred_fg.sum() == 0:
        return 1.0  # correctly predicted "nothing here"
    return _masks_iou(pred_fg, gt_fg)


def evaluate_aug_alpha(
    model,
    cfg,
    dataset_dicts: Sequence[dict],
    aug_factory: Optional[Callable],
    alpha: float,
) -> Tuple[float, List[np.ndarray]]:
    """Evaluate the model under one perturbation at one magnitude.

    Applies the standard test-time resize (``ResizeShortestEdge`` from the cfg),
    then the perturbation, runs inference, and computes per-image foreground IoU.

    Args:
        model: The (already-built) Detectron2 model.  Caller is responsible for
            setting/restoring ``eval`` mode and ``torch.no_grad``.
        cfg: Detectron2 config (for ``INPUT.MIN_SIZE_TEST`` / ``MAX_SIZE_TEST`` /
            ``FORMAT``).
        dataset_dicts: Validation entries (must carry ``file_name``, ``height``,
            ``width`` and polygon ``annotations``).
        aug_factory: Callable ``magnitude -> Augmentation`` from
            :data:`~augmentations.augmentations.PERTURBATION_REGISTRY`, or ``None``
            for the clean baseline.
        alpha: Perturbation magnitude in ``[0, 1]``.

    Returns:
        ``(mean_foreground_iou, perturbed_images)`` where ``perturbed_images`` is
        a list of HWC ``uint8`` arrays (post-resize, post-perturbation) suitable
        for KID.
    """
    image_format = cfg.INPUT.FORMAT
    resize = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    device = next(model.parameters()).device

    ious: List[float] = []
    perturbed_images: List[np.ndarray] = []

    for d in dataset_dicts:
        img = utils.read_image(d["file_name"], format=image_format)  # HWC uint8

        # Test-time resize (matches build_detection_test_loader behaviour).
        aug_input = T.AugInput(img)
        resize(aug_input)
        img_r = aug_input.image

        # Apply the perturbation (clean baseline when aug_factory is None).
        if aug_factory is not None and alpha > 0.0:
            aug = aug_factory(magnitude=float(alpha))
            img_p = aug.get_transform(img_r).apply_image(img_r)
        else:
            img_p = img_r
        if img_p.dtype != np.uint8:
            img_p = np.clip(img_p, 0, 255).astype(np.uint8)
        perturbed_images.append(img_p)

        model_input = {
            "image": torch.as_tensor(
                np.ascontiguousarray(img_p.transpose(2, 0, 1))
            ).to(device),
            "height": d["height"],
            "width": d["width"],
        }
        outputs = model([model_input])[0]
        ious.append(_foreground_iou(outputs["instances"].to("cpu"), d))

    mean_iou = float(np.mean(ious)) if ious else 0.0
    return mean_iou, perturbed_images
