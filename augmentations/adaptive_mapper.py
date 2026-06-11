"""Adaptive dataset mapper for SensAug-D2.

Ports the SensAug training-time augmentation to Detectron2's ``DatasetMapper``
pattern.  The adaptive behaviour lives in :class:`AugmentationPolicy`: for each
perturbation type it holds a set of ``L`` *intensity levels* (magnitudes) plus a
Beta-Binomial sampling distribution over them.  The
:class:`~hooks.sens_aug_hook.SensAugHook` rewrites this policy after every
sensitivity-analysis pass; this mapper samples from it for every image.

Cross-process propagation
-------------------------
Detectron2's training dataloader runs the mapper inside worker processes.  A
plain Python object updated in the main process would **not** be visible to those
workers (they hold a fork-time copy).  :class:`AugmentationPolicy` therefore
stores its levels/probabilities in **shared-memory torch tensors**
(``share_memory_()``), so in-place updates from the hook are seen by the workers
on the next batch.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from detectron2.data import DatasetMapper, detection_utils as utils
from detectron2.data import transforms as T
from detectron2.config import CfgNode

from augmentations.augmentations import PERTURBATION_REGISTRY

logger = logging.getLogger(__name__)


def _fit_length(values: np.ndarray, length: int) -> np.ndarray:
    """Truncate or pad (with the last element) ``values`` to exactly ``length``."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == length:
        return values
    if values.size == 0:
        return np.zeros(length, dtype=np.float64)
    if values.size > length:
        return values[:length]
    pad = np.full(length - values.size, values[-1], dtype=np.float64)
    return np.concatenate([values, pad])


class AugmentationPolicy:
    """Shared, mutable per-augmentation intensity-level sampling policy.

    For each perturbation name the policy stores:

    * ``levels[a]``  — ``L`` magnitudes in ``[0, 1]`` (the selected sensitive
      intensity levels), and
    * ``probs[a]``   — a length-``L`` probability vector over those levels
      (Beta-Binomial, favouring worse-performing levels).

    A scalar ``none_prob`` controls how often *no* perturbation is applied.  It
    starts at ``1.0`` so that training is clean during warmup; the hook lowers it
    once the first sensitivity-analysis pass has run.

    Args:
        perturbation_names: Augmentation types to manage (keys of
            :data:`~augmentations.augmentations.PERTURBATION_REGISTRY`).
        num_levels: Number ``L`` of intensity levels per augmentation.
        none_prob: Initial probability of skipping augmentation (default ``1.0``
            for a clean warmup).
    """

    def __init__(
        self,
        perturbation_names: List[str],
        num_levels: int,
        none_prob: float = 1.0,
    ) -> None:
        self.names: List[str] = list(perturbation_names)
        self._index: Dict[str, int] = {n: i for i, n in enumerate(self.names)}
        self.num_augs = len(self.names)
        self.L = int(num_levels)

        # Default: evenly-spread magnitudes in (0, 1], uniform sampling.  These
        # are only used if the mapper ever samples before the first SA pass
        # (it shouldn't, because none_prob starts at 1.0).
        default_levels = np.linspace(1.0 / self.L, 1.0, self.L)
        levels = np.tile(default_levels, (self.num_augs, 1))

        self.levels = torch.as_tensor(levels, dtype=torch.float32).share_memory_()
        self.probs = (
            torch.full((self.num_augs, self.L), 1.0 / self.L, dtype=torch.float32)
            .share_memory_()
        )
        self.none_prob = torch.tensor([float(none_prob)], dtype=torch.float32).share_memory_()
        self.ready = torch.zeros(1, dtype=torch.float32).share_memory_()  # 0 until first SA

    # ------------------------------------------------------------------
    # Written by the hook (main process)
    # ------------------------------------------------------------------

    def update(self, name: str, levels, probs) -> None:
        """Set the levels and sampling probabilities for one augmentation."""
        i = self._index.get(name)
        if i is None:
            logger.warning("AugmentationPolicy.update: unknown augmentation '%s'.", name)
            return
        lv = _fit_length(levels, self.L)
        pr = _fit_length(probs, self.L)
        pr = np.clip(pr, 0.0, None)
        total = pr.sum()
        pr = pr / total if total > 0 else np.full(self.L, 1.0 / self.L)
        self.levels[i].copy_(torch.as_tensor(lv, dtype=torch.float32))
        self.probs[i].copy_(torch.as_tensor(pr, dtype=torch.float32))

    def set_none_prob(self, value: float) -> None:
        """Set the probability of skipping augmentation (in ``[0, 1]``)."""
        self.none_prob.fill_(float(np.clip(value, 0.0, 1.0)))

    def mark_ready(self) -> None:
        """Flag that at least one sensitivity-analysis pass has completed."""
        self.ready.fill_(1.0)

    @property
    def is_ready(self) -> bool:
        return bool(self.ready.item() > 0.5)

    # ------------------------------------------------------------------
    # Read by the mapper (worker processes)
    # ------------------------------------------------------------------

    def sample(self) -> Tuple[Optional[str], float]:
        """Sample ``(augmentation_name | None, magnitude)`` for one image."""
        if float(np.random.uniform()) < float(self.none_prob.item()):
            return None, 0.0
        i = int(np.random.randint(self.num_augs))
        probs = self.probs[i].numpy().astype(np.float64)
        s = probs.sum()
        probs = probs / s if s > 0 else np.full(self.L, 1.0 / self.L)
        j = int(np.random.choice(self.L, p=probs))
        return self.names[i], float(self.levels[i, j].item())

    def snapshot(self) -> Dict[str, dict]:
        """Return a plain-dict view of the current policy (for logging)."""
        return {
            "none_prob": float(self.none_prob.item()),
            "ready": self.is_ready,
            "augs": {
                name: {
                    "levels": self.levels[i].tolist(),
                    "probs": self.probs[i].tolist(),
                }
                for name, i in self._index.items()
            },
        }


class AdaptiveDatasetMapper(DatasetMapper):
    """DatasetMapper that perturbs each image using a shared :class:`AugmentationPolicy`.

    Each ``__call__``:

    1. Applies the standard geometric augmentations (resize + flip) from the cfg.
    2. Asks the shared policy for ``(augmentation, magnitude)``.
    3. Applies that perturbation to the image (labels are unaffected by
       photometric transforms).

    Args:
        cfg: Detectron2 config node (used by the parent ``DatasetMapper``).
        policy: Shared :class:`AugmentationPolicy` instance, also held by the
            :class:`~hooks.sens_aug_hook.SensAugHook`.
        base_augmentations: Optional extra ``Augmentation`` objects applied before
            the adaptive perturbation.
    """

    def __init__(
        self,
        cfg: CfgNode,
        policy: AugmentationPolicy,
        base_augmentations: Optional[List[T.Augmentation]] = None,
    ) -> None:
        super().__init__(cfg, is_train=True)
        self.policy = policy
        self._extra_base_augs: List[T.Augmentation] = base_augmentations or []

    def _apply_perturbation(self, img: np.ndarray, name: str, magnitude: float) -> np.ndarray:
        aug = PERTURBATION_REGISTRY[name](magnitude=magnitude)
        return aug.get_transform(img).apply_image(img)

    def __call__(self, dataset_dict: dict) -> dict:
        dataset_dict = copy.deepcopy(dataset_dict)

        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # ---- Standard geometric augmentations (from parent config) ----
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        # ---- Optional extra base augmentations ----
        for aug in self._extra_base_augs:
            tfm = aug.get_transform(image)
            image = tfm.apply_image(image)

        # ---- Adaptive perturbation (sampled from the shared policy) ----
        name, magnitude = self.policy.sample()
        if name is not None and magnitude > 0.0 and name in PERTURBATION_REGISTRY:
            image = self._apply_perturbation(image, name, magnitude)
            dataset_dict["applied_aug"] = (name, magnitude)
        else:
            dataset_dict["applied_aug"] = ("None", 0.0)

        # ---- Pack result (mirrors DatasetMapper) ----
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
