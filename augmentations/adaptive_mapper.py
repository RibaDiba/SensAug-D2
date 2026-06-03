"""Adaptive dataset mapper for SensAug-D2.

This mapper mirrors the ``RandomAlphaTrainTransform`` / ``RandomTrainTransformNew``
logic from the original SensAug repository, ported to Detectron2's DatasetMapper
pattern.

Key design decisions
--------------------
* A *shared* :class:`AugmentationWeights` object is held by both this mapper and
  the :class:`~hooks.sens_aug_hook.SensAugHook`.  The hook updates the weights
  in-place after each sensitivity analysis interval; the mapper always samples
  from the latest weights when it processes each batch.
* Magnitude for the chosen perturbation is drawn uniformly from [0, 1] (matching
  the original ``RandomAlphaTrainTransform``), then perturbed by a small Gaussian
  (as in ``RandomTrainTransformNew``) to add diversity.
* A "none" option is always retained in the weight dict with its own weight so
  that the hook can tune how often perturbations are skipped entirely.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from detectron2.data import DatasetMapper, detection_utils as utils
from detectron2.data import transforms as T
from detectron2.config import CfgNode

from augmentations.augmentations import PERTURBATION_REGISTRY

logger = logging.getLogger(__name__)


class AugmentationWeights:
    """Shared, mutable container for per-perturbation sampling probabilities.

    The :class:`AdaptiveDatasetMapper` reads from this object every call; the
    :class:`~hooks.sens_aug_hook.SensAugHook` writes to it after each analysis
    interval.  Because both share the *same instance*, no inter-process
    communication is required when running on a single machine.

    The weights are stored as a plain ``dict[str, float]`` mapping perturbation
    names (plus the special key ``"none"``) to unnormalized non-negative weights.
    Internally, sampling always normalizes so weights always sum to 1.

    Args:
        perturbation_names: Names of augmentation types that can be selected.
            Must match keys in :data:`~augmentations.augmentations.PERTURBATION_REGISTRY`.
        none_weight: Initial weight for the "skip augmentation" option.
        initial_weight: Initial weight given to each named perturbation.
    """

    def __init__(
        self,
        perturbation_names: List[str],
        none_weight: float = 1.0,
        initial_weight: float = 1.0,
    ) -> None:
        self._weights: Dict[str, float] = {name: initial_weight for name in perturbation_names}
        self._weights["none"] = none_weight

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def names(self) -> List[str]:
        return list(self._weights.keys())

    @property
    def probabilities(self) -> np.ndarray:
        """Return a normalized probability array matching ``self.names``."""
        w = np.array([self._weights[n] for n in self.names], dtype=np.float64)
        w = np.clip(w, 0.0, None)
        total = w.sum()
        if total <= 0:
            # Uniform fallback – should never happen in practice.
            return np.ones(len(w)) / len(w)
        return w / total

    def update(self, new_weights: Dict[str, float]) -> None:
        """Replace weights for the given keys (others are left unchanged).

        Args:
            new_weights: Mapping from perturbation name (or ``"none"``) to a
                non-negative weight value.
        """
        for name, value in new_weights.items():
            if name in self._weights:
                self._weights[name] = float(max(value, 0.0))
            else:
                logger.warning("AugmentationWeights.update: unknown key '%s' ignored.", name)

    def snapshot(self) -> Dict[str, float]:
        """Return a shallow copy of the current weight dict."""
        return dict(self._weights)


class AdaptiveDatasetMapper(DatasetMapper):
    """DatasetMapper that samples one perturbation per image from a shared weight dict.

    At each ``__call__``, the mapper:

    1. Applies the *standard* geometric augmentations (resize + flip) defined by
       the Detectron2 config.
    2. Samples a perturbation type from :attr:`aug_weights` proportional to the
       current weights.
    3. Draws a magnitude uniformly from [0, 1], adds small Gaussian jitter, and
       applies the perturbation to the image only (labels are unaffected by
       photometric transforms).

    Args:
        cfg: Detectron2 config node (used by the parent ``DatasetMapper``).
        aug_weights: Shared :class:`AugmentationWeights` instance.  The
            :class:`~hooks.sens_aug_hook.SensAugHook` holds a reference to the
            same object and updates it periodically.
        magnitude_noise_std: Standard deviation of the Gaussian noise added to
            the sampled magnitude (mirrors ``RandomTrainTransformNew``).
        base_augmentations: Optional list of *additional* Detectron2
            ``Augmentation`` objects applied *before* the adaptive perturbation
            (e.g. crop, lighting).  Resize and flip come from the parent class.
    """

    def __init__(
        self,
        cfg: CfgNode,
        aug_weights: AugmentationWeights,
        magnitude_noise_std: float = 0.1,
        base_augmentations: Optional[List[T.Augmentation]] = None,
    ) -> None:
        # The parent DatasetMapper handles resize/flip/crop from cfg.
        super().__init__(cfg, is_train=True)
        self.aug_weights = aug_weights
        self.magnitude_noise_std = magnitude_noise_std
        self._extra_base_augs: List[T.Augmentation] = base_augmentations or []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_perturbation(self) -> Optional[str]:
        """Sample a perturbation name (or ``None`` for 'none')."""
        names = self.aug_weights.names
        probs = self.aug_weights.probabilities
        chosen = np.random.choice(names, p=probs)
        return None if chosen == "none" else chosen

    def _apply_perturbation(self, img: np.ndarray, name: str) -> np.ndarray:
        """Apply the named perturbation at a random magnitude."""
        aug_cls = PERTURBATION_REGISTRY[name]
        # Magnitude ~ Uniform[0, 1] + Gaussian jitter (mirrors original logic).
        magnitude = float(np.random.uniform(0.0, 1.0))
        if self.magnitude_noise_std > 0:
            magnitude = float(np.clip(
                magnitude + np.random.normal(0, self.magnitude_noise_std),
                0.0, 1.0,
            ))
        aug = aug_cls(magnitude=magnitude)
        tfm = aug.get_transform(img)
        return tfm.apply_image(img)

    # ------------------------------------------------------------------
    # Override call
    # ------------------------------------------------------------------

    def __call__(self, dataset_dict: dict) -> dict:
        """Process one dataset entry with adaptive augmentation."""
        dataset_dict = copy.deepcopy(dataset_dict)

        # Load image (parent handles file reading convention).
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

        # ---- Adaptive perturbation sampling ----
        chosen = self._sample_perturbation()
        if chosen is not None and chosen in PERTURBATION_REGISTRY:
            image = self._apply_perturbation(image, chosen)

        # ---- Pack result (mirrors DatasetMapper) ----
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        # Annotations / segmentation masks -- delegate to parent helper.
        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
