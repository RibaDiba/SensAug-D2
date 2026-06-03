"""Custom Detectron2 trainer for SensAug-D2.

This trainer wires together three components:

1.  :class:`~augmentations.adaptive_mapper.AdaptiveDatasetMapper` — a dataset
    mapper that samples one perturbation per image from a shared probability
    distribution (:class:`~augmentations.adaptive_mapper.AugmentationWeights`).

2.  :class:`~hooks.sens_aug_hook.SensAugHook` — a training hook that runs a
    sensitivity analysis pass every ``SENSAUG.EVAL_PERIOD`` iterations, computes
    per-perturbation loss sensitivity scores, and updates the shared weight dict.

3.  The existing diagnostic hooks (:class:`~hooks.IoU_val_hook.IoUHook`,
    :class:`~hooks.AP_val_hook.APVisualizationHook`,
    :class:`~hooks.loss_hook.TrainingLossHook`).

SensAug config keys (set via ``cfg.SENSAUG.*``)
-----------------------------------------------
SENSAUG.EVAL_PERIOD       int   Sensitivity analysis cadence (default 500).
SENSAUG.NUM_PROBE_BATCHES int   Probe batches per analysis pass (default 4).
SENSAUG.DIAGNOSTIC_MAG    float Fixed magnitude for the probe pass (default 0.5).
SENSAUG.TEMPERATURE       float Softmax temperature for weight conversion (default 1.0).
SENSAUG.NONE_WEIGHT       float Weight of "skip augmentation" bucket (default 0.2).
SENSAUG.ENABLED           bool  If False, fall back to the static augmentations.

All keys default gracefully if not present in the cfg node.
"""

from __future__ import annotations

import logging

from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer

from augmentations.adaptive_mapper import AdaptiveDatasetMapper, AugmentationWeights
from augmentations.augmentations import PERTURBATION_REGISTRY, build_augmentations
from hooks.IoU_val_hook import IoUHook
from hooks.AP_val_hook import APVisualizationHook
from hooks.loss_hook import TrainingLossHook
from hooks.sens_aug_hook import SensAugHook

logger = logging.getLogger(__name__)


class Trainer(DefaultTrainer):
    """Custom trainer subclass for SensAug-D2.

    Register custom hooks here. The train loader uses
    :class:`~augmentations.adaptive_mapper.AdaptiveDatasetMapper` whose
    augmentation weights are updated by :class:`~hooks.sens_aug_hook.SensAugHook`
    during training.
    """

    # Shared weight container — created once per Trainer instance and referenced
    # by both the mapper (built in build_train_loader) and the hook (registered
    # in build_hooks).  Using a class-level slot keeps them in sync even if
    # build_train_loader and build_hooks are called at different times.
    _aug_weights: AugmentationWeights | None = None

    @classmethod
    def _get_sensaug_cfg(cls, cfg):
        """Safely extract SensAug-specific config values with defaults."""
        sensaug = getattr(cfg, "SENSAUG", None)
        return {
            "enabled":          getattr(sensaug, "ENABLED",           True),
            "eval_period":      getattr(sensaug, "EVAL_PERIOD",        200),
            "num_probe_batches": getattr(sensaug, "NUM_PROBE_BATCHES", 4),
            "diagnostic_mag":   getattr(sensaug, "DIAGNOSTIC_MAG",    0.5),
            "temperature":      getattr(sensaug, "TEMPERATURE",        1.0),
            "none_weight":      getattr(sensaug, "NONE_WEIGHT",        0.2),
        }

    @classmethod
    def build_train_loader(cls, cfg):
        sa_cfg = cls._get_sensaug_cfg(cfg)

        if not sa_cfg["enabled"]:
            # Fall back to static augmentations when SensAug is disabled.
            from detectron2.data import DatasetMapper
            augs = build_augmentations()
            mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
            return build_detection_train_loader(cfg, mapper=mapper)

        # Build (or reuse) the shared weight object.
        if cls._aug_weights is None:
            cls._aug_weights = AugmentationWeights(
                perturbation_names=list(PERTURBATION_REGISTRY.keys()),
                none_weight=sa_cfg["none_weight"],
                initial_weight=1.0,
            )
            logger.info(
                "[Trainer] Created AugmentationWeights with %d perturbations + 'none'.",
                len(PERTURBATION_REGISTRY),
            )

        mapper = AdaptiveDatasetMapper(cfg, aug_weights=cls._aug_weights)
        return build_detection_train_loader(cfg, mapper=mapper)

    def build_hooks(self):
        hooks = super().build_hooks()
        cfg = self.cfg
        sa_cfg = self._get_sensaug_cfg(cfg)
        eval_period = getattr(cfg, "EVAL_PERIOD", 100)
        output_dir = getattr(cfg, "OUTPUT_DIR", "")
        job_name = getattr(cfg, "JOBNAME", "sensaug_d2")

        # ---- Diagnostic hooks (always registered) ----
        hooks.append(
            IoUHook(
                output_dir=output_dir,
                job_name=job_name,
                eval_period=eval_period,
            )
        )
        hooks.append(
            APVisualizationHook(
                output_dir=output_dir,
                eval_period=eval_period,
                job_name=job_name,
            )
        )
        hooks.append(
            TrainingLossHook(
                output_dir=output_dir,
                job_name=job_name,
            )
        )

        # ---- SensAugHook (only when adaptive augmentation is enabled) ----
        if sa_cfg["enabled"]:
            if self.__class__._aug_weights is None:
                # build_train_loader wasn't called before build_hooks — create weights now.
                self.__class__._aug_weights = AugmentationWeights(
                    perturbation_names=list(PERTURBATION_REGISTRY.keys()),
                    none_weight=sa_cfg["none_weight"],
                    initial_weight=1.0,
                )

            # Reuse the training dataloader as the probe loader.
            # A small, frozen held-out subset would be more rigorous but this
            # is a lightweight default that avoids extra setup.
            probe_loader = self.data_loader

            hooks.append(
                SensAugHook(
                    aug_weights=self.__class__._aug_weights,
                    probe_loader=probe_loader,
                    eval_period=sa_cfg["eval_period"],
                    num_probe_batches=sa_cfg["num_probe_batches"],
                    diagnostic_magnitude=sa_cfg["diagnostic_mag"],
                    temperature=sa_cfg["temperature"],
                    none_weight=sa_cfg["none_weight"],
                    output_dir=output_dir,
                    job_name=job_name,
                )
            )
            logger.info(
                "[Trainer] Registered SensAugHook — analysis every %d iterations.",
                sa_cfg["eval_period"],
            )
        else:
            logger.info("[Trainer] SensAug disabled (SENSAUG.ENABLED=False).")

        return hooks
