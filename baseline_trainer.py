"""Baseline Detectron2 trainer for SensAug-D2.

A stripped-down counterpart to :class:`trainer.Trainer`: it runs training with
the *static* augmentation pipeline and the standard diagnostic hooks, but
**without** any sensitivity-analysis machinery — no ``SensAugHook`` and no
``AdaptiveDatasetMapper``.

Use this trainer to produce a clean baseline run to compare against the full
SensAug pipeline.  Selection between the two happens in
``training_scripts/train.py`` via the required ``--mode`` flag.
"""

from __future__ import annotations

import logging

from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.engine import DefaultTrainer

from augmentations.augmentations import build_augmentations
from hooks.IoU_val_hook import IoUHook
from hooks.AP_val_hook import APVisualizationHook
from hooks.loss_hook import TrainingLossHook

logger = logging.getLogger(__name__)


class BaselineTrainer(DefaultTrainer):
    """Baseline trainer: static augmentations + diagnostic hooks, no sensitivity analysis."""

    @classmethod
    def build_train_loader(cls, cfg):
        augs = build_augmentations()
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
        return build_detection_train_loader(cfg, mapper=mapper)

    def build_hooks(self):
        hooks = super().build_hooks()
        cfg = self.cfg
        eval_period = getattr(cfg, "EVAL_PERIOD", 100)
        output_dir = getattr(cfg, "OUTPUT_DIR", "")
        job_name = getattr(cfg, "JOBNAME", "sensaug_d2")

        # ---- Diagnostic hooks (no SensAugHook) ----
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

        logger.info("[BaselineTrainer] Static augmentations, no sensitivity analysis.")
        return hooks
