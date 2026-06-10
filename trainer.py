"""Custom Detectron2 trainer for SensAug-D2.

This trainer wires together three components:

1.  :class:`~augmentations.adaptive_mapper.AdaptiveDatasetMapper` — a dataset
    mapper that samples one perturbation (and its magnitude) per image from a
    shared :class:`~augmentations.adaptive_mapper.AugmentationPolicy`.

2.  :class:`~hooks.sens_aug_hook.SensAugHook` — a training hook that runs a
    sensitivity-analysis pass every ``SENSAUG.SA_PERIOD`` iterations and rewrites
    the shared policy's per-augmentation intensity levels and sampling weights.

3.  The existing diagnostic hooks (:class:`~hooks.IoU_val_hook.IoUHook`,
    :class:`~hooks.AP_val_hook.APVisualizationHook`,
    :class:`~hooks.loss_hook.TrainingLossHook`).

SensAug config keys (set via ``cfg.SENSAUG.*``)
-----------------------------------------------
SENSAUG.ENABLED         bool  If False, fall back to the static augmentations.
SENSAUG.WARMUP_ITERS    int   Clean-training iterations before the first SA pass.
SENSAUG.SA_PERIOD       int   Iterations between sensitivity-analysis passes.
SENSAUG.NUM_LEVELS      int   Sensitive intensity levels per augmentation (L).
SENSAUG.ALPHA_GRID_SIZE int   Diagnostic magnitudes swept in [0, 1].
SENSAUG.LAMBDA          float Regularizer lambda in g(alpha).
SENSAUG.NONE_PROB       float Probability of skipping augmentation after warmup.
SENSAUG.VAL_DATASET     str   Registered validation dataset to run SA on.
SENSAUG.MAX_SA_IMAGES   int   Cap on validation images per SA pass.
SENSAUG.KID_SUBSET_SIZE int   torchmetrics KID subset size (clamped to image count).

All keys default gracefully if not present in the cfg node.
"""

from __future__ import annotations

import logging

from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer

from augmentations.adaptive_mapper import AdaptiveDatasetMapper, AugmentationPolicy
from augmentations.augmentations import PERTURBATION_REGISTRY, build_augmentations
from hooks.IoU_val_hook import IoUHook
from hooks.AP_val_hook import APVisualizationHook
from hooks.loss_hook import TrainingLossHook
from hooks.sens_aug_hook import SensAugHook
from hooks.ap_iou_final_results_hook import AP_IOU_FinalResults

logger = logging.getLogger(__name__)


class Trainer(DefaultTrainer):
    """Custom trainer subclass for SensAug-D2.

    Register custom hooks here. The train loader uses
    :class:`~augmentations.adaptive_mapper.AdaptiveDatasetMapper` whose
    augmentation weights are updated by :class:`~hooks.sens_aug_hook.SensAugHook`
    during training.
    """

    # Shared augmentation policy — created once per Trainer instance and
    # referenced by both the mapper (built in build_train_loader) and the hook
    # (registered in build_hooks).  Using a class-level slot keeps them in sync
    # even if build_train_loader and build_hooks are called at different times.
    _policy: AugmentationPolicy | None = None

    @classmethod
    def _get_sensaug_cfg(cls, cfg):
        """Safely extract SensAug-specific config values with defaults."""
        sensaug = getattr(cfg, "SENSAUG", None)
        return {
            "enabled":         getattr(sensaug, "ENABLED",         True),
            "warmup_iters":    getattr(sensaug, "WARMUP_ITERS",    1000),
            "sa_period":       getattr(sensaug, "SA_PERIOD",       1500),
            "num_levels":      getattr(sensaug, "NUM_LEVELS",      5),
            "alpha_grid_size": getattr(sensaug, "ALPHA_GRID_SIZE", 11),
            "lam":             getattr(sensaug, "LAMBDA",          0.05),
            "none_prob":       getattr(sensaug, "NONE_PROB",       0.2),
            "val_dataset":     getattr(sensaug, "VAL_DATASET",     ""),
            "max_sa_images":   getattr(sensaug, "MAX_SA_IMAGES",   200),
            "kid_subset_size": getattr(sensaug, "KID_SUBSET_SIZE", 50),
        }

    @classmethod
    def _ensure_policy(cls, sa_cfg) -> AugmentationPolicy:
        """Create the shared policy once, starting clean (none_prob = 1.0)."""
        if cls._policy is None:
            cls._policy = AugmentationPolicy(
                perturbation_names=list(PERTURBATION_REGISTRY.keys()),
                num_levels=sa_cfg["num_levels"],
                none_prob=1.0,  # clean warmup until the first SA pass
            )
            logger.info(
                "[Trainer] Created AugmentationPolicy: %d perturbations, L=%d levels.",
                len(PERTURBATION_REGISTRY), sa_cfg["num_levels"],
            )
        return cls._policy

    @classmethod
    def build_train_loader(cls, cfg):
        sa_cfg = cls._get_sensaug_cfg(cfg)

        # Global augmentation kill-switch (DATASETS.AUGMENTATION=False disables ALL
        # live augmentation — static and SensAug adaptive alike).
        if not getattr(cfg.DATASETS, "AUGMENTATION", True):
            from detectron2.data import DatasetMapper
            mapper = DatasetMapper(cfg, is_train=True, augmentations=[])
            return build_detection_train_loader(cfg, mapper=mapper)

        if not sa_cfg["enabled"]:
            # Fall back to static augmentations when SensAug is disabled.
            from detectron2.data import DatasetMapper
            augs = build_augmentations()
            mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
            return build_detection_train_loader(cfg, mapper=mapper)

        policy = cls._ensure_policy(sa_cfg)
        mapper = AdaptiveDatasetMapper(cfg, policy=policy)
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
        hooks.append(
            AP_IOU_FinalResults(
                output_dir=output_dir,
                cfg=cfg,
            )
        )


        # ---- SensAugHook (only when adaptive augmentation is enabled) ----
        if sa_cfg["enabled"]:
            policy = self.__class__._ensure_policy(sa_cfg)

            # Validation split to run sensitivity analysis on.  Falls back to the
            # configured val split (DATASETS.TEST[1], as used by the diagnostic
            # hooks) when SENSAUG.VAL_DATASET is left empty.
            val_dataset = sa_cfg["val_dataset"]
            if not val_dataset:
                test_sets = cfg.DATASETS.TEST
                val_dataset = test_sets[1] if len(test_sets) > 1 else test_sets[0]

            hooks.append(
                SensAugHook(
                    policy=policy,
                    val_dataset=val_dataset,
                    warmup_iters=sa_cfg["warmup_iters"],
                    sa_period=sa_cfg["sa_period"],
                    num_levels=sa_cfg["num_levels"],
                    alpha_grid_size=sa_cfg["alpha_grid_size"],
                    lam=sa_cfg["lam"],
                    none_prob=sa_cfg["none_prob"],
                    max_sa_images=sa_cfg["max_sa_images"],
                    kid_subset_size=sa_cfg["kid_subset_size"],
                    output_dir=output_dir,
                    job_name=job_name,
                )
            )
            logger.info(
                "[Trainer] Registered SensAugHook — warmup=%d, SA every %d iters, "
                "val='%s'.",
                sa_cfg["warmup_iters"], sa_cfg["sa_period"], val_dataset,
            )
        else:
            logger.info("[Trainer] SensAug disabled (SENSAUG.ENABLED=False).")

        return hooks
