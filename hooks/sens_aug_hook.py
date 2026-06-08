"""SensAugHook — Detectron2 hook implementing the SensAug method.

Faithful port of *Adaptive Sensitivity Analysis for Robust Augmentation against
Natural Corruptions in Image Segmentation* (Zheng et al., arXiv:2406.01425) from
MMSegmentation to Detectron2 for **instance** segmentation.

Algorithm
---------
1.  **Warmup.** For the first ``warmup_iters`` iterations the model trains on
    clean data (the shared :class:`~augmentations.adaptive_mapper.AugmentationPolicy`
    starts with ``none_prob = 1``).
2.  **Periodic sensitivity analysis** (every ``sa_period`` iterations, starting at
    warmup end).  Over a held-out **validation** split, for each perturbation
    type, sweep a grid of magnitudes ``alpha`` and measure:

      * ``MA(alpha)`` — per-image foreground mask mIoU (model accuracy), and
      * ``KID(alpha)`` — Kernel Inception Distance between clean and perturbed
        images (perceptual degradation).

    Then build the trade-off curve ``g(alpha)``, select ``L`` sensitive intensity
    levels (PCHIP), and build a Beta-Binomial sampling distribution over them that
    favours the worse-performing levels.  These update the shared policy in place,
    so the dataset mapper starts sampling the new levels immediately.
3.  Sensitivity records are logged to the Detectron2 event storage and persisted
    to JSON for post-hoc analysis.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from detectron2.data import DatasetCatalog
from detectron2.engine.hooks import HookBase
from detectron2.utils.events import get_event_storage

from augmentations.augmentations import PERTURBATION_REGISTRY
from augmentations.adaptive_mapper import AugmentationPolicy
from sensitivity.analysis import (
    build_g_curve,
    select_sensitive_levels,
    beta_binomial_weights,
    pchip_interp,
)
from sensitivity.kid import compute_kid
from sensitivity.metrics import evaluate_aug_alpha

logger = logging.getLogger(__name__)


class SensAugHook(HookBase):
    """Training hook that runs SensAug sensitivity analysis and updates the policy.

    Args:
        policy: Shared :class:`~augmentations.adaptive_mapper.AugmentationPolicy`
            (the *same* object held by the dataset mapper).
        val_dataset: Name of the registered validation dataset to run SA on.
        warmup_iters: Iterations of clean training before the first SA pass.
        sa_period: Iterations between sensitivity-analysis passes.
        num_levels: Number ``L`` of sensitive intensity levels per augmentation.
        alpha_grid_size: Number of diagnostic magnitudes swept in ``[0, 1]``.
        lam: Regularizer ``lambda`` in ``g(alpha)``.
        none_prob: Probability of skipping augmentation *after* warmup.
        max_sa_images: Cap on validation images used per SA pass (compute control).
        kid_subset_size: KID subset size (clamped to the image count).
        output_dir: If set, sensitivity records are saved here as JSON.
        job_name: Prefix for output filenames / log messages.
        perturbation_names: Subset of :data:`PERTURBATION_REGISTRY` to evaluate
            (defaults to all).
    """

    def __init__(
        self,
        policy: AugmentationPolicy,
        val_dataset: str,
        warmup_iters: int = 1000,
        sa_period: int = 1500,
        num_levels: int = 5,
        alpha_grid_size: int = 11,
        lam: float = 0.05,
        none_prob: float = 0.2,
        max_sa_images: int = 200,
        kid_subset_size: int = 50,
        output_dir: str = "",
        job_name: str = "sensaug",
        perturbation_names: Optional[List[str]] = None,
    ) -> None:
        self.policy = policy
        self.val_dataset = val_dataset
        self.warmup_iters = int(warmup_iters)
        self.sa_period = max(1, int(sa_period))
        self.num_levels = int(num_levels)
        self.alpha_grid_size = int(alpha_grid_size)
        self.lam = float(lam)
        self.none_prob = float(none_prob)
        self.max_sa_images = int(max_sa_images)
        self.kid_subset_size = int(kid_subset_size)
        self.output_dir = output_dir
        self.job_name = job_name

        if perturbation_names is None:
            self.perturbation_names = list(PERTURBATION_REGISTRY.keys())
        else:
            self.perturbation_names = perturbation_names

        self._history: List[dict] = []
        self._val_dicts_cache: Optional[List[dict]] = None

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # HookBase interface
    # ------------------------------------------------------------------

    def after_step(self) -> None:
        iteration = self.trainer.iter
        if iteration < self.warmup_iters:
            return  # clean warmup (policy.none_prob == 1.0)
        if (iteration - self.warmup_iters) % self.sa_period != 0:
            return

        logger.info("[SensAugHook] Sensitivity analysis at iter %d …", iteration)
        t0 = time.time()
        record = self._run_sensitivity_analysis()
        if record is None:
            logger.warning("[SensAugHook] SA skipped (no validation data).")
            return

        # Activate adaptive sampling now that we have a real policy.
        self.policy.set_none_prob(self.none_prob)
        self.policy.mark_ready()

        elapsed = time.time() - t0
        logger.info("[SensAugHook] SA done in %.1fs.", elapsed)
        self._log_and_save(record, iteration)

    def after_train(self) -> None:
        if self.output_dir and self._history:
            path = os.path.join(self.output_dir, f"{self.job_name}_sensitivity_history.json")
            with open(path, "w") as f:
                json.dump(self._history, f, indent=2)
            logger.info("[SensAugHook] Saved sensitivity history → %s", path)

    # ------------------------------------------------------------------
    # Core sensitivity analysis
    # ------------------------------------------------------------------

    def _val_dicts(self) -> List[dict]:
        if self._val_dicts_cache is None:
            dicts = DatasetCatalog.get(self.val_dataset)
            self._val_dicts_cache = list(dicts[: self.max_sa_images])
        return self._val_dicts_cache

    def _run_sensitivity_analysis(self) -> Optional[Dict[str, dict]]:
        model = self.trainer.model
        cfg = self.trainer.cfg
        dataset_dicts = self._val_dicts()
        if not dataset_dicts:
            return None

        device = next(model.parameters()).device
        alphas = np.linspace(0.0, 1.0, self.alpha_grid_size)

        was_training = model.training
        model.eval()
        record: Dict[str, dict] = {}
        try:
            with torch.no_grad():
                # Clean baseline (alpha = 0) — computed once, reused for every aug.
                clean_ma, clean_imgs = evaluate_aug_alpha(
                    model, cfg, dataset_dicts, aug_factory=None, alpha=0.0
                )

                for name in self.perturbation_names:
                    factory = PERTURBATION_REGISTRY[name]
                    ma_list: List[float] = []
                    kid_list: List[float] = []
                    for a in alphas:
                        if a <= 0.0:
                            ma_list.append(clean_ma)
                            kid_list.append(0.0)
                            continue
                        ma, imgs = evaluate_aug_alpha(model, cfg, dataset_dicts, factory, float(a))
                        kid = compute_kid(
                            clean_imgs, imgs,
                            subset_size=self.kid_subset_size, device=device,
                        )
                        ma_list.append(ma)
                        kid_list.append(kid)

                    g = build_g_curve(alphas, np.array(ma_list), np.array(kid_list), self.lam)
                    levels = select_sensitive_levels(alphas, g, self.num_levels)
                    ma_at_levels = pchip_interp(alphas, np.array(ma_list), levels)
                    probs = beta_binomial_weights(ma_at_levels)

                    self.policy.update(name, levels, probs)
                    record[name] = {
                        "alphas": alphas.tolist(),
                        "ma": ma_list,
                        "kid": kid_list,
                        "g": g.tolist(),
                        "levels": levels.tolist(),
                        "ma_at_levels": np.asarray(ma_at_levels).tolist(),
                        "probs": np.asarray(probs).tolist(),
                    }
        finally:
            if was_training:
                model.train()

        record["_clean_ma"] = clean_ma
        return record

    # ------------------------------------------------------------------
    # Logging / persistence
    # ------------------------------------------------------------------

    def _log_and_save(self, record: Dict[str, dict], iteration: int) -> None:
        try:
            storage = get_event_storage()
            storage.put_scalar("sensaug/clean_ma", record.get("_clean_ma", 0.0))
            for name in self.perturbation_names:
                r = record.get(name)
                if r:
                    storage.put_scalar(f"sensaug/ma_worst_{name}", float(min(r["ma"])))
                    storage.put_scalar(f"sensaug/kid_max_{name}", float(max(r["kid"])))
        except Exception:
            pass  # event storage may be unavailable

        worst = {
            name: round(float(min(r["ma"])), 4)
            for name, r in record.items()
            if isinstance(r, dict) and "ma" in r
        }
        logger.info("[SensAugHook] iter=%d clean_ma=%.4f worst MA/aug: %s",
                    iteration, record.get("_clean_ma", 0.0), worst)

        entry = {"iteration": iteration, **record}
        self._history.append(entry)
        if self.output_dir:
            path = os.path.join(
                self.output_dir, f"{self.job_name}_sensitivity_iter{iteration:07d}.json"
            )
            with open(path, "w") as f:
                json.dump(entry, f, indent=2)
            logger.info("[SensAugHook] Saved sensitivity snapshot → %s", path)
