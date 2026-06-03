"""SensAugHook — Detectron2 hook for sensitivity-informed adaptive augmentation.

Algorithm overview
------------------
Every ``eval_period`` training iterations the hook runs a lightweight
*sensitivity analysis* pass over a small held-out probe batch:

1.  Compute the model loss on the *clean* probe images (baseline).
2.  For each registered perturbation type, apply it at a fixed diagnostic
    magnitude, re-run inference, and record the loss shift:

        sensitivity_score[aug] = loss_perturbed - loss_clean

    A *high* score means the model is still losing a lot relative to clean
    under that perturbation — the model is *sensitive* to it, so we should
    increase its sampling probability during training.

3.  Convert raw scores to sampling weights using softmax (temperature-scaled),
    then update the shared :class:`~augmentations.adaptive_mapper.AugmentationWeights`
    object that the :class:`~augmentations.adaptive_mapper.AdaptiveDatasetMapper`
    reads from.

4.  Optionally persist the sensitivity scores to a JSON file for post-hoc
    analysis.

Design decisions
----------------
* The probe batch is **re-used** from the training dataloader's current
  iterator state — we do not need a separate dataloader.  This keeps the hook
  lightweight and avoids dataset duplication.
* Perturbations are applied **on the raw pixel tensor** after decoding, before
  any normalization, mirroring the original SensAug approach.
* The hook is **loss-based**, not metric-based.  This is intentional: computing
  AP/mIoU on every probe call would be too expensive for frequent updates.
  If you want metric-based updates, combine this hook with
  :class:`~hooks.IoU_val_hook.IoUHook` at a longer cadence.

References
----------
Zheng, L. et al. "SensAug: Sensitivity-Informed Augmentation for Robust
Perception under Distribution Shift." (2024).
"""

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from detectron2.engine.hooks import HookBase
from detectron2.utils.events import get_event_storage

from augmentations.augmentations import PERTURBATION_REGISTRY
from augmentations.adaptive_mapper import AugmentationWeights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Fixed magnitude used during the sensitivity diagnostic pass.
#: Mirrors the original SensAug choice of using a mid-range perturbation level.
DIAGNOSTIC_MAGNITUDE: float = 0.5

#: Softmax temperature for converting raw sensitivity scores to sampling
#: probabilities.  Lower T → sharper focus on the most-sensitive augmentations.
DEFAULT_TEMPERATURE: float = 1.0

#: Weight given to the "no augmentation" option after re-weighting.
#: Set to 0.0 to always apply some perturbation.
DEFAULT_NONE_WEIGHT: float = 0.2


class SensAugHook(HookBase):
    """Detectron2 training hook that implements sensitivity-informed adaptive augmentation.

    This hook:

    * Runs a sensitivity analysis pass every ``eval_period`` iterations.
    * Updates the :class:`~augmentations.adaptive_mapper.AugmentationWeights`
      shared with :class:`~augmentations.adaptive_mapper.AdaptiveDatasetMapper`.
    * Logs sensitivity scores to the Detectron2 event storage and optionally
      persists them to disk.

    Args:
        aug_weights: Shared :class:`~augmentations.adaptive_mapper.AugmentationWeights`
            instance.  Must be the *same* object passed to
            :class:`~augmentations.adaptive_mapper.AdaptiveDatasetMapper`.
        probe_loader: Dataloader (or any iterable of batches) used to draw the
            probe batch for sensitivity analysis.  Using the same train loader is
            fine; a small held-out subset is better but not required.
        eval_period: Number of training iterations between sensitivity analysis
            runs.  Defaults to 500.
        num_probe_batches: Number of batches to average the sensitivity scores
            over.  More batches → more stable estimates, but slower analysis.
        diagnostic_magnitude: Fixed perturbation magnitude used during analysis.
        temperature: Softmax temperature for score → weight conversion.
        none_weight: Weight of the "skip augmentation" bucket after updating.
        output_dir: If provided, sensitivity scores are saved as JSON files here.
        job_name: Used as a prefix for output filenames and log messages.
        perturbation_names: Subset of :data:`~augmentations.augmentations.PERTURBATION_REGISTRY`
            keys to evaluate.  Defaults to all registered perturbations.
    """

    def __init__(
        self,
        aug_weights: AugmentationWeights,
        probe_loader,
        eval_period: int = 500,
        num_probe_batches: int = 4,
        diagnostic_magnitude: float = DIAGNOSTIC_MAGNITUDE,
        temperature: float = DEFAULT_TEMPERATURE,
        none_weight: float = DEFAULT_NONE_WEIGHT,
        output_dir: str = "",
        job_name: str = "sensaug",
        perturbation_names: Optional[List[str]] = None,
    ) -> None:
        self.aug_weights = aug_weights
        self.probe_loader = probe_loader
        self.eval_period = eval_period
        self.num_probe_batches = num_probe_batches
        self.diagnostic_magnitude = diagnostic_magnitude
        self.temperature = temperature
        self.none_weight = none_weight
        self.output_dir = output_dir
        self.job_name = job_name

        # Which perturbations to evaluate during the analysis pass.
        if perturbation_names is None:
            self.perturbation_names: List[str] = list(PERTURBATION_REGISTRY.keys())
        else:
            self.perturbation_names = perturbation_names

        # Internal state
        self._sensitivity_history: List[Dict[str, float]] = []
        self._probe_iter = iter(self.probe_loader)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # HookBase interface
    # ------------------------------------------------------------------

    def after_step(self) -> None:
        """Run sensitivity analysis every ``eval_period`` steps."""
        iteration = self.trainer.iter
        if iteration == 0 or iteration % self.eval_period != 0:
            return

        logger.info(
            "[SensAugHook] Running sensitivity analysis at iteration %d …", iteration
        )

        scores = self._compute_sensitivity_scores()
        if scores is None:
            logger.warning("[SensAugHook] Sensitivity analysis returned None — skipping weight update.")
            return

        new_weights = self._scores_to_weights(scores)
        self.aug_weights.update(new_weights)
        self.aug_weights.update({"none": self.none_weight})

        # Logging
        self._log_scores(scores, iteration)
        self._sensitivity_history.append({"iteration": iteration, **scores})

        if self.output_dir:
            self._save_scores(scores, iteration)

    def after_train(self) -> None:
        """Save complete sensitivity history to disk."""
        if self.output_dir and self._sensitivity_history:
            path = os.path.join(self.output_dir, f"{self.job_name}_sensitivity_history.json")
            with open(path, "w") as f:
                json.dump(self._sensitivity_history, f, indent=2)
            logger.info("[SensAugHook] Saved sensitivity history → %s", path)

    # ------------------------------------------------------------------
    # Core sensitivity analysis
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_sensitivity_scores(self) -> Optional[Dict[str, float]]:
        """Compute per-perturbation sensitivity scores.

        Returns:
            Dict mapping perturbation name → sensitivity score (loss increase
            relative to clean baseline), or ``None`` if the probe batch is
            unavailable.
        """
        model = self.trainer.model
        was_training = model.training
        model.train()  # Detectron2 models need train mode for loss computation

        try:
            probe_batches = self._draw_probe_batches()
        except StopIteration:
            self._probe_iter = iter(self.probe_loader)
            try:
                probe_batches = self._draw_probe_batches()
            except StopIteration:
                return None

        if not probe_batches:
            return None

        # ---- Baseline (clean) loss ----
        clean_loss = self._mean_loss(model, probe_batches)

        # ---- Per-perturbation loss ----
        scores: Dict[str, float] = {}
        for name in self.perturbation_names:
            perturbed_batches = self._apply_perturbation_to_batches(probe_batches, name)
            perturbed_loss = self._mean_loss(model, perturbed_batches)
            # Sensitivity = how much loss *increases* relative to clean.
            # We max-clip at 0 so that perturbations that happen to *reduce*
            # loss don't get negative weights.
            scores[name] = float(max(perturbed_loss - clean_loss, 0.0))
            logger.debug(
                "[SensAugHook] %s: clean_loss=%.4f, perturbed_loss=%.4f, Δ=%.4f",
                name, clean_loss, perturbed_loss, scores[name],
            )

        if not was_training:
            model.eval()

        return scores

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _draw_probe_batches(self) -> List[List[dict]]:
        """Draw ``num_probe_batches`` batches from the probe loader."""
        batches = []
        for _ in range(self.num_probe_batches):
            batches.append(next(self._probe_iter))
        return batches

    @torch.no_grad()
    def _mean_loss(self, model, batches: List[List[dict]]) -> float:
        """Compute average total loss across a list of batches."""
        total, n = 0.0, 0
        for batch in batches:
            loss_dict = model(batch)
            # Detectron2 models return a dict of scalar losses.
            loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
            total += loss.item()
            n += 1
        return total / max(n, 1)

    def _apply_perturbation_to_batches(
        self, batches: List[List[dict]], perturbation_name: str
    ) -> List[List[dict]]:
        """Return new batch lists with the given perturbation applied to images.

        Images in Detectron2 batches are stored as CHW ``torch.Tensor`` objects
        with values in [0, 255] (uint8 or float32 depending on model config).
        We convert to HWC numpy, apply the perturbation, and convert back.
        """
        aug_factory = PERTURBATION_REGISTRY.get(perturbation_name)
        if aug_factory is None:
            logger.warning("[SensAugHook] Unknown perturbation '%s' — skipping.", perturbation_name)
            return batches

        aug = aug_factory(magnitude=self.diagnostic_magnitude)
        # Get the transform object; we need a dummy image for get_transform().
        # We'll use per-image get_transform calls below.

        new_batches = []
        for batch in batches:
            new_batch = []
            for sample in batch:
                sample = copy.copy(sample)  # shallow copy to avoid mutating originals
                img_tensor: torch.Tensor = sample["image"]  # C H W, float or uint8
                # Convert to HWC numpy uint8 for our augmentation classes.
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                if img_np.dtype != np.uint8:
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                # Re-instantiate per image because get_transform may be stateful.
                aug_i = aug_factory(magnitude=self.diagnostic_magnitude)
                tfm = aug_i.get_transform(img_np)
                img_np = tfm.apply_image(img_np)
                sample["image"] = torch.as_tensor(
                    np.ascontiguousarray(img_np.transpose(2, 0, 1)),
                    dtype=img_tensor.dtype,
                )
                new_batch.append(sample)
            new_batches.append(new_batch)
        return new_batches

    def _scores_to_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Convert raw sensitivity scores to sampling weights via softmax.

        Args:
            scores: Per-perturbation sensitivity scores (loss deltas ≥ 0).

        Returns:
            Dict of unnormalized sampling weights (will be normalized by
            :class:`~augmentations.adaptive_mapper.AugmentationWeights`).
        """
        names = list(scores.keys())
        values = np.array([scores[n] for n in names], dtype=np.float64)

        # Softmax with temperature — higher score → higher weight.
        exp_values = np.exp(values / max(self.temperature, 1e-6))
        weights = exp_values / exp_values.sum()

        return {name: float(w) for name, w in zip(names, weights)}

    def _log_scores(self, scores: Dict[str, float], iteration: int) -> None:
        """Log sensitivity scores to Detectron2 event storage and stdout."""
        try:
            storage = get_event_storage()
            for name, score in scores.items():
                storage.put_scalar(f"sensaug/score_{name}", score)
        except Exception:
            pass  # event storage might not be available outside training

        score_str = "  ".join(f"{n}={v:.4f}" for n, v in sorted(scores.items()))
        logger.info("[SensAugHook] iter=%d  scores: %s", iteration, score_str)

        weights_snapshot = self.aug_weights.snapshot()
        weight_str = "  ".join(f"{n}={v:.4f}" for n, v in sorted(weights_snapshot.items()))
        logger.info("[SensAugHook] iter=%d  weights: %s", iteration, weight_str)

    def _save_scores(self, scores: Dict[str, float], iteration: int) -> None:
        """Persist sensitivity scores for the current iteration to JSON."""
        filename = f"{self.job_name}_sensitivity_iter{iteration:07d}.json"
        path = os.path.join(self.output_dir, filename)
        payload = {
            "iteration": iteration,
            "diagnostic_magnitude": self.diagnostic_magnitude,
            "scores": scores,
            "weights": self.aug_weights.snapshot(),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("[SensAugHook] Saved sensitivity snapshot → %s", path)
