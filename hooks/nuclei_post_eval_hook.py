"""NucleiPostTrainHook — post-training robustness evaluation on *unseen* corruptions.

After training finishes, this hook evaluates the **test** split
(``cfg.DATASETS.TEST[0]`` — never seen during training; SensAug's sensitivity
analysis uses the *val* split) under a set of **held-out** photometric corruptions
the model has never trained on (``augmentations.heldout_augmentations.HELDOUT_PERTURBATION_REGISTRY``,
disjoint from the trained ``PERTURBATION_REGISTRY``).  This mirrors the SensAug
paper's held-out / ImageNet-C-style robustness evaluation (Zheng et al.,
arXiv:2406.01425): the model trains on the basis augmentations but is *evaluated*
on corruptions swept at uniform parameter intervals.

It produces two complementary outputs, both written to a **dedicated folder**
``<output_dir>/json_<job>_posteval/`` so they never mix with the clean
final-results folder produced by :class:`~hooks.ap_iou_final_results_hook.AP_IOU_FinalResults`:

1. **Degradation curves** (``posteval_curves.json`` + ``.png``): per-image foreground
   IoU (the SensAug accuracy metric ``MA``) across a magnitude grid, via
   :func:`sensitivity.metrics.evaluate_aug_alpha`.
2. **Final AP + IoU statistics** (``ap_results.json`` + ``iou_results.json``), in the
   same format as :mod:`hooks.ap_iou_final_results_hook`, computed with the same
   evaluators (:class:`~detectron2.evaluation.COCOEvaluator` and
   :class:`~hooks.iou_evaluator.PerImageIoUEvaluator`) but on **perturbed** test data,
   keyed by ``_clean`` / per-corruption / ``_aggregate``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    build_detection_test_loader,
    detection_utils as utils,
)
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset
from detectron2.utils.events import get_event_storage

from augmentations.heldout_augmentations import HELDOUT_PERTURBATION_REGISTRY
from hooks.iou_evaluator import PerImageIoUEvaluator
from hooks.plot_style import apply_style
from sensitivity.metrics import evaluate_aug_alpha

logger = logging.getLogger(__name__)

_AP_KEYS = ("AP", "AP50", "AP75", "APs", "APm", "APl")


class NucleiPostTrainHook(HookBase):
    """Evaluate the trained model on held-out (unseen) corruptions of the test set.

    Args:
        output_dir: Base directory; results go in ``<output_dir>/json_<job>_posteval/``.
        cfg: Detectron2 config (for INPUT/test-loader settings and the evaluators).
        test_dataset: Registered test split to perturb and evaluate.
        corruption_names: Subset of :data:`HELDOUT_PERTURBATION_REGISTRY` to use
            (defaults to all).
        grid_size: Number of uniform magnitudes in ``[0, 1]`` for the degradation
            curves (index 0 is the clean baseline).
        severities: Magnitudes at which to run the (heavier) AP+IoU evaluators.
            Defaults to ``(1.0,)`` — the strongest / worst-case corruption.
        max_images: Cap on test images used for the lightweight curve sweep.
        job_name: Drives the dedicated output-folder name and plot titles.
        save_json: If False, only the plot is written (curves/stats JSON skipped).
    """

    def __init__(
        self,
        output_dir: str,
        cfg,
        test_dataset: Optional[str] = None,
        corruption_names: Optional[Sequence[str]] = None,
        grid_size: Optional[int] = None,
        severities: Optional[Sequence[float]] = None,
        max_images: Optional[int] = None,
        job_name: str = "",
        save_json: bool = True,
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.cfg = cfg

        # Anything not passed explicitly falls back to the cfg.POST_EVAL namespace,
        # so the hook works both when auto-loaded from cfg.ADDTIONAL_HOOKS (which
        # only forwards cfg/output_dir/job_name) and when constructed directly.
        pe = getattr(cfg, "POST_EVAL", None)
        self.enabled = bool(getattr(pe, "ENABLED", True))

        test_sets = list(getattr(cfg.DATASETS, "TEST", []) or [])
        self.test_dataset = test_dataset or (test_sets[0] if test_sets else None)

        if grid_size is None:
            grid_size = getattr(pe, "GRID_SIZE", 6)
        self.grid_size = max(2, int(grid_size))

        if severities is None:
            severities = list(getattr(pe, "SEVERITIES", [1.0]) or [1.0])
        self.severities = [float(s) for s in severities]

        if max_images is None:
            max_images = getattr(pe, "MAX_IMAGES", 200)
        self.max_images = int(max_images)

        if corruption_names is None:
            corruption_names = list(getattr(pe, "CORRUPTIONS", []) or [])
        self.corruption_names = (
            [str(c) for c in corruption_names]
            if corruption_names
            else list(HELDOUT_PERTURBATION_REGISTRY.keys())
        )

        self.job_name = job_name or getattr(cfg, "JOBNAME", "") or "model"
        self.save_json = save_json

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # HookBase interface
    # ------------------------------------------------------------------
    def after_train(self) -> None:
        if not self.enabled:
            logger.info("[NucleiPostEval] disabled (POST_EVAL.ENABLED=False); skipping.")
            return
        if not self.test_dataset:
            logger.warning("[NucleiPostEval] no test dataset configured; skipping.")
            return

        json_dir = os.path.join(self.output_dir, f"json_{self.job_name}_posteval")
        os.makedirs(json_dir, exist_ok=True)
        self._coco_tmp_dir = os.path.join(json_dir, "_coco_tmp")
        os.makedirs(self._coco_tmp_dir, exist_ok=True)

        model = self.trainer.model
        cfg = self.cfg
        was_training = model.training
        model.eval()
        curves: Optional[Dict] = None
        ap_record: Dict = {}
        iou_record: Dict = {}
        try:
            with torch.no_grad():
                curves = self._run_curves(model, cfg)
                ap_record, iou_record = self._run_final_stats(model, cfg)
        except Exception:  # post-train eval must not mask a successful training run
            logger.exception("[NucleiPostEval] evaluation failed.")
        finally:
            if was_training:
                model.train()

        # Save whatever completed — the curves and the AP/IoU stats are independent,
        # so a failure in one still persists the other.
        if curves is not None:
            self._save_curves(json_dir, curves)
            self._log_summary(curves, ap_record, iou_record)
        if ap_record or iou_record:
            self._save_stats(json_dir, ap_record, iou_record)

    # ------------------------------------------------------------------
    # (a) Degradation curves — foreground IoU (MA) vs magnitude
    # ------------------------------------------------------------------
    def _run_curves(self, model, cfg) -> Dict:
        dataset_dicts = list(DatasetCatalog.get(self.test_dataset))[: self.max_images]
        alphas = np.linspace(0.0, 1.0, self.grid_size)

        clean_ma, _ = evaluate_aug_alpha(model, cfg, dataset_dicts, aug_factory=None, alpha=0.0)

        corruptions: Dict[str, dict] = {}
        for name in self.corruption_names:
            factory = HELDOUT_PERTURBATION_REGISTRY[name]
            ma_list: List[float] = []
            for a in alphas:
                if a <= 0.0:
                    ma_list.append(clean_ma)
                    continue
                ma, _ = evaluate_aug_alpha(model, cfg, dataset_dicts, factory, float(a))
                ma_list.append(ma)
            corrupted = [ma_list[i] for i in range(len(alphas)) if alphas[i] > 0.0]
            corruptions[name] = {
                "alphas": alphas.tolist(),
                "ma": ma_list,
                "mean_ma_corrupted": float(np.mean(corrupted)) if corrupted else clean_ma,
                "worst_ma": float(min(ma_list)),
            }

        per_corruption_means = [v["mean_ma_corrupted"] for v in corruptions.values()]
        mean_corruption_ma = float(np.mean(per_corruption_means)) if per_corruption_means else clean_ma
        return {
            "test_dataset": self.test_dataset,
            "num_images": len(dataset_dicts),
            "alphas": alphas.tolist(),
            "clean_ma": clean_ma,
            "corruptions": corruptions,
            "mean_corruption_ma": mean_corruption_ma,
            "robustness_gap": clean_ma - mean_corruption_ma,
        }

    # ------------------------------------------------------------------
    # (b) Final AP + IoU statistics on perturbed test data
    # ------------------------------------------------------------------
    def _perturbed_test_loader(self, cfg, corruption_aug):
        """Test loader that applies ``corruption_aug`` after the standard test resize."""
        augs = utils.build_augmentation(cfg, is_train=False)  # [ResizeShortestEdge]
        if corruption_aug is not None:
            augs.append(corruption_aug)
        mapper = DatasetMapper(cfg, is_train=False, augmentations=augs)
        return build_detection_test_loader(cfg, self.test_dataset, mapper=mapper)

    def _eval_one(self, model, cfg, corruption_aug) -> Tuple[dict, dict]:
        """Run COCO AP + per-image IoU in a single inference pass over perturbed data.

        A failure of a single condition (e.g. an evaluator/pycocotools error) is
        logged and reported as zeros so the remaining corruptions still evaluate.
        """
        try:
            loader = self._perturbed_test_loader(cfg, corruption_aug)
            coco_eval = COCOEvaluator(
                self.test_dataset,
                distributed=(cfg.MODEL.DEVICE != "cpu"),
                output_dir=self._coco_tmp_dir,
            )
            iou_eval = PerImageIoUEvaluator(self.test_dataset)
            results = inference_on_dataset(model, loader, DatasetEvaluators([coco_eval, iou_eval]))
        except Exception:
            logger.exception("[NucleiPostEval] AP/IoU eval failed for one condition; recording zeros.")
            results = {}

        segm = results.get("segm", {})
        ap_d = {k: float(segm.get(k, 0.0)) for k in _AP_KEYS}
        iou_d = {
            "mean_iou": float(results.get("mean_iou", 0.0)),
            "mean_foreground_iou": float(results.get("mean_foreground_iou", 0.0)),
            "bucket_counts": results.get("bucket_counts", {}),
            "total_evaluated_images": int(results.get("total_evaluated_images", 0)),
        }
        return ap_d, iou_d

    def _run_final_stats(self, model, cfg) -> Tuple[Dict, Dict]:
        ap_record: Dict[str, dict] = {}
        iou_record: Dict[str, dict] = {}

        # Clean baseline (no corruption), evaluated once.
        ap_record["_clean"], iou_record["_clean"] = self._eval_one(model, cfg, None)

        sevs = [s for s in self.severities if s > 0.0]
        for name in self.corruption_names:
            factory = HELDOUT_PERTURBATION_REGISTRY[name]
            ap_record[name] = {}
            iou_record[name] = {}
            for sev in sevs:
                ap_d, iou_d = self._eval_one(model, cfg, factory(magnitude=float(sev)))
                key = f"{sev:.2f}"
                ap_record[name][key] = ap_d
                iou_record[name][key] = iou_d

        # Aggregate over corruptions at the strongest severity = "mean corruption" robustness.
        top_key = f"{max(sevs):.2f}" if sevs else None
        ap_record["_aggregate"] = self._aggregate_ap(ap_record, top_key)
        iou_record["_aggregate"] = self._aggregate_iou(iou_record, top_key)
        return ap_record, iou_record

    @staticmethod
    def _aggregate_ap(record: Dict[str, dict], sev_key: Optional[str]) -> dict:
        vals = {k: [] for k in _AP_KEYS}
        for name, sev_map in record.items():
            if name.startswith("_") or sev_key is None:
                continue
            d = sev_map.get(sev_key)
            if d:
                for k in _AP_KEYS:
                    vals[k].append(d.get(k, 0.0))
        return {k: (float(np.mean(v)) if v else 0.0) for k, v in vals.items()}

    @staticmethod
    def _aggregate_iou(record: Dict[str, dict], sev_key: Optional[str]) -> dict:
        mi, mfi = [], []
        for name, sev_map in record.items():
            if name.startswith("_") or sev_key is None:
                continue
            d = sev_map.get(sev_key)
            if d:
                mi.append(d.get("mean_iou", 0.0))
                mfi.append(d.get("mean_foreground_iou", 0.0))
        return {
            "mean_iou": float(np.mean(mi)) if mi else 0.0,
            "mean_foreground_iou": float(np.mean(mfi)) if mfi else 0.0,
        }

    # ------------------------------------------------------------------
    # Persistence / logging
    # ------------------------------------------------------------------
    def _save_curves(self, json_dir: str, curves: Dict) -> None:
        self._save_plot(curves, os.path.join(json_dir, "posteval_curves.png"))
        if self.save_json:
            with open(os.path.join(json_dir, "posteval_curves.json"), "w") as f:
                json.dump(curves, f, indent=4)

    def _save_stats(self, json_dir: str, ap_record: Dict, iou_record: Dict) -> None:
        if not self.save_json:
            return
        with open(os.path.join(json_dir, "ap_results.json"), "w") as f:
            json.dump(ap_record, f, indent=4)
        with open(os.path.join(json_dir, "iou_results.json"), "w") as f:
            json.dump(iou_record, f, indent=4)
        logger.info("[NucleiPostEval] Saved AP/IoU stats + curves → %s", json_dir)

    def _save_plot(self, curves: Dict, path: str) -> None:
        apply_style()
        alphas = curves["alphas"]
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, d in curves["corruptions"].items():
            ax.plot(alphas, d["ma"], marker="o", label=name)
        ax.axhline(curves["clean_ma"], ls="--", color="#333333", linewidth=1.5, label="clean")
        ax.set_title(f"Held-out corruption robustness — {self.job_name} — Test set")
        ax.set_xlabel("Corruption magnitude")
        ax.set_ylabel("Foreground IoU (MA)")
        ax.legend(fontsize=8, ncol=2)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def _log_summary(self, curves: Dict, ap_record: Dict, iou_record: Dict) -> None:
        clean_ap = ap_record.get("_clean", {})
        agg_ap = ap_record.get("_aggregate", {})
        clean_iou = iou_record.get("_clean", {})
        agg_iou = iou_record.get("_aggregate", {})
        logger.info(
            "[NucleiPostEval] test=%s | AP %.3f -> %.3f (mean-corruption) | "
            "fgIoU %.3f -> %.3f | curve robustness_gap=%.4f",
            self.test_dataset,
            clean_ap.get("AP", 0.0), agg_ap.get("AP", 0.0),
            clean_iou.get("mean_foreground_iou", 0.0), agg_iou.get("mean_foreground_iou", 0.0),
            curves.get("robustness_gap", 0.0),
        )
        try:
            storage = get_event_storage()
            storage.put_scalar("posteval/clean_AP", clean_ap.get("AP", 0.0))
            storage.put_scalar("posteval/mean_corruption_AP", agg_ap.get("AP", 0.0))
            storage.put_scalar("posteval/robustness_gap", curves.get("robustness_gap", 0.0))
        except Exception:
            pass  # event storage may be unavailable after training
