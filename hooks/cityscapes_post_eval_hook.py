"""CityscapesPostTrainHook — post-training robustness eval for the Cityscapes dataset.

This is the Cityscapes counterpart of :class:`~hooks.nuclei_post_eval_hook.NucleiPostTrainHook`.
It reuses essentially all of that hook's machinery (perturbed test loader,
COCO AP + per-image IoU evaluators, aggregation, persistence) and changes only
the two things that are dataset-specific:

1.  **The held-out corruption set.**  Instead of the nuclei-microscopy registry it
    sweeps :data:`augmentations.cityscapes_heldout_augmentations.CITYSCAPES_HELDOUT_PERTURBATION_REGISTRY`
    — the ImageNet-C photometric corruptions plus the adverse-weather / driving
    additions (fog, snow, frost, low-light/night, rain-spatter, AdvSteer) that the
    SensAug paper's Cityscapes robustness benchmark centers on (Zheng et al.,
    arXiv:2406.01425).

2.  **The degradation curves report both metrics.**  Cityscapes has 8 instance
    classes, so the class-agnostic foreground-IoU proxy (merge-all-instances) used
    for nuclei is kept only as a *fast secondary* signal, swept on a fine magnitude
    grid over the capped test subset.  The **headline** curve is class-aware COCO
    **mask AP** as a function of corruption severity, taken from the per-severity
    AP that :meth:`_run_final_stats` already computes over the *full* test split
    (the statistically correct COCO AP — restricting AP to a subset would penalise
    unseen images).  Both series are plotted side-by-side and persisted.

Outputs are identical in shape to the nuclei hook, written to
``<output_dir>/json_<job>_posteval/``: ``posteval_curves.{json,png}``,
``ap_results.json``, ``iou_results.json``.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from augmentations.cityscapes_heldout_augmentations import (
    CITYSCAPES_HELDOUT_PERTURBATION_REGISTRY,
)
from hooks.nuclei_post_eval_hook import NucleiPostTrainHook
from hooks.plot_style import apply_style


class CityscapesPostTrainHook(NucleiPostTrainHook):
    """Held-out corruption robustness eval tailored to Cityscapes instance segmentation.

    Constructed exactly like :class:`NucleiPostTrainHook` (and auto-loadable via
    ``cfg.ADDTIONAL_HOOKS`` / ``cfg.POST_EVAL.*``); it deliberately does **not**
    override ``__init__`` so the additional-hooks loader's signature inspection
    keeps forwarding only ``cfg`` / ``output_dir`` / ``job_name``.
    """

    #: Swap the swept registry for the Cityscapes / driving corruption set.
    registry: dict = CITYSCAPES_HELDOUT_PERTURBATION_REGISTRY

    #: Per-severity AP captured in :meth:`_run_final_stats`, consumed by the plot.
    _last_ap_record: dict = {}

    # ------------------------------------------------------------------
    # Capture the (full-test-set) per-severity AP so the curve plot can use it.
    # _run_final_stats runs before _save_curves in NucleiPostTrainHook.after_train,
    # so self._last_ap_record is populated by the time we plot.
    # ------------------------------------------------------------------
    def _run_final_stats(self, model, cfg):
        ap_record, iou_record = super()._run_final_stats(model, cfg)
        self._last_ap_record = ap_record
        return ap_record, iou_record

    def _ap_curve_from_record(self, ap_record: Dict) -> Dict:
        """Reshape the per-corruption/per-severity AP record into a plottable curve."""
        if not ap_record:
            return {}
        sevs = sorted(s for s in self.severities if s > 0.0)
        keys = [f"{s:.2f}" for s in sevs]
        clean_ap = float(ap_record.get("_clean", {}).get("AP", 0.0))
        corruptions = {
            name: [float(ap_record.get(name, {}).get(k, {}).get("AP", 0.0)) for k in keys]
            for name in self.corruption_names
        }
        return {"severities": sevs, "clean_ap": clean_ap, "corruptions": corruptions}

    # ------------------------------------------------------------------
    # Inject the AP-vs-severity series so it is both plotted and persisted in
    # posteval_curves.json (keeping that file self-contained).
    # ------------------------------------------------------------------
    def _save_curves(self, json_dir: str, curves: Dict) -> None:
        enriched = dict(curves)
        enriched["ap_by_severity"] = self._ap_curve_from_record(
            getattr(self, "_last_ap_record", {})
        )
        super()._save_curves(json_dir, enriched)

    def _save_plot(self, curves: Dict, path: str) -> None:
        apply_style()
        ap_curve = curves.get("ap_by_severity") or {}
        has_ap = bool(ap_curve.get("corruptions"))

        ncols = 2 if has_ap else 1
        fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 6))
        axes = np.atleast_1d(axes)
        col = 0

        # Headline panel: class-aware mask AP vs severity.
        if has_ap:
            ax = axes[col]
            col += 1
            sevs = ap_curve["severities"]
            for name, ap_list in ap_curve["corruptions"].items():
                ax.plot(sevs, ap_list, marker="o", label=name)
            ax.axhline(
                ap_curve.get("clean_ap", 0.0), ls="--", color="#333333", linewidth=1.5, label="clean"
            )
            ax.set_title("Mask AP vs corruption severity")
            ax.set_xlabel("Corruption severity")
            ax.set_ylabel("Mask AP (segm)")
            ax.legend(fontsize=7, ncol=2)

        # Secondary panel: fast class-agnostic foreground-IoU proxy vs magnitude.
        ax = axes[col]
        alphas = curves["alphas"]
        for name, d in curves["corruptions"].items():
            ax.plot(alphas, d["ma"], marker="o", label=name)
        ax.axhline(curves["clean_ma"], ls="--", color="#333333", linewidth=1.5, label="clean")
        ax.set_title("Foreground IoU (MA, class-agnostic) vs magnitude")
        ax.set_xlabel("Corruption magnitude")
        ax.set_ylabel("Foreground IoU (MA)")
        ax.legend(fontsize=7, ncol=2)

        fig.suptitle(
            f"Cityscapes held-out corruption robustness — {self.job_name} — Test set",
            y=1.02,
        )
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
