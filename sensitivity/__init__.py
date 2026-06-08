"""Sensitivity-analysis primitives for the Detectron2 SensAug port.

This package isolates the math and evaluation building blocks used by
:class:`~hooks.sens_aug_hook.SensAugHook`:

* :mod:`sensitivity.analysis` — pure (numpy/scipy) functions that turn measured
  accuracy/degradation curves into per-augmentation intensity levels and a
  Beta-Binomial sampling distribution.  No torch, easy to unit-test.
* :mod:`sensitivity.kid`      — Kernel Inception Distance wrapper (torchmetrics).
* :mod:`sensitivity.metrics`  — per-image mask-mIoU evaluation of the model under
  a perturbation at a given magnitude.
"""

from .analysis import (
    build_g_curve,
    select_sensitive_levels,
    beta_binomial_weights,
    pchip_interp,
)

__all__ = [
    "build_g_curve",
    "select_sensitive_levels",
    "beta_binomial_weights",
    "pchip_interp",
]
