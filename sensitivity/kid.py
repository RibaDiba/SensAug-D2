"""Kernel Inception Distance (KID) — perceptual degradation measure.

SensAug normalizes the model's accuracy drop by *how much the image was actually
degraded*, measured as the KID between the clean and perturbed image sets.  This
module is a thin wrapper over :class:`torchmetrics.image.kid.KernelInceptionDistance`
that handles the bookkeeping our hook needs: variable-sized HWC ``uint8`` images,
resizing to the InceptionV3 input, device placement, and clamping the KID
``subset_size`` to the (often small) number of available images.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

#: InceptionV3 input resolution used by the KID feature extractor.
_INCEPTION_SIZE = 299

#: Weights torchmetrics/torch_fidelity lazily download on first KID use.
_INCEPTION_WEIGHTS_FILENAME = "weights-inception-2015-12-05-6726825d.pth"
_INCEPTION_WEIGHTS_URL = (
    "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/"
    + _INCEPTION_WEIGHTS_FILENAME
)

#: One metric instance per (subset_size, device) — constructing
#: KernelInceptionDistance reloads the ~91 MB InceptionV3 from disk every time,
#: and an SA pass calls compute_kid once per (perturbation, alpha) pair.
_KID_CACHE: Dict[Tuple[int, str], object] = {}


def _check_inception_weights_cached() -> None:
    """Fail fast if the InceptionV3 weights are not in the torch.hub cache.

    Cluster compute nodes have no internet access, so the lazy download inside
    ``KernelInceptionDistance`` would otherwise crash the run only *after* the
    warmup iterations, when the first sensitivity analysis fires.
    """
    path = os.path.join(torch.hub.get_dir(), "checkpoints", _INCEPTION_WEIGHTS_FILENAME)
    if not os.path.isfile(path):
        raise RuntimeError(
            f"InceptionV3 weights for KID not found at {path} and compute nodes "
            f"cannot download them. Pre-download on a node with internet access:\n"
            f"  mkdir -p {os.path.dirname(path)}\n"
            f"  wget -O {path} {_INCEPTION_WEIGHTS_URL}\n"
            f"(set TORCH_HOME to control the cache location)"
        )


def _get_kid_metric(subset_size: int, device: torch.device):
    # Imported lazily so the rest of the package works without torchmetrics
    # installed (e.g. when only running the pure-math unit tests).
    from torchmetrics.image.kid import KernelInceptionDistance

    key = (subset_size, str(device))
    metric = _KID_CACHE.get(key)
    if metric is None:
        _check_inception_weights_cached()
        metric = KernelInceptionDistance(subset_size=subset_size, normalize=False).to(device)
        _KID_CACHE[key] = metric
    else:
        metric.reset()
    return metric


def _to_uint8_nchw(images: Sequence[np.ndarray]) -> torch.Tensor:
    """Stack a list of HWC ``uint8`` images into an ``(N, 3, 299, 299)`` tensor.

    Images may have differing spatial sizes (val images at native resolution),
    so each is resized to the InceptionV3 input size before stacking.  Grayscale
    inputs are promoted to 3 channels.
    """
    out = []
    for img in images:
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = cv2.resize(arr, (_INCEPTION_SIZE, _INCEPTION_SIZE), interpolation=cv2.INTER_AREA)
        out.append(arr.transpose(2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(np.ascontiguousarray(np.stack(out))).to(torch.uint8)


def compute_kid(
    clean_images: Sequence[np.ndarray],
    perturbed_images: Sequence[np.ndarray],
    subset_size: int = 50,
    device: Optional[torch.device] = None,
) -> float:
    """Compute the KID between a clean and a perturbed set of images.

    Args:
        clean_images: List of HWC ``uint8`` images (the reference / "real" set).
        perturbed_images: List of HWC ``uint8`` images (the augmented / "fake" set).
        subset_size: KID subset size; clamped to ``<= min(len(clean), len(pert))``.
            Smaller sets (e.g. during smoke tests) are handled gracefully.
        device: Device to run the Inception feature extractor on.  Defaults to
            CUDA when available.

    Returns:
        The mean KID (a non-negative float; ``0.0`` if there are too few images
        to estimate it).
    """
    n = min(len(clean_images), len(perturbed_images))
    if n < 2:
        return 0.0

    subset = max(2, min(int(subset_size), n))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clean = _to_uint8_nchw(clean_images).to(device)
    perturbed = _to_uint8_nchw(perturbed_images).to(device)

    kid = _get_kid_metric(subset, device)
    kid.update(clean, real=True)
    kid.update(perturbed, real=False)
    mean, _std = kid.compute()
    return float(mean.detach().cpu().item())
