"""Held-out corruptions for the post-training robustness evaluation.

These are the **unseen** corruptions: they are *not* in
:data:`augmentations.augmentations.PERTURBATION_REGISTRY` and are therefore never
applied during SensAug training.  The :class:`~hooks.nuclei_post_eval_hook.NucleiPostTrainHook`
sweeps them over the test split to measure how the trained model degrades on corruptions it
has never seen — mirroring the SensAug paper's held-out / ImageNet-C-style robustness
evaluation (Zheng et al., arXiv:2406.01425).

Every corruption here is **photometric** (it changes pixel values only, never moves them), so
the original ground-truth masks/boxes remain valid under perturbation.  This is what lets the
evaluators read GT at the original resolution while the model sees the corrupted image.

The set is curated for **nuclei microscopy** (BBBC038): out-of-focus and motion blur, photon /
sensor / impulse noise, staining-contrast and illumination shifts, gamma response, and digital
storage / resolution artifacts.  Each corruption is mechanically distinct from the trained
basis set (Gaussian blur, Gaussian noise, per-channel RGB/HSV lighter/darker).

Interface mirrors :mod:`augmentations.augmentations`: each corruption is a Detectron2
``Augmentation`` taking a single ``magnitude`` in ``[0, 1]`` (``magnitude == 0`` ⇒ identity),
and :data:`HELDOUT_PERTURBATION_REGISTRY` maps a short name to a ``magnitude -> Augmentation``
factory, so ``evaluate_aug_alpha`` and a Detectron2 test-loader augmentation list can both use
them unchanged.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from detectron2.data.transforms import Augmentation, Transform, NoOpTransform

# ---------------------------------------------------------------------------
# Magnitude → native-strength caps (tunable; mirror MAX_BLUR/MAX_NOISE/MAX_COLOR
# in augmentations.augmentations).  magnitude == 0 is always identity (NoOp).
# ---------------------------------------------------------------------------
MAX_DEFOCUS_RADIUS = 8     # disk-blur radius (px) at magnitude 1
MAX_MOTION_LEN = 20        # motion-blur kernel length (px) at magnitude 1
SHOT_C_CLEAN = 60.0        # Poisson photon count at magnitude→0 (low noise)
SHOT_C_WORST = 3.0         # Poisson photon count at magnitude 1 (high noise)
MAX_IMPULSE = 0.15         # salt&pepper fraction of pixels at magnitude 1
MAX_SPECKLE = 0.6          # multiplicative-noise sigma at magnitude 1
MAX_CONTRAST_DROP = 0.85   # contrast reduced to (1 - this) at magnitude 1
MAX_BRIGHT_SHIFT = 0.4     # additive brightness as fraction of 255 at magnitude 1
MAX_GAMMA = 2.0            # gamma exponent = 1 + this at magnitude 1 (darkening)
JPEG_Q_CLEAN = 95          # JPEG quality at magnitude→0
JPEG_Q_WORST = 5           # JPEG quality at magnitude 1
PIXELATE_MIN_SCALE = 0.1   # downsample scale at magnitude 1


def _clamp01(magnitude: float) -> float:
    """Clamp a magnitude into ``[0, 1]`` (mirrors ``_clamp_magnitude`` in augmentations.py)."""
    return float(np.clip(abs(magnitude), 0.0, 1.0))


def _lerp(lo: float, hi: float, m: float) -> float:
    """Linear interpolation ``lo -> hi`` as ``m`` goes ``0 -> 1``."""
    return float(lo + (hi - lo) * m)


def _disk_kernel(radius: int) -> Optional[np.ndarray]:
    """Normalized circular (disk) averaging kernel of the given radius."""
    r = int(radius)
    if r < 1:
        return None
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    mask = (xx * xx + yy * yy) <= (r * r)
    k = mask.astype(np.float32)
    total = k.sum()
    return k / total if total > 0 else None


def _motion_kernel(length: int, angle_deg: float) -> Optional[np.ndarray]:
    """Normalized linear motion-blur kernel of the given length and angle."""
    L = int(length)
    if L < 2:
        return None
    k = np.zeros((L, L), dtype=np.float32)
    k[L // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((L / 2.0 - 0.5, L / 2.0 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, rot, (L, L))
    total = k.sum()
    return k / total if total > 0 else None


# ---------------------------------------------------------------------------
# Transforms (photometric: coords + segmentation pass through unchanged so the
# corruption never moves/relabels the ground truth).
# ---------------------------------------------------------------------------
class _PhotometricTransform(Transform):
    """Base for pixel-only corruptions; coords and segmentation are untouched."""

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class DefocusBlurTransform(_PhotometricTransform):
    def __init__(self, radius: int):
        super().__init__()
        self.kernel = _disk_kernel(radius)

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if self.kernel is None:
            return img
        return cv2.filter2D(img, -1, self.kernel)


class MotionBlurTransform(_PhotometricTransform):
    def __init__(self, length: int, angle_deg: float):
        super().__init__()
        self.kernel = _motion_kernel(length, angle_deg)

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if self.kernel is None:
            return img
        return cv2.filter2D(img, -1, self.kernel)


class ShotNoiseTransform(_PhotometricTransform):
    """Poisson (photon shot) noise; smaller ``c`` = stronger noise."""

    def __init__(self, c: float, rng: np.random.Generator):
        super().__init__()
        self.c = float(c)
        self.rng = rng

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        x = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)
        noisy = self.rng.poisson(x * self.c) / float(self.c)
        return np.clip(noisy * 255.0, 0, 255).astype(img.dtype)


class ImpulseNoiseTransform(_PhotometricTransform):
    """Salt-and-pepper (impulse) noise — dead/hot sensor pixels."""

    def __init__(self, amount: float, rng: np.random.Generator):
        super().__init__()
        self.amount = float(amount)
        self.rng = rng

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        out = img.copy()
        r = self.rng.random(img.shape[:2])
        out[r < self.amount / 2.0] = 0
        out[r > 1.0 - self.amount / 2.0] = 255
        return out


class SpeckleNoiseTransform(_PhotometricTransform):
    """Multiplicative (speckle) sensor noise: ``img * (1 + N(0, sigma))``."""

    def __init__(self, sigma: float, rng: np.random.Generator):
        super().__init__()
        self.sigma = float(sigma)
        self.rng = rng

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(0.0, self.sigma, img.shape).astype(np.float32)
        out = img.astype(np.float32) * (1.0 + noise)
        return np.clip(out, 0, 255).astype(img.dtype)


class ContrastTransform(_PhotometricTransform):
    """Reduce contrast toward the per-channel mean (weak/variable staining)."""

    def __init__(self, factor: float):
        super().__init__()
        self.factor = float(factor)

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32)
        mean = x.mean(axis=(0, 1), keepdims=True)
        out = mean + (x - mean) * self.factor
        return np.clip(out, 0, 255).astype(img.dtype)


class BrightnessTransform(_PhotometricTransform):
    """Additive brightness shift (illumination / exposure variation)."""

    def __init__(self, delta: float):
        super().__init__()
        self.delta = float(delta)

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        out = img.astype(np.float32) + self.delta
        return np.clip(out, 0, 255).astype(img.dtype)


class GammaTransform(_PhotometricTransform):
    """Nonlinear intensity (gamma) remap via a uint8 lookup table."""

    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = float(gamma)
        lut = (np.power(np.arange(256, dtype=np.float32) / 255.0, self.gamma) * 255.0)
        self.lut = np.clip(lut, 0, 255).astype(np.uint8)

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            base = np.clip(img, 0, 255).astype(np.uint8)
            return cv2.LUT(base, self.lut).astype(img.dtype)
        return cv2.LUT(img, self.lut)


class JpegCompressionTransform(_PhotometricTransform):
    """JPEG compression artifacts (digital-pathology storage)."""

    def __init__(self, quality: int):
        super().__init__()
        self.quality = int(quality)

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return img
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return dec.astype(img.dtype)


class PixelateTransform(_PhotometricTransform):
    """Downsample (box) then upsample (nearest) — magnification/resolution loss."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        nh, nw = max(1, int(round(h * self.scale))), max(1, int(round(w * self.scale)))
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Augmentation wrappers (magnitude in [0, 1] -> native strength).
# ---------------------------------------------------------------------------
class DefocusBlurPerturbation(Augmentation):
    def __init__(self, magnitude: float = 0.5):
        super().__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> Transform:
        m = _clamp01(self.magnitude)
        if m <= 0.0:
            return NoOpTransform()
        return DefocusBlurTransform(radius=1 + int(m * MAX_DEFOCUS_RADIUS))


class MotionBlurPerturbation(Augmentation):
    def __init__(self, magnitude: float = 0.5):
        super().__init__()
        self._init(locals())
        self._rng = np.random.default_rng()

    def get_transform(self, image: np.ndarray) -> Transform:
        m = _clamp01(self.magnitude)
        if m <= 0.0:
            return NoOpTransform()
        length = 2 + int(m * MAX_MOTION_LEN)
        angle = float(self._rng.uniform(0.0, 180.0))
        return MotionBlurTransform(length=length, angle_deg=angle)


class ShotNoisePerturbation(Augmentation):
    def __init__(self, magnitude: float = 0.5):
        super().__init__()
        self._init(locals())
        self._rng = np.random.default_rng()

    def get_transform(self, image: np.ndarray) -> Transform:
        m = _clamp01(self.magnitude)
        if m <= 0.0:
            return NoOpTransform()
        return ShotNoiseTransform(c=_lerp(SHOT_C_CLEAN, SHOT_C_WORST, m), rng=self._rng)


class ImpulseNoisePerturbation(Augmentation):
    def __init__(self, magnitude: float = 0.5):
        super().__init__()
        self._init(locals())
        self._rng = np.random.default_rng()

    def get_transform(self, image: np.ndarray) -> Transform:
        m = _clamp01(self.magnitude)
        if m <= 0.0:
            return NoOpTransform()
        return ImpulseNoiseTransform(amount=m * MAX_IMPULSE, rng=self._rng)


class SpeckleNoisePerturbation(Augmentation):
    def __init__(self, magnitude: float = 0.5):
        super().__init__()
        self._init(locals())
        self._rng = np.random.default_rng()

    def get_transform(self, image: np.ndarray) -> Transform:
        m = _clamp01(self.magnitude)
        if m <= 0.0:
            return NoOpTransform()
        return SpeckleNoiseTransform(sigma=m * MAX_SPECKLE, rng=self._rng)


class ContrastPerturbation(Augmentation):
    def __init__(self, magnitude: float = 0.5):
        super().__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> Transform:
        m = _clamp01(self.magnitude)
        if m <= 0.0:
            return NoOpTransform()
        return ContrastTransform(factor=1.0 - m * MAX_CONTRAST_DROP)


class BrightnessPerturbation(Augmentation):
    def __init__(self, magnitude: float = 0.5):
        super().__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> Transform:
        m = _clamp01(self.magnitude)
        if m <= 0.0:
            return NoOpTransform()
        return BrightnessTransform(delta=m * MAX_BRIGHT_SHIFT * 255.0)


class GammaPerturbation(Augmentation):
    def __init__(self, magnitude: float = 0.5):
        super().__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> Transform:
        m = _clamp01(self.magnitude)
        if m <= 0.0:
            return NoOpTransform()
        return GammaTransform(gamma=1.0 + m * MAX_GAMMA)


class JpegCompressionPerturbation(Augmentation):
    def __init__(self, magnitude: float = 0.5):
        super().__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> Transform:
        m = _clamp01(self.magnitude)
        if m <= 0.0:
            return NoOpTransform()
        return JpegCompressionTransform(quality=int(round(_lerp(JPEG_Q_CLEAN, JPEG_Q_WORST, m))))


class PixelatePerturbation(Augmentation):
    def __init__(self, magnitude: float = 0.5):
        super().__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> Transform:
        m = _clamp01(self.magnitude)
        if m <= 0.0:
            return NoOpTransform()
        return PixelateTransform(scale=_lerp(1.0, PIXELATE_MIN_SCALE, m))


# ---------------------------------------------------------------------------
# Registry of held-out corruptions.  Each value is a ``magnitude -> Augmentation``
# factory, matching the calling convention of
# ``augmentations.augmentations.PERTURBATION_REGISTRY`` (and
# ``sensitivity.metrics.evaluate_aug_alpha``'s ``aug_factory(magnitude=alpha)``).
# Names are intentionally disjoint from PERTURBATION_REGISTRY (the trained set).
# ---------------------------------------------------------------------------
HELDOUT_PERTURBATION_REGISTRY: dict = {
    "defocus_blur":     lambda magnitude: DefocusBlurPerturbation(magnitude=magnitude),
    "motion_blur":      lambda magnitude: MotionBlurPerturbation(magnitude=magnitude),
    "shot_noise":       lambda magnitude: ShotNoisePerturbation(magnitude=magnitude),
    "impulse_noise":    lambda magnitude: ImpulseNoisePerturbation(magnitude=magnitude),
    "speckle_noise":    lambda magnitude: SpeckleNoisePerturbation(magnitude=magnitude),
    "contrast":         lambda magnitude: ContrastPerturbation(magnitude=magnitude),
    "brightness":       lambda magnitude: BrightnessPerturbation(magnitude=magnitude),
    "gamma":            lambda magnitude: GammaPerturbation(magnitude=magnitude),
    "jpeg_compression": lambda magnitude: JpegCompressionPerturbation(magnitude=magnitude),
    "pixelate":         lambda magnitude: PixelatePerturbation(magnitude=magnitude),
}
