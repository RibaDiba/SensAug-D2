"""Held-out corruptions for the **Cityscapes** post-training robustness eval.

Where :mod:`augmentations.heldout_augmentations` curates corruptions for *nuclei
microscopy*, this module curates the held-out set for **Cityscapes / driving
scenes**, replicating the robustness benchmark the SensAug paper runs on
Cityscapes (Zheng et al., arXiv:2406.01425):

* the **ImageNet-C** synthetic corruptions (here: the *photometric* ones — blur,
  noise, contrast, brightness, jpeg, pixelate — reused unchanged from the nuclei
  registry, plus the **weather group** ``fog`` / ``snow`` / ``frost`` added below);
* a **low-light / night** condition mirroring ACDC's *night* split;
* a rain-occlusion ``spatter`` mirroring ACDC's *rain* split;
* an ``advsteer`` composite (intense combination of photometric perturbations),
  mirroring the paper's AdvSteer benchmark.

The real-world ACDC / IDD datasets in the paper are *external data*, not synthetic
perturbations, so they are out of scope here.

Every corruption is **photometric** — it changes pixel values only and never moves
them (``apply_coords`` / ``apply_segmentation`` pass through via
:class:`~augmentations.heldout_augmentations._PhotometricTransform`).  This keeps
the original instance masks/boxes valid under perturbation, the same invariant the
nuclei registry relies on, so geometric ImageNet-C corruptions
(elastic_transform, zoom/glass blur) are deliberately excluded.

Interface mirrors :mod:`augmentations.heldout_augmentations`: each corruption is a
Detectron2 ``Augmentation`` taking a single ``magnitude`` in ``[0, 1]``
(``magnitude == 0`` ⇒ identity), and
:data:`CITYSCAPES_HELDOUT_PERTURBATION_REGISTRY` maps a short name to a
``magnitude -> Augmentation`` factory, so ``evaluate_aug_alpha`` and a Detectron2
test-loader augmentation list can both use them unchanged.  Names are disjoint
from the trained ``augmentations.augmentations.PERTURBATION_REGISTRY``.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from detectron2.data.transforms import (
    Augmentation,
    NoOpTransform,
    Transform,
    TransformList,
)

# Reuse the shared helpers, photometric base, and the ImageNet-C photometric
# corruptions already implemented for the nuclei registry rather than
# re-implementing them.
from augmentations.heldout_augmentations import (
    MAX_BRIGHT_SHIFT,
    MAX_CONTRAST_DROP,
    MAX_MOTION_LEN,
    BrightnessPerturbation,
    BrightnessTransform,
    ContrastPerturbation,
    ContrastTransform,
    DefocusBlurPerturbation,
    ImpulseNoisePerturbation,
    JpegCompressionPerturbation,
    MotionBlurPerturbation,
    MotionBlurTransform,
    PixelatePerturbation,
    ShotNoisePerturbation,
    _PhotometricTransform,
    _clamp01,
    _lerp,
    _motion_kernel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Magnitude → native-strength caps for the weather/driving additions
# (mirror the MAX_* convention in augmentations.heldout_augmentations).
# ---------------------------------------------------------------------------
FOG_MAX_INTENSITY = 3.0      # additive fog haze strength at magnitude 1
FOG_PLASMA_MAPSIZE = 512     # fog map is generated at this (cheap) size then upscaled
SNOW_MAX_INTENSITY = 0.5     # snow-layer coverage/whiteness at magnitude 1
FROST_MAX_WEIGHT = 0.8       # frost-texture blend weight at magnitude 1
FROST_MIN_IMG_WEIGHT = 0.6   # original-image weight at magnitude 1 (1.0 at clean)
NIGHT_MIN_GAIN = 0.35        # multiplicative darkening at magnitude 1
NIGHT_MAX_GAMMA = 1.8        # gamma darkening exponent at magnitude 1
SPATTER_MAX_INTENSITY = 0.4  # rain-occlusion coverage at magnitude 1
ADVSTEER_MIN_STRENGTH = 0.4  # composite strength floor (magnitude→0+ is still "extreme")

_FROST_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "frost"
_FROST_ASSETS: Optional[List[np.ndarray]] = None
_FROST_LOCK = threading.Lock()
_FROST_WARNED = False


# ---------------------------------------------------------------------------
# Procedural helpers (no external assets needed for fog/snow).
# ---------------------------------------------------------------------------
def _plasma_fractal(mapsize: int, wibbledecay: float, rng: np.random.Generator) -> np.ndarray:
    """Diamond-square plasma fractal in ``[0, 1]`` (ImageNet-C fog generator).

    ``mapsize`` must be a power of two.  Smaller ``wibbledecay`` ⇒ more
    low-frequency (larger, denser) fog structures.
    """
    assert (mapsize & (mapsize - 1)) == 0, "mapsize must be a power of two"
    maparray = np.zeros((mapsize, mapsize), dtype=np.float64)
    stepsize = mapsize
    wibble = 100.0

    def wibbledmean(array):
        return array / 4.0 + wibble * rng.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, 1, axis=0)
        squareaccum += np.roll(squareaccum, 1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ldrsum + lulsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(tdrsum + tulsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    peak = maparray.max()
    return (maparray / peak) if peak > 0 else maparray


def _load_frost_assets() -> List[np.ndarray]:
    """Lazily load (and cache) frost texture images from ``assets/frost/``.

    Returns an empty list if the directory is missing or holds no images; the
    :class:`FrostPerturbation` falls back to a NoOp (warned once) in that case.
    """
    global _FROST_ASSETS
    if _FROST_ASSETS is not None:
        return _FROST_ASSETS
    with _FROST_LOCK:
        if _FROST_ASSETS is not None:
            return _FROST_ASSETS
        assets: List[np.ndarray] = []
        if _FROST_ASSET_DIR.is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
                for p in sorted(_FROST_ASSET_DIR.glob(ext)):
                    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if img is not None:
                        assets.append(img)
        _FROST_ASSETS = assets
        return _FROST_ASSETS


def _warn_no_frost_once() -> None:
    global _FROST_WARNED
    if not _FROST_WARNED:
        _FROST_WARNED = True
        logger.warning(
            "[CityscapesHeldout] 'frost' corruption has no textures in %s; "
            "falling back to identity. Drop frost PNG(s) there to enable it.",
            _FROST_ASSET_DIR,
        )


# ---------------------------------------------------------------------------
# Weather / driving transforms (photometric: coords + segmentation untouched).
# ---------------------------------------------------------------------------
class FogTransform(_PhotometricTransform):
    """Additive low-frequency haze blended toward atmospheric light (ImageNet-C fog)."""

    def __init__(self, intensity: float, wibbledecay: float, rng: np.random.Generator):
        super().__init__()
        self.intensity = float(intensity)
        self.wibbledecay = float(wibbledecay)
        self.rng = rng

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        x = img.astype(np.float32) / 255.0
        max_val = float(x.max()) if x.size else 1.0
        fog_small = _plasma_fractal(FOG_PLASMA_MAPSIZE, self.wibbledecay, self.rng).astype(np.float32)
        fog = cv2.resize(fog_small, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
        x = x + self.intensity * fog
        x = x * max_val / (max_val + self.intensity)
        return np.clip(x * 255.0, 0, 255).astype(img.dtype)


class SnowTransform(_PhotometricTransform):
    """Sparse bright snow layer with wind-driven motion-blur streaks (ImageNet-C / ACDC snow)."""

    def __init__(self, intensity: float, rng: np.random.Generator):
        super().__init__()
        self.intensity = float(intensity)
        self.rng = rng

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        x = img.astype(np.float32) / 255.0

        # Sparse bright speckles → more flakes as intensity rises.
        layer = self.rng.normal(loc=0.5, scale=0.3, size=(h, w)).astype(np.float32)
        layer = np.where(layer > (1.0 - self.intensity), np.clip(layer, 0.0, 1.0), 0.0)

        # Wind-driven streaks: near-vertical motion blur (reuses the shared kernel).
        kernel = _motion_kernel(
            length=3 + int(self.intensity * 12),
            angle_deg=90.0 + float(self.rng.uniform(-25.0, 25.0)),
        )
        if kernel is not None:
            layer = cv2.filter2D(layer, -1, kernel)
        layer = np.clip(layer, 0.0, 1.0)[..., None]

        # Slight overcast wash toward white, then screen-blend the snow on top.
        x = x * (1.0 - 0.2 * self.intensity) + 0.2 * self.intensity
        out = 1.0 - (1.0 - x) * (1.0 - layer)
        return np.clip(out * 255.0, 0, 255).astype(img.dtype)


class FrostTransform(_PhotometricTransform):
    """Alpha-blend a (pre-sized) frost texture over the image (ImageNet-C frost)."""

    def __init__(self, frost_patch: np.ndarray, img_weight: float, frost_weight: float):
        super().__init__()
        self.frost_patch = frost_patch
        self.img_weight = float(img_weight)
        self.frost_weight = float(frost_weight)

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        out = self.img_weight * img.astype(np.float32) + self.frost_weight * self.frost_patch.astype(np.float32)
        return np.clip(out, 0, 255).astype(img.dtype)


class LowLightTransform(_PhotometricTransform):
    """Night-time darkening via multiplicative gain + gamma (mirrors ACDC night)."""

    def __init__(self, gain: float, gamma: float):
        super().__init__()
        self.gain = float(gain)
        self.gamma = float(gamma)

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        x = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)
        x = np.power(x * self.gain, self.gamma)
        return np.clip(x * 255.0, 0, 255).astype(img.dtype)


class SpatterTransform(_PhotometricTransform):
    """Translucent rain occlusion: blurred droplet mask lightened over the image (ACDC rain proxy)."""

    def __init__(self, intensity: float, rng: np.random.Generator):
        super().__init__()
        self.intensity = float(intensity)
        self.rng = rng

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        liquid = self.rng.normal(loc=0.5, scale=0.3, size=(h, w)).astype(np.float32)
        liquid = cv2.GaussianBlur(liquid, (0, 0), sigmaX=_lerp(1.0, 3.0, self.intensity))
        mask = np.where(liquid > (1.0 - self.intensity), 1.0, 0.0).astype(np.float32)

        # Rain streaks: near-vertical motion blur of the droplet mask.
        kernel = _motion_kernel(length=3 + int(self.intensity * 14), angle_deg=90.0)
        if kernel is not None:
            mask = cv2.filter2D(mask, -1, kernel)
        mask = np.clip(mask, 0.0, 1.0)[..., None]

        alpha = 0.6 * self.intensity
        out = img.astype(np.float32) * (1.0 - alpha * mask) + 255.0 * alpha * mask
        return np.clip(out, 0, 255).astype(img.dtype)


# ---------------------------------------------------------------------------
# Augmentation wrappers (magnitude in [0, 1] -> native strength).
# ---------------------------------------------------------------------------
class FogPerturbation(Augmentation):
    def __init__(self, magnitude: float = 0.5):
        super().__init__()
        self._init(locals())
        self._rng = np.random.default_rng()

    def get_transform(self, image: np.ndarray) -> Transform:
        m = _clamp01(self.magnitude)
        if m <= 0.0:
            return NoOpTransform()
        # Heavier fog = more intensity and lower decay (larger, denser structures).
        return FogTransform(
            intensity=_lerp(0.2, FOG_MAX_INTENSITY, m),
            wibbledecay=_lerp(2.5, 1.4, m),
            rng=self._rng,
        )


class SnowPerturbation(Augmentation):
    def __init__(self, magnitude: float = 0.5):
        super().__init__()
        self._init(locals())
        self._rng = np.random.default_rng()

    def get_transform(self, image: np.ndarray) -> Transform:
        m = _clamp01(self.magnitude)
        if m <= 0.0:
            return NoOpTransform()
        return SnowTransform(intensity=_lerp(0.05, SNOW_MAX_INTENSITY, m), rng=self._rng)


class FrostPerturbation(Augmentation):
    def __init__(self, magnitude: float = 0.5):
        super().__init__()
        self._init(locals())
        self._rng = np.random.default_rng()

    def get_transform(self, image: np.ndarray) -> Transform:
        m = _clamp01(self.magnitude)
        if m <= 0.0:
            return NoOpTransform()
        assets = _load_frost_assets()
        if not assets:
            _warn_no_frost_once()
            return NoOpTransform()

        h, w = image.shape[:2]
        frost = assets[int(self._rng.integers(len(assets)))]
        fh, fw = frost.shape[:2]
        if fh >= h and fw >= w:
            top = int(self._rng.integers(0, fh - h + 1))
            left = int(self._rng.integers(0, fw - w + 1))
            patch = frost[top:top + h, left:left + w]
        else:
            patch = cv2.resize(frost, (w, h), interpolation=cv2.INTER_LINEAR)
        return FrostTransform(
            frost_patch=patch,
            img_weight=_lerp(1.0, FROST_MIN_IMG_WEIGHT, m),
            frost_weight=_lerp(0.15, FROST_MAX_WEIGHT, m),
        )


class LowLightPerturbation(Augmentation):
    def __init__(self, magnitude: float = 0.5):
        super().__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> Transform:
        m = _clamp01(self.magnitude)
        if m <= 0.0:
            return NoOpTransform()
        return LowLightTransform(
            gain=_lerp(1.0, NIGHT_MIN_GAIN, m),
            gamma=_lerp(1.0, NIGHT_MAX_GAMMA, m),
        )


class SpatterPerturbation(Augmentation):
    def __init__(self, magnitude: float = 0.5):
        super().__init__()
        self._init(locals())
        self._rng = np.random.default_rng()

    def get_transform(self, image: np.ndarray) -> Transform:
        m = _clamp01(self.magnitude)
        if m <= 0.0:
            return NoOpTransform()
        return SpatterTransform(intensity=_lerp(0.05, SPATTER_MAX_INTENSITY, m), rng=self._rng)


class AdvSteerPerturbation(Augmentation):
    """Composite of intense photometric perturbations (the paper's AdvSteer benchmark).

    Chains a contrast drop, a darkening shift, and a motion blur — a deliberately
    *extreme*, out-of-training-distribution combination — into a single
    :class:`~detectron2.data.transforms.TransformList`.
    """

    def __init__(self, magnitude: float = 0.5):
        super().__init__()
        self._init(locals())
        self._rng = np.random.default_rng()

    def get_transform(self, image: np.ndarray) -> Transform:
        m = _clamp01(self.magnitude)
        if m <= 0.0:
            return NoOpTransform()
        s = _lerp(ADVSTEER_MIN_STRENGTH, 1.0, m)
        transforms: List[Transform] = [
            ContrastTransform(factor=1.0 - s * MAX_CONTRAST_DROP),
            BrightnessTransform(delta=-s * MAX_BRIGHT_SHIFT * 255.0 * 0.5),
        ]
        kernel_len = 2 + int(s * MAX_MOTION_LEN * 0.5)
        if kernel_len >= 2:
            transforms.append(
                MotionBlurTransform(length=kernel_len, angle_deg=float(self._rng.uniform(0.0, 180.0)))
            )
        return TransformList(transforms)


# ---------------------------------------------------------------------------
# Registry of held-out Cityscapes corruptions.  Each value is a
# ``magnitude -> Augmentation`` factory, matching the calling convention of
# ``augmentations.heldout_augmentations.HELDOUT_PERTURBATION_REGISTRY`` and
# ``sensitivity.metrics.evaluate_aug_alpha``'s ``aug_factory(magnitude=alpha)``.
# Names are disjoint from the trained ``PERTURBATION_REGISTRY``.
# ---------------------------------------------------------------------------
CITYSCAPES_HELDOUT_PERTURBATION_REGISTRY: dict = {
    # --- Reused ImageNet-C photometric corruptions (driving-relevant) ---
    "motion_blur":      lambda magnitude: MotionBlurPerturbation(magnitude=magnitude),
    "defocus_blur":     lambda magnitude: DefocusBlurPerturbation(magnitude=magnitude),
    "shot_noise":       lambda magnitude: ShotNoisePerturbation(magnitude=magnitude),
    "impulse_noise":    lambda magnitude: ImpulseNoisePerturbation(magnitude=magnitude),
    "contrast":         lambda magnitude: ContrastPerturbation(magnitude=magnitude),
    "brightness":       lambda magnitude: BrightnessPerturbation(magnitude=magnitude),
    "jpeg_compression": lambda magnitude: JpegCompressionPerturbation(magnitude=magnitude),
    "pixelate":         lambda magnitude: PixelatePerturbation(magnitude=magnitude),
    # --- Weather / driving additions (ACDC + ImageNet-C weather group) ---
    "fog":              lambda magnitude: FogPerturbation(magnitude=magnitude),
    "snow":             lambda magnitude: SnowPerturbation(magnitude=magnitude),
    "frost":            lambda magnitude: FrostPerturbation(magnitude=magnitude),
    "low_light":        lambda magnitude: LowLightPerturbation(magnitude=magnitude),
    "spatter":          lambda magnitude: SpatterPerturbation(magnitude=magnitude),
    "advsteer":         lambda magnitude: AdvSteerPerturbation(magnitude=magnitude),
}
