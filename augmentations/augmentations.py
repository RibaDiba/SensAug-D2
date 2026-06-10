from typing import Optional, List
from pathlib import Path

import yaml
import numpy as np
import cv2
from detectron2.data.transforms import Augmentation, Transform
import detectron2.data.transforms as T

AUG_CFG_PATH = Path(__file__).parent / "augmentations.yml"

# Maximum native strengths, mirroring the original SensAug (Zheng et al.).
# A single ``magnitude`` in [0, 1] is mapped onto each augmentation's native
# parameter using these caps, exactly as in the upstream implementation.
MAX_COLOR = 1.0   # RGB/HSV alpha cap
MAX_BLUR = 49     # max (odd) Gaussian blur kernel size
MAX_NOISE = 50.0  # max Gaussian noise sigma


def _clamp_magnitude(magnitude: float) -> float:
    """Clamp a magnitude into [0, 1] (mirrors the original's abs()/clip usage)."""
    return float(np.clip(abs(magnitude), 0.0, 1.0))


def blur_kernel_from_magnitude(magnitude: float) -> int:
    """Map magnitude in [0, 1] to an odd cv2 Gaussian-blur kernel size.

    Mirrors the original ``Blur`` transform:
        kernel_size = int(magnitude * 49) // 2 * 2 + 1
    """
    m = _clamp_magnitude(magnitude)
    return int(m * MAX_BLUR) // 2 * 2 + 1


def noise_sigma_from_magnitude(magnitude: float) -> float:
    """Map magnitude in [0, 1] to a Gaussian-noise sigma (mirrors original ``Noise``)."""
    return MAX_NOISE * _clamp_magnitude(magnitude)


def color_alpha_from_magnitude(magnitude: float) -> float:
    """Map magnitude in [0, 1] to an RGB/HSV perturbation alpha (mirrors ``MAX_COLOR``)."""
    return _clamp_magnitude(magnitude) * MAX_COLOR


class GaussianBlurTransform(Transform):
    def __init__(self, kernel_size: int):
        super().__init__()
        # Ensure kernel size is odd
        self.kernel_size = int(kernel_size) | 1

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class GaussianNoiseTransform(Transform):
    def __init__(self, sigma: float, rng: np.random.Generator):
        super().__init__()
        self.sigma = sigma
        self.rng = rng

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(0, self.sigma, img.shape).astype(img.dtype)
        return np.clip(img.astype(np.float32) + noise.astype(np.float32), 0, 255).astype(img.dtype)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class RGBPerturbationTransform(Transform):
    """Lighten or darken a single RGB channel by a factor alpha.

    direction=1 (lighter): channel = channel + alpha * (255 - channel)
    direction=0 (darker):  channel = channel * (1 - alpha)
    """

    def __init__(self, channel: int, alpha: float, direction: int):
        super().__init__()
        self.channel = channel
        self.alpha = alpha
        self.direction = direction

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        out = img.astype(np.float32)
        ch = out[:, :, self.channel]
        if self.direction == 1:
            ch = ch + self.alpha * (255.0 - ch)
        else:
            ch = ch * (1.0 - self.alpha)
        out[:, :, self.channel] = np.clip(ch, 0, 255)
        return out.astype(img.dtype)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class HSVPerturbationTransform(Transform):
    """Lighten or darken a single HSV channel by a factor alpha.

    Converts to HSV, applies the perturbation, converts back to BGR.
    direction=1 (lighter): channel = channel + alpha * (max - channel)
    direction=0 (darker):  channel = channel * (1 - alpha)
    """

    def __init__(self, channel: int, alpha: float, direction: int):
        super().__init__()
        self.channel = channel
        self.alpha = alpha
        self.direction = direction

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        ch = hsv[:, :, self.channel]
        max_val = 180.0 if self.channel == 0 else 255.0
        if self.direction == 1:
            ch = ch + self.alpha * (max_val - ch)
        elif self.channel == 2:
            # V-darker special case (mirrors original): keep a small floor so
            # the value channel never collapses fully to black.
            ch = ch * (1.0 - self.alpha) + 10.0 * self.alpha
        else:
            ch = ch * (1.0 - self.alpha)
        hsv[:, :, self.channel] = np.clip(ch, 0, max_val)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


# --- Detectron2 Augmentation wrappers ---

class GaussianBlurPerturbation(Augmentation):
    """Apply Gaussian blur whose strength is set by a magnitude in [0, 1].

    Args:
        magnitude: perturbation strength in [0, 1]; mapped to an odd cv2 kernel
            size via ``blur_kernel_from_magnitude``.
    """

    def __init__(self, magnitude: float = 0.1):
        super().__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> Transform:
        return GaussianBlurTransform(blur_kernel_from_magnitude(self.magnitude))


class GaussianNoisePerturbation(Augmentation):
    """Add Gaussian noise whose strength is set by a magnitude in [0, 1].

    Args:
        magnitude: perturbation strength in [0, 1]; mapped to sigma via
            ``noise_sigma_from_magnitude``.
    """

    def __init__(self, magnitude: float = 0.2):
        super().__init__()
        self._init(locals())
        self._rng = np.random.default_rng()

    def get_transform(self, image: np.ndarray) -> Transform:
        return GaussianNoiseTransform(noise_sigma_from_magnitude(self.magnitude), self._rng)


class RGBPerturbation(Augmentation):
    """Lighten or darken a specific RGB channel.

    Args:
        channel: 0=R, 1=G, 2=B
        magnitude: perturbation strength in [0, 1]; mapped to alpha via
            ``color_alpha_from_magnitude``.
        direction: 1=lighter, 0=darker
    """

    def __init__(self, channel: int = 0, magnitude: float = 0.3, direction: int = 1):
        super().__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> Transform:
        return RGBPerturbationTransform(
            self.channel, color_alpha_from_magnitude(self.magnitude), self.direction
        )


class HSVPerturbation(Augmentation):
    """Lighten or darken a specific HSV channel.

    Args:
        channel: 0=H, 1=S, 2=V
        magnitude: perturbation strength in [0, 1]; mapped to alpha via
            ``color_alpha_from_magnitude``.
        direction: 1=lighter, 0=darker
    """

    def __init__(self, channel: int = 0, magnitude: float = 0.3, direction: int = 1):
        super().__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> Transform:
        return HSVPerturbationTransform(
            self.channel, color_alpha_from_magnitude(self.magnitude), self.direction
        )


# ---------------------------------------------------------------------------
# Perturbation registry
# ---------------------------------------------------------------------------
# Maps a short name → Augmentation class.  Matches the ``NEW_PERTURBATIONS``
# dict from the original SensAug repository so that hooks and mappers can look
# up classes by name.  Each class must accept a single ``magnitude`` kwarg in
# [0, 1].
PERTURBATION_REGISTRY: dict = {
    # Gaussian blur / noise
    "blur":  GaussianBlurPerturbation,
    "noise": GaussianNoisePerturbation,
    # RGB channel perturbations  (channel 0=R, 1=G, 2=B in OpenCV BGR images)
    "lighter_R": lambda magnitude: RGBPerturbation(channel=2, magnitude=magnitude, direction=1),
    "darker_R":  lambda magnitude: RGBPerturbation(channel=2, magnitude=magnitude, direction=0),
    "lighter_G": lambda magnitude: RGBPerturbation(channel=1, magnitude=magnitude, direction=1),
    "darker_G":  lambda magnitude: RGBPerturbation(channel=1, magnitude=magnitude, direction=0),
    "lighter_B": lambda magnitude: RGBPerturbation(channel=0, magnitude=magnitude, direction=1),
    "darker_B":  lambda magnitude: RGBPerturbation(channel=0, magnitude=magnitude, direction=0),
    # HSV channel perturbations  (channel 0=H, 1=S, 2=V)
    "lighter_H": lambda magnitude: HSVPerturbation(channel=0, magnitude=magnitude, direction=1),
    "darker_H":  lambda magnitude: HSVPerturbation(channel=0, magnitude=magnitude, direction=0),
    "lighter_S": lambda magnitude: HSVPerturbation(channel=1, magnitude=magnitude, direction=1),
    "darker_S":  lambda magnitude: HSVPerturbation(channel=1, magnitude=magnitude, direction=0),
    "lighter_V": lambda magnitude: HSVPerturbation(channel=2, magnitude=magnitude, direction=1),
    "darker_V":  lambda magnitude: HSVPerturbation(channel=2, magnitude=magnitude, direction=0),
}

# Names that each accept a ``magnitude`` kwarg — the lambdas above act as
# factory callables so the registry is uniform: ``PERTURBATION_REGISTRY[name](magnitude=m)``.
# For the two direct classes we wrap them the same way for consistency.
PERTURBATION_REGISTRY["blur"]  = lambda magnitude: GaussianBlurPerturbation(magnitude=magnitude)
PERTURBATION_REGISTRY["noise"] = lambda magnitude: GaussianNoisePerturbation(magnitude=magnitude)


class UniformPerturbation(Augmentation):
    """Sample one corruption uniformly from PERTURBATION_REGISTRY and apply it.

    Each image draws a corruption name uniformly over all registry entries and a
    magnitude uniformly in [0, 1]. This is the non-adaptive uniform baseline:
    one perturbation per image, always applied (mirroring the original SensAug's
    training procedure, which has no skip/"none" probability after warmup).

    Args:
        names: Subset of perturbation names to sample from. Defaults to every key
            in :data:`PERTURBATION_REGISTRY`.
    """

    def __init__(self, names: Optional[List[str]] = None):
        super().__init__()
        names = list(names) if names else list(PERTURBATION_REGISTRY.keys())
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> Transform:
        # Use the global np.random so Detectron2's per-worker seeding applies,
        # matching AugmentationPolicy.sample() in adaptive_mapper.py.
        name = self.names[np.random.randint(len(self.names))]
        magnitude = float(np.random.uniform(0.0, 1.0))
        aug = PERTURBATION_REGISTRY[name](magnitude=magnitude)
        return aug.get_transform(image)


def build_augmentations(cfg_path: str = None) -> List[Augmentation]:

    """Build the augmentation list from the YAML config.

    Args:
        cfg_path: Path to augmentations YAML. Defaults to augmentations.yml
                  next to this file.

    Returns:
        List of Detectron2 Augmentation objects.
    """
    path = Path(cfg_path) if cfg_path else AUG_CFG_PATH
    with open(path, "r") as f:
        aug_cfg = yaml.safe_load(f)

    resize = aug_cfg["resize"]
    flip = aug_cfg["random_flip"]

    # Geometric augs always apply; corruptions are sampled uniformly — one per
    # image, random type and magnitude — via UniformPerturbation. (The per-
    # corruption magnitudes in the YAML are no longer read here.)
    return [
        T.ResizeShortestEdge(resize["min_size"], resize["max_size"], resize["sample_style"]),
        T.RandomFlip(prob=flip["prob"], horizontal=flip["horizontal"], vertical=flip["vertical"]),
        UniformPerturbation(),
    ]
