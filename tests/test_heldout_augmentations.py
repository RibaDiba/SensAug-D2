"""Unit tests for the held-out (unseen) corruption registry.

These corruptions are what the post-training robustness eval applies to the test
split.  The contract the hook relies on:

* ``magnitude == 0`` is an exact identity (so the clean baseline is truly clean),
* ``magnitude == 1`` actually corrupts the image while preserving shape/dtype/range,
* the registry is **disjoint** from the trained ``PERTURBATION_REGISTRY`` (these are
  genuinely *unseen* during training).
"""

import numpy as np
import pytest

from augmentations.heldout_augmentations import HELDOUT_PERTURBATION_REGISTRY
from augmentations.augmentations import PERTURBATION_REGISTRY

NAMES = sorted(HELDOUT_PERTURBATION_REGISTRY.keys())


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


def _apply(name: str, magnitude: float, img: np.ndarray) -> np.ndarray:
    aug = HELDOUT_PERTURBATION_REGISTRY[name](magnitude=magnitude)
    return aug.get_transform(img).apply_image(img)


@pytest.mark.parametrize("name", NAMES)
def test_identity_at_zero(name, img):
    out = _apply(name, 0.0, img)
    np.testing.assert_array_equal(out, img)


@pytest.mark.parametrize("name", NAMES)
def test_corrupts_and_preserves_shape_dtype_range(name, img):
    out = _apply(name, 1.0, img)
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert out.min() >= 0 and out.max() <= 255
    assert not np.array_equal(out, img), f"{name} did not change the image at magnitude 1"


def test_registry_disjoint_from_trained():
    assert set(HELDOUT_PERTURBATION_REGISTRY) & set(PERTURBATION_REGISTRY) == set()


def test_registry_matches_expected_microscopy_set():
    expected = {
        "defocus_blur", "motion_blur", "shot_noise", "impulse_noise", "speckle_noise",
        "contrast", "brightness", "gamma", "jpeg_compression", "pixelate",
    }
    assert set(HELDOUT_PERTURBATION_REGISTRY) == expected


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
