"""Unit tests for the Cityscapes held-out (unseen) corruption registry.

These corruptions are what :class:`~hooks.cityscapes_post_eval_hook.CityscapesPostTrainHook`
applies to the test split.  The contract the hook relies on:

* ``magnitude == 0`` is an exact identity (so the clean baseline is truly clean),
* ``magnitude == 1`` actually corrupts the image while preserving shape/dtype/range,
* every corruption is **photometric** — ``apply_coords`` / ``apply_segmentation``
  pass through unchanged, so the original instance GT stays valid,
* the registry is **disjoint** from the trained ``PERTURBATION_REGISTRY``.

``frost`` needs bundled texture assets; with none present it falls back to a NoOp,
so the "actually corrupts" assertion is skipped for it in that case.
"""

import numpy as np
import pytest

from augmentations.augmentations import PERTURBATION_REGISTRY
from augmentations.cityscapes_heldout_augmentations import (
    CITYSCAPES_HELDOUT_PERTURBATION_REGISTRY as REGISTRY,
    _load_frost_assets,
)

NAMES = sorted(REGISTRY.keys())


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)


def _apply(name: str, magnitude: float, img: np.ndarray) -> np.ndarray:
    aug = REGISTRY[name](magnitude=magnitude)
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

    if name == "frost" and not _load_frost_assets():
        pytest.skip("no frost textures bundled; frost falls back to identity")

    # Several corruptions are stochastic (noise, snow, spatter, motion angle);
    # retry a few times so a rare empty draw doesn't flake the test.
    changed = any(not np.array_equal(_apply(name, 1.0, img), img) for _ in range(5))
    assert changed, f"{name} did not change the image at magnitude 1"


@pytest.mark.parametrize("name", NAMES)
def test_preserves_coords_and_segmentation(name, img):
    """Photometric corruptions must leave GT coords + masks untouched."""
    tfm = REGISTRY[name](magnitude=1.0).get_transform(img)

    coords = np.array([[0.0, 0.0], [10.0, 5.0], [95.0, 63.0]], dtype=np.float64)
    np.testing.assert_array_equal(tfm.apply_coords(coords.copy()), coords)

    seg = np.arange(img.shape[0] * img.shape[1], dtype=np.int32).reshape(img.shape[:2])
    np.testing.assert_array_equal(tfm.apply_segmentation(seg.copy()), seg)


def test_registry_disjoint_from_trained():
    assert set(REGISTRY) & set(PERTURBATION_REGISTRY) == set()


def test_registry_matches_expected_cityscapes_set():
    expected = {
        # reused ImageNet-C photometric corruptions
        "motion_blur", "defocus_blur", "shot_noise", "impulse_noise",
        "contrast", "brightness", "jpeg_compression", "pixelate",
        # weather / driving additions
        "fog", "snow", "frost", "low_light", "spatter", "advsteer",
    }
    assert set(REGISTRY) == expected


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
