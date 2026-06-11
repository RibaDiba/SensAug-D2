# Session Showcase: Porting the `magnitude` Parameter from SensAug

**Date:** 2026-06-11
**Branch:** `claude/sens-aug-augmentations-review-woJ3o`
**Commit:** `473387c` — *Add magnitude parameter to augmentations mirroring original SensAug*

---

## Goal

Port the single **`magnitude`** knob from Laura Zheng's original
[SensAug](https://github.com/laurayuzheng/SensAug) (ICML 2025, Zheng et al.) into
this Detectron2 edition's augmentation pipeline — **without** adding any new
augmentation types. The scaffolding (Transform classes, Augmentation wrappers, a
YAML-driven builder) already existed; it just lacked the unified strength
parameter.

**Reference paper:** [Sensitivity-Informed Augmentation for Robust Segmentation](https://arxiv.org/html/2406.01425v5)

---

## Background: how the original works

In the upstream `sensaug/dataset/augmentations.py`, every perturbation is driven
by one scalar `magnitude ∈ [0, 1]` (sampled per-image, e.g. via
`np.random.uniform(0, 1)` in `RandomAlphaTrainTransform`). Each augmentation maps
that one value onto its own native parameter using fixed caps:

```python
MAX_COLOR = 1.0   # RGB/HSV alpha cap
MAX_BLUR  = 49    # max (odd) Gaussian blur kernel
MAX_NOISE = 50.0  # max Gaussian noise sigma
```

| Perturbation | Original class      | magnitude → native parameter                       |
|--------------|---------------------|----------------------------------------------------|
| Blur         | `Blur`              | `kernel_size = int(m * 49) // 2 * 2 + 1` (odd, cv2) |
| Noise        | `Noise`             | `sigma = 50.0 * m`                                  |
| RGB          | `perturb_rgb`       | `alpha = m` → `ch - alpha·ch + 255·dir·alpha`       |
| HSV          | `HSVPerturbation`   | `alpha = m`; `max_val = 180 if H else 255`; V-darker uses `+10·alpha` floor |

`channel` and `direction` *select which* perturbation (e.g. `LighterR` = channel
2 / direction 1); `magnitude` only controls *strength*.

---

## What changed in this repo

All changes are in `augmentations/`. No new augmentation types were introduced.

### 1. `augmentations.py` — magnitude constants + mapping helpers

```python
MAX_COLOR = 1.0   # RGB/HSV alpha cap
MAX_BLUR  = 49    # max (odd) Gaussian blur kernel size
MAX_NOISE = 50.0  # max Gaussian noise sigma

def _clamp_magnitude(magnitude: float) -> float:
    return float(np.clip(abs(magnitude), 0.0, 1.0))

def blur_kernel_from_magnitude(magnitude: float) -> int:
    m = _clamp_magnitude(magnitude)
    return int(m * MAX_BLUR) // 2 * 2 + 1          # odd cv2 kernel

def noise_sigma_from_magnitude(magnitude: float) -> float:
    return MAX_NOISE * _clamp_magnitude(magnitude)

def color_alpha_from_magnitude(magnitude: float) -> float:
    return _clamp_magnitude(magnitude) * MAX_COLOR
```

### 2. Wrappers are now **magnitude-only**

The four `Augmentation` wrappers take `magnitude` instead of raw
`kernel_size`/`sigma`/`alpha`; the native value is derived in `get_transform()`:

```python
class GaussianBlurPerturbation(Augmentation):
    def __init__(self, magnitude: float = 0.1): ...
    def get_transform(self, image):
        return GaussianBlurTransform(blur_kernel_from_magnitude(self.magnitude))
```

(`GaussianNoisePerturbation`, `RGBPerturbation`, `HSVPerturbation` follow the
same pattern. `channel`/`direction` are retained as selectors.)

### 3. HSV value-darker floor (fidelity tweak)

`HSVPerturbationTransform` now mirrors the original's small floor so the value
channel never collapses fully to black:

```python
if self.direction == 1:
    ch = ch + self.alpha * (max_val - ch)
elif self.channel == 2:                              # V-darker special case
    ch = ch * (1.0 - self.alpha) + 10.0 * self.alpha
else:
    ch = ch * (1.0 - self.alpha)
```

### 4. `augmentations.yml` — exposes `magnitude` per perturbation

```yaml
gaussian_blur:
  magnitude: 0.1   # [0, 1] -> kernel_size = int(m*49)//2*2+1
gaussian_noise:
  magnitude: 0.2   # [0, 1] -> sigma = 50.0*m
rgb_perturbation:
  channel: 0       # 0=R, 1=G, 2=B
  magnitude: 0.3   # [0, 1] -> alpha
  direction: 1     # 1=lighter, 0=darker
hsv_perturbation:
  channel: 1       # 0=H, 1=S, 2=V
  magnitude: 0.3   # [0, 1] -> alpha
  direction: 0     # 1=lighter, 0=darker
```

`build_augmentations()` reads these `magnitude` keys and passes them through.
`resize` and `random_flip` are unchanged.

---

## Verification notes

- The magnitude→native formulas match the upstream `Blur`/`Noise`/`perturb_rgb`/
  `HSVPerturbation` line-for-line (spot-checked: `m=0.1 → kernel 5`,
  `m=1.0 → kernel 49`, `m=0.2 → sigma 10.0`, `m=0.3 → alpha 0.3`; out-of-range
  inputs clamp to `[0, 1]`).
- This environment had **no numpy / detectron2 installed**, so a live import /
  smoke run could not be executed here. The mapping helpers are pure-Python
  arithmetic and were reasoned through directly.

---

## How it was built (workflow)

1. Located the original repo ([laurayuzheng/SensAug](https://github.com/laurayuzheng/SensAug))
   and read its real `dataset/augmentations.py` + `parameters.py`.
2. Diffed the original's magnitude approach against this repo's existing
   scaffolding to scope the change precisely (magnitude only, no new augs).
3. Implemented constants, mapping helpers, magnitude-only wrappers, the HSV
   floor, and the YAML schema update.
4. Committed and pushed to the feature branch; removed accidentally-tracked
   `__pycache__/*.pyc` artifacts via an amend.
