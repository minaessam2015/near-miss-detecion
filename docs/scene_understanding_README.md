# Scene Understanding Module

**Package:** `scene_understanding/`
**Version:** 1.0
**Status:** Active — runs independently of the detection / tracking / near-miss pipeline.

---

## Overview

`SceneUnderstanding` estimates a ground plane from monocular video using road segmentation and monocular depth, then provides geometric visualisations tied to that plane.  It is designed to be run standalone first (to verify accuracy and tune thresholds), and then optionally integrated with the near-miss pipeline to add metric-scale context.

```
video frame
     │
     ├─► RoadSegmenter.predict()  ──► road mask (H×W bool)
     │
     └─► DepthEstimator.predict() ──► depth map (H×W float)
                │                              │
                └──────────┬───────────────────┘
                           ▼
                 backproject road pixels
                    (N × 3) points
                           │
                           ▼
                   RANSAC plane fit
                  PlaneParams (n, d)
                           │
                    validate + EMA smooth
                           │
                   ScenePlaneTracker
                    (stable plane)
                           │
          ┌────────────────┼──────────────────────┐
          ▼                ▼                      ▼
    mask overlay     ground grid             BEV scatter
    depth colourmap  ray-plane probes        diagnostics plot
```

---

## Package Structure

```
scene_understanding/
├── __init__.py                  # public API re-exports
├── config.py                   # SceneConfig dataclass
├── models.py                   # RoadSegmenter, DepthEstimator, dummy + torch wrappers
├── geometry.py                 # CameraIntrinsics, backproject, ray-plane intersection
├── plane_fit.py                # RANSAC, EMA smoothing, ScenePlaneTracker
├── visualize.py                # all visual outputs
└── run_scene_understanding.py  # CLI entry point
tests/
├── test_scene_understanding_geometry.py
└── test_scene_understanding_plane_fit.py
```

---

## Quick Start

### 1. Bootstrap — verify with dummy models

No real models or video download required:

```bash
python -m scene_understanding.run_scene_understanding \
    --video_path data/traffic_video.mp4 \
    --mode bootstrap \
    --num_bootstrap_frames 30 \
    --output_dir outputs/scene_understanding
```

Outputs written to `outputs/scene_understanding/`:

| Path | Contents |
|---|---|
| `frames_overlay/` | Road mask + grid overlay per frame |
| `frames_depth/`   | Depth colourmap per frame |
| `frames_grid/`    | 2×2 composite (mask, depth, road-depth, depth+contour) |
| `bev_plots/`      | Bird's-eye-view scatter plots |
| `diagnostics/`    | RANSAC inlier/outlier dots; pixel probe annotations |
| `plane_params.json` | Full per-frame plane parameters + final stable plane |
| `plane_diagnostics.png` | 3-panel time-series (inlier ratio, angle change, d) |
| `summary_report.md` | Human-readable summary |
| `output_video_overlay.mp4` | Annotated video |

### 2. Full run

```bash
python -m scene_understanding.run_scene_understanding \
    --video_path data/traffic_video.mp4 \
    --intrinsics 800 800 640 360 \
    --mode run \
    --stride_seg 2 \
    --stride_depth 2 \
    --output_dir outputs/scene_understanding \
    --probe_pixels 320 450 640 480
```

`--probe_pixels u1 v1 u2 v2 ...` annotates chosen pixels with their estimated ground distance.

### 3. From a frames directory

```bash
python -m scene_understanding.run_scene_understanding \
    --frames_dir data/frames/ \
    --mode run \
    --output_dir outputs/scene_understanding
```

---

## Plugging in Real Models

### DepthAnything v2 (via Hugging Face transformers)

```python
# In scene_understanding/run_scene_understanding.py, edit _build_models():
from transformers import pipeline as hf_pipeline
import numpy as np
from scene_understanding.models import DepthEstimator

class DepthAnythingEstimator(DepthEstimator):
    def __init__(self):
        self._pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device="cpu",
        )

    def predict(self, frame: np.ndarray) -> np.ndarray:
        import PIL.Image
        img_pil = PIL.Image.fromarray(frame[:, :, ::-1])  # BGR → RGB
        result = self._pipe(img_pil)
        return np.array(result["depth"], dtype=np.float32)
```

### SegFormer road segmenter (via transformers)

```python
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from scene_understanding.models import TorchRoadSegmenter

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")

class CityscapesSegmenter(TorchRoadSegmenter):
    # Cityscapes class 0 = road
    def __init__(self):
        super().__init__(model=model, road_class_ids=[0], device="cpu")

    def _preprocess(self, frame):
        import torch, PIL.Image
        img_pil = PIL.Image.fromarray(frame[:, :, ::-1])
        return processor(images=img_pil, return_tensors="pt").pixel_values.to(self.device)

    def _postprocess(self, output, original_hw):
        import torch, numpy as np, cv2
        logits = output.logits  # (1, num_classes, H', W')
        pred = logits.argmax(dim=1).squeeze().cpu().numpy()
        mask = (pred == 0)  # road class
        return cv2.resize(
            mask.astype(np.uint8),
            (original_hw[1], original_hw[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
```

Replace the return values in `_build_models()`:

```python
def _build_models(args):
    segmenter = CityscapesSegmenter()
    estimator = DepthAnythingEstimator()
    return segmenter, estimator
```

---

## Geometry Reference

All coordinates are in **camera space**:

| Axis | Direction |
|---|---|
| +X | right |
| +Y | down (OpenCV convention) |
| +Z | forward (optical axis) |

### Backprojection

Given pixel (u, v) and depth Z (distance along optical axis):

```
X = (u - cx) / fx * Z
Y = (v - cy) / fy * Z
P = [X, Y, Z]
```

### Plane equation

```
n · P + d = 0    (n = unit normal, d = scalar offset)
```

The closest point on the plane to the camera origin is `-d * n`.

### Ray–plane intersection

```
ray direction:  r = normalize([(u-cx)/fx, (v-cy)/fy, 1])
denominator:    denom = n · r
if |denom| < eps: ray is parallel → no intersection
t = -d / denom
if t < 0: intersection is behind camera → no intersection
P_intersect = t * r
```

---

## Algorithm — Step by Step

### Step 1 — Road segmentation

`RoadSegmenter.predict(frame)` returns a `(H, W)` boolean mask.  The mask is morphologically dilated by `road_dilate_px` to fill small holes before backprojection.

### Step 2 — Depth estimation

`DepthEstimator.predict(frame)` returns a `(H, W)` float32 depth map.  For relative-depth models the values are in arbitrary units; absolute metres are only available when camera intrinsics + model scale are both known.

### Step 3 — Caching

Both mask and depth map are saved as `.npz` files keyed by `(video_path_hash, frame_idx, suffix)`.  On subsequent runs the inference step is skipped entirely for cached frames.

### Step 4 — Backprojection

Road pixels are backprojected to 3-D camera-space points using the pinhole formula above.  Up to `sample_points_per_frame` points are randomly subsampled before passing to RANSAC.

### Step 5 — RANSAC plane fitting

1. Sample 3 random points, compute the plane through them.
2. Count inliers: `|n · P + d| < inlier_thresh`.
3. Keep the best plane; early-stop if `inlier_ratio >= 0.8`.
4. Refit the kept plane to **all** inliers via SVD (least-squares).

### Step 6 — Plane validation

A newly fitted plane is accepted only when:
1. `inlier_ratio >= min_inlier_ratio`
2. The angle between the new normal and the last stable normal is `<= max_normal_angle_change_deg`

### Step 7 — EMA smoothing

When the new plane is valid it is blended into the stable estimate:

```
n_stable = normalize((1-α)*n_prev + α*n_new)
d_stable = (1-α)*d_prev + α*d_new
```

`α = ema_alpha` (default 0.20 — the stable estimate moves slowly toward new observations).

### Step 8 — Freeze on invalid frame

When validation fails the stable plane from the last valid frame is retained unchanged.  The `plane_valid` flag in the history log marks the frame as frozen.

---

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `fx, fy, cx, cy` | `None` | Camera intrinsics. `None` = auto-estimate from image size. |
| `fps` | 30 | Source video FPS (used for timestamp labelling only). |
| `stride_seg` | 1 | Run segmenter every N frames. |
| `stride_depth` | 1 | Run depth estimator every N frames. |
| `num_bootstrap_frames` | 30 | Frame count for bootstrap mode. |
| `sample_points_per_frame` | 2000 | Max road pixels passed to RANSAC. |
| `ransac_iters` | 150 | RANSAC trial count. |
| `inlier_thresh` | 0.05 | Inlier distance threshold (depth units). |
| `min_inlier_ratio` | 0.30 | Minimum inlier fraction to accept a plane. |
| `max_normal_angle_change_deg` | 15.0 | Maximum normal change between updates. |
| `ema_alpha` | 0.20 | EMA weight for new plane observation. |
| `road_dilate_px` | 5 | Morphological dilation radius for road mask. |
| `snap_radius_px` | 50 | Pixel search radius for probe snapping. |
| `grid_spacing` | 1.0 | Ground-grid line spacing (depth units). |
| `grid_extent` | 10.0 | Ground-grid half-extent (depth units). |
| `output_dir` | `outputs/scene_understanding` | Root output directory. |
| `inference_max_dim` | `None` | Resize cap before inference; `None` = no resize. |
| `save_overlay_video` | `True` | Write annotated MP4. |

---

## Running Unit Tests

```bash
# From the repo root
python -m pytest tests/test_scene_understanding_geometry.py -v
python -m pytest tests/test_scene_understanding_plane_fit.py -v

# Or all at once
python -m pytest tests/ -v
```

No GPU, no video, no model downloads required.  The tests use synthetic data with known analytic ground truths.

---

## Known Limitations

1. **Relative depth only.** Without a metric-scale depth model and calibrated intrinsics, distances are in arbitrary "depth units", not metres.

2. **Flat-road assumption.** The RANSAC plane assumes the road is locally flat in each frame.  Hilly terrain or ramps will produce inaccurate or frequently-frozen planes.

3. **Segmentation quality.** The plane estimate is only as good as the road mask.  Dummy models produce a crude bottom-half mask; real SegFormer-class models are strongly recommended for production use.

4. **Stride latency.** When `stride_seg > 1` or `stride_depth > 1`, the pipeline reuses the last cached mask/depth for intermediate frames.  Fast-moving cameras may see stale masks.

5. **No temporal association.** The plane tracker has no notion of object identity or scene continuity beyond EMA smoothing.  A sudden viewpoint change (e.g. camera pan) will cause a brief freeze until a consistent plane is re-established.
