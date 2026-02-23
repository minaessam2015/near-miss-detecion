# Tatweer — Near-Miss Traffic Incident Detection

CPU-first computer-vision pipeline for detecting and analyzing traffic near-miss events from monocular video.

[![Open near_miss_detection_demo.ipynb in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minaessam2015/near-miss-detecion/blob/main/notebooks/near_miss_detection_demo.ipynb)
[![Open local_test.ipynb in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minaessam2015/near-miss-detecion/blob/main/notebooks/local_test.ipynb)

## What This Project Contains

- Detection layer with multiple model backends
- Multi-object tracking with pluggable tracker implementations
- Near-miss detection algorithms (`v1.0`, `v1.1`, `v2.0`, `v3.0`, `v4.0`)
- Visualization and reporting (annotated videos, charts, HTML report)
- Scene-understanding module for road-plane estimation and 3D grounding
- Colab-style demo notebook and local development notebook

## Repository Structure

```text
tatweer/
├── src/
│   ├── detector.py            # Detection abstraction + backends
│   ├── tracker.py             # Tracker abstraction + centroid/byte/botsort (+ v2/v3)
│   ├── near_miss.py           # Near-miss algorithms (v1.0/v1.1/v2.0/v3.0/v4.0)
│   ├── visualizer.py          # Overlay drawing + video/chart/report helpers
│   ├── video_utils.py         # Video I/O helpers
│   └── optimizer.py           # Benchmarking / optimization helpers
├── scene_understanding/
│   ├── run_scene_understanding.py
│   ├── geometry.py
│   ├── plane_fit.py
│   ├── calibration.py
│   ├── visualize.py
│   └── models/                # Depth + road-segmentation wrappers
├── notebooks/
│   ├── near_miss_detection_demo.ipynb   # Submission/demo-style pipeline notebook
│   └── local_test.ipynb                 # Local experimentation notebook
├── docs/
│   ├── near_miss_detection_v1.md
│   ├── near_miss_detection_v11.md
│   └── scene_understanding_README.md
├── tests/
│   ├── test_scene_understanding_geometry.py
│   ├── test_scene_understanding_plane_fit.py
│   └── test_near_miss_v11_v40.py
├── requirements.txt
└── .gitignore
```

## Core Pipeline

1. Detect objects (`vehicle` / `pedestrian`) per frame.
2. Track detections across frames with persistent IDs.
3. Compute near-miss metrics (proximity, closest approach, TTC/risk signals).
4. Apply false-positive filtering.
5. Export events + annotated outputs.

## Trackers

Use `create_tracker(...)` from `src/tracker.py`:

- `centroid`
- `centroid_v2` (3D-aware hybrid matching)
- `centroid_v3` (hybrid + Kalman 3D prediction)
- `bytetrack`
- `botsort`

## Near-Miss Algorithms

Implemented in `src/near_miss.py`:

- `NearMissDetector` (`v1.0` baseline)
- `NearMissDetectorV11` (improved geometry + leaky confirmation)
- `NearMissDetectorV20` (3D ground-point mode + 2D fallback)
- `NearMissDetectorV30` (v20 + predicted/dual-predicted source control)
- `NearMissDetectorV40` (optical-flow-enhanced velocity estimation)

Recent update:
- Added size-aware adjacent-lane false-positive suppression (`clearance_scale`) in `v1.1` and `v4.0`.

## Scene Understanding (3D Ground Plane)

`scene_understanding/` can run independently to estimate a stable road plane from monocular depth + road segmentation, then provide 3D grounding for tracking/near-miss.

See:
- `docs/scene_understanding_README.md`
- `scene_understanding/run_scene_understanding.py`

## Notebooks

- `notebooks/near_miss_detection_demo.ipynb`
  - End-to-end demo flow
  - Uses latest `NearMissDetectorV11` settings in this repo
- `notebooks/local_test.ipynb`
  - Multi-version comparison and debugging workflow
  - Includes `v11`, `v40`, and `v20` evaluation paths

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/near_miss_detection_demo.ipynb
```

## Tests

```bash
python3 -m pytest -q tests/test_near_miss_v11_v40.py \
  tests/test_scene_understanding_geometry.py \
  tests/test_scene_understanding_plane_fit.py
```

## Notes

- Heavy files (weights, videos, outputs, caches) are excluded by `.gitignore`.
- Primary code-level docs:
  - `docs/near_miss_detection_v1.md`
  - `docs/near_miss_detection_v11.md`
  - `docs/scene_understanding_README.md`
