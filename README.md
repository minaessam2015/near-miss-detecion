# Tatweer — Near-Miss Traffic Incident Detection

A computer-vision pipeline for detecting near-miss events in traffic video.
Runs on Google Colab free tier (CPU) and locally.

---

## Project Structure

```
tatweer/
├── src/
│   ├── detector.py        # Object detection — unified detector abstraction
│   ├── tracker.py         # Multi-object tracking — unified tracker abstraction
│   ├── near_miss.py       # Near-miss detection & false-positive filtering
│   ├── visualizer.py      # Annotated video, charts, HTML report
│   ├── video_utils.py     # Video I/O, metadata, frame sampling
│   └── optimizer.py       # ONNX export, FPS benchmarking
├── notebooks/
│   ├── near_miss_detection_demo.ipynb   # Colab-ready end-to-end demo
│   └── local_test.ipynb                 # Local dev: faster defaults, no download
├── data/                  # Input videos (not committed)
├── outputs/               # Generated videos, charts, reports
└── requirements.txt
```

---

## Detection Layer (`src/detector.py`)

All detectors share a `BaseDetector` interface:

```python
det = create_detector("yolo26l.pt")
dets = det.detect(frame)   # → list of detection dicts
```

### Detection dict schema

| Field | Type | Description |
|---|---|---|
| `class` | `str` | Unified category: `"vehicle"` or `"pedestrian"` |
| `label` | `str` | Fine-grained sub-type: `"car"`, `"truck"`, `"bus"`, `"person"`, … |
| `bbox` | `[x1,y1,x2,y2]` | Bounding box in original-frame pixels |
| `confidence` | `float` | Detection score |
| `center` | `(cx, cy)` | Bbox centroid |
| `contour` | `ndarray\|None` | Segmentation polygon — seg models only |

**Vehicle aggregation.** All COCO vehicle classes (car, motorcycle, bus, truck, bicycle) are unified under `class="vehicle"`. This ensures downstream tracking and near-miss logic work the same regardless of which detection model is used. The original label is preserved for debugging and reporting.

### Backends

| Backend key | Class | Use case |
|---|---|---|
| `"ultralytics"` | `UltralyticsDetector` | Standard detection — default |
| `"ultralytics-seg"` | `UltralyticsSegDetector` | Adds `"contour"` field (polygon mask) |
| `"sahi"` | `SAHIDetector` | Sliced inference — better recall for small/distant objects |

### Model selection

YOLO26 was selected as the recommended model for this project after comparing published benchmarks. It was released in January 2026 and is currently the best detection backbone available for CPU inference.

| Model | Params | mAP (50-95) | CPU ms/frame | Notes |
|---|---|---|---|---|
| `yolo26n.pt` | 2.4M | 40.9 | 38.9 | Fastest, quick tests |
| `yolo26s.pt` | 9.5M | 48.6 | 87.2 | |
| `yolo26l.pt` | 24.8M | 55.0 | 286.2 | **Recommended** |
| `yolo26x.pt` | 55.7M | 57.5 | 525.8 | Max accuracy |
| `yolo12l.pt` | 26.4M | 53.7 | — | Lower mAP than yolo26l; skip on CPU |
| `rtdetr-l.pt` | — | — | high | Transformer; strong on small objects but slow on CPU |

YOLO26 advantages over alternatives:
- NMS-free end-to-end architecture (no separate NMS pass needed)
- ProgLoss + STAL training recipe improves small-object recall — relevant for distant vehicles
- 43% faster on CPU vs YOLO11 at equivalent accuracy tiers
- YOLO12 has no published CPU benchmarks and lower mAP than YOLO26 at the same size tier

### Post-processing

Two post-processing steps are applied after every `detect()` call via `_postprocess()`:

1. **Per-class confidence filtering.** `conf` can be a plain float (same threshold for all classes) or a dict (different thresholds per category, e.g. `{"vehicle": 0.4, "pedestrian": 0.7}`). When a dict is used, the minimum value is passed to the model so no candidates are dropped prematurely; per-class filtering then happens in `_postprocess()`.

2. **Within-class NMS.** Applied independently per category so a vehicle and a pedestrian with overlapping bboxes are both kept. Controlled by `nms_iou` (default 0.45, set to `None` to disable).

---

## Tracking Layer (`src/tracker.py`)

### Motivation

The initial implementation used a simple centroid tracker. As the project grew it became clear that:

- Different videos and camera angles have different occlusion characteristics.
- A simple greedy centroid match produces many ID switches during partial occlusions.
- Comparing tracker quality required running the same detections through multiple trackers without re-running expensive inference.

Three trackers were implemented behind a **single unified interface** so they are drop-in replacements for each other and can be compared fairly on the same detection cache.

### Unified interface

```python
from src.tracker import create_tracker

tracker = create_tracker("bytetrack", track_buffer=90, min_hits=3)

for frame_idx, frame in frame_generator(VIDEO_PATH, stride=2):
    dets    = detector.detect(frame)
    tracked = tracker.update(dets, frame_idx, frame)
    # tracked: OrderedDict[int, dict]  — same schema regardless of tracker
```

`update()` always returns the same output dict schema:

| Field | Description |
|---|---|
| `id` | Persistent integer track ID |
| `class` | `"vehicle"` or `"pedestrian"` |
| `label` | Fine-grained sub-type |
| `bbox` | `[x1,y1,x2,y2]` |
| `confidence` | Detection score |
| `center` | `(cx, cy)` |
| `trajectory` | `[(frame_idx, cx, cy), …]` — full centroid history |

### Track activation — `min_hits`

Every tracker supports a `min_hits` parameter implementing the **tentative → confirmed** state machine from the DeepSORT literature:

```
TENTATIVE  ──( min_hits consecutive matches )──►  CONFIRMED
               ▲ missed frame resets streak                │
                                                           │ (one-way)
                                               CONFIRMED stays CONFIRMED
                                               until max_disappeared misses
                                                           │
                                                           ▼
                                                        DELETED
```

`min_hits=1` (default) is backward-compatible — every detection is immediately active. `min_hits=3` is a practical setting that filters single- and double-frame noise without delaying legitimate tracks. Crucially, once a track is confirmed it is **never demoted back to tentative** — a temporary miss only counts towards the deletion counter, not re-confirmation. This eliminates the "flickering" behaviour that the earlier rolling-streak approach produced.

### Tracker implementations

#### 1. CentroidTracker (`'centroid'`)

The original tracker. Assigns IDs by matching each detection to the nearest existing centroid using L2 distance (greedy, via `scipy.cdist`). Deregisters tracks after `max_disappeared` consecutive misses.

- No extra dependencies
- Fastest of the three
- No Kalman prediction: ID switches occur on occlusion
- No ReID: once lost, the track restarts with a new ID

```python
create_tracker('centroid', max_disappeared=15, max_distance=200, min_hits=3)
```

#### 2. ByteTracker (`'bytetrack'`)

Wraps `boxmot.ByteTrack`. Implements the ByteTrack two-stage association algorithm:

1. **Stage 1 (high-confidence detections):** IoU matching against existing tracks for detections above `track_thresh`.
2. **Stage 2 (low-confidence detections):** Unmatched tracks from stage 1 are given a second chance by matching against lower-confidence detections — this is ByteTrack's key innovation for recovering from occlusion without ReID.
3. A Kalman filter predicts each track's position during missed frames.

```python
create_tracker('bytetrack', track_thresh=0.25, track_buffer=90,
               match_thresh=0.8, frame_rate=20, min_hits=3)
```

#### 3. BoTSORTTracker (`'botsort'`)

Wraps `boxmot.BoTSORT`. Adds **ReID appearance features** on top of ByteTrack-style Kalman IoU matching. The ReID model (`osnet_x0_25_msmt17.pt`, ~20 MB, auto-downloaded) embeds each detection crop into an appearance vector. Tracks are matched jointly on IoU + appearance similarity, allowing correct re-identification after longer occlusions.

Requires the actual frame to be passed to `update()` so the ReID model can crop object patches.

```python
create_tracker('botsort', device='cpu', half=False, min_hits=3)
```

### Comparison

The three trackers were compared on the same 100-frame detection cache (no repeated inference) in notebook section §2d.

| Property | CentroidTracker | ByteTracker | BoTSORTTracker |
|---|---|---|---|
| Matching | Greedy L2 (centroid) | 2-stage IoU + L2 | IoU + Kalman + ReID |
| Occlusion handling | None | Kalman prediction | Kalman + appearance |
| ReID | No | No | Yes (~20 MB weights) |
| Extra dependencies | None | `boxmot` | `boxmot` |
| CPU speed | Fastest | Fast | Moderate |
| ID stability | Low (switches on occlusion) | Good | Best |
| Short-track noise | Highest (with min_hits=1) | Lower | Lower |

**Key observations from the §2d comparison:**

- **Total IDs assigned:** CentroidTracker consistently assigns the most IDs due to ID switches on occlusion. ByteTracker and BoT-SORT assign fewer total IDs at equivalent `min_hits` settings.
- **Short track fraction (≤3 frames):** CentroidTracker produces the most short-lived tracks, many of which are noise or re-registered versions of the same physical object. ByteTrack's low-confidence second stage recovers many of these.
- **Active track count:** BoT-SORT and ByteTracker track active-track counts closer to the true number of objects because they maintain tracks through brief occlusions rather than dropping and re-registering.
- **`min_hits` effect:** Setting `min_hits=3–5` dramatically reduces noise tracks for all three trackers. Because confirmation is one-way, confirmed tracks remain stable even when the object briefly leaves the detector's view.

**Recommended setting:** `ByteTracker` with `min_hits=3`, `track_buffer=60–90` for typical traffic footage at 20–30 FPS. BoT-SORT is worth enabling if the video has significant occlusions and the ~20 MB weight download is acceptable.

### Rerun stability

A subtle bug was fixed: `boxmot.ByteTrack` and `boxmot.BoTSORT` use a **class-level** `_count` variable on their internal `STrack`/`BaseTrack` class to assign IDs. This counter persists across Python instances within the same kernel session. Creating a new tracker object in a Jupyter cell did not reset it, causing IDs to continue incrementing across reruns (making statistics appear doubled).

Fixed in `_BoxmotWrapper.__init__` by `_reset_boxmot_id_counter()`, which probes all known boxmot module paths and resets any matching class-level counter to 0 before the tracker is used.

---

## Near-Miss Detection (`src/near_miss.py`)

See [docs/near_miss_detection_v1.md](docs/near_miss_detection_v1.md) for the full algorithm specification.

Quick summary:

1. **Proximity gate** — centroid distance < `proximity_px` OR IoU > 0
2. **Multi-criteria check** — at least 2 of: distance threshold, TTC threshold, minimum speed
3. **TTC estimation** — 1-D closing speed projected onto the inter-centroid axis
4. **Risk score** — weighted composite of normalised distance, TTC, and speed (0.0–1.0 → Low / Medium / High)
5. **FP filters** — stationary filter, direction filter, confidence gate, confirmation buffer
6. **Debounce** — suppresses repeated events for the same pair within `debounce_frames`

---

## Quick Start

```bash
pip install -r requirements.txt
# place traffic video at data/traffic_video.mp4
jupyter notebook notebooks/local_test.ipynb
```

All tunable parameters (model, thresholds, tracker, frame stride) are gathered at the top of each notebook in a single config cell.
