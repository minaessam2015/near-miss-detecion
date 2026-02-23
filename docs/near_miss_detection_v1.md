# Near-Miss Detection Algorithm — v1.0

**Module:** `src/near_miss.py` — `NearMissDetector`
**Version:** 1.0
**Status:** Active

---

## Overview

The near-miss detector operates on the output of the tracking layer. It is called once per processed frame with the current set of tracked objects and evaluates every pair of objects for collision risk. When a pair meets the risk criteria — and survives the false-positive filters — an event is recorded.

The algorithm is **stateful**: it accumulates a trajectory history per track ID (provided by the tracker), maintains a confirmation buffer for each candidate pair, and enforces a debounce window so that a single sustained near-miss does not generate a flood of repeated events.

---

## Inputs and Outputs

### Input (per frame)

`process_frame(frame_idx, tracked_objects)` receives:

- `frame_idx` — integer video frame index
- `tracked_objects` — `OrderedDict[int, dict]` from any `BaseTracker.update()` call

Each tracked object dict contains at minimum:

```
{
  "id":         int
  "class":      "vehicle" | "pedestrian"
  "label":      "car" | "truck" | "bus" | "person" | …
  "bbox":       [x1, y1, x2, y2]
  "confidence": float
  "center":     (cx, cy)
  "trajectory": [(frame_idx, cx, cy), …]   ← full centroid history
}
```

### Output (per frame)

A list of event dicts emitted **this frame** (may be empty). All events are also accumulated internally and retrievable via `get_events_dataframe()`.

Event dict schema:

| Field | Type | Description |
|---|---|---|
| `frame_index` | `int` | Frame at which the event was emitted |
| `timestamp_sec` | `float` | `frame_index / fps` |
| `object_id_1` | `int` | Track ID of first object |
| `object_id_2` | `int` | Track ID of second object |
| `class_1`, `class_2` | `str` | Category of each object |
| `label_1`, `label_2` | `str` | Fine-grained sub-type |
| `distance_px` | `float` | Centroid distance at event time |
| `ttc_sec` | `float\|None` | Estimated time-to-collision; `None` if objects are diverging |
| `risk_score` | `float` | Composite score in [0, 1] |
| `risk_level` | `str` | `"High"` / `"Medium"` / `"Low"` |
| `conf_1`, `conf_2` | `float` | Detection confidence of each object |

---

## Algorithm — Step by Step

The detector evaluates all **unordered pairs** `(id1, id2)` of currently active tracked objects. Pairs where `id1 == id2` are skipped; each unordered pair is represented by the canonical key `(min(id1,id2), max(id1,id2))` to ensure debounce and confirmation state are shared regardless of order.

### Step 1 — Proximity Gate

```
distance = hypot(cx2 - cx1, cy2 - cy1)
iou      = bbox_iou(bbox1, bbox2)
proximate = (distance < proximity_px) OR (iou > 0)
```

A pair is **proximate** if their centroids are closer than `proximity_px` pixels **or** their bounding boxes overlap at all (IoU > 0). The IoU check catches cases where bounding boxes overlap but centroids are farther apart than the threshold — common for large vehicles viewed obliquely.

If the pair is not proximate the confirmation buffer for that pair is reset to 0 and evaluation stops.

Default: `proximity_px = 100 px`

### Step 2 — Trajectory Analysis

For each object in the pair, the last 5 trajectory points are used:

**Speed estimation**

```
speed = mean of step-to-step centroid distances over the last 5 frames
      = mean( hypot(pts[i+1] - pts[i]) for i in range(n-1) )
      unit: px / frame
```

If fewer than 2 trajectory points exist, speed defaults to 0.

**Heading estimation**

```
dx = last_x - first_x  (over the last 5 trajectory points)
dy = last_y - first_y
heading = atan2(dy, dx)  [degrees, range: -180 to +180]
```

The heading represents the net direction of travel, not instantaneous. Using 5-point windows smooths out jitter from imperfect detections.

### Step 3 — Time-to-Collision (TTC)

TTC is estimated using **1-D projected closing speed** along the inter-centroid axis. This approach avoids the perspective distortion complexities of converting pixel velocities to real-world speeds.

```
d⃗   = (cx2 - cx1, cy2 - cy1)          ← vector from obj1 to obj2
dist = |d⃗|
û    = d⃗ / dist                        ← unit vector

v⃗1 = speed1 * (cos(heading1), sin(heading1))   ← velocity of obj1 (px/frame)
v⃗2 = speed2 * (cos(heading2), sin(heading2))   ← velocity of obj2 (px/frame)

relative_velocity = v⃗2 - v⃗1
closing_speed     = -(relative_velocity · û)    ← positive = approaching
```

If `closing_speed <= 0` the objects are diverging and `ttc = math.inf` (no collision risk).

Otherwise:

```
ttc_frames = dist / closing_speed
ttc_sec    = ttc_frames / fps
```

TTC is used both in the multi-criteria check (Step 4) and in the risk score (Step 5). When objects are diverging, TTC does not contribute to the risk score.

### Step 4 — Multi-Criteria Gate

To reduce spurious events, at least **2 out of 3** criteria must be satisfied simultaneously:

| Criterion | Condition |
|---|---|
| Proximity | `distance < proximity_px` |
| TTC | `ttc_sec < ttc_threshold` |
| Motion | `max(speed1, speed2) > 5.0 px/frame` |

This prevents flagging stationary objects that happen to be parked close together (proximity met but neither TTC nor motion), or fast-moving objects that are far apart (motion met but not proximity).

If fewer than 2 criteria are met the confirmation buffer resets and evaluation stops.

Default: `ttc_threshold = 2.0 s`

### Step 5 — Risk Score

The risk score combines three normalised signals into a scalar in [0, 1]:

```
norm_dist  = min(distance / proximity_px, 1.0)        # 0 = very close, 1 = at threshold
norm_ttc   = min(ttc_sec / ttc_threshold, 1.0)         # 0 = imminent, 1 = at/beyond threshold
               (1.0 when ttc = inf, i.e. diverging)
speed_factor = min(max(speed1, speed2) / 30.0, 1.0)   # 30 px/frame as reference maximum

risk_score = (1 - norm_dist)  * 0.4
           + (1 - norm_ttc)   * 0.4
           + speed_factor     * 0.2
```

**Weight rationale:**

- Distance and TTC each carry 40% weight — they are equally important primary indicators.
- Speed carries 20% — it is a contributing factor but not sufficient alone (a fast object far away should not score high).
- Speed is capped at 30 px/frame to prevent very fast detections from dominating the score.

**Risk levels:**

| Score range | Level |
|---|---|
| ≥ 0.7 | High |
| 0.4 – 0.69 | Medium |
| < 0.4 | Low |

### Step 6 — False-Positive Filters (v1.6)

All four filters are applied only when `filters_enabled=True`. They can be toggled off for comparison (Section 6 of the notebook runs with/without filters on the same `frame_data` cache).

**Filter 1 — Stationary object filter**

```
discard if: speed1 < stationary_speed_px AND speed2 < stationary_speed_px
```

Default: `stationary_speed_px = 5.0 px/frame`

Suppresses false alerts from stopped vehicles that are close to each other — e.g. at an intersection waiting for a light. If both objects are essentially stationary there is no imminent collision.

**Filter 2 — Direction filter**

```
angle_diff = absolute difference in headings (normalised to 0–180°)
same_direction = angle_diff < same_direction_deg

closing_speed = -(relative_velocity · û)

discard if: same_direction AND closing_speed < 2.0 px/frame
```

Default: `same_direction_deg = 30°`

Suppresses false alerts from vehicles traveling in the same direction (e.g. in adjacent lanes) that happen to be close but are not actually converging. The `closing_speed < 2.0` guard ensures the filter does not suppress cases where same-direction vehicles are genuinely merging.

**Filter 3 — Confidence gate**

```
discard if: confidence_1 < min_confidence OR confidence_2 < min_confidence
```

Default: `min_confidence = 0.5`

Suppresses events involving low-confidence detections that may be false positives from the detector itself. A near-miss involving an uncertain detection is not reliable enough to report.

**Filter 4 — Confirmation buffer**

```
confirmation_buffer[(id1, id2)] += 1
discard if: confirmation_buffer < confirm_frames
```

Default: `confirm_frames = 5`

The near-miss condition must persist for at least 5 consecutive processed frames before the event is emitted. A single-frame proximity (e.g. a detection jitter) does not produce an event. The buffer resets to 0 whenever the pair fails the proximity gate or any earlier filter.

### Step 7 — Debounce

```
last_frame = last_event_frame.get((id1, id2), -debounce_frames - 1)
discard if: frame_idx - last_frame < debounce_frames
```

Default: `debounce_frames = 30`

After an event is emitted for a pair, the same pair cannot emit another event until `debounce_frames` processed frames have elapsed. This prevents a single sustained near-miss encounter from flooding the event log with dozens of identical entries.

### Step 8 — Emit Event

If the pair passes all checks, an event dict is constructed and appended to the internal event log. The `last_event_frame` for the pair is updated.

---

## Configuration Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `proximity_px` | float | 100 | Centroid distance gate (pixels) |
| `ttc_threshold` | float | 2.0 | Maximum TTC to flag (seconds) |
| `fps` | float | 15 | Effective FPS after stride, used for TTC conversion |
| `debounce_frames` | int | 30 | Minimum processed frames between events for same pair |
| `filters_enabled` | bool | True | Toggle all FP filters on/off |
| `stationary_speed_px` | float | 5.0 | Max speed (px/frame) for stationary classification |
| `confirm_frames` | int | 5 | Consecutive frames required before emitting an event |
| `same_direction_deg` | float | 30.0 | Heading difference (°) considered "same direction" |
| `min_confidence` | float | 0.5 | Minimum detection confidence for either object |

---

## Data Flow Diagram

```
┌──────────────────┐        ┌──────────────────────────┐
│  BaseTracker     │        │  NearMissDetector         │
│  .update(dets,   │──────► │  .process_frame(          │
│   frame_idx,     │        │     frame_idx,            │
│   frame)         │        │     tracked_objects)      │
│                  │        │                           │
│  Returns:        │        │  For every pair (A, B):   │
│  {id: state_dict}│        │  1. Proximity gate        │
└──────────────────┘        │  2. Trajectory analysis   │
                            │  3. TTC estimation        │
                            │  4. Multi-criteria gate   │
                            │  5. Risk scoring          │
                            │  6. FP filters (×4)       │
                            │  7. Confirmation buffer   │
                            │  8. Debounce              │
                            │  9. Emit event            │
                            │                           │
                            │  _events: [event_dict, …] │
                            └──────────────────────────┘
                                         │
                      ┌──────────────────┴────────────────┐
                      │                                    │
               .get_events_dataframe()            .active_pairs(frame_idx)
               pandas DataFrame                  pairs active within
               (all events, sorted)              last debounce_frames
                                                 (used for annotation)
```

---

## Known Limitations and Assumptions

1. **Pixel-space only.** All distance and speed estimates are in pixel coordinates. Without camera calibration or homography mapping the estimates are not in real-world units (metres, km/h). A vehicle far from the camera appears smaller and its pixel speed is underestimated relative to an identical vehicle close to the camera.

2. **Linear velocity model.** TTC is computed assuming constant velocity over the next `ttc_sec` seconds. Vehicles that are braking or turning will produce inaccurate TTC estimates.

3. **Trajectory window (5 frames).** Speed and heading are estimated from the last 5 tracked positions. Very short trajectories (< 2 points, e.g. newly confirmed tracks) default to speed = 0 and heading = 0°. This can delay near-miss detection by a few frames for newly appeared objects.

4. **2-D projected closing speed.** The TTC computation projects relative velocity onto the 2-D centroid-to-centroid axis. Objects approaching at an angle (e.g. one coming from the side) will have their closing speed underestimated by `cos(approach_angle)`. Head-on and rear-end approaches are measured accurately.

5. **Fixed `proximity_px` threshold.** The proximity gate is a fixed pixel distance and does not account for the object's apparent size. A proximity of 100 px is significant for small motorcycles but trivial for two large buses. A proportional threshold (e.g. based on the average bbox diagonal) would be more physically meaningful.

6. **No 3-D context.** Objects in different lanes that appear close in the 2-D image projection may be physically far apart. Without depth estimation or lane segmentation, such pairs may generate false positives. The direction filter partially mitigates this for parallel-lane scenarios.

7. **`confirm_frames` latency.** The confirmation buffer introduces a detection latency of `confirm_frames / effective_fps` seconds. At 10 effective FPS with `confirm_frames=5`, the minimum latency is 0.5 s. This is a deliberate trade-off against false-positive rate.

---

## Notebook Validation

The algorithm is evaluated in two ways in `notebooks/local_test.ipynb`:

- **Section 4** — full event log with risk scores and labels for visual inspection
- **Section 6 (FP Filter Comparison)** — runs the same `frame_data` through a second `NearMissDetector` with `filters_enabled=False`, then prints a before/after table showing how many events each filter removes

- **Section 8 (Threshold Sweep)** — reruns over `frame_data` for `proximity_px` in [60, 80, 100, 120, 150, 180, 220] to show sensitivity of event count to the primary threshold, without any additional inference cost.
