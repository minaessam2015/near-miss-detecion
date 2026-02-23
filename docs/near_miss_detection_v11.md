# Near-Miss Detection Algorithm — v1.1

**Module:** `src/near_miss.py` — `NearMissDetectorV11`
**Inherits from:** `NearMissDetector` (v1.0)
**Status:** Active (used in dual-pipeline comparison against v2.0)

---

## Overview

`NearMissDetectorV11` is a drop-in replacement for v1.0 (`NearMissDetector`) with
six targeted algorithmic improvements. It shares the same call signature, the same
`process_frame()` / `get_events_dataframe()` interface, and produces the same event
dict schema — except it adds one extra field: `d_min_px` (the predicted minimum
footpoint distance at closest approach).

The improvements fall into two categories:

**Geometry accuracy**
1. Footpoint distance instead of centroid distance (#3)
2. 2-D closest-approach trajectory instead of 1-D projected TTC (#2)
3. Scale-aware proximity threshold (#4)

**False-positive reduction**
4. Tightened IoU gate (#5)
5. Leaky confirmation buffer (#6)
6. Convergence guard — fires only on approach, never on diverge (#7)

> **What was NOT changed:**
> - Heading units (#1) — already correct in v1.0; `_estimate_heading` returns degrees
>   and all callers wrap with `math.radians()`.
> - Homography / BEV ground-plane (#8) — deferred to v2.0.

---

## Inputs and Outputs

### Input (per frame)

`process_frame(frame_idx, tracked_objects)` is identical to v1.0:

```
frame_idx        : int
tracked_objects  : OrderedDict[int, dict]   ← from any BaseTracker.update()
```

Each tracked object dict must contain:

```
{
  "id":         int
  "class":      "vehicle" | "pedestrian"
  "label":      "car" | "truck" | "bus" | "person" | …
  "bbox":       [x1, y1, x2, y2]
  "confidence": float
  "center":     (cx, cy)
  "trajectory": [(frame_idx, cx, cy), …]   ← centroid history
}
```

### Output schema (per event)

All v1.0 fields plus one addition:

| Field          | Type          | Description                                           |
|----------------|---------------|-------------------------------------------------------|
| `frame_index`  | `int`         | Frame at which the event was emitted                  |
| `timestamp_sec`| `float`       | `frame_index / fps`                                   |
| `object_id_1`  | `int`         | Track ID of first object                              |
| `object_id_2`  | `int`         | Track ID of second object                             |
| `class_1/2`    | `str`         | Category of each object                               |
| `label_1/2`    | `str`         | Fine-grained sub-type                                 |
| `distance_px`  | `float`       | **Footpoint** distance at event time (v1.1 change)    |
| `ttc_sec`      | `float\|None` | Time to closest approach `t_star` in seconds          |
| `d_min_px`     | `float`       | **NEW v1.1** — predicted minimum footpoint distance   |
| `risk_score`   | `float`       | Composite score in [0, 1]                             |
| `risk_level`   | `str`         | `"High"` / `"Medium"` / `"Low"`                       |
| `conf_1`, `conf_2` | `float`   | Detection confidence of each object                   |

---

## Algorithm — Step by Step

The detector evaluates all unordered pairs `(id1, id2)` of currently active tracked
objects. The canonical pair key is `(min(id1,id2), max(id1,id2))` so all state
(buffer, debounce) is shared regardless of iteration order.

```
┌─────────────────────────────────────────────────────────────────┐
│  For every unordered pair (A, B) of currently tracked objects:  │
│                                                                   │
│  Step 1 ── Footpoints & scale-aware proximity ──────────────────┤
│  Step 2 ── Proximity gate (footpoint dist + IoU) ───────────────┤
│              │ fail → _leak_buffer → SKIP PAIR                   │
│  Step 3 ── Trajectory analysis (speed + heading) ───────────────┤
│  Step 4 ── 2-D closest-approach ────────────────────────────────┤
│  Step 5 ── v1.1 risk score ─────────────────────────────────────┤
│  Step 6 ── Multi-criteria gate (2 of 3) ────────────────────────┤
│              │ fail → _leak_buffer → SKIP PAIR                   │
│  Step 7 ── False-positive filters (conf / stationary / dir / converge)
│              │ fail → _leak_buffer → SKIP PAIR                   │
│  Step 8 ── Leaky confirmation buffer ───────────────────────────┤
│              │ < confirm_frames → SKIP PAIR                      │
│  Step 9 ── Debounce ────────────────────────────────────────────┤
│              │ too soon → SKIP PAIR                              │
│  Step 10 ── Emit event ──────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────┘
```

---

### Step 1 — Footpoints and Scale-Aware Proximity (#3, #4)

#### Footpoint (improvement #3)

v1.0 measured the **centroid** distance between the two bounding boxes. v1.1 uses
the **bottom-center** of each bounding box instead — the pixel where the object
nominally touches the ground:

```
             ┌──────────┐
             │          │
             │  object  │
             │          │
             └────┬─────┘
                  │ ← footpoint  =  ((x1+x2)/2 ,  y2)
                  ▼
              ground plane
```

**Why footpoints?**

For tall objects viewed from a slightly elevated camera, the centroid is in the
middle of the object (e.g. the roof of a bus) rather than at its base. The physical
ground proximity is better approximated by the base, especially for mixed pairs
(pedestrian vs. vehicle) where the height difference makes centroid-based distance
systematically wrong:

```
     [TRUCK]  [PEDESTRIAN]
      ┌───┐      ┌─┐
      │   │      │ │     centroid of truck is much higher than pedestrian centroid
      │ × │      │ │     → centroid distance overestimates ground proximity
      │   │      │×│
      └─┬─┘      └┬┘
    ────┴──────────┴────  ground  ← footpoint distance is physically accurate
```

Code: `_footpoint(obj)` returns `((x1+x2)/2, y2)`.

#### Scale-Aware Proximity (#4)

v1.0 used a fixed `proximity_px` threshold for all objects at all distances. v1.1
computes an **effective proximity** that adapts to object size:

```
diag_1    = hypot(x2_a - x1_a, y2_a - y1_a)   ← bbox diagonal of object A
diag_2    = hypot(x2_b - x1_b, y2_b - y1_b)   ← bbox diagonal of object B
adaptive  = proximity_scale × mean(diag_1, diag_2)
eff_prox  = max(proximity_px, adaptive)          ← proximity_px is a hard floor
```

Default `proximity_scale = 0.5` means the adaptive threshold is half the average
bounding-box diagonal:

```
  Far from camera (small boxes, diag ≈ 60 px):
    adaptive = 0.5 × 60 = 30 px  → eff_prox = max(100, 30) = 100 px  (floor wins)

  Near camera (large boxes, diag ≈ 300 px):
    adaptive = 0.5 × 300 = 150 px → eff_prox = max(100, 150) = 150 px  (adaptive wins)
```

This corrects the perspective bias present in v1.0: distant objects appeared falsely
close (small pixel distance despite large real distance) while nearby large objects
appeared falsely safe (large pixel distance despite physical proximity).

---

### Step 2 — Proximity Gate (#5 IoU)

```python
dist      = hypot(fp2.x - fp1.x, fp2.y - fp1.y)   # footpoint distance
iou       = bbox_iou(bbox1, bbox2)
proximate = (dist < eff_prox) OR (iou > min_iou)   # v1.1: min_iou=0.05 not 0
```

v1.0 checked `iou > 0` — any single pixel of overlap triggered the IoU branch.
Bounding box coordinates from YOLO are integer-rounded, so a 1-pixel jitter in a
nearby detection could cause `iou = 1e-5 > 0` and false-trigger the gate.

v1.1 uses `iou > min_iou = 0.05` as a de-noise threshold.

If not proximate: `_leak_buffer(pair_key)` (see Step 8) and skip.

---

### Step 3 — Trajectory Analysis

Identical to v1.0 but the heading is now used with footpoints in the direction filter
(Step 7). Speed and heading are estimated from the last 5 centroid trajectory points:

```
pts  = trajectory[-5:]    ← last 5 (frame_idx, cx, cy) entries

speed = mean of step distances
      = mean( hypot(pts[i+1].cx - pts[i].cx, pts[i+1].cy - pts[i].cy)
              for i in 0..len-2 )
      unit: px/frame

heading = atan2(pts[-1].cy - pts[0].cy,
                pts[-1].cx - pts[0].cx)
        unit: degrees,  range [-180, +180]
```

Returns 0 for both if fewer than 2 trajectory points exist (newly registered track).

---

### Step 4 — 2-D Closest-Approach (#2)

This is the most significant algorithmic difference from v1.0.

#### v1.0 approach (1-D projected TTC)

v1.0 computed TTC by projecting the relative velocity onto the centroid-to-centroid
axis — a 1-D scalar closing speed — then dividing the current distance by it:

```
closing_speed = -(Δv · û)    ← scalar projection onto inter-centroid unit vector
TTC           = dist / closing_speed
```

**Limitation:** This only asks "are they approaching along the line joining them
right now?" It misses **crossing trajectories** where two objects will intersect
close to each other's path but their current headings are not directly toward each
other.

```
  v1.0 TTC misses this case:

      A ──────►                 A is moving right
               ↑                B is moving up
               B                They will pass very close, but û = diagonal,
                                 and closing_speed = -(Δv · û) ≈ small → TTC = ∞
                                 → No alert!
```

#### v1.1 approach (2-D constant-velocity closest approach)

v1.1 models both objects as moving at constant velocity and finds the **time t\***
at which the inter-object distance is minimum:

```
Notation:
  fp1, fp2  : current footpoints (2-D pixel positions)
  v1, v2    : velocity vectors = speed × (cos h, sin h)  [px/frame]

  p = fp2 - fp1                  ← relative position
  v = v2  - v1                   ← relative velocity

  Parametric future position of B relative to A:
  p(t) = p + v · t

  Minimise |p(t)|²:
    d/dt |p + vt|² = 0
    2(p + vt) · v = 0
    t_raw = -(p · v) / (v · v)    ← unclamped time to closest approach
```

```
  Graph of inter-footpoint distance over time for a crossing conflict:

  distance (px)
    │
300 ┤ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─         eff_prox threshold
    │              ╲
200 ┤               ╲
    │                ╲   d_min ←──── predicted minimum distance
100 ┤                 ╲______/
    │                            (objects pass each other and diverge)
  0 ┤
    └──────┬───────┬──────┬──── time (frames)
           0      t*     t_horizon
           now  closest  horizon
                approach
```

**t\* clamping:**

```
t_horizon_frames = t_horizon_sec × fps       # default 5s × fps
t_star           = clamp(t_raw, 0, t_horizon_frames)
```

Clamping to 0 handles "already-past" cases (t_raw < 0). Clamping to the horizon
prevents spurious far-future predictions from contributing to risk.

**Predicted minimum distance:**

```
d_min = |p + v · t_star|    ← footpoint distance at closest approach
```

**Convergence flag:**

```
converging = (t_raw > 0)
```

`converging = False` means the closest approach has already passed; the objects
are diverging. The convergence guard in Step 7 (#7) uses this to suppress
post-encounter alerts.

**Returns:** `(t_star_sec, d_min_px, converging)`

---

### Step 5 — v1.1 Risk Score

v1.0 weighted current distance and TTC equally (40% each) and speed at 20%.
v1.1 promotes `d_min` (predicted closest approach) as the primary signal:

```
norm_dmin = min(d_min   / eff_prox,          1.0)   ← 0 = will get very close
norm_dist = min(dist    / eff_prox,          1.0)   ← 0 = already very close
norm_t    = min(t_star  / ttc_threshold,     1.0)   ← 0 = imminent
speed_fac = min(max_speed / 30.0,            1.0)   ← capped at 30 px/frame

risk_score = (1 - norm_dmin) × 0.45
           + (1 - norm_dist) × 0.15
           + (1 - norm_t)    × 0.30
           + speed_fac       × 0.10
```

**Weight rationale vs v1.0:**

```
Signal                  v1.0 weight    v1.1 weight    Rationale
──────────────────────  ───────────    ───────────    ─────────────────────────────────
d_min (predicted)            –             45%        Earlier warning; catches crossing
distance (current)          40%            15%        Secondary; d_min is more informative
TTC / time urgency          40%            30%        How soon does the close approach happen?
speed factor                20%            10%        Kinetic severity proxy; still useful
```

**Risk level thresholds** (unchanged from v1.0):

```
Score ≥ 0.70  →  High
Score ≥ 0.40  →  Medium
Score < 0.40  →  Low
```

---

### Step 6 — Multi-Criteria Gate

At least **2 of 3** criteria must be satisfied:

| Criterion | v1.0 condition            | v1.1 condition               |
|-----------|---------------------------|------------------------------|
| Proximity | `dist < proximity_px`     | `dist < eff_prox`            |
| Closest approach | _not present_       | `d_min < eff_prox`           |
| Motion    | `max(speed1,speed2)>5 px/fr` | same                      |

v1.0 had `{proximity, ttc < threshold, motion}` as the three criteria. v1.1
replaces the TTC criterion with `d_min < eff_prox` because `d_min` is more
informative (it captures future trajectories, not just a threshold on time).

If fewer than 2 criteria are met: `_leak_buffer(pair_key)` and skip.

---

### Step 7 — False-Positive Filters

All four filters are applied only when `filters_enabled=True`. Order of application:

#### Filter 1 — Confidence gate

```python
discard if: conf1 < min_confidence OR conf2 < min_confidence
```

Default: `min_confidence = 0.5`

Unchanged from v1.0. Suppresses events involving uncertain detections.

#### Filter 2 — Stationary filter

```python
discard if: speed1 < stationary_speed_px AND speed2 < stationary_speed_px
```

Default: `stationary_speed_px = 5.0 px/frame`

Unchanged from v1.0. Suppresses parked-vehicle proximity.

#### Filter 3 — Direction filter (#3 footpoints change)

v1.1 uses **footpoints** to compute the inter-object axis instead of centroids, but
the logic is otherwise the same as v1.0:

```
angle_diff    = |heading1 - heading2| mod 180°
same_direction = angle_diff < same_direction_deg

axis = (fp2 - fp1) / |fp2 - fp1|            ← unit vector from A to B via footpoints
closing_speed = -(Δv · axis)

discard if: same_direction AND closing_speed < 2.0 px/frame
```

Default: `same_direction_deg = 30°`

```
  Example — parallel-lane vehicles (same direction, barely closing):

  ──────► A  }               heading_A ≈ heading_B ≈ 0°
  ──────► B  }  same lane    angle_diff ≈ 0° < 30°  → same_direction = True
                             closing_speed ≈ 0        < 2.0 → DISCARD

  Example — merging vehicles (same direction but converging):

  ──────► A  \               heading_A ≈ heading_B ≈ 0°
  ────────►  B \             angle_diff ≈ 0° → same_direction = True
                             closing_speed ≈ 5.0 px/fr > 2.0 → KEEP
```

#### Filter 4 — Convergence guard (#7)

```python
discard if: NOT converging    # i.e. t_raw <= 0
```

This filter is **new in v1.1** (no equivalent in v1.0).

`converging = False` means `t_raw ≤ 0`, i.e. the closest-approach moment is in
the past. The objects have already passed each other and are now diverging. v1.0
would still emit an alert here (current distance may still be < `proximity_px` for
a few frames after the crossing). v1.1 suppresses it:

```
  Timeline of a fast crossing event:

  distance
    │
    │\  ← alert window
    │ ╲
    │  ╲  ← t* (closest approach)
    │   ╲___________/ diverging
    │
    └────────────────────────────── time
          │    │
      approach  already past → v1.0 still fires here, v1.1 does NOT
```

On miss: `_leak_buffer(pair_key)` (same as all other filters).

---

### Step 8 — Leaky Confirmation Buffer (#6)

**v1.0 — hard reset:**

```
buffer += 1    on each frame where pair passes all checks
buffer = 0     on ANY miss (proximity fail, any filter fail)

Emit event when buffer ≥ confirm_frames
```

The hard reset means a single missed frame (e.g. from detection jitter — YOLO
drops a detection for one frame) zeroes out accumulated evidence and restarts the
5-frame counter. In practice, brief jitter increased false-negative rate.

**v1.1 — leaky integrator:**

```
buffer += 1.0   on each frame where pair passes all checks
buffer = max(buffer - buffer_decay, 0)   on ANY miss   ← "leaky" decay, not reset

Emit event when buffer ≥ confirm_frames
```

Default `buffer_decay = 0.5` — each missed frame removes half a frame's worth of
evidence, but accumulated evidence from previous frames is partially preserved:

```
  Example (confirm_frames=5, buffer_decay=0.5):

  Frame:   F1  F2  F3  |miss|  F4  F5  F6
  v1.0:     1   2   3     0    1   2   3    ← reset → need 5 more → slow!
  v1.1:     1   2   3    2.5   3.5 4.5 5.5 → EMIT at F6 (only 1 extra frame)
```

```
  Buffer evolution graph (v1.0 vs v1.1):

  buffer value
  6 ┤
    │                            v1.1 ─────────────────►(emit at ~5)
  5 ┤──────────────────────────────────────
    │                 ╱─────────────────
  4 ┤                ╱
    │               ╱
  3 ┤    v1.0 ─────╱             v1.1 drop on miss (−0.5, not −3)
    │         \   ╱         ╱
  2 ┤          ╲ ╱         ╱
    │           ╳─────────╱     v1.0 drop on miss (full reset)
  1 ┤          ╱
    │         ╱
  0 ┤────────────────────────────────────────────── frames
        1  2  miss  3  4  5  6  7
```

The internal `_confirmation_buffer` type is changed from `defaultdict(int)` to
`defaultdict(float)` in v1.1 to support sub-integer decay values.

---

### Step 9 — Debounce

Unchanged from v1.0:

```python
last = _last_event_frame.get(pair_key, -debounce_frames - 1)
discard if: frame_idx - last < debounce_frames
```

Default: `debounce_frames = 30`

After an event is emitted for a pair, that pair cannot emit another event until
30 processed frames have elapsed. This prevents a sustained near-miss encounter
from flooding the event log.

---

### Step 10 — Emit Event

If the pair passes all checks, an event is emitted and appended to `self._events`.
`_last_event_frame[pair_key]` and `_last_event_data[pair_key]` are updated (the
latter is used by `active_pair_data()` for live visualizer annotations).

---

## Improvement Summary Table

| # | Improvement | v1.0 behaviour | v1.1 behaviour | Effect |
|---|-------------|----------------|----------------|--------|
| 2 | TTC model | 1-D projected closing speed | 2-D closest-approach `t*` + `d_min` | Catches crossing conflicts |
| 3 | Distance reference point | Centroid `(cx, cy)` | Footpoint `(bottom_cx, y2)` | Accurate for tall objects |
| 4 | Proximity threshold | Fixed `proximity_px` | `max(prox_px, scale × mean_diag)` | Corrects perspective bias |
| 5 | IoU gate | `iou > 0` | `iou > min_iou` (0.05) | Removes bbox-jitter FP |
| 6 | Confirmation buffer | Hard reset to 0 | Leaky decay `−buffer_decay` | Tolerates detection jitter |
| 7 | Convergence guard | Not present | Discard if `t_raw ≤ 0` | No post-encounter alerts |

---

## Full Pipeline Data-Flow Diagram

```
Tracker output (one frame)
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  NearMissDetectorV11.process_frame(frame_idx, tracked_objects)        │
│                                                                        │
│  For every unordered pair (A, B):                                      │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Step 1  Footpoints & eff_prox                                    │ │
│  │         fp = (bottom_cx, y2)                                     │ │
│  │         eff_prox = max(prox_px, scale × mean_diag)               │ │
│  └────────────────────────┬─────────────────────────────────────────┘ │
│                           │                                            │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Step 2  Proximity gate                                           │ │
│  │         dist = |fp2 - fp1|                                       │ │
│  │         iou  = bbox_iou(b1, b2)                                  │ │
│  │         pass = (dist < eff_prox) OR (iou > min_iou)             │ │
│  └──────────────────── fail: leak_buffer ──► SKIP ─────────────────┘ │
│                           │                                            │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Step 3  Trajectory analysis (last 5 pts)                         │ │
│  │         speed   = mean step distance  [px/frame]                 │ │
│  │         heading = atan2(Δy, Δx)       [degrees]                  │ │
│  └────────────────────────┬─────────────────────────────────────────┘ │
│                           │                                            │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Step 4  2-D closest-approach                                     │ │
│  │         p = fp2 − fp1,  v = v2 − v1                              │ │
│  │         t_raw  = −(p·v)/(v·v)                                    │ │
│  │         t_star = clamp(t_raw, 0, horizon_frames) / fps  [sec]   │ │
│  │         d_min  = |p + v · t_star|               [px]            │ │
│  │         converging = (t_raw > 0)                                 │ │
│  └────────────────────────┬─────────────────────────────────────────┘ │
│                           │                                            │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Step 5  v1.1 risk score                                          │ │
│  │         risk = (1−d_min/eff_prox)×0.45                          │ │
│  │              + (1−dist/eff_prox) ×0.15                          │ │
│  │              + (1−t_star/ttc_th) ×0.30                          │ │
│  │              + (speed/30)        ×0.10                          │ │
│  └────────────────────────┬─────────────────────────────────────────┘ │
│                           │                                            │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Step 6  Multi-criteria gate (2 of 3 required)                    │ │
│  │         C1: dist  < eff_prox                                     │ │
│  │         C2: d_min < eff_prox                                     │ │
│  │         C3: max(speed1, speed2) > 5.0 px/frame                  │ │
│  └──────────────────── fail: leak_buffer ──► SKIP ─────────────────┘ │
│                           │                                            │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Step 7  False-positive filters  (filters_enabled=True)           │ │
│  │         F1  confidence: conf1<min_conf OR conf2<min_conf → skip  │ │
│  │         F2  stationary: both speed < stationary_speed_px → skip  │ │
│  │         F3  direction:  same_dir AND closing_speed<2 → skip      │ │
│  │         F4  converging: t_raw ≤ 0 → skip    ← NEW in v1.1       │ │
│  └──────────────────── fail: leak_buffer ──► SKIP ─────────────────┘ │
│                           │                                            │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Step 8  Leaky confirmation buffer                                │ │
│  │         hit:  buffer[pair] += 1.0                                │ │
│  │         miss: buffer[pair] = max(buffer − decay, 0)             │ │
│  │         gate: buffer[pair] < confirm_frames → skip              │ │
│  └────────────────────────┬─────────────────────────────────────────┘ │
│                           │                                            │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Step 9  Debounce                                                 │ │
│  │         frame_idx − last_event_frame[pair] < debounce → skip    │ │
│  └────────────────────────┬─────────────────────────────────────────┘ │
│                           │                                            │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Step 10  Emit event dict → self._events                          │ │
│  │          update _last_event_frame, _last_event_data              │ │
│  └────────────────────────┬─────────────────────────────────────────┘ │
└───────────────────────────┼────────────────────────────────────────────┘
                            │
          ┌─────────────────┴───────────────────────┐
          │                                         │
   get_events_dataframe()                   active_pairs(frame_idx)
   pandas DataFrame                         active_pair_data(frame_idx)
   (all events, sorted by timestamp)        (for live visualizer annotations)
```

---

## 2-D Closest-Approach — Visual Reference

### Head-on collision (TTC well-estimated by both v1.0 and v1.1)

```
  A ──────►          ◄────── B
  fp_A →                   ← fp_B

  p = fp_B − fp_A  (pointing right)
  v = v_B  − v_A  (pointing left, negative x)
  p · v < 0  →  t_raw > 0  (converging)
  d_min  ≈  0 (they will collide on current paths)
  t_star ≈  dist / |closing_speed|
```

### Crossing conflict (missed by v1.0, caught by v1.1)

```
         B ↑
           │       ← path of B
           │
  A ──────►│──────►
           │
           ▼
  fp_A and fp_B may be > proximity_px apart right now.
  However p + v·t* → small at t_raw ≈ dist_to_intersection / relative_speed
  → d_min < eff_prox  → v1.1 raises C2, v1.0's C1+TTC would miss it.
```

### Post-encounter diverge (emitted by v1.0, suppressed by v1.1)

```
  Before crossing:  t_raw > 0, converging=True  → both fire
  After  crossing:  t_raw < 0, converging=False → v1.1 SUPPRESSES, v1.0 fires
                               ^^^^^^^^^^^^^^^^^^^^ convergence guard #7
```

---

## Configuration Reference

| Parameter            | Type    | Default | Description |
|----------------------|---------|---------|-------------|
| `proximity_px`       | float   | 100.0   | Hard-floor proximity threshold (px) |
| `ttc_threshold`      | float   | 2.0     | TTC / `t_star` threshold for risk normalisation (sec) |
| `fps`                | float   | 15.0    | Effective FPS after stride; used for TTC conversion |
| `debounce_frames`    | int     | 30      | Min processed frames between events for the same pair |
| `filters_enabled`    | bool    | True    | Toggle all FP filters on/off |
| `stationary_speed_px`| float   | 5.0     | Max speed (px/frame) for stationary classification |
| `confirm_frames`     | int     | 5       | Buffer threshold before emitting event |
| `same_direction_deg` | float   | 30.0    | Heading difference (°) considered "same direction" |
| `min_confidence`     | float   | 0.5     | Minimum detection confidence for either object |
| `proximity_scale`    | float   | 0.5     | Scale factor for adaptive proximity (0.0 = disabled) |
| `min_iou`            | float   | 0.05    | Minimum IoU overlap to trigger bbox-overlap gate |
| `buffer_decay`       | float   | 0.5     | Amount subtracted from buffer on each miss frame |
| `t_horizon_sec`      | float   | 5.0     | Max look-ahead horizon for closest-approach (sec) |

---

## Known Limitations

1. **Pixel-space geometry.** All distances and speeds remain in pixel coordinates.
   Perspective distortion is partially compensated by the scale-aware proximity (#4)
   but not eliminated. Objects far from the camera still have proportionally smaller
   pixel displacement per real-world metre. Resolved in v2.0 by ray-casting to the
   ground plane.

2. **Constant-velocity assumption.** The 2-D closest-approach formula assumes
   linear trajectories. Vehicles braking, accelerating, or turning will cause
   `t_star` and `d_min` to diverge from reality. The `t_horizon_sec` clamp (default
   5 s) limits the prediction window to reduce the impact of this assumption.

3. **Footpoint approximation.** The bottom-center of a bounding box is only a proxy
   for the ground contact point. For an obliquely viewed vehicle, the actual
   wheel-contact pixel may be offset from the bbox center-bottom by tens of pixels.

4. **Trajectory warmup latency.** Speed and heading default to 0 for tracks with
   fewer than 2 trajectory points. New tracks (within the first 2 processed frames
   of their life) will fail the motion criterion (C3) in Step 6, delaying detection.
   Combined with `confirm_frames=5`, the minimum detection latency is
   `(min_hits + confirm_frames) / effective_fps`.

5. **2-D only — no depth.** Objects in adjacent but physically separate lanes can
   appear close in the 2-D projection (same x, similar y) despite being in different
   depth planes. The direction filter (F3) partially mitigates parallel-lane cases
   but cannot distinguish objects in different depth planes. Resolved in v2.0.

---

## Relationship to Other Versions

```
NearMissDetector (v1.0)
        │
        └── NearMissDetectorV11 (v1.1)  ← this document
                │
                └── NearMissDetectorV20 (v2.0)    inherits v1.1;
                        │                          overrides process_frame
                        │                          to operate in 3-D ground space
                        └── NearMissDetectorV30    wraps V20; adds
                                                   ground_pt_source awareness
```

v2.0 falls back to v1.1 pixel logic transparently for any pair where either object
is missing a `ground_pt` field. All improvements in v1.1 are therefore active in
v2.0's 2-D fallback path.
