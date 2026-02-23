"""
Tasks 1.4 + 1.6 — Near-Miss Detection & False-Positive Filtering

Detects near-miss events between tracked objects using:
  - Proximity (centroid distance + IoU)
  - Time-To-Collision (TTC) estimation
  - Risk scoring (0.0–1.0 → High / Medium / Low)

Bonus (Task 1.6) false-positive filters:
  - Stationary object filter
  - Minimum duration confirmation buffer
  - Direction-of-travel filter
  - Confidence gate
"""

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None  # optical flow unavailable; v4.0 falls back to trajectory

_log_v20 = logging.getLogger("near_miss.v20")
_log_v40 = logging.getLogger("near_miss.v40")


# ---------------------------------------------------------------------------
# Constants / default thresholds
# ---------------------------------------------------------------------------

_RISK_HIGH = 0.7
_RISK_MEDIUM = 0.4


class NearMissDetector:
    """
    Stateful near-miss detector. Call process_frame() once per processed frame.
    """

    def __init__(
        self,
        proximity_px: float = 100.0,
        ttc_threshold: float = 2.0,
        fps: float = 15.0,
        debounce_frames: int = 30,
        # FP filter thresholds
        filters_enabled: bool = True,
        stationary_speed_px: float = 5.0,
        confirm_frames: int = 5,
        same_direction_deg: float = 30.0,
        min_confidence: float = 0.5,
        moving_speed_px: float = 5.0,
        risk_high_threshold: float = _RISK_HIGH,
        risk_medium_threshold: float = _RISK_MEDIUM,
    ):
        """
        Args:
            proximity_px: Centroid distance (px) below which objects are "proximate".
            ttc_threshold: Time-to-collision threshold in seconds.
            fps: Effective FPS after frame-stride (used to convert px/frame → seconds).
            debounce_frames: Min frames between events for the same object pair.
            filters_enabled: Toggle all FP filters on/off for comparison.
            stationary_speed_px: Max px/frame speed to consider an object stationary.
            confirm_frames: Frames a condition must persist before emitting an event.
            same_direction_deg: Heading difference (°) considered "same direction".
            min_confidence: Detection confidence gate.
            moving_speed_px: Minimum speed (px/processed-frame) to count as
                             "moving" in the 2-of-3 near-miss criteria gate.
            risk_high_threshold: Risk-score cutoff for "High" severity.
            risk_medium_threshold: Risk-score cutoff for "Medium" severity.
                                   Scores below this are labeled "Low".
        """
        self.proximity_px = proximity_px
        self.ttc_threshold = ttc_threshold
        self.fps = fps
        self.debounce_frames = debounce_frames
        self.filters_enabled = filters_enabled
        self.stationary_speed_px = stationary_speed_px
        self.confirm_frames = confirm_frames
        self.same_direction_deg = same_direction_deg
        self.min_confidence = min_confidence
        self.moving_speed_px = moving_speed_px
        self.risk_high_threshold = float(risk_high_threshold)
        self.risk_medium_threshold = float(risk_medium_threshold)

        if not (0.0 <= self.risk_medium_threshold <= 1.0):
            raise ValueError(
                f"risk_medium_threshold must be in [0, 1], got {self.risk_medium_threshold}"
            )
        if not (0.0 <= self.risk_high_threshold <= 1.0):
            raise ValueError(
                f"risk_high_threshold must be in [0, 1], got {self.risk_high_threshold}"
            )
        if self.risk_high_threshold < self.risk_medium_threshold:
            raise ValueError(
                "risk_high_threshold must be >= risk_medium_threshold "
                f"(got high={self.risk_high_threshold}, medium={self.risk_medium_threshold})"
            )

        # (id1, id2) → frame index of last emitted event
        self._last_event_frame: Dict[Tuple[int, int], int] = {}
        # (id1, id2) → most recent emitted event dict (for live visualization)
        self._last_event_data: Dict[Tuple[int, int], Dict[str, Any]] = {}
        # (id1, id2) → consecutive frames where near-miss condition was met (for confirmation buffer)
        self._confirmation_buffer: Dict[Tuple[int, int], int] = defaultdict(int)
        # All logged events
        self._events: List[Dict[str, Any]] = []
        # Global processed-frame timing (used to normalize trajectory frame gaps)
        self._nominal_frame_step: Optional[int] = None
        self._last_frame_idx_seen: Optional[int] = None

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _centroid_distance(obj1: Dict, obj2: Dict) -> float:
        cx1, cy1 = obj1["center"]
        cx2, cy2 = obj2["center"]
        return math.hypot(cx2 - cx1, cy2 - cy1)

    @staticmethod
    def _compute_iou(bbox1: List[int], bbox2: List[int]) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    # ------------------------------------------------------------------
    # Trajectory analysis helpers
    # ------------------------------------------------------------------

    def _update_frame_step(self, frame_idx: int) -> None:
        """
        Track nominal processed-frame step from process_frame() calls.

        Handles same-frame re-entry (delta==0) and sequence rewinds (delta<0).
        """
        if self._last_frame_idx_seen is None:
            self._last_frame_idx_seen = frame_idx
            return
        delta = frame_idx - self._last_frame_idx_seen
        if delta > 0:
            if self._nominal_frame_step is None:
                self._nominal_frame_step = delta
            else:
                self._nominal_frame_step = min(self._nominal_frame_step, delta)
            self._last_frame_idx_seen = frame_idx
        elif delta == 0:
            # Same frame can happen in wrappers that delegate per-pair.
            return
        else:
            # New/rewound sequence.
            self._nominal_frame_step = None
            self._last_frame_idx_seen = frame_idx

    def _estimate_speed(self, trajectory: List[Tuple], window: int = 5) -> float:
        """
        Average speed in px/processed-frame over the last `window` points.

        Uses trajectory frame indices to avoid overestimating speed when the
        tracker misses intermittent frames. The smallest positive frame delta in
        the window is treated as the nominal processed-frame step.
        """
        pts = trajectory[-window:]
        if len(pts) < 2:
            return 0.0

        nominal_dt = self._nominal_frame_step if self._nominal_frame_step and self._nominal_frame_step > 0 else 1

        speeds = []
        for i in range(len(pts) - 1):
            try:
                dt_raw = int(pts[i + 1][0]) - int(pts[i][0])
            except (TypeError, ValueError, IndexError):
                dt_raw = nominal_dt
            if dt_raw <= 0:
                continue

            dx = pts[i + 1][1] - pts[i][1]
            dy = pts[i + 1][2] - pts[i][2]
            dist = math.hypot(dx, dy)
            dt_norm = dt_raw / nominal_dt
            speeds.append(dist / dt_norm)

        if not speeds:
            return 0.0
        return sum(speeds) / len(speeds)

    @staticmethod
    def _estimate_heading(trajectory: List[Tuple], window: int = 5) -> float:
        """Heading angle in degrees over the last `window` trajectory points."""
        pts = trajectory[-window:]
        if len(pts) < 2:
            return 0.0
        dx = pts[-1][1] - pts[0][1]
        dy = pts[-1][2] - pts[0][2]
        return math.degrees(math.atan2(dy, dx))

    def _compute_ttc(
        self,
        obj1: Dict,
        obj2: Dict,
        speed1: float,
        speed2: float,
        heading1: float,
        heading2: float,
    ) -> float:
        """
        Estimate Time-To-Collision (seconds).
        Projects relative velocity onto the line connecting the two centroids.
        Returns math.inf if objects are diverging.
        """
        cx1, cy1 = obj1["center"]
        cx2, cy2 = obj2["center"]

        dist = math.hypot(cx2 - cx1, cy2 - cy1)
        if dist < 1e-6:
            return 0.0

        # Unit vector from obj1 → obj2
        ux, uy = (cx2 - cx1) / dist, (cy2 - cy1) / dist

        # Velocity vectors (px/frame)
        vx1 = speed1 * math.cos(math.radians(heading1))
        vy1 = speed1 * math.sin(math.radians(heading1))
        vx2 = speed2 * math.cos(math.radians(heading2))
        vy2 = speed2 * math.sin(math.radians(heading2))

        # Relative velocity of obj2 w.r.t. obj1
        rvx, rvy = vx2 - vx1, vy2 - vy1

        # Closing speed = negative dot product of relative velocity with unit vector
        closing_speed = -(rvx * ux + rvy * uy)  # positive = approaching

        if closing_speed <= 0:
            return math.inf  # diverging → no collision risk

        ttc_frames = dist / closing_speed
        ttc_sec = ttc_frames / self.fps
        return ttc_sec

    # ------------------------------------------------------------------
    # Risk scoring
    # ------------------------------------------------------------------

    def _risk_score(
        self, distance: float, ttc: float, speed: float
    ) -> float:
        """
        Compute a risk score in [0, 1].
        distance: centroid distance in px
        ttc: time-to-collision in seconds (use self.ttc_threshold as norm)
        speed: max speed of the two objects in px/frame
        """
        # Normalize distance: 0 = very close, 1 = at threshold
        norm_dist = min(distance / self.proximity_px, 1.0)

        # Normalize TTC: 0 = imminent, 1 = at threshold or beyond
        if ttc == math.inf:
            norm_ttc = 1.0
        else:
            norm_ttc = min(ttc / self.ttc_threshold, 1.0)

        # Speed factor: cap at 30 px/frame as reference
        speed_factor = min(speed / 30.0, 1.0)

        score = (
            (1 - norm_dist) * 0.4
            + (1 - norm_ttc) * 0.4
            + speed_factor * 0.2
        )
        return round(min(max(score, 0.0), 1.0), 4)

    def _risk_level(self, score: float) -> str:
        if score >= self.risk_high_threshold:
            return "High"
        if score >= self.risk_medium_threshold:
            return "Medium"
        return "Low"

    # ------------------------------------------------------------------
    # False-positive filters (Task 1.6)
    # ------------------------------------------------------------------

    def _filter_stationary(self, speed1: float, speed2: float) -> bool:
        """Return True (→ discard) if both objects are essentially stationary."""
        return speed1 < self.stationary_speed_px and speed2 < self.stationary_speed_px

    def _filter_direction(
        self,
        heading1: float,
        heading2: float,
        obj1: Dict,
        obj2: Dict,
        speed1: float,
        speed2: float,
    ) -> bool:
        """
        Return True (→ discard) if both objects travel in roughly the same
        direction AND are not significantly closing in on each other.
        """
        angle_diff = abs(heading1 - heading2) % 360
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        same_direction = angle_diff < self.same_direction_deg

        # Closing speed
        cx1, cy1 = obj1["center"]
        cx2, cy2 = obj2["center"]
        dist = math.hypot(cx2 - cx1, cy2 - cy1)
        if dist < 1e-6:
            return False
        ux, uy = (cx2 - cx1) / dist, (cy2 - cy1) / dist
        vx1 = speed1 * math.cos(math.radians(heading1))
        vy1 = speed1 * math.sin(math.radians(heading1))
        vx2 = speed2 * math.cos(math.radians(heading2))
        vy2 = speed2 * math.sin(math.radians(heading2))
        rvx, rvy = vx2 - vx1, vy2 - vy1
        closing_speed = -(rvx * ux + rvy * uy)

        return same_direction and closing_speed < 2.0  # barely approaching

    def _filter_confidence(self, conf1: float, conf2: float) -> bool:
        """Return True (→ discard) if either detection confidence is too low."""
        return conf1 < self.min_confidence or conf2 < self.min_confidence

    # ------------------------------------------------------------------
    # Main processing method
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_idx: int,
        tracked_objects: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all pairs of tracked objects for near-miss conditions.

        Args:
            frame_idx: Current frame index.
            tracked_objects: OrderedDict from CentroidTracker.update().

        Returns:
            List of event dicts emitted this frame (may be empty).
        """
        self._update_frame_step(frame_idx)
        ids = list(tracked_objects.keys())
        new_events = []

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                pair_key = (min(id1, id2), max(id1, id2))

                obj1 = tracked_objects[id1]
                obj2 = tracked_objects[id2]

                # ---- Proximity check ----
                dist = self._centroid_distance(obj1, obj2)
                iou = self._compute_iou(obj1["bbox"], obj2["bbox"])
                proximate = dist < self.proximity_px or iou > 0.0

                if not proximate:
                    self._confirmation_buffer[pair_key] = 0
                    continue

                # ---- Trajectory-based analysis ----
                traj1 = obj1.get("trajectory", [])
                traj2 = obj2.get("trajectory", [])
                speed1 = self._estimate_speed(traj1)
                speed2 = self._estimate_speed(traj2)
                heading1 = self._estimate_heading(traj1)
                heading2 = self._estimate_heading(traj2)

                ttc = self._compute_ttc(obj1, obj2, speed1, speed2, heading1, heading2)
                risk_score = self._risk_score(dist, ttc, max(speed1, speed2))

                # At least 2 criteria must be met
                criteria_met = sum([
                    dist < self.proximity_px,
                    ttc < self.ttc_threshold,
                    max(speed1, speed2) > self.moving_speed_px,
                ])
                if criteria_met < 2:
                    self._confirmation_buffer[pair_key] = 0
                    continue

                # ---- False-positive filters (Task 1.6) ----
                if self.filters_enabled:
                    if self._filter_confidence(obj1["confidence"], obj2["confidence"]):
                        self._confirmation_buffer[pair_key] = 0
                        continue
                    if self._filter_stationary(speed1, speed2):
                        self._confirmation_buffer[pair_key] = 0
                        continue
                    if self._filter_direction(heading1, heading2, obj1, obj2, speed1, speed2):
                        self._confirmation_buffer[pair_key] = 0
                        continue

                # ---- Confirmation buffer (must persist ≥ confirm_frames) ----
                self._confirmation_buffer[pair_key] += 1
                if self.filters_enabled and self._confirmation_buffer[pair_key] < self.confirm_frames:
                    continue

                # ---- Debounce ----
                last_frame = self._last_event_frame.get(pair_key, -self.debounce_frames - 1)
                if frame_idx - last_frame < self.debounce_frames:
                    continue

                # ---- Emit event ----
                timestamp_sec = frame_idx / self.fps
                event = {
                    "frame_index": frame_idx,
                    "timestamp_sec": round(timestamp_sec, 2),
                    "object_id_1": id1,
                    "object_id_2": id2,
                    "class_1": obj1["class"],
                    "class_2": obj2["class"],
                    "label_1": obj1.get("label", ""),
                    "label_2": obj2.get("label", ""),
                    "distance_px": round(dist, 2),
                    "ttc_sec": round(ttc, 3) if ttc != math.inf else None,
                    "risk_score": risk_score,
                    "risk_level": self._risk_level(risk_score),
                    "conf_1": round(obj1["confidence"], 3),
                    "conf_2": round(obj2["confidence"], 3),
                }
                self._events.append(event)
                self._last_event_frame[pair_key] = frame_idx
                self._last_event_data[pair_key] = event
                new_events.append(event)

        return new_events

    def get_events_dataframe(self) -> pd.DataFrame:
        """Return all logged events as a sorted pandas DataFrame."""
        if not self._events:
            return pd.DataFrame(columns=[
                "frame_index", "timestamp_sec",
                "object_id_1", "object_id_2",
                "class_1", "class_2", "label_1", "label_2",
                "distance_px", "ttc_sec",
                "risk_score", "risk_level", "conf_1", "conf_2",
            ])
        df = pd.DataFrame(self._events)
        return df.sort_values("timestamp_sec").reset_index(drop=True)

    def summary(self) -> Dict[str, Any]:
        df = self.get_events_dataframe()
        if df.empty:
            return {"total": 0, "high": 0, "medium": 0, "low": 0}
        counts = df["risk_level"].value_counts().to_dict()
        return {
            "total": len(df),
            "high": counts.get("High", 0),
            "medium": counts.get("Medium", 0),
            "low": counts.get("Low", 0),
        }

    def active_pairs(self, frame_idx: int) -> List[Tuple[int, int]]:
        """Return pairs that had an event within the last debounce_frames frames."""
        return [
            pair for pair, last in self._last_event_frame.items()
            if frame_idx - last < self.debounce_frames
        ]

    def active_pair_data(self, frame_idx: int) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """
        Return the most recent event dict for every pair active within
        the last debounce_frames frames.

        Keys are canonical (min_id, max_id) pairs; values are the event dicts
        produced by process_frame() and stored in get_events_dataframe().
        Pass the result as pair_data= to visualizer.annotate_frame() to render
        live TTC, target-ID, and risk information on each near-miss object.
        """
        return {
            pair: self._last_event_data[pair]
            for pair, last in self._last_event_frame.items()
            if frame_idx - last < self.debounce_frames
            and pair in self._last_event_data
        }


# ---------------------------------------------------------------------------
# NearMissDetectorV11 — six targeted improvements over v1.0
# ---------------------------------------------------------------------------

class NearMissDetectorV11(NearMissDetector):
    """
    Near-Miss Detector v1.1 — six algorithmic improvements over v1.0.

    Applied (✓) / skipped (✗) with rationale:

    ✓ #2  2D closest-approach replaces 1-D projected TTC.
          t* = clamp(-(p·v)/||v||², 0, T_horizon);  d_min = ||p + v·t*||
          Catches crossing conflicts that v1.0's projection axis misses.

    ✓ #3  Footpoint distance (bbox bottom-center) instead of centroid.
          Reduces systematic geometric error for tall objects (buses, trucks)
          and mixed vehicle/pedestrian pairs.

    ✓ #4  Scale-aware proximity: eff_prox = max(proximity_px, scale·mean_diag).
          proximity_px is a hard floor.  Reduces perspective-induced bias
          (over-detection near camera, under-detection far from camera).

    ✓ #5  Tightened IoU gate: iou > min_iou (default 0.05) instead of iou > 0.
          Eliminates single-pixel bbox-jitter false triggers.

    ✓ #6  Leaky confirmation buffer: buffer -= decay on miss instead of reset=0.
          Preserves evidence across brief detection jitter (↑ recall, ≈ FPR).

    ✓ #7  Convergence guard: only fires when t_raw > 0 (objects still approaching).
          Eliminates "already-passed" alerts emitted after closest approach.

    ✗ #1  Heading units: already correct in v1.0.  _estimate_heading stores
          degrees; every caller wraps with math.radians() before sin/cos.

    ✗ #8  Homography / BEV ground-plane: requires one-time camera calibration.
          Deferred to v2.0.

    Drop-in replacement for NearMissDetector — same call signature, same output
    dict schema plus an extra "d_min_px" field (predicted minimum footpoint
    distance at closest approach).
    """

    def __init__(
        self,
        # ── inherited v1.0 parameters ────────────────────────────────────────
        proximity_px:        float = 100.0,
        ttc_threshold:       float = 2.0,
        fps:                 float = 15.0,
        debounce_frames:     int   = 30,
        filters_enabled:     bool  = True,
        stationary_speed_px: float = 5.0,
        confirm_frames:      int   = 5,
        same_direction_deg:  float = 30.0,
        min_confidence:      float = 0.5,
        moving_speed_px:     float = 5.0,
        risk_high_threshold: float = _RISK_HIGH,
        risk_medium_threshold: float = _RISK_MEDIUM,
        # ── v1.1 additions ────────────────────────────────────────────────────
        proximity_scale: float = 0.5,
        min_iou:         float = 0.05,
        buffer_decay:    float = 0.5,
        t_horizon_sec:   float = 5.0,
        clearance_scale: float = 1.1,
    ):
        """
        v1.1-specific parameters (all v1.0 params are unchanged):

        proximity_scale:
            Adaptive proximity multiplier.
            eff_prox = max(proximity_px, proximity_scale * mean_bbox_diagonal).
            proximity_px acts as a hard floor.
            0.0 → disabled, falls back to fixed proximity_px.
            Default 0.5.

        min_iou:
            Minimum IoU overlap required to trigger the bbox-overlap gate.
            Replaces the v1.0 ``iou > 0`` condition.
            Default 0.05.

        buffer_decay:
            Amount subtracted from the confirmation buffer on each failed frame
            (leaky integrator). Hard reset in v1.0 is equivalent to decay = ∞.
            Default 0.5 (lose half a hit's evidence per missed frame).

        t_horizon_sec:
            Look-ahead horizon for the 2D closest-approach calculation.
            t* is clamped to [0, t_horizon_sec].
            Default 5.0 s.

        clearance_scale:
            Size-aware closest-approach guard.
            A pair is discarded when ``d_min_px`` is larger than:
            ``clearance_scale * (bbox_width_1 + bbox_width_2) / 2`` (capped by
            ``eff_prox``), unless current IoU already exceeds ``min_iou``.
            This suppresses adjacent-lane side-by-side false positives.
            Set <= 0.0 to disable.
        """
        super().__init__(
            proximity_px=proximity_px,
            ttc_threshold=ttc_threshold,
            fps=fps,
            debounce_frames=debounce_frames,
            filters_enabled=filters_enabled,
            stationary_speed_px=stationary_speed_px,
            confirm_frames=confirm_frames,
            same_direction_deg=same_direction_deg,
            min_confidence=min_confidence,
            moving_speed_px=moving_speed_px,
            risk_high_threshold=risk_high_threshold,
            risk_medium_threshold=risk_medium_threshold,
        )
        self.proximity_scale = proximity_scale
        self.min_iou         = min_iou
        self.buffer_decay    = buffer_decay
        self.t_horizon_sec   = t_horizon_sec
        self.clearance_scale = max(float(clearance_scale), 0.0)
        # Override parent's defaultdict(int) with float for leaky arithmetic
        self._confirmation_buffer = defaultdict(float)

    # ------------------------------------------------------------------
    # v1.1 geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _footpoint(obj: Dict[str, Any]) -> Tuple[float, float]:
        """
        Ground-contact proxy: bottom-center of the bounding box.

        More physically meaningful than the bbox centroid for measuring
        how close two objects are on the ground plane, especially for
        tall/large objects (buses, trucks) viewed from a slightly elevated camera.
        """
        x1, _y1, x2, y2 = obj["bbox"]
        return ((x1 + x2) / 2.0, float(y2))

    def _effective_proximity(
        self, obj1: Dict[str, Any], obj2: Dict[str, Any]
    ) -> float:
        """
        Scale-aware proximity threshold.

        eff_prox = max(proximity_px, proximity_scale × mean(diag₁, diag₂))

        proximity_px acts as a hard floor so tiny detections (small diagonal)
        still get at least the configured baseline threshold.
        """
        if self.proximity_scale <= 0.0:
            return self.proximity_px
        x1a, y1a, x2a, y2a = obj1["bbox"]
        x1b, y1b, x2b, y2b = obj2["bbox"]
        diag1 = math.hypot(x2a - x1a, y2a - y1a)
        diag2 = math.hypot(x2b - x1b, y2b - y1b)
        adaptive = self.proximity_scale * (diag1 + diag2) / 2.0
        return max(self.proximity_px, adaptive)

    def _clearance_threshold_px(
        self,
        obj1: Dict[str, Any],
        obj2: Dict[str, Any],
        eff_prox: float,
    ) -> float:
        """
        Size-aware minimum-clearance threshold in pixels.

        Uses bbox widths as a proxy for occupied lane width in image space.
        The returned threshold is capped by ``eff_prox`` so the physical gate
        cannot be looser than the configured proximity gate.
        """
        if self.clearance_scale <= 0.0:
            return eff_prox

        w1 = max(float(obj1["bbox"][2] - obj1["bbox"][0]), 1.0)
        w2 = max(float(obj2["bbox"][2] - obj2["bbox"][0]), 1.0)
        clearance = self.clearance_scale * 0.5 * (w1 + w2)
        return min(clearance, eff_prox)

    def _closest_approach(
        self,
        fp1:     Tuple[float, float],
        fp2:     Tuple[float, float],
        speed1:  float,
        speed2:  float,
        heading1: float,
        heading2: float,
    ) -> Tuple[float, float, bool]:
        """
        2D constant-velocity closest-approach calculation.

        Position: footpoints (px).
        Velocity: derived from centroid trajectory (heading °, speed px/frame).

        Returns
        -------
        t_star_sec : float
            Time to closest approach in seconds, clamped to [0, t_horizon_sec].
        d_min_px : float
            Predicted footpoint distance (px) at t_star.
        converging : bool
            True if the unclamped closest-approach time is in the future
            (t_raw > 0), meaning objects are still approaching each other.
            False when already past closest approach or no relative motion.
        """
        px, py = fp2[0] - fp1[0], fp2[1] - fp1[1]          # relative position (px)

        # Velocity vectors in px/frame (heading is in degrees — converted here)
        vx1 = speed1 * math.cos(math.radians(heading1))
        vy1 = speed1 * math.sin(math.radians(heading1))
        vx2 = speed2 * math.cos(math.radians(heading2))
        vy2 = speed2 * math.sin(math.radians(heading2))

        vx = vx2 - vx1                                        # relative velocity (px/frame)
        vy = vy2 - vy1
        v_sq = vx * vx + vy * vy

        if v_sq < 0.01:                                       # essentially no relative motion
            return (0.0, math.hypot(px, py), False)

        # Unclamped closest-approach time (frames)
        t_raw = -(px * vx + py * vy) / v_sq
        converging = t_raw > 0.0

        # Clamp to horizon
        t_horizon_frames = self.t_horizon_sec * self.fps
        t_clamped = max(0.0, min(t_raw, t_horizon_frames))

        # Predicted footpoint distance at t_clamped
        d_min = math.hypot(px + vx * t_clamped, py + vy * t_clamped)

        return (t_clamped / self.fps, d_min, converging)

    def _leak_buffer(self, pair_key: Tuple[int, int]) -> None:
        """
        Decay the confirmation buffer by buffer_decay (leaky integrator, floor 0).
        Called wherever v1.0 would do a hard reset to 0.
        """
        old = self._confirmation_buffer.get(pair_key, 0.0)
        if old <= 0.0:
            return
        new = max(old - self.buffer_decay, 0.0)
        if new <= 0.0:
            self._confirmation_buffer.pop(pair_key, None)
        else:
            self._confirmation_buffer[pair_key] = new

    def _risk_score_v11(
        self,
        dist:      float,
        t_star:    float,
        d_min:     float,
        speed:     float,
        eff_prox:  float,
    ) -> float:
        """
        v1.1 risk score.

        Primary signal is d_min (predicted closest approach) rather than the
        current distance, giving earlier warning for crossing conflicts.

        Weights
        -------
        d_min proximity  45 % — how close will they actually get?
        time urgency     30 % — how soon does that happen?
        current dist     15 % — how close are they right now?
        speed factor     10 % — kinetic severity proxy
        """
        norm_dmin = min(d_min  / eff_prox,         1.0)
        norm_dist = min(dist   / eff_prox,          1.0)
        norm_t    = min(t_star / self.ttc_threshold, 1.0) if t_star < math.inf else 1.0
        speed_fac = min(speed  / 30.0,              1.0)

        score = (
              (1.0 - norm_dmin) * 0.45
            + (1.0 - norm_dist) * 0.15
            + (1.0 - norm_t)    * 0.30
            + speed_fac         * 0.10
        )
        return round(min(max(score, 0.0), 1.0), 4)

    def _filter_direction_v11(
        self,
        heading1: float,
        heading2: float,
        fp1:      Tuple[float, float],
        fp2:      Tuple[float, float],
        speed1:   float,
        speed2:   float,
    ) -> bool:
        """
        Direction filter using footpoints instead of centroids (v1.1).

        Identical logic to v1.0's _filter_direction but uses footpoint positions
        to compute the inter-object axis, consistent with the rest of v1.1.
        """
        angle_diff = abs(heading1 - heading2) % 360
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        same_direction = angle_diff < self.same_direction_deg

        dist = math.hypot(fp2[0] - fp1[0], fp2[1] - fp1[1])
        if dist < 1e-6:
            return False
        ux = (fp2[0] - fp1[0]) / dist
        uy = (fp2[1] - fp1[1]) / dist

        vx1 = speed1 * math.cos(math.radians(heading1))
        vy1 = speed1 * math.sin(math.radians(heading1))
        vx2 = speed2 * math.cos(math.radians(heading2))
        vy2 = speed2 * math.sin(math.radians(heading2))

        closing_speed = -((vx2 - vx1) * ux + (vy2 - vy1) * uy)
        return same_direction and closing_speed < 2.0

    # ------------------------------------------------------------------
    # Main processing method (v1.1 override)
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_idx: int,
        tracked_objects: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        v1.1 process_frame — same interface as v1.0, six internal improvements.
        """
        self._update_frame_step(frame_idx)
        ids        = list(tracked_objects.keys())
        new_events = []

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                pair_key = (min(id1, id2), max(id1, id2))

                obj1 = tracked_objects[id1]
                obj2 = tracked_objects[id2]

                # ── Footpoints & scale-aware threshold (#3, #4) ───────────────
                fp1      = self._footpoint(obj1)
                fp2      = self._footpoint(obj2)
                dist     = math.hypot(fp2[0] - fp1[0], fp2[1] - fp1[1])
                eff_prox = self._effective_proximity(obj1, obj2)

                # ── Proximity gate (#3 footpoints, #4 scale, #5 min_iou) ──────
                iou       = self._compute_iou(obj1["bbox"], obj2["bbox"])
                proximate = (dist < eff_prox) or (iou > self.min_iou)

                if not proximate:
                    self._leak_buffer(pair_key)                          # #6
                    continue

                # ── Trajectory analysis ───────────────────────────────────────
                traj1  = obj1.get("trajectory", [])
                traj2  = obj2.get("trajectory", [])
                speed1 = self._estimate_speed(traj1)
                speed2 = self._estimate_speed(traj2)
                h1     = self._estimate_heading(traj1)
                h2     = self._estimate_heading(traj2)

                # ── 2D closest approach (#2) ──────────────────────────────────
                t_star, d_min, converging = self._closest_approach(
                    fp1, fp2, speed1, speed2, h1, h2
                )

                risk_score = self._risk_score_v11(
                    dist, t_star, d_min, max(speed1, speed2), eff_prox
                )

                # ── Multi-criteria gate: 2 of 3 required ─────────────────────
                criteria_met = sum([
                    dist  < eff_prox,
                    d_min < eff_prox,
                    max(speed1, speed2) > self.moving_speed_px,
                ])
                if criteria_met < 2:
                    self._leak_buffer(pair_key)                          # #6
                    continue

                # ── Convergence guard (#7) — hard physical constraint ─────────
                # Objects already past their closest approach cannot collide;
                # this is physics, not a heuristic — always active.
                if not converging:
                    self._leak_buffer(pair_key)
                    continue

                # ── Size-aware closest-approach guard ────────────────────────
                # Prevents false positives for adjacent-lane trajectories that
                # remain well-separated even though pixel distance < eff_prox.
                if iou <= self.min_iou:
                    clearance_px = self._clearance_threshold_px(obj1, obj2, eff_prox)
                    if d_min > clearance_px:
                        self._leak_buffer(pair_key)
                        continue

                # ── False-positive heuristic filters ──────────────────────────
                if self.filters_enabled:
                    if self._filter_confidence(obj1["confidence"], obj2["confidence"]):
                        self._leak_buffer(pair_key)
                        continue
                    if self._filter_stationary(speed1, speed2):
                        self._leak_buffer(pair_key)
                        continue
                    if self._filter_direction_v11(h1, h2, fp1, fp2, speed1, speed2):
                        self._leak_buffer(pair_key)
                        continue

                # ── Leaky confirmation buffer (#6) ────────────────────────────
                self._confirmation_buffer[pair_key] += 1.0
                if (self.filters_enabled
                        and self._confirmation_buffer[pair_key] < self.confirm_frames):
                    continue

                # ── Debounce ──────────────────────────────────────────────────
                last_frame = self._last_event_frame.get(
                    pair_key, -self.debounce_frames - 1
                )
                if frame_idx - last_frame < self.debounce_frames:
                    continue

                # ── Emit event ────────────────────────────────────────────────
                event = {
                    "frame_index":   frame_idx,
                    "timestamp_sec": round(frame_idx / self.fps, 2),
                    "object_id_1":   id1,
                    "object_id_2":   id2,
                    "class_1":       obj1["class"],
                    "class_2":       obj2["class"],
                    "label_1":       obj1.get("label", ""),
                    "label_2":       obj2.get("label", ""),
                    "distance_px":   round(dist, 2),
                    "ttc_sec":       round(t_star, 3) if t_star < math.inf else None,
                    "d_min_px":      round(d_min, 2),    # v1.1: predicted min footpoint dist
                    "risk_score":    risk_score,
                    "risk_level":    self._risk_level(risk_score),
                    "conf_1":        round(obj1["confidence"], 3),
                    "conf_2":        round(obj2["confidence"], 3),
                }
                self._events.append(event)
                self._last_event_frame[pair_key] = frame_idx
                self._last_event_data[pair_key]  = event
                new_events.append(event)

        return new_events

    def get_events_dataframe(self) -> pd.DataFrame:
        """v1.1 override: empty-DataFrame fallback includes d_min_px column."""
        if not self._events:
            return pd.DataFrame(columns=[
                "frame_index", "timestamp_sec",
                "object_id_1", "object_id_2",
                "class_1", "class_2", "label_1", "label_2",
                "distance_px", "ttc_sec", "d_min_px",
                "risk_score", "risk_level", "conf_1", "conf_2",
            ])
        df = pd.DataFrame(self._events)
        return df.sort_values("timestamp_sec").reset_index(drop=True)


# ---------------------------------------------------------------------------
# NearMissDetectorV20 — 3-D ground-plane aware near-miss detection
# ---------------------------------------------------------------------------

class NearMissDetectorV20(NearMissDetectorV11):
    """
    Near-Miss Detector v2.0 — operates in 3-D ground-plane space.

    When tracked objects carry a ``ground_pt`` field (populated by
    :class:`src.ground_projection.GroundProjector`), all proximity and TTC
    calculations are performed in 3-D camera-space units rather than pixels.
    This removes perspective distortion: a car 20 m away is compared at the
    same physical scale as one 5 m away.

    Falls back transparently to v1.1 pixel logic for any pair where either
    object is missing ``ground_pt`` (e.g. during the warm-up frames before the
    ground plane is first fitted).

    New parameters
    --------------
    proximity_3d : float (default 2.0)
        Ground-plane proximity threshold in depth units.  Pairs closer than
        this on the ground plane trigger proximity gating.

        **Depth-scale calibration** — DepthAnythingV2 with p95=10 normalization
        produces RELATIVE (non-metric) depth values.  Ground-plane ray–plane
        intersections typically yield t ≈ 0.2–0.35 depth units for road pixels,
        meaning the default of 2.0 is ~50–100× too large and will mark all
        object pairs as proximate permanently.  For this depth model, use
        ``proximity_3d ≈ proximity_px × t_avg / fx``, e.g. ``0.05`` for a
        120-px threshold with fx≈1088 and t_avg≈0.25.

    speed_3d_min : float (default 0.05)
        Minimum 3-D speed (units/frame) for an object to be considered moving.

        **Depth-scale calibration** — with the relative depth scale above,
        a car moving 15 px/frame has speed_3d ≈ 15 × 0.25 / 1088 ≈ 0.003
        depth-units/frame.  The default of 0.05 is ~50× too large and causes
        the stationary filter to silently discard every 3-D pair, making all
        events fall back to pixel-mode.  Use ``speed_3d_min ≈ 0.0005`` for
        DepthAnythingV2 + p95=10.

    Extra event fields
    ------------------
    distance_3d  : float | None   current ground-plane distance (depth units)
    d_min_3d     : float | None   predicted closest ground-plane approach
    speed_3d_1   : float | None   3-D speed of object 1 (units/frame)
    speed_3d_2   : float | None   3-D speed of object 2 (units/frame)
    mode         : "3d" | "2d"    which code path produced this event
    """

    def __init__(
        self,
        proximity_px:        float = 100.0,
        ttc_threshold:       float = 2.0,
        fps:                 float = 15.0,
        debounce_frames:     int   = 30,
        filters_enabled:     bool  = True,
        stationary_speed_px: float = 5.0,
        confirm_frames:      int   = 5,
        same_direction_deg:  float = 30.0,
        min_confidence:      float = 0.5,
        moving_speed_px:     float = 5.0,
        risk_high_threshold: float = _RISK_HIGH,
        risk_medium_threshold: float = _RISK_MEDIUM,
        proximity_scale:     float = 0.5,
        min_iou:             float = 0.05,
        buffer_decay:        float = 0.5,
        t_horizon_sec:       float = 5.0,
        # v2.0
        proximity_3d:        float = 2.0,
        speed_3d_min:        float = 0.05,
    ):
        super().__init__(
            proximity_px=proximity_px,
            ttc_threshold=ttc_threshold,
            fps=fps,
            debounce_frames=debounce_frames,
            filters_enabled=filters_enabled,
            stationary_speed_px=stationary_speed_px,
            confirm_frames=confirm_frames,
            same_direction_deg=same_direction_deg,
            min_confidence=min_confidence,
            moving_speed_px=moving_speed_px,
            risk_high_threshold=risk_high_threshold,
            risk_medium_threshold=risk_medium_threshold,
            proximity_scale=proximity_scale,
            min_iou=min_iou,
            buffer_decay=buffer_decay,
            t_horizon_sec=t_horizon_sec,
        )
        self.proximity_3d = proximity_3d
        self.speed_3d_min = speed_3d_min

    # ── 3-D helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _gp(obj: Dict[str, Any]) -> Optional[np.ndarray]:
        """Return ground_pt as float64 ndarray, or None."""
        gp = obj.get("ground_pt")
        return np.array(gp, dtype=np.float64) if gp is not None else None

    @staticmethod
    def _traj3d_speed(traj_3d: List[Tuple], window: int = 5) -> float:
        """Average 3-D speed (units/frame) from trajectory_3d = [(fi,X,Y,Z), ...]."""
        pts = traj_3d[-window:]
        if len(pts) < 2:
            return 0.0
        dists = [
            float(np.linalg.norm(
                np.array(pts[i + 1][1:], dtype=np.float64)
                - np.array(pts[i][1:],   dtype=np.float64)
            ))
            for i in range(len(pts) - 1)
        ]
        return float(np.mean(dists))

    @staticmethod
    def _traj3d_heading_vec(traj_3d: List[Tuple], window: int = 5) -> Optional[np.ndarray]:
        """Unit 3-D direction vector from trajectory_3d.  None when indeterminate."""
        pts = traj_3d[-window:]
        if len(pts) < 2:
            return None
        delta = (np.array(pts[-1][1:], dtype=np.float64)
                 - np.array(pts[0][1:],  dtype=np.float64))
        norm = float(np.linalg.norm(delta))
        return delta / norm if norm > 1e-6 else None

    def _closest_approach_3d(
        self,
        gp1: np.ndarray,
        gp2: np.ndarray,
        v1:  np.ndarray,
        v2:  np.ndarray,
    ) -> Tuple[float, float, bool]:
        """3-D constant-velocity closest-approach.

        Returns (t_star_sec, d_min_3d, converging).
        """
        p    = gp2 - gp1
        v    = v2  - v1
        v_sq = float(np.dot(v, v))

        if v_sq < 1e-6:
            return 0.0, float(np.linalg.norm(p)), False

        t_raw      = -float(np.dot(p, v)) / v_sq
        converging = t_raw > 0.0
        t_clamped  = max(0.0, min(t_raw, self.t_horizon_sec * self.fps))
        d_min      = float(np.linalg.norm(p + v * t_clamped))
        return t_clamped / self.fps, d_min, converging

    # ── Main processing override ──────────────────────────────────────────────

    def process_frame(
        self,
        frame_idx: int,
        tracked_objects: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """v2.0 process_frame: 3-D for ground-pt pairs, 2-D fallback otherwise."""
        self._update_frame_step(frame_idx)
        ids        = list(tracked_objects.keys())
        new_events: List[Dict[str, Any]] = []

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2   = ids[i], ids[j]
                pair_key   = (min(id1, id2), max(id1, id2))
                obj1, obj2 = tracked_objects[id1], tracked_objects[id2]
                gp1, gp2   = self._gp(obj1), self._gp(obj2)

                if gp1 is not None and gp2 is not None:
                    ev = self._process_pair_3d(
                        frame_idx, pair_key, id1, id2, obj1, obj2, gp1, gp2
                    )
                    if ev is not None:
                        new_events.append(ev)
                else:
                    _log_v20.warning(
                        "[fr=%d] pair=(%d,%d) fallback=2d "
                        "(id%d ground_pt=%s  id%d ground_pt=%s)",
                        frame_idx, id1, id2,
                        id1, "ok" if gp1 is not None else "MISSING",
                        id2, "ok" if gp2 is not None else "MISSING",
                    )
                    # Parent processes just this pair via a temporary single-pair dict
                    sub_evs = super().process_frame(frame_idx, {id1: obj1, id2: obj2})
                    for ev in sub_evs:
                        ev["mode"] = "2d"
                        # Parent already appended to self._events; avoid duplicate
                    new_events.extend(sub_evs)

        return new_events

    def _process_pair_3d(
        self,
        frame_idx: int,
        pair_key:  Tuple[int, int],
        id1: int,
        id2: int,
        obj1: Dict[str, Any],
        obj2: Dict[str, Any],
        gp1:  np.ndarray,
        gp2:  np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """Full 3-D near-miss logic for one pair.  Returns event dict or None."""

        dist_3d   = float(np.linalg.norm(gp2 - gp1))
        iou       = self._compute_iou(obj1["bbox"], obj2["bbox"])
        proximate = dist_3d < self.proximity_3d or iou > self.min_iou

        if not proximate:
            self._leak_buffer(pair_key)
            _log_v20.debug(
                "[fr=%d] pair=(%d,%d) mode=3d skip=proximity "
                "dist_3d=%.3f threshold=%.3f",
                frame_idx, id1, id2, dist_3d, self.proximity_3d,
            )
            return None

        # ── 3-D kinematics ────────────────────────────────────────────────────
        traj3d_1  = obj1.get("trajectory_3d", [])
        traj3d_2  = obj2.get("trajectory_3d", [])
        speed3d_1 = self._traj3d_speed(traj3d_1)
        speed3d_2 = self._traj3d_speed(traj3d_2)
        h1 = self._traj3d_heading_vec(traj3d_1)
        h2 = self._traj3d_heading_vec(traj3d_2)
        v1 = h1 * speed3d_1 if h1 is not None else np.zeros(3)
        v2 = h2 * speed3d_2 if h2 is not None else np.zeros(3)

        t_star, d_min_3d, converging = self._closest_approach_3d(gp1, gp2, v1, v2)

        # ── Risk (v1.1 formula scaled to 3-D units) ───────────────────────────
        max_speed = max(speed3d_1, speed3d_2)
        # Scale 3-D speed to px/frame range: multiply by 60 (empirical; tune per scene)
        risk = self._risk_score_v11(
            dist_3d, t_star, d_min_3d,
            speed=max_speed * 60.0,
            eff_prox=self.proximity_3d,
        )

        # ── Multi-criteria gate ────────────────────────────────────────────────
        criteria = sum([
            dist_3d  < self.proximity_3d,
            d_min_3d < self.proximity_3d,
            max(speed3d_1, speed3d_2) > self.speed_3d_min,
        ])
        if criteria < 2:
            self._leak_buffer(pair_key)
            _log_v20.debug(
                "[fr=%d] pair=(%d,%d) mode=3d skip=criteria(%d/3) "
                "dist=%.3f d_min=%.3f speed=%.5f",
                frame_idx, id1, id2, criteria, dist_3d, d_min_3d, max_speed,
            )
            return None

        _log_v20.debug(
            "[fr=%d] pair=(%d,%d) mode=3d dist_3d=%.3f d_min_3d=%.3f "
            "t*=%.2fs sp1=%.4f sp2=%.4f risk=%.3f",
            frame_idx, id1, id2, dist_3d, d_min_3d, t_star,
            speed3d_1, speed3d_2, risk,
        )

        # ── Convergence guard — hard physical constraint ───────────────────────
        if not converging:
            self._leak_buffer(pair_key)
            _log_v20.debug(
                "[fr=%d] pair=(%d,%d) skip=not_converging",
                frame_idx, id1, id2,
            )
            return None

        # ── False-positive heuristic filters ──────────────────────────────────
        if self.filters_enabled:
            c1, c2 = obj1["confidence"], obj2["confidence"]
            if self._filter_confidence(c1, c2):
                self._leak_buffer(pair_key)
                _log_v20.debug(
                    "[fr=%d] pair=(%d,%d) skip=confidence c1=%.2f c2=%.2f",
                    frame_idx, id1, id2, c1, c2,
                )
                return None

            if speed3d_1 < self.speed_3d_min and speed3d_2 < self.speed_3d_min:
                self._leak_buffer(pair_key)
                _log_v20.debug(
                    "[fr=%d] pair=(%d,%d) skip=stationary "
                    "sp1=%.5f sp2=%.5f < %.5f",
                    frame_idx, id1, id2, speed3d_1, speed3d_2, self.speed_3d_min,
                )
                return None

            # 3-D direction filter
            v_rel = v2 - v1
            if float(np.dot(v_rel, v_rel)) > 1e-6:
                axis    = (gp2 - gp1) / (dist_3d + 1e-12)
                closing = -float(np.dot(v_rel, axis))
                nv1     = float(np.linalg.norm(v1))
                nv2     = float(np.linalg.norm(v2))
                if nv1 > 1e-6 and nv2 > 1e-6:
                    cos_a = float(np.dot(v1 / nv1, v2 / nv2))
                    ang   = math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))
                    if ang < self.same_direction_deg and closing < 0.01:
                        self._leak_buffer(pair_key)
                        _log_v20.debug(
                            "[fr=%d] pair=(%d,%d) skip=direction "
                            "ang=%.1f° closing=%.5f",
                            frame_idx, id1, id2, ang, closing,
                        )
                        return None

        # ── Leaky confirmation buffer ──────────────────────────────────────────
        self._confirmation_buffer[pair_key] += 1.0
        if (self.filters_enabled
                and self._confirmation_buffer[pair_key] < self.confirm_frames):
            _log_v20.debug(
                "[fr=%d] pair=(%d,%d) skip=buffer(%.1f/%.0f)",
                frame_idx, id1, id2,
                self._confirmation_buffer[pair_key], self.confirm_frames,
            )
            return None

        # ── Debounce ──────────────────────────────────────────────────────────
        last = self._last_event_frame.get(pair_key, -self.debounce_frames - 1)
        if frame_idx - last < self.debounce_frames:
            _log_v20.debug(
                "[fr=%d] pair=(%d,%d) skip=debounce gap=%d < %d",
                frame_idx, id1, id2, frame_idx - last, self.debounce_frames,
            )
            return None

        # ── Footpoint pixel distance (kept for visualizer backward compat) ────
        fp1   = self._footpoint(obj1)
        fp2   = self._footpoint(obj2)
        fp_dist = math.hypot(fp2[0] - fp1[0], fp2[1] - fp1[1])

        # ── Emit event ────────────────────────────────────────────────────────
        event: Dict[str, Any] = {
            "frame_index":   frame_idx,
            "timestamp_sec": round(frame_idx / self.fps, 2),
            "object_id_1":   id1,
            "object_id_2":   id2,
            "class_1":       obj1["class"],
            "class_2":       obj2["class"],
            "label_1":       obj1.get("label", ""),
            "label_2":       obj2.get("label", ""),
            "distance_px":   round(fp_dist, 2),
            "ttc_sec":       round(t_star, 3) if t_star < math.inf else None,
            "d_min_px":      round(fp_dist, 2),   # approx; 3-D is more meaningful
            "distance_3d":   round(dist_3d,  4),
            "d_min_3d":      round(d_min_3d, 4),
            "speed_3d_1":    round(speed3d_1, 5),
            "speed_3d_2":    round(speed3d_2, 5),
            "mode":          "3d",
            "risk_score":    risk,
            "risk_level":    self._risk_level(risk),
            "conf_1":        round(obj1["confidence"], 3),
            "conf_2":        round(obj2["confidence"], 3),
        }
        self._events.append(event)
        self._last_event_frame[pair_key] = frame_idx
        self._last_event_data[pair_key]  = event

        _log_v20.info(
            "[fr=%d] EVENT pair=(%d,%d) dist_3d=%.3f d_min_3d=%.3f "
            "risk=%.3f level=%s",
            frame_idx, id1, id2, dist_3d, d_min_3d, risk, event["risk_level"],
        )
        return event

    def get_events_dataframe(self) -> pd.DataFrame:
        """v2.0: includes 3-D columns in the empty-DataFrame schema."""
        if not self._events:
            return pd.DataFrame(columns=[
                "frame_index", "timestamp_sec",
                "object_id_1", "object_id_2",
                "class_1", "class_2", "label_1", "label_2",
                "distance_px", "ttc_sec", "d_min_px",
                "distance_3d", "d_min_3d", "speed_3d_1", "speed_3d_2",
                "mode", "risk_score", "risk_level", "conf_1", "conf_2",
            ])
        df = pd.DataFrame(self._events)
        return df.sort_values("timestamp_sec").reset_index(drop=True)


# ---------------------------------------------------------------------------
# NearMissDetectorV30  — source-aware wrapper for CentroidTrackerV3 output
# ---------------------------------------------------------------------------

_log_v30 = logging.getLogger("near_miss.v30")


class NearMissDetectorV30(NearMissDetectorV20):
    """v3.0 near-miss detector — consumes CentroidTrackerV3 output.

    V3.0 is identical to V2.0 except it is **source-aware**: the
    ``ground_pt_source`` field added by ``CentroidTrackerV3`` is propagated
    into events and optionally used to suppress alerts when *both* objects'
    ground points are Kalman-predictions (both coasting).

    Dual-predicted pairs are suppressed by default because two coasting
    tracks have accumulated position error from consecutive Kalman steps and
    are therefore unreliable enough to generate false positives.

    New constructor parameter
    -------------------------
    skip_dual_predicted : bool (default True)
        When True, any pair where both ``ground_pt_source`` values are
        ``"predicted"`` is skipped silently (DEBUG log emitted).
        Set False to allow Kalman-predicted positions to trigger events.

    Extra event fields
    ------------------
    ground_pt_source_1 : "measured" | "predicted" | None
        Source of object 1's ground point.
    ground_pt_source_2 : "measured" | "predicted" | None
        Source of object 2's ground point.
    """

    def __init__(
        self,
        *args,
        skip_dual_predicted: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.skip_dual_predicted = skip_dual_predicted

    # ── Override: inject source-awareness before passing to V20 logic ─────────

    def _process_pair_3d(
        self,
        frame_idx: int,
        pair_key:  Tuple[int, int],
        id1: int,
        id2: int,
        obj1: Dict[str, Any],
        obj2: Dict[str, Any],
        gp1:  np.ndarray,
        gp2:  np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        src1 = obj1.get("ground_pt_source")
        src2 = obj2.get("ground_pt_source")

        if self.skip_dual_predicted and src1 == "predicted" and src2 == "predicted":
            self._leak_buffer(pair_key)
            _log_v30.debug(
                "[fr=%d] pair=(%d,%d) skip=dual_predicted (both coasting)",
                frame_idx, id1, id2,
            )
            return None

        ev = super()._process_pair_3d(frame_idx, pair_key, id1, id2, obj1, obj2, gp1, gp2)
        if ev is not None:
            ev["ground_pt_source_1"] = src1
            ev["ground_pt_source_2"] = src2
            _log_v30.debug(
                "[fr=%d] pair=(%d,%d) src1=%s src2=%s",
                frame_idx, id1, id2, src1, src2,
            )
        return ev

    def get_events_dataframe(self) -> pd.DataFrame:
        """v3.0: includes ground_pt_source columns."""
        if not self._events:
            return pd.DataFrame(columns=[
                "frame_index", "timestamp_sec",
                "object_id_1", "object_id_2",
                "class_1", "class_2", "label_1", "label_2",
                "distance_px", "ttc_sec", "d_min_px",
                "distance_3d", "d_min_3d", "speed_3d_1", "speed_3d_2",
                "mode", "risk_score", "risk_level", "conf_1", "conf_2",
                "ground_pt_source_1", "ground_pt_source_2",
            ])
        df = pd.DataFrame(self._events)
        return df.sort_values("timestamp_sec").reset_index(drop=True)


# ---------------------------------------------------------------------------
# NearMissDetectorV40  — optical flow velocity estimation
# ---------------------------------------------------------------------------

class NearMissDetectorV40(NearMissDetectorV11):
    """Near-Miss Detector v4.0 — optical flow motion vectors for velocity.

    Improvements over v1.1
    ----------------------
    * Dense Gunnar Farnebäck optical flow replaces trajectory-averaged centroid
      displacement for speed and heading estimation.  This eliminates the
      trajectory warm-up latency and captures sub-centroid motion patterns.
    * Global background flow (median over all pixels) is subtracted before bbox
      sampling to compensate for camera motion (useful for dashcam footage).
    * Blended mode: ``flow_weight × optical_flow + (1−flow_weight) × trajectory``
      lets the user tune between pure-flow (1.0) and pure-trajectory (0.0).
    * Works with ANY tracker — ByteTrack, CentroidTrackerV2, etc. — because it
      only needs ``bbox`` and ``trajectory`` from each tracked object dict.

    When ``frame`` is omitted from ``process_frame()`` the detector degrades
    silently to v1.1 trajectory-based velocity (full backward compatibility).

    Parameters (v4.0 additions)
    ---------------------------
    flow_margin : float (default 0.15)
        Fractional inward margin applied to each bbox side before sampling flow.
        Removes edge pixels that often contain background bleed-in.
    flow_min_magnitude : float (default 1.0)
        Pixels whose flow magnitude is below this value (px/frame) are treated
        as background/static and excluded from the aggregation.
    flow_percentile : float (default 50.0)
        Aggregation percentile over valid flow pixels (50 = median).  Raising
        toward 75–90 extracts faster-moving pixels; lowering toward 25 is more
        conservative.
    flow_min_pixels : int (default 20)
        Minimum number of valid (above-threshold) pixels needed to trust the
        flow estimate.  Falls back to trajectory when the count is too low
        (e.g. very small or partially-occluded bounding boxes).
    flow_weight : float (default 0.7)
        Blend weight.  Final velocity vector =
        ``flow_weight × v_flow + (1 − flow_weight) × v_trajectory``.
        Setting 1.0 uses flow exclusively (with trajectory as fallback when
        flow is unavailable).
    compensate_camera : bool (default True)
        Subtract the global median flow from each bbox region to remove
        apparent motion caused by camera movement.

    Extra event fields
    ------------------
    vel_source_1, vel_source_2 : "flow" | "flow+traj" | "trajectory"
        How the velocity was computed for each object.
    speed_1_px, speed_2_px : float
        Final blended speed used in the computation (px/processed-frame).
    optical_flow_used : bool
        True when the detector had access to a valid flow map this frame.
    """

    def __init__(
        self,
        # ── inherited v1.1 parameters ────────────────────────────────────────
        proximity_px:        float = 100.0,
        ttc_threshold:       float = 2.0,
        fps:                 float = 15.0,
        debounce_frames:     int   = 30,
        filters_enabled:     bool  = True,
        stationary_speed_px: float = 5.0,
        confirm_frames:      int   = 5,
        same_direction_deg:  float = 30.0,
        min_confidence:      float = 0.5,
        moving_speed_px:     float = 5.0,
        risk_high_threshold: float = _RISK_HIGH,
        risk_medium_threshold: float = _RISK_MEDIUM,
        proximity_scale:     float = 0.5,
        min_iou:             float = 0.05,
        buffer_decay:        float = 0.5,
        t_horizon_sec:       float = 5.0,
        clearance_scale:     float = 1.1,
        # ── v4.0 additions ────────────────────────────────────────────────────
        flow_margin:         float = 0.15,
        flow_min_magnitude:  float = 1.0,
        flow_percentile:     float = 50.0,
        flow_min_pixels:     int   = 20,
        flow_weight:         float = 0.7,
        compensate_camera:   bool  = True,
    ) -> None:
        super().__init__(
            proximity_px=proximity_px,
            ttc_threshold=ttc_threshold,
            fps=fps,
            debounce_frames=debounce_frames,
            filters_enabled=filters_enabled,
            stationary_speed_px=stationary_speed_px,
            confirm_frames=confirm_frames,
            same_direction_deg=same_direction_deg,
            min_confidence=min_confidence,
            moving_speed_px=moving_speed_px,
            risk_high_threshold=risk_high_threshold,
            risk_medium_threshold=risk_medium_threshold,
            proximity_scale=proximity_scale,
            min_iou=min_iou,
            buffer_decay=buffer_decay,
            t_horizon_sec=t_horizon_sec,
            clearance_scale=clearance_scale,
        )
        self.flow_margin        = flow_margin
        self.flow_min_magnitude = flow_min_magnitude
        self.flow_percentile    = flow_percentile
        self.flow_min_pixels    = flow_min_pixels
        self.flow_weight        = min(max(float(flow_weight), 0.0), 1.0)
        self.compensate_camera  = compensate_camera

        self._prev_gray: Optional[np.ndarray] = None
        self._last_flow_frame_idx: Optional[int] = None

    # ── Optical flow helpers ──────────────────────────────────────────────────

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert BGR (or already-gray) frame to single-channel uint8."""
        if frame.ndim == 2:
            return frame
        if _cv2 is None:
            raise RuntimeError("cv2 required for optical flow")
        return _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _compute_farneback(
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> np.ndarray:
        """Dense Gunnar Farnebäck flow → (H, W, 2) float32 array [dx, dy] px/frame."""
        if _cv2 is None:
            raise RuntimeError("cv2 required for optical flow")
        return _cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

    @staticmethod
    def _global_background_flow(
        flow: np.ndarray,
    ) -> Tuple[float, float]:
        """Median flow over all pixels — proxy for camera motion."""
        return (
            float(np.median(flow[..., 0])),
            float(np.median(flow[..., 1])),
        )

    def _extract_bbox_flow(
        self,
        flow:    np.ndarray,
        bbox:    List[float],
        bg_flow: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Aggregate optical flow within a bounding box.

        Parameters
        ----------
        flow    : (H, W, 2) dense flow map.
        bbox    : [x1, y1, x2, y2] pixel coordinates.
        bg_flow : camera-motion offset to subtract (or None).

        Returns
        -------
        (vx, vy) in px/processed-frame, or None when the region has too few
        valid (above-threshold) pixels to be reliable.
        """
        H, W = flow.shape[:2]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Inward margin — shrink bbox to avoid background bleed at edges
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        mx = max(1, int(bw * self.flow_margin))
        my = max(1, int(bh * self.flow_margin))
        ix1 = max(0, x1 + mx)
        iy1 = max(0, y1 + my)
        ix2 = min(W, x2 - mx)
        iy2 = min(H, y2 - my)

        if ix2 <= ix1 or iy2 <= iy1:
            return None

        region = flow[iy1:iy2, ix1:ix2].copy()   # (rH, rW, 2) float32

        # Camera-motion compensation
        if bg_flow is not None:
            region[..., 0] -= bg_flow[0]
            region[..., 1] -= bg_flow[1]

        # Filter: keep pixels whose compensated flow exceeds the noise floor
        magnitudes = np.sqrt(
            region[..., 0] ** 2 + region[..., 1] ** 2
        )
        valid = magnitudes > self.flow_min_magnitude

        if int(valid.sum()) < self.flow_min_pixels:
            _log_v40.debug(
                "bbox_flow: too few valid pixels (%d < %d) in region %dx%d",
                int(valid.sum()), self.flow_min_pixels,
                iy2 - iy1, ix2 - ix1,
            )
            return None

        # Aggregate: Nth percentile of each component over valid pixels
        vx = float(np.percentile(region[..., 0][valid], self.flow_percentile))
        vy = float(np.percentile(region[..., 1][valid], self.flow_percentile))
        return vx, vy

    def _get_velocity_v4(
        self,
        obj:     Dict[str, Any],
        flow:    Optional[np.ndarray],
        bg_flow: Optional[Tuple[float, float]],
    ) -> Tuple[float, float, str]:
        """
        Return (speed_px_per_frame, heading_degrees, source_label).

        Blends optical flow and trajectory velocity in vector space:
            v_blend = flow_weight × v_flow + (1 − flow_weight) × v_traj

        source_label: "flow" | "flow+traj" | "trajectory"
        """
        traj     = obj.get("trajectory", [])
        spd_traj = self._estimate_speed(traj)
        hdg_traj = self._estimate_heading(traj)
        vx_traj  = spd_traj * math.cos(math.radians(hdg_traj))
        vy_traj  = spd_traj * math.sin(math.radians(hdg_traj))

        if flow is not None and self.flow_weight > 0.0:
            flow_result = self._extract_bbox_flow(flow, obj["bbox"], bg_flow)
            if flow_result is not None:
                fvx, fvy = flow_result
                w        = self.flow_weight
                vx       = w * fvx + (1.0 - w) * vx_traj
                vy       = w * fvy + (1.0 - w) * vy_traj
                spd      = math.hypot(vx, vy)
                hdg      = math.degrees(math.atan2(vy, vx))
                source   = "flow" if w >= 1.0 else "flow+traj"
                _log_v40.debug(
                    "vel id=%d src=%s flow=(%.2f,%.2f) traj_spd=%.2f → spd=%.2f hdg=%.1f°",
                    obj.get("id", -1), source, fvx, fvy, spd_traj, spd, hdg,
                )
                return spd, hdg, source

        return spd_traj, hdg_traj, "trajectory"

    def reset_flow_state(self) -> None:
        """
        Clear optical-flow temporal state.

        Call before processing a new, unrelated sequence with the same detector
        instance (e.g., notebook reruns with a different video).
        """
        self._prev_gray = None
        self._last_flow_frame_idx = None

    # ── Main processing override ──────────────────────────────────────────────

    def process_frame(
        self,
        frame_idx:       int,
        tracked_objects: Dict[int, Dict[str, Any]],
        frame:           Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        v4.0 process_frame — optical flow velocity + v1.1 geometry.

        Parameters
        ----------
        frame_idx       : current video frame index.
        tracked_objects : OrderedDict from any BaseTracker.update().
        frame           : current BGR frame (optional).  When provided, dense
                          optical flow is computed against the previous processed
                          frame and used for velocity estimation.  When None,
                          degrades transparently to v1.1 trajectory-based logic.
        """
        self._update_frame_step(frame_idx)

        if _cv2 is None and frame is not None:
            _log_v40.warning(
                "cv2 not available — optical flow disabled, using trajectory fallback"
            )
            frame = None

        # ── Optical flow computation ──────────────────────────────────────────
        flow    = None
        bg_flow = None
        if frame is not None:
            # Guard against notebook/video restarts with the same detector instance.
            # If frame indices rewind, stale prev_gray would corrupt the first flow.
            if (self._last_flow_frame_idx is not None
                    and frame_idx <= self._last_flow_frame_idx):
                _log_v40.info(
                    "[fr=%d] non-monotonic frame index after %d -> reset flow state",
                    frame_idx, self._last_flow_frame_idx,
                )
                self._prev_gray = None

            curr_gray = self._to_gray(frame)
            if (self._prev_gray is not None
                    and self._prev_gray.shape == curr_gray.shape):
                try:
                    flow = self._compute_farneback(self._prev_gray, curr_gray)
                    if self.compensate_camera:
                        bg_flow = self._global_background_flow(flow)
                        _log_v40.debug(
                            "[fr=%d] bg_flow=(%.2f,%.2f)",
                            frame_idx, bg_flow[0], bg_flow[1],
                        )
                except Exception as exc:
                    _log_v40.warning(
                        "[fr=%d] flow computation failed: %s", frame_idx, exc
                    )
                    flow = None
            self._prev_gray = curr_gray
            self._last_flow_frame_idx = frame_idx

        _log_v40.debug(
            "[fr=%d] flow=%s  bg_comp=%s  n_tracks=%d",
            frame_idx,
            "yes" if flow is not None else "no",
            "yes" if bg_flow is not None else "no",
            len(tracked_objects),
        )

        # ── Per-pair evaluation ───────────────────────────────────────────────
        ids        = list(tracked_objects.keys())
        new_events: List[Dict[str, Any]] = []
        vel_cache: Dict[int, Tuple[float, float, str]] = {}

        def _velocity(obj_id: int, obj: Dict[str, Any]) -> Tuple[float, float, str]:
            if obj_id not in vel_cache:
                vel_cache[obj_id] = self._get_velocity_v4(obj, flow, bg_flow)
            return vel_cache[obj_id]

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                pair_key = (min(id1, id2), max(id1, id2))

                obj1 = tracked_objects[id1]
                obj2 = tracked_objects[id2]

                # ── Footpoints & scale-aware threshold (#3, #4) ───────────────
                fp1      = self._footpoint(obj1)
                fp2      = self._footpoint(obj2)
                dist     = math.hypot(fp2[0] - fp1[0], fp2[1] - fp1[1])
                eff_prox = self._effective_proximity(obj1, obj2)

                # ── Proximity gate (#5 IoU) ───────────────────────────────────
                iou       = self._compute_iou(obj1["bbox"], obj2["bbox"])
                proximate = (dist < eff_prox) or (iou > self.min_iou)
                if not proximate:
                    self._leak_buffer(pair_key)
                    continue

                # ── Optical-flow / trajectory velocity ────────────────────────
                spd1, h1, src1 = _velocity(id1, obj1)
                spd2, h2, src2 = _velocity(id2, obj2)

                _log_v40.debug(
                    "[fr=%d] pair=(%d,%d) src=(%s,%s) sp=(%.2f,%.2f) h=(%.1f°,%.1f°)",
                    frame_idx, id1, id2, src1, src2, spd1, spd2, h1, h2,
                )

                # ── 2-D closest-approach (#2) ─────────────────────────────────
                t_star, d_min, converging = self._closest_approach(
                    fp1, fp2, spd1, spd2, h1, h2
                )

                risk_score = self._risk_score_v11(
                    dist, t_star, d_min, max(spd1, spd2), eff_prox
                )

                # ── Multi-criteria gate (2 of 3) ──────────────────────────────
                criteria_met = sum([
                    dist  < eff_prox,
                    d_min < eff_prox,
                    max(spd1, spd2) > self.moving_speed_px,
                ])
                if criteria_met < 2:
                    self._leak_buffer(pair_key)
                    continue

                # ── Convergence guard — hard physical constraint ──────────────
                if not converging:
                    self._leak_buffer(pair_key)
                    _log_v40.debug(
                        "[fr=%d] pair=(%d,%d) skip=not_converging",
                        frame_idx, id1, id2,
                    )
                    continue

                # ── Size-aware closest-approach guard (adjacent-lane suppressor)
                if iou <= self.min_iou:
                    clearance_px = self._clearance_threshold_px(obj1, obj2, eff_prox)
                    if d_min > clearance_px:
                        self._leak_buffer(pair_key)
                        _log_v40.debug(
                            "[fr=%d] pair=(%d,%d) skip=clearance d_min=%.1f > %.1f",
                            frame_idx, id1, id2, d_min, clearance_px,
                        )
                        continue

                # ── False-positive heuristic filters ──────────────────────────
                if self.filters_enabled:
                    if self._filter_confidence(
                        obj1["confidence"], obj2["confidence"]
                    ):
                        self._leak_buffer(pair_key)
                        continue
                    if self._filter_stationary(spd1, spd2):
                        self._leak_buffer(pair_key)
                        _log_v40.debug(
                            "[fr=%d] pair=(%d,%d) skip=stationary "
                            "sp1=%.2f sp2=%.2f < %.2f",
                            frame_idx, id1, id2, spd1, spd2,
                            self.stationary_speed_px,
                        )
                        continue
                    if self._filter_direction_v11(h1, h2, fp1, fp2, spd1, spd2):
                        self._leak_buffer(pair_key)
                        _log_v40.debug(
                            "[fr=%d] pair=(%d,%d) skip=direction",
                            frame_idx, id1, id2,
                        )
                        continue

                # ── Leaky confirmation buffer (#6) ────────────────────────────
                self._confirmation_buffer[pair_key] += 1.0
                if (self.filters_enabled
                        and self._confirmation_buffer[pair_key] < self.confirm_frames):
                    _log_v40.debug(
                        "[fr=%d] pair=(%d,%d) skip=buffer(%.1f/%.0f)",
                        frame_idx, id1, id2,
                        self._confirmation_buffer[pair_key], self.confirm_frames,
                    )
                    continue

                # ── Debounce ──────────────────────────────────────────────────
                last = self._last_event_frame.get(
                    pair_key, -self.debounce_frames - 1
                )
                if frame_idx - last < self.debounce_frames:
                    _log_v40.debug(
                        "[fr=%d] pair=(%d,%d) skip=debounce gap=%d",
                        frame_idx, id1, id2, frame_idx - last,
                    )
                    continue

                # ── Emit event ────────────────────────────────────────────────
                event: Dict[str, Any] = {
                    "frame_index":       frame_idx,
                    "timestamp_sec":     round(frame_idx / self.fps, 2),
                    "object_id_1":       id1,
                    "object_id_2":       id2,
                    "class_1":           obj1["class"],
                    "class_2":           obj2["class"],
                    "label_1":           obj1.get("label", ""),
                    "label_2":           obj2.get("label", ""),
                    "distance_px":       round(dist, 2),
                    "ttc_sec":           round(t_star, 3) if t_star < math.inf else None,
                    "d_min_px":          round(d_min, 2),
                    "risk_score":        risk_score,
                    "risk_level":        self._risk_level(risk_score),
                    "conf_1":            round(obj1["confidence"], 3),
                    "conf_2":            round(obj2["confidence"], 3),
                    # v4.0 fields
                    "vel_source_1":      src1,
                    "vel_source_2":      src2,
                    "speed_1_px":        round(spd1, 2),
                    "speed_2_px":        round(spd2, 2),
                    "optical_flow_used": flow is not None,
                }
                self._events.append(event)
                self._last_event_frame[pair_key] = frame_idx
                self._last_event_data[pair_key]  = event
                new_events.append(event)

                _log_v40.info(
                    "[fr=%d] EVENT pair=(%d,%d) dist=%.1f d_min=%.1f "
                    "risk=%.3f level=%s vel_src=(%s,%s)",
                    frame_idx, id1, id2, dist, d_min,
                    risk_score, self._risk_level(risk_score), src1, src2,
                )

        return new_events

    def get_events_dataframe(self) -> pd.DataFrame:
        """v4.0: includes optical flow source columns in empty-DataFrame schema."""
        if not self._events:
            return pd.DataFrame(columns=[
                "frame_index", "timestamp_sec",
                "object_id_1", "object_id_2",
                "class_1", "class_2", "label_1", "label_2",
                "distance_px", "ttc_sec", "d_min_px",
                "risk_score", "risk_level", "conf_1", "conf_2",
                "vel_source_1", "vel_source_2",
                "speed_1_px", "speed_2_px",
                "optical_flow_used",
            ])
        df = pd.DataFrame(self._events)
        return df.sort_values("timestamp_sec").reset_index(drop=True)
