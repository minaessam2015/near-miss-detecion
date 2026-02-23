"""
Multi-Object Tracking — pluggable tracker abstraction.

Architecture:
  BaseTracker (ABC)
  ├── CentroidTracker   — greedy centroid matching (no extra deps)
  ├── ByteTracker       — ByteTrack via boxmot (two-stage IoU + Kalman)
  └── BoTSORTTracker    — BoT-SORT via boxmot (Kalman + ReID appearance)

Factory:
  create_tracker(name, **kwargs) -> BaseTracker
  name: 'centroid' | 'bytetrack' | 'botsort'

All three share the same update() interface and produce the same output dict
schema, so they are drop-in replacements in the full pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output dict schema (per tracked object)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "id":         int
  "class":      "vehicle" | "pedestrian"
  "label":      "car" | "truck" | "bus" | "motorcycle" | "bicycle" | "person"
  "bbox":       [x1, y1, x2, y2]
  "confidence": float
  "center":     (cx, cy)
  "trajectory": [(frame_idx, cx, cy), ...]
}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Track activation — min_hits  (tentative → confirmed state machine)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A new track starts as *tentative* and only appears in update()'s output once
it has been matched for min_hits **consecutive** frames.  This is the same
n_init / n_init mechanism used by DeepSORT, StrongSORT, OC-SORT, and the
ByteTrack reference implementation.

  min_hits=1  — every detection is immediately active (default, backward-compat)
  min_hits=2  — must be seen 2 frames in a row before first exposure
  min_hits=3  — typical setting to suppress single-/double-frame noise

State transitions:
  TENTATIVE → CONFIRMED   after min_hits consecutive matched frames
              ↑                  (transition is ONE-WAY)
  missed frame resets streak only while still TENTATIVE

  CONFIRMED → DELETED     after max_disappeared missed frames
              (confirmed tracks are never demoted back to tentative)

This eliminates the "flickering" problem where a confirmed track temporarily
disappears and would otherwise require another min_hits warmup to reappear.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Comparison
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tracker       Matching         Occlusion handling     ReID    Speed (CPU)
CentroidTrack greedy L2        none                   no      fastest
ByteTrack     2-stage IoU+L2   low-conf 2nd pass      no      fast
BoT-SORT      IoU+Kalman       Kalman prediction       yes*    moderate
                                                      (*auto-downloaded ~20 MB)
"""

from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Tuple
import importlib
import logging

import numpy as np
from scipy.spatial.distance import cdist

_log = logging.getLogger("tracker.v2")


def _reset_boxmot_id_counter() -> None:
    """
    Reset boxmot's class-level track-ID counter.

    ByteTrack and BoT-SORT assign IDs via ``STrack._count`` (or
    ``BaseTrack._count``), which is a *class* variable shared across all
    instances.  Creating a fresh tracker instance does NOT reset it, so
    reruns in a Jupyter notebook accumulate IDs from previous runs.

    This function probes the known module paths across boxmot versions and
    resets the counter to 0 so every new tracker wrapper starts from ID 0.
    """
    _candidates = [
        ("boxmot.trackers.bytetrack.byte_tracker", ("STrack", "BaseTrack")),
        ("boxmot.trackers.botsort.bot_sort",        ("STrack", "BaseTrack")),
        ("boxmot.trackers.basetrack",               ("BaseTrack",)),
        ("boxmot.trackers.bytetrack.basetrack",     ("BaseTrack",)),
        ("boxmot.trackers.botsort.basetrack",       ("BaseTrack",)),
    ]
    for mod_path, class_names in _candidates:
        try:
            mod = importlib.import_module(mod_path)
            for cls_name in class_names:
                cls = getattr(mod, cls_name, None)
                if cls is not None and hasattr(cls, "_count"):
                    cls._count = 0
        except Exception:
            pass

# Numeric class encoding for boxmot input arrays (cls column)
_CLASS_TO_CLS: Dict[str, int] = {"pedestrian": 0, "vehicle": 1}
_CLS_TO_CLASS: Dict[int, str] = {0: "pedestrian", 1: "vehicle"}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseTracker(ABC):
    """
    Common interface for all trackers.

    update() must be called once per processed frame.
    All trackers produce the same output dict schema (see module docstring).
    """

    @abstractmethod
    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int,
        frame: Optional[np.ndarray] = None,
    ) -> "OrderedDict[int, Dict[str, Any]]":
        """
        Process detections for one frame and return currently tracked objects.

        Args:
            detections: Detection dicts from any BaseDetector.
            frame_idx:  Current video frame index (used in trajectory logging).
            frame:      BGR frame array. Required by BoT-SORT for ReID crops;
                        ignored by CentroidTracker and ByteTracker.

        Returns:
            OrderedDict[track_id -> state_dict]  (only currently visible tracks)
        """
        ...

    @property
    @abstractmethod
    def tracker_name(self) -> str:
        """Human-readable tracker identifier."""
        ...

    @property
    def total_ids_assigned(self) -> int:
        """Total unique track IDs created so far (includes deregistered)."""
        return 0

    def active_count(self) -> int:
        """Number of currently visible tracks."""
        return 0

    @property
    def all_trajectories(self) -> Dict[int, List[Tuple[int, int, int]]]:
        """Full trajectory history including deregistered tracks."""
        return {}


# ---------------------------------------------------------------------------
# CentroidTracker  (original, no extra dependencies)
# ---------------------------------------------------------------------------

class CentroidTracker(BaseTracker):
    """
    Greedy centroid-distance tracker.

    Assigns persistent IDs by matching each new detection to the nearest
    existing centroid (L2 distance). Deregisters tracks that have been
    absent for more than max_disappeared frames.

    Pros:  no extra dependencies, very fast, simple to reason about.
    Cons:  no Kalman prediction, no re-ID; ID switches on occlusion.
    """

    def __init__(
        self,
        max_disappeared: int   = 10,
        max_distance:    float = 150.0,
        min_hits:        int   = 1,
    ):
        """
        Args:
            max_disappeared: Frames an unmatched track can survive before
                             being deregistered.
            max_distance:    Maximum centroid distance (px) for a match.
            min_hits:        Consecutive frames required to confirm a new track.
                             min_hits=1 → every detection is immediately active
                                          (default, backward-compatible).
                             min_hits=3 → track must appear in 3 consecutive frames
                                          before first being exposed to the pipeline.
                             Once confirmed, a track is never demoted back to
                             tentative — missed frames only count toward
                             max_disappeared deletion, not re-confirmation.
        """
        self.next_id:        int = 0
        self.max_disappeared     = max_disappeared
        self.max_distance        = max_distance
        self.min_hits            = min_hits
        self.objects:  OrderedDict = OrderedDict()
        self.disappeared:   Dict   = {}
        self.trajectories:  Dict   = {}
        self._class_history: Dict  = {}
        self._hit_streak:    Dict  = {}   # obj_id → consecutive-hit count (tentative only)
        self._confirmed_ids: set   = set()  # IDs that have crossed the min_hits threshold

    @property
    def tracker_name(self) -> str:
        return f"CentroidTracker(max_dist={self.max_distance}, max_gone={self.max_disappeared})"

    # ── internals ────────────────────────────────────────────────────────

    def _register(self, detection: Dict[str, Any], frame_idx: int) -> None:
        obj_id = self.next_id
        self.next_id += 1
        cx, cy = detection["center"]
        self.objects[obj_id] = {**detection, "id": obj_id,
                                "trajectory": [(frame_idx, cx, cy)]}
        self.disappeared[obj_id]    = 0
        self.trajectories[obj_id]   = [(frame_idx, cx, cy)]
        self._class_history[obj_id] = [detection["class"]]
        self._hit_streak[obj_id]    = 1   # first hit
        if self.min_hits <= 1:
            self._confirmed_ids.add(obj_id)  # immediate confirmation

    def _deregister(self, obj_id: int) -> None:
        del self.objects[obj_id]
        del self.disappeared[obj_id]
        del self._class_history[obj_id]
        self._hit_streak.pop(obj_id, None)
        self._confirmed_ids.discard(obj_id)

    def _majority_class(self, obj_id: int) -> str:
        return Counter(self._class_history[obj_id][-10:]).most_common(1)[0][0]

    def _active_objects(self) -> OrderedDict:
        """Return only confirmed tracks (those that have met min_hits at creation)."""
        if self.min_hits <= 1:
            return self.objects
        return OrderedDict(
            (k, v) for k, v in self.objects.items()
            if k in self._confirmed_ids
        )

    # ── public API ───────────────────────────────────────────────────────

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int,
        frame: Optional[np.ndarray] = None,  # unused
    ) -> OrderedDict:
        if not detections:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if obj_id not in self._confirmed_ids:
                    self._hit_streak[obj_id] = 0  # tentative only
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return self._active_objects()

        if not self.objects:
            for det in detections:
                self._register(det, frame_idx)
            return self._active_objects()

        existing_ids       = list(self.objects.keys())
        existing_centroids = np.array([self.objects[i]["center"] for i in existing_ids], float)
        new_centroids      = np.array([d["center"] for d in detections], float)

        D    = cdist(existing_centroids, new_centroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            obj_id = existing_ids[row]
            det    = detections[col]
            cx, cy = det["center"]

            self.objects[obj_id].update(
                {k: det[k] for k in ("class", "label", "bbox", "confidence", "center")}
            )
            self.trajectories[obj_id].append((frame_idx, cx, cy))
            self.objects[obj_id]["trajectory"] = self.trajectories[obj_id]
            self._class_history[obj_id].append(det["class"])
            self.objects[obj_id]["class"] = self._majority_class(obj_id)
            self.disappeared[obj_id] = 0
            # Advance tentative streak; confirmed tracks need no further tracking
            if obj_id not in self._confirmed_ids:
                self._hit_streak[obj_id] = self._hit_streak.get(obj_id, 0) + 1
                if self._hit_streak[obj_id] >= self.min_hits:
                    self._confirmed_ids.add(obj_id)
            used_rows.add(row)
            used_cols.add(col)

        for row in range(len(existing_ids)):
            if row not in used_rows:
                obj_id = existing_ids[row]
                self.disappeared[obj_id] += 1
                # Reset streak only while still tentative — confirmed tracks keep their status
                if obj_id not in self._confirmed_ids:
                    self._hit_streak[obj_id] = 0
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)

        for col in range(len(detections)):
            if col not in used_cols:
                self._register(detections[col], frame_idx)

        return self._active_objects()

    @property
    def all_trajectories(self) -> Dict[int, List[Tuple[int, int, int]]]:
        return self.trajectories

    @property
    def total_ids_assigned(self) -> int:
        return self.next_id

    def active_count(self) -> int:
        return len(self.objects)


# ---------------------------------------------------------------------------
# CentroidTrackerV2  — 3-D ground-plane matching
# ---------------------------------------------------------------------------

class CentroidTrackerV2(CentroidTracker):
    """
    CentroidTracker with 3-D ground-plane distance matching.

    Matching is computed per pair:

    - if both candidates have ``ground_pt`` (X, Y, Z), use 3-D distance;
    - otherwise use 2-D centroid distance.

    This keeps 3-D matching active for valid pairs even when some detections
    in the same frame are missing ``ground_pt``.

    All other behaviour (trajectory, deregistration, min_hits, majority-vote
    class label) is inherited unchanged from CentroidTracker.

    Extra schema fields added to every tracked-object dict
    -------------------------------------------------------
    ground_pt     : Optional[Tuple[float, float, float]]
        3-D camera-space footpoint for the current frame, or None.
    trajectory_3d : [(frame_idx, X, Y, Z), ...]
        Full 3-D trajectory history (only frames where ground_pt was valid).

    Extra constructor parameter
    ---------------------------
    max_distance_3d : float (default 3.0)
        Maximum 3-D ground distance (depth units) accepted as a match when
        3-D mode is active.  Analogous to ``max_distance`` for 2-D mode.
    """

    def __init__(
        self,
        max_disappeared: int   = 10,
        max_distance:    float = 150.0,
        min_hits:        int   = 1,
        max_distance_3d: float = 3.0,
    ):
        super().__init__(
            max_disappeared=max_disappeared,
            max_distance=max_distance,
            min_hits=min_hits,
        )
        self.max_distance_3d = max_distance_3d
        # 3-D trajectory store: obj_id → [(frame_idx, X, Y, Z), ...]
        self._trajectories_3d: Dict[int, List[Tuple]] = {}

    @property
    def tracker_name(self) -> str:
        return (
            f"CentroidTrackerV2(max_dist_3d={self.max_distance_3d}, "
            f"max_dist_2d={self.max_distance}, max_gone={self.max_disappeared})"
        )

    # ── internals ────────────────────────────────────────────────────────────

    def _register(self, detection: Dict[str, Any], frame_idx: int) -> None:
        """Register a new track; initialise ground_pt and trajectory_3d."""
        super()._register(detection, frame_idx)
        obj_id = self.next_id - 1          # super() already incremented next_id
        gp = detection.get("ground_pt")
        self.objects[obj_id]["ground_pt"]     = gp
        self.objects[obj_id]["trajectory_3d"] = []
        self._trajectories_3d[obj_id] = []
        if gp is not None:
            entry = (frame_idx, *gp)
            self.objects[obj_id]["trajectory_3d"].append(entry)
            self._trajectories_3d[obj_id].append(entry)
            _log.debug(
                "[fr=%d] register id=%d ground_pt=(%.3f,%.3f,%.3f)",
                frame_idx, obj_id, *gp,
            )
        else:
            _log.debug("[fr=%d] register id=%d ground_pt=None", frame_idx, obj_id)

    def _deregister(self, obj_id: int) -> None:
        super()._deregister(obj_id)
        self._trajectories_3d.pop(obj_id, None)

    # ── public API ────────────────────────────────────────────────────────────

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int,
        frame: Optional[np.ndarray] = None,
    ) -> OrderedDict:
        """Update tracks with 3-D ground-plane matching when ground_pts available."""

        # ── No detections: delegate to parent (handles disappeared counts) ────
        if not detections:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if obj_id not in self._confirmed_ids:
                    self._hit_streak[obj_id] = 0
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return self._active_objects()

        # ── No existing tracks: register all ─────────────────────────────────
        if not self.objects:
            for det in detections:
                self._register(det, frame_idx)
            return self._active_objects()

        existing_ids = list(self.objects.keys())

        # ── Build per-pair hybrid cost matrix ────────────────────────────────
        existing_gps = [self.objects[i].get("ground_pt") for i in existing_ids]
        new_gps      = [d.get("ground_pt") for d in detections]
        N = len(existing_ids)
        M = len(detections)

        e_has_3d = np.array([gp is not None for gp in existing_gps], dtype=bool)
        n_has_3d = np.array([gp is not None for gp in new_gps],      dtype=bool)
        mask_3d  = np.outer(e_has_3d, n_has_3d)  # (N, M): True -> 3-D cost

        # Normalised cost matrix:
        #   3-D cells: dist_3d / max_distance_3d
        #   2-D cells: dist_2d / max_distance
        # A match is accepted when cost < 1.0.
        D = np.full((N, M), np.inf, dtype=np.float64)

        if mask_3d.any():
            pts_exist_3d = np.array(
                [gp if gp is not None else (0.0, 0.0, 0.0) for gp in existing_gps],
                dtype=np.float64,
            )
            pts_new_3d = np.array(
                [gp if gp is not None else (0.0, 0.0, 0.0) for gp in new_gps],
                dtype=np.float64,
            )
            D_3d = cdist(pts_exist_3d, pts_new_3d) / self.max_distance_3d
            D = np.where(mask_3d, D_3d, D)

        mask_2d = ~mask_3d
        if mask_2d.any():
            pts_exist_2d = np.array(
                [self.objects[i]["center"] for i in existing_ids], dtype=np.float64
            )
            pts_new_2d = np.array([d["center"] for d in detections], dtype=np.float64)
            D_2d = cdist(pts_exist_2d, pts_new_2d) / self.max_distance
            D = np.where(mask_2d, D_2d, D)

        _log.debug(
            "[fr=%d] match mode=hybrid existing=%d new=%d 3d_cells=%d 2d_cells=%d",
            frame_idx, N, M, int(mask_3d.sum()), int(mask_2d.sum()),
        )

        # ── Greedy matching (same logic as CentroidTracker) ───────────────────
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows: set = set()
        used_cols: set = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] >= 1.0:
                continue
            obj_id = existing_ids[row]
            det    = detections[col]
            cx, cy = det["center"]
            gp     = det.get("ground_pt")

            # Update 2-D fields
            self.objects[obj_id].update(
                {k: det[k] for k in ("class", "label", "bbox", "confidence", "center")}
            )
            self.trajectories[obj_id].append((frame_idx, cx, cy))
            self.objects[obj_id]["trajectory"] = self.trajectories[obj_id]
            self._class_history[obj_id].append(det["class"])
            self.objects[obj_id]["class"] = self._majority_class(obj_id)
            self.disappeared[obj_id] = 0

            # Update 3-D fields
            match_mode = "3d" if mask_3d[row, col] else "2d"
            self.objects[obj_id]["ground_pt"] = gp
            if gp is not None:
                entry = (frame_idx, *gp)
                if obj_id not in self._trajectories_3d:
                    self._trajectories_3d[obj_id] = []
                self._trajectories_3d[obj_id].append(entry)
                self.objects[obj_id]["trajectory_3d"] = self._trajectories_3d[obj_id]
                _log.debug(
                    "[fr=%d] match id=%d cost=%.4f [%s] gp=(%.3f,%.3f,%.3f)",
                    frame_idx, obj_id, D[row, col], match_mode, *gp,
                )
            else:
                if obj_id not in self._trajectories_3d:
                    self._trajectories_3d[obj_id] = []
                self.objects[obj_id]["trajectory_3d"] = self._trajectories_3d[obj_id]
                _log.debug(
                    "[fr=%d] match id=%d cost=%.4f [%s] gp=None",
                    frame_idx, obj_id, D[row, col], match_mode,
                )

            # Tentative → confirmed state machine
            if obj_id not in self._confirmed_ids:
                self._hit_streak[obj_id] = self._hit_streak.get(obj_id, 0) + 1
                if self._hit_streak[obj_id] >= self.min_hits:
                    self._confirmed_ids.add(obj_id)

            used_rows.add(row)
            used_cols.add(col)

        # ── Unmatched existing tracks ─────────────────────────────────────────
        for row in range(len(existing_ids)):
            if row not in used_rows:
                obj_id = existing_ids[row]
                self.disappeared[obj_id] += 1
                if obj_id not in self._confirmed_ids:
                    self._hit_streak[obj_id] = 0
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)

        # ── Unmatched new detections → register ───────────────────────────────
        for col in range(len(detections)):
            if col not in used_cols:
                self._register(detections[col], frame_idx)

        return self._active_objects()

    @property
    def all_trajectories_3d(self) -> Dict[int, List[Tuple]]:
        """Full 3-D trajectory history including deregistered tracks."""
        return self._trajectories_3d


# ---------------------------------------------------------------------------
# Shared boxmot wrapper
# ---------------------------------------------------------------------------

class _BoxmotWrapper(BaseTracker):
    """
    Internal base for boxmot-backed trackers (ByteTrack, BoT-SORT, …).

    Handles detection dict ↔ numpy conversion and maintains per-track
    trajectory history (boxmot does not expose this itself).

    boxmot input:  (N, 6) float32  [x1, y1, x2, y2, conf, cls]
    boxmot output: (M, 8) float32  [x1, y1, x2, y2, id, conf, cls, det_ind]
      det_ind = index into input dets for matched tracks, -1 for predicted ones.
    """

    def __init__(self, boxmot_tracker, min_hits: int = 1):
        self._tracker    = boxmot_tracker
        self.min_hits    = min_hits
        # boxmot IDs are process-global (shared class counter). Remap each raw
        # ID into a tracker-local contiguous ID space so multiple tracker
        # instances can run in parallel without cross-instance ID inflation.
        self._raw_to_local: Dict[int, int] = {}
        self._next_local_id: int = 0
        self._trajectories:    Dict[int, List[Tuple[int, int, int]]] = {}
        self._trajectories_3d: Dict[int, List[Tuple]] = {}           # (fi,X,Y,Z)
        self._id_label:      Dict[int, Tuple[str, str]] = {}  # id → (class, label)
        self._hit_streak:    Dict[int, int] = {}              # tentative id → consecutive hits
        self._confirmed_ids: set = set()                      # IDs that passed min_hits
        self._objects:       OrderedDict = OrderedDict()

    @property
    def total_ids_assigned(self) -> int:
        return self._next_local_id

    def active_count(self) -> int:
        return len(self._objects)

    @property
    def all_trajectories(self) -> Dict[int, List[Tuple[int, int, int]]]:
        return self._trajectories

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int,
        frame: Optional[np.ndarray] = None,
    ) -> OrderedDict:
        self._objects = OrderedDict()
        if not detections:
            return self._objects

        # Build (N, 6) array expected by boxmot
        det_arr = np.array(
            [[*d["bbox"], d["confidence"], float(_CLASS_TO_CLS[d["class"]])]
             for d in detections],
            dtype=np.float32,
        )

        # Some trackers use the frame for ReID crops; pass a small dummy if None
        img = frame if frame is not None else np.zeros((2, 2, 3), dtype=np.uint8)

        try:
            raw = self._tracker.update(det_arr, img)
        except Exception as exc:
            print(f"[{self.tracker_name}] update error: {exc}")
            return self._objects

        if raw is None or len(raw) == 0:
            # All tracks missed — reset streak only for tentative ones
            for oid in list(self._hit_streak):
                if oid not in self._confirmed_ids:
                    self._hit_streak[oid] = 0
            return self._objects

        visible_ids: set = set()

        for track in raw:
            # Columns: x1 y1 x2 y2 id conf cls det_ind
            x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
            raw_id  = int(track[4])
            if raw_id not in self._raw_to_local:
                self._raw_to_local[raw_id] = self._next_local_id
                self._next_local_id += 1
            obj_id  = self._raw_to_local[raw_id]
            conf    = float(track[5])
            cls_int = int(track[6]) if len(track) > 6 else 1
            det_idx = int(track[7]) if len(track) > 7 else -1

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Trajectory
            if obj_id not in self._trajectories:
                self._trajectories[obj_id] = []
            self._trajectories[obj_id].append((frame_idx, cx, cy))

            # Advance tentative streak; confirmed tracks need no further tracking
            if obj_id not in self._confirmed_ids:
                self._hit_streak[obj_id] = self._hit_streak.get(obj_id, 0) + 1
                if self._hit_streak[obj_id] >= self.min_hits:
                    self._confirmed_ids.add(obj_id)
            visible_ids.add(obj_id)

            # Class / label — prefer original detection; fall back to history
            if 0 <= det_idx < len(detections):
                orig     = detections[det_idx]
                category = orig["class"]
                label    = orig["label"]
                gp       = orig.get("ground_pt")
            elif obj_id in self._id_label:
                category, label = self._id_label[obj_id]
                gp = None
            else:
                category = _CLS_TO_CLASS.get(cls_int, "vehicle")
                label    = category
                gp = None

            self._id_label[obj_id] = (category, label)

            # 3-D trajectory
            if obj_id not in self._trajectories_3d:
                self._trajectories_3d[obj_id] = []
            if gp is not None:
                self._trajectories_3d[obj_id].append((frame_idx, *gp))

            self._objects[obj_id] = {
                "id":           obj_id,
                "class":        category,
                "label":        label,
                "bbox":         [x1, y1, x2, y2],
                "confidence":   conf,
                "center":       (cx, cy),
                "trajectory":   self._trajectories[obj_id],
                "ground_pt":    gp,
                "trajectory_3d": self._trajectories_3d[obj_id],
            }

        # Reset streak only for tentative tracks that disappeared this frame
        for oid in list(self._hit_streak):
            if oid not in visible_ids and oid not in self._confirmed_ids:
                self._hit_streak[oid] = 0

        # Return only confirmed tracks
        if self.min_hits <= 1:
            return self._objects
        return OrderedDict(
            (k, v) for k, v in self._objects.items()
            if k in self._confirmed_ids
        )


# ---------------------------------------------------------------------------
# ByteTracker
# ---------------------------------------------------------------------------

class ByteTracker(_BoxmotWrapper):
    """
    ByteTrack via the boxmot library.

    Two-stage association: high-confidence detections are matched first (IoU),
    then remaining low-confidence detections are matched to unresolved tracks.
    Uses a Kalman filter to predict positions during occlusion.

    Requires: pip install boxmot

    Args:
        track_thresh:  Confidence gate for first-stage matching.
                       Detections between track_thresh and match_thresh go to
                       the second (low-conf) stage.
        track_buffer:  Max frames a track can be "lost" before deregistration
                       (analogous to CentroidTracker.max_disappeared).
        match_thresh:  Minimum IoU to accept a match in either stage.
        frame_rate:    Source video FPS (scales internal Kalman noise).
    """

    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int   = 30,
        match_thresh: float = 0.8,
        frame_rate:   int   = 30,
        min_hits:     int   = 1,
    ):
        try:
            from boxmot import ByteTrack as _BT
        except ImportError:
            raise ImportError(
                "ByteTracker requires boxmot: pip install boxmot"
            )

        super().__init__(_BT(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
        ), min_hits=min_hits)
        self._params = dict(track_thresh=track_thresh, track_buffer=track_buffer,
                            match_thresh=match_thresh, min_hits=min_hits)

    @property
    def tracker_name(self) -> str:
        p = self._params
        return (f"ByteTrack(thresh={p['track_thresh']}, "
                f"buf={p['track_buffer']}, iou={p['match_thresh']}, "
                f"min_hits={p['min_hits']})")


# ---------------------------------------------------------------------------
# BoTSORTTracker
# ---------------------------------------------------------------------------

class BoTSORTTracker(_BoxmotWrapper):
    """
    BoT-SORT via the boxmot library.

    Combines Kalman-filter IoU matching with optional ReID appearance
    features for robust re-identification across occlusions.

    ReID weights (~20 MB) are auto-downloaded by boxmot on first use.
    Pass reid_weights=None to use boxmot's default model (osnet_x0_25_msmt17).

    Requires: pip install boxmot

    Args:
        reid_weights:  Path to a .pt ReID model, or None for boxmot default.
        device:        'cpu' or 'cuda:0' for the ReID feature extractor.
        half:          Use FP16 for the ReID model (GPU only).

    Note:
        For purely motion-based tracking with no appearance model, use
        ByteTracker — it is faster and requires no weight downloads.
    """

    def __init__(
        self,
        reid_weights: Optional[str] = None,
        device: str  = "cpu",
        half:   bool = False,
        min_hits: int = 1,
    ):
        try:
            from boxmot import BotSort as _BS
        except ImportError as e:
            raise ImportError(
                f"BoTSORTTracker requires boxmot: pip install boxmot {e}"
            )

        from pathlib import Path
        rw = Path(reid_weights) if reid_weights else Path("osnet_x0_25_msmt17.pt")

        super().__init__(_BS(
            reid_weights=rw,
            device=device,
            half=half,
        ), min_hits=min_hits)
        self._device = device
        self._rw_name = rw.name

    @property
    def tracker_name(self) -> str:
        return f"BoT-SORT(reid={self._rw_name}, device={self._device})"


# ---------------------------------------------------------------------------
# Kalman filter helper (3-D constant-velocity model)
# ---------------------------------------------------------------------------

class _KalmanFilter3D:
    """Minimal constant-velocity Kalman filter for 3-D ground-plane points.

    State  x  = [X, Y, Z, vX, vY, vZ]   (6-D)
    Observation = [X, Y, Z]               (3-D)

    The filter is initialised with velocity = 0.  Velocity is learned from
    successive measurements via the Kalman update equations.  During frames
    where no measurement arrives the predict() step extrapolates position
    forward using the last estimated velocity.

    All numeric values use the same depth-unit scale as ground_pt.
    """

    def __init__(
        self,
        initial_pos: np.ndarray,
        q_pos: float = 1e-4,
        q_vel: float = 1e-3,
        r_pos: float = 5e-4,
    ) -> None:
        # State vector [X, Y, Z, vX, vY, vZ]
        self.x    = np.zeros(6, dtype=np.float64)
        self.x[:3] = initial_pos

        # Covariance — start with high uncertainty in velocity
        self.P = np.diag([r_pos, r_pos, r_pos, q_vel * 10, q_vel * 10, q_vel * 10])

        # Transition  x_{t+1} = F x_t   (constant velocity, dt=1 frame)
        self.F       = np.eye(6, dtype=np.float64)
        self.F[0, 3] = 1.0   # X  += vX
        self.F[1, 4] = 1.0   # Y  += vY
        self.F[2, 5] = 1.0   # Z  += vZ

        # Observation  z = H x   (we only observe position)
        self.H       = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Noise matrices
        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(np.float64)
        self.R = np.eye(3, dtype=np.float64) * r_pos

    def predict(self) -> np.ndarray:
        """Predict next state.  Returns predicted position (3-D)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    def update(self, z: np.ndarray) -> None:
        """Correct state with measurement z (3-D)."""
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6, dtype=np.float64) - K @ self.H) @ self.P

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()


# ---------------------------------------------------------------------------
# CentroidTrackerV3  — per-pair hybrid cost + Kalman prediction
# ---------------------------------------------------------------------------

_log_v3 = logging.getLogger("tracker.v3")


class CentroidTrackerV3(CentroidTrackerV2):
    """Ground-plane tracker with per-pair cost fusion and Kalman prediction.

    Improvements over CentroidTrackerV2
    ------------------------------------
    1. **Per-pair cost matrix** — each cell in the assignment matrix
       independently uses:

       * 3-D ground distance (normalised by ``max_distance_3d``) when *both*
         the existing track and the new detection have a ``ground_pt``, or
       * 2-D centroid distance (normalised by ``max_distance``) otherwise.

       A cell is accepted when the normalised cost < 1.0.  This replaces V2's
       binary per-frame mode switch ("either all objects have ground_pt, or
       fall back to 2-D for the whole frame").

    2. **Kalman prediction** — each track maintains a constant-velocity Kalman
       filter in 3-D.  Every frame, all existing tracks are *predicted* forward
       before matching.  Tracks that are missed ("coasting") carry a
       Kalman-extrapolated ``ground_pt`` instead of the stale last-measured
       value, improving match quality during brief occlusions.

    3. **``ground_pt_source``** — every object dict carries
       ``"measured" | "predicted" | None`` so that downstream consumers
       (e.g. ``NearMissDetectorV30``) can weight confidence accordingly.

    Extra constructor parameters
    ----------------------------
    kf_q_pos : float (default 1e-4)
        Kalman process noise for position axes (depth-unit²/frame).
    kf_q_vel : float (default 1e-3)
        Kalman process noise for velocity axes (depth-unit²/frame³).
    kf_r_pos : float (default 5e-4)
        Kalman measurement noise (depth-unit²).  Increase for noisier scenes.
    """

    def __init__(
        self,
        max_disappeared: int   = 10,
        max_distance:    float = 150.0,
        min_hits:        int   = 1,
        max_distance_3d: float = 3.0,
        kf_q_pos:        float = 1e-4,
        kf_q_vel:        float = 1e-3,
        kf_r_pos:        float = 5e-4,
    ):
        super().__init__(
            max_disappeared=max_disappeared,
            max_distance=max_distance,
            min_hits=min_hits,
            max_distance_3d=max_distance_3d,
        )
        self._kf_q_pos = kf_q_pos
        self._kf_q_vel = kf_q_vel
        self._kf_r_pos = kf_r_pos
        self._kf: Dict[int, _KalmanFilter3D] = {}

    @property
    def tracker_name(self) -> str:
        return (
            f"CentroidTrackerV3(max_dist_3d={self.max_distance_3d}, "
            f"max_dist_2d={self.max_distance}, max_gone={self.max_disappeared}, "
            f"kf=true)"
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _register(self, detection: Dict[str, Any], frame_idx: int) -> None:
        """Register new track; add ground_pt_source and Kalman filter."""
        super()._register(detection, frame_idx)
        obj_id = self.next_id - 1
        gp     = detection.get("ground_pt")
        self.objects[obj_id]["ground_pt_source"] = "measured" if gp is not None else None
        if gp is not None:
            self._kf[obj_id] = _KalmanFilter3D(
                np.array(gp, dtype=np.float64),
                q_pos=self._kf_q_pos,
                q_vel=self._kf_q_vel,
                r_pos=self._kf_r_pos,
            )
            _log_v3.debug(
                "[fr=%d] register id=%d kf=init gp=(%.3f,%.3f,%.3f)",
                frame_idx, obj_id, *gp,
            )
        else:
            _log_v3.debug(
                "[fr=%d] register id=%d kf=none (no ground_pt)", frame_idx, obj_id
            )

    def _deregister(self, obj_id: int) -> None:
        super()._deregister(obj_id)
        self._kf.pop(obj_id, None)

    # ── public API ────────────────────────────────────────────────────────────

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_idx:  int,
        frame:      Optional[np.ndarray] = None,
    ) -> OrderedDict:
        """Per-pair hybrid cost + Kalman-prediction update."""

        # ── No detections: advance Kalman for all coasting tracks ────────────
        if not detections:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if obj_id not in self._confirmed_ids:
                    self._hit_streak[obj_id] = 0
                if obj_id in self._kf:
                    pred = self._kf[obj_id].predict()
                    self.objects[obj_id]["ground_pt"]        = tuple(pred.tolist())
                    self.objects[obj_id]["ground_pt_source"] = "predicted"
                    _log_v3.debug(
                        "[fr=%d] id=%d coast predict gp=(%.3f,%.3f,%.3f)",
                        frame_idx, obj_id, *pred,
                    )
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return self._active_objects()

        # ── No existing tracks: register all ─────────────────────────────────
        if not self.objects:
            for det in detections:
                self._register(det, frame_idx)
            return self._active_objects()

        existing_ids = list(self.objects.keys())
        N            = len(existing_ids)
        M            = len(detections)

        # ── Step 1: Kalman predict for every existing track ───────────────────
        # After this, ground_pt holds the predicted position (or None if no KF).
        # The matching in Step 2 uses these predicted positions.
        for obj_id in existing_ids:
            if obj_id in self._kf:
                pred = self._kf[obj_id].predict()
                self.objects[obj_id]["ground_pt"]        = tuple(pred.tolist())
                self.objects[obj_id]["ground_pt_source"] = "predicted"

        # ── Step 2: Build per-pair hybrid cost matrix (N × M) ────────────────
        existing_gps = [self.objects[eid].get("ground_pt") for eid in existing_ids]
        new_gps      = [d.get("ground_pt") for d in detections]

        e_has_3d = np.array([gp is not None for gp in existing_gps], dtype=bool)
        n_has_3d = np.array([gp is not None for gp in new_gps],      dtype=bool)
        mask_3d  = np.outer(e_has_3d, n_has_3d)   # (N, M) — True → use 3-D cost

        D = np.full((N, M), np.inf, dtype=np.float64)

        if mask_3d.any():
            gp_e = np.array(
                [gp if gp is not None else (0.0, 0.0, 0.0) for gp in existing_gps],
                dtype=np.float64,
            )
            gp_n = np.array(
                [gp if gp is not None else (0.0, 0.0, 0.0) for gp in new_gps],
                dtype=np.float64,
            )
            D_3d = cdist(gp_e, gp_n) / self.max_distance_3d   # normalised
            D    = np.where(mask_3d, D_3d, D)

        mask_2d = ~mask_3d
        if mask_2d.any():
            c_e  = np.array([self.objects[eid]["center"] for eid in existing_ids],
                            dtype=np.float64)
            c_n  = np.array([d["center"] for d in detections], dtype=np.float64)
            D_2d = cdist(c_e, c_n) / self.max_distance        # normalised
            D    = np.where(mask_2d, D_2d, D)

        _log_v3.debug(
            "[fr=%d] cost matrix %dx%d  3d-cells=%d  2d-cells=%d",
            frame_idx, N, M, int(mask_3d.sum()), int(mask_2d.sum()),
        )

        # ── Step 3: Greedy matching (normalised threshold = 1.0) ─────────────
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows: set = set()
        used_cols: set = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] >= 1.0:          # normalised cost threshold
                continue

            obj_id = existing_ids[row]
            det    = detections[col]
            cx, cy = det["center"]
            gp     = det.get("ground_pt")

            # 2-D fields
            self.objects[obj_id].update(
                {k: det[k] for k in ("class", "label", "bbox", "confidence", "center")}
            )
            self.trajectories[obj_id].append((frame_idx, cx, cy))
            self.objects[obj_id]["trajectory"] = self.trajectories[obj_id]
            self._class_history[obj_id].append(det["class"])
            self.objects[obj_id]["class"] = self._majority_class(obj_id)
            self.disappeared[obj_id] = 0

            # 3-D fields
            if gp is not None:
                # Kalman correction
                if obj_id in self._kf:
                    self._kf[obj_id].update(np.array(gp, dtype=np.float64))
                else:
                    # First measurement for this track → init Kalman
                    self._kf[obj_id] = _KalmanFilter3D(
                        np.array(gp, dtype=np.float64),
                        q_pos=self._kf_q_pos,
                        q_vel=self._kf_q_vel,
                        r_pos=self._kf_r_pos,
                    )
                self.objects[obj_id]["ground_pt"]        = gp
                self.objects[obj_id]["ground_pt_source"] = "measured"
                entry = (frame_idx, *gp)
                if obj_id not in self._trajectories_3d:
                    self._trajectories_3d[obj_id] = []
                self._trajectories_3d[obj_id].append(entry)
                self.objects[obj_id]["trajectory_3d"] = self._trajectories_3d[obj_id]
                _log_v3.debug(
                    "[fr=%d] match id=%d cost=%.3f [3d] gp=(%.3f,%.3f,%.3f)",
                    frame_idx, obj_id, D[row, col], *gp,
                )
            else:
                # Detection has no ground_pt; keep Kalman-predicted value if available
                if obj_id not in self._kf:
                    self.objects[obj_id]["ground_pt"]        = None
                    self.objects[obj_id]["ground_pt_source"] = None
                # else: predicted ground_pt set in Step 1 stays (source="predicted")
                if obj_id not in self._trajectories_3d:
                    self._trajectories_3d[obj_id] = []
                self.objects[obj_id]["trajectory_3d"] = self._trajectories_3d[obj_id]
                _log_v3.debug(
                    "[fr=%d] match id=%d cost=%.3f [2d%s]",
                    frame_idx, obj_id, D[row, col],
                    ", kf-pred" if obj_id in self._kf else ", no-gp",
                )

            # Tentative → confirmed state machine
            if obj_id not in self._confirmed_ids:
                self._hit_streak[obj_id] = self._hit_streak.get(obj_id, 0) + 1
                if self._hit_streak[obj_id] >= self.min_hits:
                    self._confirmed_ids.add(obj_id)

            used_rows.add(row)
            used_cols.add(col)

        # ── Step 4: Unmatched existing tracks → coast on Kalman prediction ────
        # ground_pt already holds the predicted value from Step 1.
        for row in range(N):
            if row not in used_rows:
                obj_id = existing_ids[row]
                self.disappeared[obj_id] += 1
                if obj_id not in self._confirmed_ids:
                    self._hit_streak[obj_id] = 0
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)

        # ── Step 5: Unmatched new detections → register ───────────────────────
        for col in range(M):
            if col not in used_cols:
                self._register(detections[col], frame_idx)

        return self._active_objects()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_tracker(name: str = "centroid", **kwargs) -> BaseTracker:
    """
    Create a tracker by name.

    Args:
        name:     'centroid' | 'centroid_v2' | 'centroid_v3' | 'bytetrack' | 'botsort'.
        **kwargs: Forwarded to the tracker constructor.

                  centroid:    max_disappeared (int), max_distance (float),
                               min_hits (int)
                  centroid_v2: same as centroid, plus
                               max_distance_3d (float, default 3.0) —
                               max 3-D ground distance for a match
                  centroid_v3: same as centroid_v2, plus
                               kf_q_pos (float, default 1e-4) — Kalman position noise
                               kf_q_vel (float, default 1e-3) — Kalman velocity noise
                               kf_r_pos (float, default 5e-4) — Kalman measurement noise
                  bytetrack:   track_thresh (float), track_buffer (int),
                               match_thresh (float), frame_rate (int),
                               min_hits (int)
                  botsort:     reid_weights (str|None), device (str), half (bool),
                               min_hits (int)

    Examples
    --------
    t = create_tracker('centroid', max_disappeared=15, max_distance=200)
    t = create_tracker('centroid_v2', max_distance_3d=3.0)
    t = create_tracker('centroid_v3', max_distance_3d=0.1, kf_r_pos=1e-3)
    t = create_tracker('bytetrack', track_buffer=30, match_thresh=0.8)
    t = create_tracker('botsort', device='cpu')
    """
    _registry = {
        "centroid":    CentroidTracker,
        "bytetrack":   ByteTracker,
        "byte":        ByteTracker,
        "botsort":     BoTSORTTracker,
        "bot-sort":    BoTSORTTracker,
        "centroid_v2": CentroidTrackerV2,
        "centroid_v3": CentroidTrackerV3,
    }
    key = name.lower().strip()
    if key not in _registry:
        raise ValueError(
            f"Unknown tracker '{name}'. "
            f"Choose from: 'centroid', 'centroid_v2', 'centroid_v3', "
            f"'bytetrack', 'botsort'."
        )
    return _registry[key](**kwargs)
