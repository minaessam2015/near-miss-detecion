"""
Object Detection — pluggable detector abstraction.

Architecture:
  BaseDetector (ABC)              — common interface + stats tracking
  ├── UltralyticsDetector         — any Ultralytics detection model (YOLO26, YOLO12, …)
  │   └── UltralyticsSegDetector  — seg-model variant; adds 'contour' to each detection
  └── SAHIDetector                — sliced inference (best recall for small/far objects)

Factory:
  create_detector(model_name, backend, **kwargs) -> BaseDetector
  backend: "ultralytics" | "ultralytics-seg" | "sahi"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Detection dict schema
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "id":         None  (filled in by tracker)
  "class":      "vehicle" | "pedestrian"           ← unified category
  "label":      "car" | "truck" | "bus" | ...      ← fine-grained sub-type
  "bbox":       [x1, y1, x2, y2]                  ← original-frame pixels
  "confidence": float
  "center":     (cx, cy)
  "contour":    np.ndarray shape (N,1,2) | None    ← seg models only
}

Vehicle aggregation:
  All COCO vehicle classes (car=2, motorcycle=3, bus=5, truck=7, bicycle=1)
  are unified under class="vehicle" so downstream code doesn't need to
  enumerate individual labels. The original label is preserved in "label".

Per-class confidence thresholds:
  Pass conf={"vehicle": 0.4, "pedestrian": 0.3} to set different thresholds
  per category. A plain float applies the same threshold to all categories.

NMS:
  Within-class NMS is applied after inference. Adjust nms_iou (default 0.45)
  or set nms_iou=1.0 to disable.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Supported models (all auto-downloaded on first use via ultralytics)
CPU speed measured on COCO val, ONNX, 1-thread. mAP = COCO val2017 50-95.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOLO26 — ✅ RECOMMENDED for this project  (released Jan 2026)
  NMS-free end-to-end · ProgLoss+STAL (best small-object recall) · 43% faster CPU vs YOLO11
  Model           Params   mAP    CPU (ms)
  yolo26n.pt       2.4M   40.9     38.9      ← fastest, ok for quick tests
  yolo26s.pt       9.5M   48.6     87.2
  yolo26m.pt      20.4M   53.1    220.0
  yolo26l.pt      24.8M   55.0    286.2      ← best accuracy/speed for this use-case
  yolo26x.pt      55.7M   57.5    525.8      ← max accuracy, slow on CPU

  Seg variants (UltralyticsSegDetector):
  yolo26n-seg.pt · yolo26s-seg.pt · yolo26l-seg.pt · yolo26x-seg.pt

YOLO12  (released 2025 — attention-based, GPU-optimised, no CPU benchmark published)
  yolo12n.pt       2.6M   40.6      —
  yolo12l.pt      26.4M   53.7      —        ← lower mAP than yolo26l, skip on CPU

YOLO11  (released 2024)
  yolo11n.pt · yolo11s.pt · yolo11m.pt · yolo11l.pt · yolo11x.pt
  Seg: yolo11n-seg.pt · yolo11s-seg.pt · yolo11l-seg.pt

YOLOv8  (2023, battle-tested baseline)
  yolov8n.pt · yolov8s.pt · yolov8m.pt · yolov8l.pt · yolov8x.pt
  Seg: yolov8n-seg.pt · yolov8s-seg.pt · yolov8l-seg.pt

YOLOv9  yolov9c.pt · yolov9e.pt
YOLOv10 yolov10n.pt · yolov10s.pt · yolov10m.pt · yolov10b.pt · yolov10l.pt · yolov10x.pt

RT-DETR (transformer, strong on small objects, slow on CPU)
  rtdetr-l.pt · rtdetr-x.pt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# COCO class mapping — shared across all backends
# ---------------------------------------------------------------------------
# All COCO vehicle-like classes are unified under the "vehicle" category.
_VEHICLE_IDS    = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}
_PEDESTRIAN_IDS = {0: "person"}
_TARGET_IDS     = {**_VEHICLE_IDS, **_PEDESTRIAN_IDS}


# ---------------------------------------------------------------------------
# Within-class NMS
# ---------------------------------------------------------------------------

def _iou(b1: List[int], b2: List[int]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def _nms_detections(
    detections: List[Dict[str, Any]],
    iou_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Within-class Non-Maximum Suppression.

    Applied independently per category so a vehicle and a pedestrian with
    overlapping bboxes are both kept. Within the same category, lower-
    confidence duplicates are suppressed when IoU > iou_threshold.

    Args:
        detections:    List of detection dicts (needs 'class', 'bbox', 'confidence').
        iou_threshold: Boxes with IoU above this are suppressed (0.45 is typical).
                       Pass 1.0 to disable NMS entirely.
    """
    if len(detections) <= 1 or iou_threshold >= 1.0:
        return detections

    groups: Dict[str, List[Dict]] = defaultdict(list)
    for det in detections:
        groups[det["class"]].append(det)

    kept: List[Dict] = []
    for cls_dets in groups.values():
        cls_dets.sort(key=lambda d: d["confidence"], reverse=True)
        suppressed = [False] * len(cls_dets)
        for i, di in enumerate(cls_dets):
            if suppressed[i]:
                continue
            kept.append(di)
            for j in range(i + 1, len(cls_dets)):
                if not suppressed[j] and _iou(di["bbox"], cls_dets[j]["bbox"]) > iou_threshold:
                    suppressed[j] = True

    return kept


# ---------------------------------------------------------------------------
# Detection builder helper
# ---------------------------------------------------------------------------

def _build_detection(
    cls_id: int,
    conf: float,
    x1: int, y1: int, x2: int, y2: int,
) -> Optional[Dict[str, Any]]:
    """
    Build a detection dict for a target class. Returns None if not a target.

    Maps all vehicle sub-types to class="vehicle" and pedestrians to
    class="pedestrian". The original label (e.g. "car", "truck") is
    preserved in the "label" field for debugging/filtering.
    """
    if cls_id not in _TARGET_IDS:
        return None
    category = "pedestrian" if cls_id in _PEDESTRIAN_IDS else "vehicle"
    return {
        "id":         None,
        "class":      category,             # unified: "vehicle" | "pedestrian"
        "label":      _TARGET_IDS[cls_id],  # specific: "car" | "truck" | "person" …
        "bbox":       [x1, y1, x2, y2],
        "confidence": conf,
        "center":     ((x1 + x2) // 2, (y1 + y2) // 2),
    }


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseDetector(ABC):
    """
    Abstract detector interface with automatic stats tracking.

    Args:
        conf:       Confidence threshold. Either a single float applied to all
                    classes, or a dict mapping category names to thresholds:
                        {"vehicle": 0.4, "pedestrian": 0.3}
                    When a dict is provided, the minimum value is used as the
                    model-level threshold so no candidates are dropped too early.
        nms_iou:    IoU threshold for within-class NMS applied after inference.
                    Set to 1.0 to disable. Default 0.45.
        input_size: Frame resize dimension before inference.
    """

    def __init__(
        self,
        conf: Union[float, Dict[str, float]] = 0.4,
        nms_iou: Optional[float] = 0.45,
        input_size: int = 640,
    ):
        if isinstance(conf, dict):
            self._conf_map: Dict[str, float] = {k: float(v) for k, v in conf.items()}
        else:
            self._conf_map = {"vehicle": float(conf), "pedestrian": float(conf)}

        # Minimum across categories — used as the model-level filter so detections
        # that pass a per-class threshold are not discarded by the model first.
        self.conf = min(self._conf_map.values())

        self.nms_iou    = nms_iou
        self.input_size = input_size
        self._latencies: List[float] = []

    def _conf_for(self, category: str) -> float:
        """Return the confidence threshold for a given category name."""
        return self._conf_map.get(category, self.conf)

    def _postprocess(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply per-class confidence filtering then within-class NMS.
        Every concrete detect() implementation must call this before returning.
        """
        filtered = [
            d for d in detections
            if d["confidence"] >= self._conf_for(d["class"])
        ]
        if self.nms_iou is not None:
            return _nms_detections(filtered, self.nms_iou)
        return filtered

    # ── must override ────────────────────────────────────────────────────────

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference on a BGR frame. Returns list of detection dicts."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...

    @property
    def model_size_mb(self) -> float:
        """Weight file size in MB. Override if the path is known."""
        return float("nan")

    # ── stats API ────────────────────────────────────────────────────────────

    def _record(self, elapsed: float) -> None:
        self._latencies.append(elapsed)

    def reset_stats(self) -> None:
        """Clear all accumulated latency measurements."""
        self._latencies = []

    def reset_timing(self) -> None:  # backward-compat alias
        self.reset_stats()

    @property
    def avg_inference_ms(self) -> float:
        return mean(self._latencies) * 1000 if self._latencies else 0.0

    @property
    def throughput_fps(self) -> float:
        avg = mean(self._latencies) if self._latencies else None
        return (1.0 / avg) if avg else 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        """
        Summary of all performance stats.

        Keys: model_name, model_size_mb, n_requests, avg_latency_ms,
              median_latency_ms, p95_latency_ms, min_latency_ms,
              max_latency_ms, stdev_ms, throughput_fps.
        """
        base: Dict[str, Any] = {
            "model_name":        self.model_name,
            "model_size_mb":     self.model_size_mb,
            "n_requests":        0,
            "avg_latency_ms":    0.0,
            "median_latency_ms": 0.0,
            "p95_latency_ms":    0.0,
            "min_latency_ms":    0.0,
            "max_latency_ms":    0.0,
            "stdev_ms":          0.0,
            "throughput_fps":    0.0,
        }
        if not self._latencies:
            return base

        ms = [t * 1000 for t in self._latencies]
        sorted_ms = sorted(ms)
        p95_idx = max(0, int(len(sorted_ms) * 0.95) - 1)

        base.update({
            "n_requests":        len(ms),
            "avg_latency_ms":    round(mean(ms), 2),
            "median_latency_ms": round(median(ms), 2),
            "p95_latency_ms":    round(sorted_ms[p95_idx], 2),
            "min_latency_ms":    round(min(ms), 2),
            "max_latency_ms":    round(max(ms), 2),
            "stdev_ms":          round(stdev(ms) if len(ms) > 1 else 0.0, 2),
            "throughput_fps":    round(1000 / mean(ms), 2),
        })
        return base

    def print_stats(self) -> None:
        """Pretty-print the stats dict."""
        s = self.stats
        print(f"\n── Detector stats: {s['model_name']} ──────────────────────")
        print(f"  Model size      : {s['model_size_mb']:.1f} MB")
        print(f"  Requests        : {s['n_requests']}")
        print(f"  Avg latency     : {s['avg_latency_ms']:.1f} ms")
        print(f"  Median latency  : {s['median_latency_ms']:.1f} ms")
        print(f"  p95 latency     : {s['p95_latency_ms']:.1f} ms")
        print(f"  Min / Max       : {s['min_latency_ms']:.1f} / {s['max_latency_ms']:.1f} ms")
        print(f"  Std dev         : {s['stdev_ms']:.1f} ms")
        print(f"  Throughput      : {s['throughput_fps']:.1f} FPS")


# ---------------------------------------------------------------------------
# Ultralytics backend  (YOLO26, YOLO12, YOLO11, YOLOv8, YOLOv9, RT-DETR …)
# ---------------------------------------------------------------------------

class UltralyticsDetector(BaseDetector):
    """
    Detector backed by any Ultralytics *detection* model.

    Examples
    --------
    UltralyticsDetector("yolo26l.pt")
    UltralyticsDetector("yolo26l.pt", conf={"vehicle": 0.4, "pedestrian": 0.3})
    UltralyticsDetector("yolo26x.pt", conf=0.3, input_size=1280)
    UltralyticsDetector("rtdetr-l.pt")
    UltralyticsDetector("yolov8n.pt")
    """

    def __init__(
        self,
        model_name: str = "yolo26l.pt",
        conf: Union[float, Dict[str, float]] = 0.35,
        nms_iou: Optional[float] = 0.45,
        input_size: int = 640,
        device: str = "cpu",
    ):
        super().__init__(conf=conf, nms_iou=nms_iou, input_size=input_size)
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("pip install ultralytics")

        self._model_name = model_name
        self.device = device
        print(f"Loading {model_name} on {device} ...")
        self._model = YOLO(model_name)
        self._model.to(device)
        print("Model ready.")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_size_mb(self) -> float:
        for candidate in [
            self._model_name,
            os.path.join(os.path.expanduser("~"), ".cache", "ultralytics", self._model_name),
        ]:
            if os.path.exists(candidate):
                return os.path.getsize(candidate) / 1e6
        return float("nan")

    def _run_model(self, resized: np.ndarray):
        """Run the underlying YOLO model and return the first Results object."""
        return self._model(
            resized,
            conf=self.conf,   # minimum threshold; per-class filtering in _postprocess
            verbose=False,
            device=self.device,
        )[0]

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        h_orig, w_orig = frame.shape[:2]
        # resized = cv2.resize(frame, (self.input_size, self.input_size))
        resized = frame

        t0 = time.perf_counter()
        results = self._run_model(resized)
        self._record(time.perf_counter() - t0)

        if results.boxes is None:
            return []

        # sx = w_orig / self.input_size
        sx = 1
        # sy = h_orig / self.input_size
        sy = 1
        raw: List[Dict] = []
        for box in results.boxes:
            cls_id   = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            det = _build_detection(
                cls_id, conf_val,
                int(x1 * sx), int(y1 * sy),
                int(x2 * sx), int(y2 * sy),
            )
            if det:
                raw.append(det)

        return self._postprocess(raw)


# ---------------------------------------------------------------------------
# Segmentation variant
# ---------------------------------------------------------------------------

class UltralyticsSegDetector(UltralyticsDetector):
    """
    Extends UltralyticsDetector with instance segmentation contour output.

    Use with *-seg model variants:
      yolo26n-seg.pt · yolo26s-seg.pt · yolo26l-seg.pt · yolo26x-seg.pt
      yolo11n-seg.pt · yolo11s-seg.pt · yolo11l-seg.pt
      yolov8n-seg.pt · yolov8s-seg.pt · yolov8l-seg.pt

    Additional field in each detection dict
    ----------------------------------------
    "contour" : np.ndarray | None
        Shape (N, 1, 2), dtype int32. Polygon contour in *original-frame*
        pixel coordinates — directly compatible with OpenCV contour APIs:
          cv2.drawContours(frame, [det["contour"]], -1, color, 2)
          cv2.contourArea(det["contour"])
          cv2.fillPoly(mask, [det["contour"]], 255)
        None when the model did not produce a mask for this detection.

    Examples
    --------
    seg = UltralyticsSegDetector("yolo26l-seg.pt")
    seg = UltralyticsSegDetector(
        "yolo26l-seg.pt",
        conf={"vehicle": 0.4, "pedestrian": 0.3},
        nms_iou=0.5,
    )
    dets = seg.detect(frame)
    for d in dets:
        if d["contour"] is not None:
            cv2.drawContours(frame, [d["contour"]], -1, (0, 255, 0), 2)
            area_px = cv2.contourArea(d["contour"])
    """

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        h_orig, w_orig = frame.shape[:2]
        resized = cv2.resize(frame, (self.input_size, self.input_size))

        t0 = time.perf_counter()
        results = self._run_model(resized)
        self._record(time.perf_counter() - t0)

        if results.boxes is None:
            return []

        sx = w_orig / self.input_size
        sy = h_orig / self.input_size

        # masks.xy: list of (N_i, 2) float arrays in resized-image pixel space
        masks_xy = results.masks.xy if (results.masks is not None) else None

        raw: List[Dict] = []
        for idx, box in enumerate(results.boxes):
            cls_id   = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            det = _build_detection(
                cls_id, conf_val,
                int(x1 * sx), int(y1 * sy),
                int(x2 * sx), int(y2 * sy),
            )
            if det is None:
                continue

            # Scale contour points from resized-image space to original-frame space
            contour = None
            if masks_xy is not None and idx < len(masks_xy):
                pts = masks_xy[idx]           # (N, 2) float, resized coords
                if len(pts) >= 3:             # need at least a triangle
                    pts_scaled = pts * np.array([sx, sy], dtype=np.float32)
                    contour = pts_scaled.astype(np.int32).reshape(-1, 1, 2)

            det["contour"] = contour
            raw.append(det)

        return self._postprocess(raw)


# ---------------------------------------------------------------------------
# SAHI backend  (sliced inference — best for small/distant objects)
# ---------------------------------------------------------------------------

class SAHIDetector(BaseDetector):
    """
    Wraps any Ultralytics model with SAHI sliced inference.

    SAHI divides each frame into overlapping tiles, runs detection on each,
    then merges results with NMM. Significantly better recall for small and
    distant objects (far-away cars, motorcycles at the edge of the frame).

    Requires: pip install sahi

    Examples
    --------
    SAHIDetector("yolo26l.pt", slice_height=512, slice_width=512, overlap=0.2)
    SAHIDetector("yolo26l.pt", conf={"vehicle": 0.4, "pedestrian": 0.3})
    """

    def __init__(
        self,
        model_name: str = "yolo26l.pt",
        conf: Union[float, Dict[str, float]] = 0.35,
        nms_iou: float = 0.45,
        input_size: int = 640,
        slice_height: int = 512,
        slice_width:  int = 512,
        overlap: float = 0.2,
        device: str = "cpu",
    ):
        super().__init__(conf=conf, nms_iou=nms_iou, input_size=input_size)
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
        except ImportError:
            raise ImportError("pip install sahi")

        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        self._predict_fn = get_sliced_prediction

        self._model_name  = model_name
        self.slice_height = slice_height
        self.slice_width  = slice_width
        self.overlap      = overlap
        self.device       = device

        print(f"Loading SAHI wrapper: {model_name} ...")
        self._sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_name,
            confidence_threshold=self.conf,
            device=device,
        )
        print("SAHI model ready.")

    @property
    def model_name(self) -> str:
        return f"sahi/{self._model_name}"

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        result = self._predict_fn(
            rgb,
            detection_model=self._sahi_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap,
            overlap_width_ratio=self.overlap,
            perform_standard_pred=True,
            postprocess_type="GREEDYNMM",
            verbose=0,
        )
        self._record(time.perf_counter() - t0)

        raw: List[Dict] = []
        for obj in result.object_prediction_list:
            cls_id   = obj.category.id
            conf_val = float(obj.score.value)
            bbox     = obj.bbox
            det = _build_detection(
                cls_id, conf_val,
                int(bbox.minx), int(bbox.miny),
                int(bbox.maxx), int(bbox.maxy),
            )
            if det:
                raw.append(det)

        return self._postprocess(raw)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_detector(
    model_name: str = "yolo26l.pt",
    backend: str = "ultralytics",
    **kwargs,
) -> BaseDetector:
    """
    Create a detector by name and backend.

    Args:
        model_name: Weight file, e.g. "yolo26l.pt", "yolo26l-seg.pt".
        backend:    One of:
                      "ultralytics"     — standard detection model (default)
                      "ultralytics-seg" — segmentation model; adds 'contour' to dets
                      "sahi"            — sliced inference for small/far objects
        **kwargs:   Forwarded to the detector constructor.
                    Common:    conf, nms_iou, input_size, device
                    SAHI-only: slice_height, slice_width, overlap

    Examples
    --------
    det = create_detector("yolo26l.pt")
    det = create_detector("yolo26l.pt", conf={"vehicle": 0.4, "pedestrian": 0.3})
    det = create_detector("yolo26l.pt", nms_iou=0.5)
    det = create_detector("yolo26l-seg.pt", backend="ultralytics-seg")
    det = create_detector("yolo26l.pt", backend="sahi", slice_height=512)
    det = create_detector("rtdetr-l.pt", conf=0.4)
    """
    _backends = {
        "ultralytics":     UltralyticsDetector,
        "ultralytics-seg": UltralyticsSegDetector,
        "sahi":            SAHIDetector,
    }
    if backend not in _backends:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from: {list(_backends)}"
        )
    return _backends[backend](model_name=model_name, **kwargs)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def draw_detections(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    draw_contours: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes (and segmentation contours if present) on a copy of the frame.

    Vehicles=blue, Pedestrians=green. Contours are drawn when available and
    draw_contours=True (only seg-model detections carry a 'contour' field).

    Args:
        frame:          BGR image.
        detections:     Detection dicts from any detector.
        draw_contours:  Draw segmentation contour when det['contour'] is not None.
    """
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = (255, 100, 0) if det["class"] == "vehicle" else (0, 200, 50)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out, f"{det['label']} {det['confidence']:.2f}",
            (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

        if draw_contours and det.get("contour") is not None:
            cv2.drawContours(out, [det["contour"]], -1, color, 1)

    return out


# ---------------------------------------------------------------------------
# Backward-compat alias  (existing code that imports ObjectDetector still works)
# ---------------------------------------------------------------------------
ObjectDetector = UltralyticsDetector
