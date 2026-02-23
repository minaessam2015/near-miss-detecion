"""
Task 1.7 (Bonus) — Real-Time Optimization
ONNX export, ONNX-based inference, and FPS benchmarking.
"""

import time
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_to_onnx(yolo_model, output_path: str = "outputs/yolov8n.onnx") -> str:
    """
    Export a loaded Ultralytics YOLO model to ONNX format.

    Args:
        yolo_model: A loaded `ultralytics.YOLO` instance.
        output_path: Destination path for the .onnx file.

    Returns:
        Absolute path to the exported ONNX file.
    """
    import os

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    exported = yolo_model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    print(f"ONNX model exported to: {exported}")
    return str(exported)


# ---------------------------------------------------------------------------
# ONNX Detector (same interface as detector.ObjectDetector)
# ---------------------------------------------------------------------------

# Same COCO class mapping as detector.py
_VEHICLE_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}
_PEDESTRIAN_IDS = {0: "person"}
_TARGET_IDS = {**_VEHICLE_IDS, **_PEDESTRIAN_IDS}


class ONNXDetector:
    """
    CPU inference using onnxruntime.
    Offers the same `.detect()` interface as `ObjectDetector`.
    """

    def __init__(
        self,
        onnx_path: str,
        conf: float = 0.4,
        input_size: int = 640,
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")

        self.conf = conf
        self.input_size = input_size
        self._inference_times: List[float] = []

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"ONNXDetector loaded: {onnx_path}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize, convert to RGB float32, normalize to [0,1], add batch dim."""
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC → CHW
        blob = np.expand_dims(blob, axis=0)   # CHW → NCHW
        return blob

    def _postprocess(
        self,
        output: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> List[Dict[str, Any]]:
        """
        Parse YOLOv8 ONNX output tensor.
        Shape: (1, 84, N) where 84 = 4 bbox + 80 classes, N = anchors.
        """
        preds = output[0]  # shape (84, N) or (1, 84, N)
        if preds.ndim == 3:
            preds = preds[0]  # (84, N)

        detections = []
        n_anchors = preds.shape[1]
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        for i in range(n_anchors):
            cx, cy, bw, bh = preds[:4, i]
            class_scores = preds[4:, i]
            cls_id = int(np.argmax(class_scores))
            conf = float(class_scores[cls_id])

            if conf < self.conf:
                continue
            if cls_id not in _TARGET_IDS:
                continue

            # Convert from center format to xyxy, scale to original size
            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            x2 = int((cx + bw / 2) * scale_x)
            y2 = int((cy + bh / 2) * scale_y)

            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, orig_w)
            y2 = min(y2, orig_h)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            category = "pedestrian" if cls_id in _PEDESTRIAN_IDS else "vehicle"
            detections.append({
                "id": None,
                "class": category,
                "label": _TARGET_IDS[cls_id],
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "center": (center_x, center_y),
            })

        return detections

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        h_orig, w_orig = frame.shape[:2]
        t0 = time.perf_counter()
        blob = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: blob})
        t1 = time.perf_counter()
        self._inference_times.append(t1 - t0)
        return self._postprocess(outputs[0], w_orig, h_orig)

    @property
    def avg_inference_ms(self) -> float:
        if not self._inference_times:
            return 0.0
        return (sum(self._inference_times) / len(self._inference_times)) * 1000

    def reset_timing(self) -> None:
        self._inference_times = []


# ---------------------------------------------------------------------------
# Benchmarking utility
# ---------------------------------------------------------------------------

def benchmark_pipeline(
    video_path: str,
    configs: List[Dict[str, Any]],
    n_frames: int = 100,
) -> pd.DataFrame:
    """
    Benchmark end-to-end detection throughput for different configurations.

    Args:
        video_path: Path to the input video file.
        configs: List of config dicts. Each dict must have:
            - "stride": int — process every N-th frame.
            - "input_size": int — resize dimension (e.g. 320 or 640).
            - "backend": str — "pytorch" or "onnx".
            - "onnx_path": str (required if backend == "onnx").
        n_frames: Number of video frames to read per config (for speed).

    Returns:
        DataFrame with columns: config_name, stride, input_size, backend,
        fps, avg_inference_ms, frames_processed.
    """
    rows = []

    for cfg in configs:
        stride = cfg.get("stride", 2)
        input_size = cfg.get("input_size", 640)
        backend = cfg.get("backend", "pytorch")
        config_name = f"{backend}_s{stride}_r{input_size}"

        print(f"\nBenchmarking: {config_name} ...")

        # Build detector
        if backend == "pytorch":
            from src.detector import ObjectDetector
            detector = ObjectDetector(conf=0.4, input_size=input_size)
        elif backend == "onnx":
            onnx_path = cfg.get("onnx_path", "outputs/yolov8n.onnx")
            detector = ONNXDetector(onnx_path=onnx_path, conf=0.4, input_size=input_size)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        detector.reset_timing()

        cap = cv2.VideoCapture(video_path)
        t_start = time.perf_counter()
        frames_processed = 0
        frame_idx = 0

        while frames_processed < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                detector.detect(frame)
                frames_processed += 1
            frame_idx += 1

        t_end = time.perf_counter()
        cap.release()

        elapsed = t_end - t_start
        fps = frames_processed / elapsed if elapsed > 0 else 0.0

        rows.append({
            "config_name": config_name,
            "stride": stride,
            "input_size": input_size,
            "backend": backend,
            "fps": round(fps, 2),
            "avg_inference_ms": round(detector.avg_inference_ms, 1),
            "frames_processed": frames_processed,
        })
        print(f"  → {fps:.1f} FPS, avg inference: {detector.avg_inference_ms:.1f} ms/frame")

    return pd.DataFrame(rows)


def recommend_config(benchmark_df: pd.DataFrame) -> str:
    """
    Return the recommended config name (best FPS ≥ 15, or best available).
    """
    real_time = benchmark_df[benchmark_df["fps"] >= 15]
    if real_time.empty:
        best = benchmark_df.loc[benchmark_df["fps"].idxmax()]
    else:
        # Among real-time configs, prefer lower stride (better accuracy)
        best = real_time.sort_values(["stride", "fps"], ascending=[True, False]).iloc[0]
    return best["config_name"]
