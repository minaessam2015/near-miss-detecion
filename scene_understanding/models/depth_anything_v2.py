"""
scene_understanding.models.depth_anything_v2
---------------------------------------------
Production-grade wrapper for Depth Anything V2 (DAv2) monocular depth.

Model IDs (HuggingFace Hub)
---------------------------
Small  (~24 M params, fastest, CPU-friendly):
    ``depth-anything/Depth-Anything-V2-Small-hf``

Base   (~97 M params, balanced):
    ``depth-anything/Depth-Anything-V2-Base-hf``

Large  (~335 M params, highest quality):
    ``depth-anything/Depth-Anything-V2-Large-hf``

Local checkpoint
----------------
Pass a local directory path to ``model_id`` — the directory must contain
``config.json`` and the model weights (``model.safetensors`` or ``pytorch_model.bin``).

Depth scale
-----------
DAv2 outputs *inverse* relative depth (larger value = closer to camera).
By default this wrapper **inverts the raw output** so that larger values
correspond to farther distances, which is the convention expected by the
plane-fitting pipeline.  Set ``invert=False`` to get the raw model output.

Performance
-----------
- ``resize_long_side`` controls the maximum dimension sent to the model.
  768–1024 px gives a good speed/quality trade-off.
- ``fp16=True`` halves memory and speeds up inference on CUDA.
- The model is moved to the specified device once during construction; each
  ``predict()`` call uses ``torch.inference_mode()`` and does not rebuild
  the computation graph.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from . import DepthEstimator

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model identifier
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_long_side(
    frame: np.ndarray,
    long_side: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Resize so that max(H, W) == long_side, preserving aspect ratio.

    Returns (resized_frame, (original_H, original_W)).
    """
    H, W = frame.shape[:2]
    original_hw = (H, W)
    max_dim = max(H, W)
    if max_dim <= long_side:
        return frame, original_hw
    scale = long_side / max_dim
    new_h = max(1, round(H * scale))
    new_w = max(1, round(W * scale))
    # Use INTER_AREA for down-scaling (less aliasing)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA), original_hw


def _to_rgb_pil(bgr_frame: np.ndarray):
    """Convert BGR uint8 ndarray → PIL Image (RGB)."""
    from PIL import Image
    return Image.fromarray(bgr_frame[:, :, ::-1])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DepthAnythingV2Estimator(DepthEstimator):
    """Depth Anything V2 depth estimator.

    Parameters
    ----------
    model_id : str
        HuggingFace Hub model ID or path to a local checkpoint directory.
        Defaults to ``depth-anything/Depth-Anything-V2-Small-hf``.
    device : str
        ``'cuda'``, ``'cpu'``, or ``'auto'`` (uses CUDA if available).
    fp16 : bool
        Use half-precision (float16) when running on CUDA.  Ignored on CPU
        because float16 arithmetic on CPU is slower than float32.
    resize_long_side : int
        Resize the input so that max(H, W) equals this value before
        inference.  Set to ``None`` to pass the original resolution.
    invert : bool
        DAv2 produces *inverse* relative depth (larger = closer).  When
        ``True`` (default) the output is inverted so larger = farther.

    Example
    -------
    >>> est = DepthAnythingV2Estimator(device="cuda", fp16=True)
    >>> depth = est.predict(bgr_frame)   # (H, W) float32

    Plugging in a local checkpoint::

        est = DepthAnythingV2Estimator(
            model_id="/path/to/depth_anything_v2_large",
            device="cpu",
        )
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        fp16: bool = True,
        resize_long_side: int = 768,
        invert: bool = True,
    ) -> None:
        self.model_id = model_id
        self.resize_long_side = resize_long_side
        self.invert = invert

        self._device = self._resolve_device(device)
        self._fp16 = fp16 and self._device == "cuda"
        self._model = None
        self._processor = None
        self._loaded = False

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load model and processor on first use."""
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise ImportError(
                "Depth Anything V2 requires 'transformers' and 'torch'. "
                "Install them with: pip install transformers torch"
            ) from e

        _logger.info("Loading Depth Anything V2 from '%s' …", self.model_id)

        dtype = torch.float16 if self._fp16 else torch.float32

        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        ).to(self._device).eval()

        _logger.info(
            "DepthAnythingV2 ready — device=%s  fp16=%s  resize_long_side=%s",
            self._device, self._fp16, self.resize_long_side,
        )
        self._loaded = True

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Run DAv2 on a single BGR frame.

        Parameters
        ----------
        frame : (H, W, 3) uint8 BGR image.

        Returns
        -------
        depth : (H, W) float32 depth map aligned to original resolution.
            Larger values = farther from camera (after inversion).
            Values are in relative (arbitrary) units.
        """
        self._ensure_loaded()

        import torch
        import torch.nn.functional as F

        original_hw = frame.shape[:2]

        # ── Resize for inference ───────────────────────────────────────────────
        infer_frame = frame
        if self.resize_long_side is not None:
            infer_frame, _ = _resize_long_side(frame, self.resize_long_side)

        # ── Preprocess ────────────────────────────────────────────────────────
        pil_image = _to_rgb_pil(infer_frame)
        inputs = self._processor(images=pil_image, return_tensors="pt")

        # Move inputs to device / dtype
        inputs = {
            k: (v.to(self._device, dtype=torch.float16)
                if self._fp16 and v.dtype == torch.float32
                else v.to(self._device))
            for k, v in inputs.items()
        }

        # ── Inference ─────────────────────────────────────────────────────────
        with torch.inference_mode():
            outputs = self._model(**inputs)

        # ``predicted_depth`` shape: (1, H_inf, W_inf)
        predicted_depth: torch.Tensor = outputs.predicted_depth

        # ── Resize to original resolution ────────────────────────────────────
        depth_tensor = F.interpolate(
            predicted_depth.unsqueeze(1).float(),
            size=original_hw,
            mode="bicubic",
            align_corners=False,
        ).squeeze()  # (H, W)

        depth = depth_tensor.cpu().numpy().astype(np.float32)

        # ── Optionally invert (raw DAv2 is inverse depth) ─────────────────────
        if self.invert:
            # Avoid division by zero: clip raw values to ≥ 1% of the 1st
            # percentile so no single extreme pixel dominates the scale.
            eps = float(np.percentile(depth[depth > 0], 1) * 0.01) if (depth > 0).any() else 1e-3
            depth = 1.0 / np.maximum(depth, eps)
            # Robust normalisation: scale so the 95th-percentile equals 10.0.
            # The previous approach used d_max = 1/eps (dominated by the single
            # closest pixel in the frame), which collapsed road depths to
            # ~0.001 depth units and made the ground plane appear at the camera
            # origin, breaking cube projection and grid visualisation.
            p95 = float(np.percentile(depth, 95))
            if p95 > 0:
                depth = depth / p95 * 10.0

        return depth

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def warmup(self, height: int = 480, width: int = 640) -> None:
        """Run one dummy inference to JIT-compile and warm up CUDA kernels.

        Useful before timing benchmarks.
        """
        self._ensure_loaded()
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        self.predict(dummy)
        _logger.info("DepthAnythingV2 warmup complete.")

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        return (
            f"DepthAnythingV2Estimator(model_id='{self.model_id}', "
            f"device='{self._device}', fp16={self._fp16}, "
            f"resize_long_side={self.resize_long_side}, invert={self.invert})"
        )
