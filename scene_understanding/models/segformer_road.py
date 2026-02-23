"""
scene_understanding.models.segformer_road
------------------------------------------
SegFormer-based road / floor segmenter.

Two operating modes
-------------------

**Generic mode** (``seg_mode='cityscapes'`` or ``seg_mode='ade20k'``)
    Uses an off-the-shelf SegFormer checkpoint trained on Cityscapes or
    ADE20K and maps the relevant semantic classes to a binary road/floor mask.

    Recommended defaults:

    +------------------+-------------------------------------------------+
    | ``seg_mode``     | ``model_id``                                    |
    +==================+=================================================+
    | ``'cityscapes'`` | ``nvidia/segformer-b2-finetuned-cityscapes-1024-1024`` |
    +------------------+-------------------------------------------------+
    | ``'ade20k'``     | ``nvidia/segformer-b2-finetuned-ade-512-512``   |
    +------------------+-------------------------------------------------+

    Class→road mappings used:

    *Cityscapes 19-class labels:*
        - 0  = road  ✓
        - 1  = sidewalk  (included when ``include_sidewalk=True``)

    *ADE20K 150-class labels:*
        - 6  = road, path
        - 3  = floor, flooring
        - 11 = earth, ground
        - 29 = field

**Binary fine-tune mode** (``seg_mode='binary'``)
    Loads a checkpoint with 2 output classes (background=0, road=1).
    Suitable for a model fine-tuned specifically for road/floor binary
    segmentation.  Pass the local directory to ``model_id``.

Post-processing
---------------
After semantic → binary conversion the following are applied in order:

1. Optional morphological dilation (``road_dilate_px``).
2. Removal of small connected components (``min_area_px``).

Both steps are skipped when the respective parameter is zero.

Local checkpoints
-----------------
Pass a local directory path (instead of a HuggingFace Hub ID) to
``model_id`` — the directory must contain ``config.json`` and model weights.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import RoadSegmenter

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model IDs
# ---------------------------------------------------------------------------

CITYSCAPES_MODEL_ID = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
ADE20K_MODEL_ID     = "nvidia/segformer-b2-finetuned-ade-512-512"

# Class IDs that are considered "road / floor" per dataset
_CITYSCAPES_ROAD_CLASSES = [0]         # road
_CITYSCAPES_SIDEWALK_CLASSES = [1]     # sidewalk (optional)
_ADE20K_ROAD_CLASSES = [6, 3, 11, 29] # road/path, floor, earth/ground, field


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Morphological dilation with an elliptical kernel."""
    if radius_px <= 0:
        return mask
    ksize = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Remove connected components whose pixel area is below ``min_area_px``."""
    if min_area_px <= 0 or not mask.any():
        return mask
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    result = np.zeros_like(mask_u8)
    for i in range(1, n_labels):          # label 0 = background
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area_px:
            result[labels == i] = 1
    return result.astype(bool)


def _postprocess_mask(
    mask: np.ndarray,
    road_dilate_px: int,
    min_area_px: int,
) -> np.ndarray:
    """Apply dilation then small-component removal."""
    if road_dilate_px > 0:
        mask = _dilate_mask(mask, road_dilate_px)
    if min_area_px > 0:
        mask = _remove_small_components(mask, min_area_px)
    return mask


def _resize_long_side(
    frame: np.ndarray,
    long_side: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    H, W = frame.shape[:2]
    max_dim = max(H, W)
    if max_dim <= long_side:
        return frame, (H, W)
    scale = long_side / max_dim
    new_h = max(1, round(H * scale))
    new_w = max(1, round(W * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA), (H, W)


def _to_rgb_pil(bgr_frame: np.ndarray):
    from PIL import Image
    return Image.fromarray(bgr_frame[:, :, ::-1])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SegFormerRoadSegmenter(RoadSegmenter):
    """SegFormer road / floor segmenter.

    Parameters
    ----------
    model_id : str
        HuggingFace Hub model ID or path to a local checkpoint directory.
        Defaults depend on ``seg_mode``:
        - ``'cityscapes'`` → ``CITYSCAPES_MODEL_ID``
        - ``'ade20k'``     → ``ADE20K_MODEL_ID``
        - ``'binary'``     → **must be provided** (no sensible default).
    seg_mode : str
        ``'cityscapes'``, ``'ade20k'``, or ``'binary'``.
    device : str
        ``'cuda'``, ``'cpu'``, or ``'auto'``.
    fp16 : bool
        Use float16 on CUDA.
    resize_long_side : int
        Resize input so that max(H, W) = this before inference.
    road_dilate_px : int
        Morphological dilation radius applied to the final mask.
    min_area_px : int
        Remove connected components smaller than this area.
    include_sidewalk : bool
        Cityscapes mode only — also include sidewalk pixels (class 1).
    custom_road_class_ids : list[int]
        Override the default class mapping for the chosen dataset.
        Takes precedence over ``include_sidewalk``.

    Examples
    --------
    >>> seg = SegFormerRoadSegmenter(seg_mode="cityscapes", device="cpu")
    >>> mask = seg.predict(bgr_frame)   # (H, W) bool

    Custom local binary checkpoint::

        seg = SegFormerRoadSegmenter(
            model_id="/path/to/binary_segformer",
            seg_mode="binary",
            device="cuda",
        )
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        seg_mode: str = "cityscapes",
        device: str = "auto",
        fp16: bool = True,
        resize_long_side: int = 1024,
        road_dilate_px: int = 5,
        min_area_px: int = 500,
        include_sidewalk: bool = False,
        custom_road_class_ids: Optional[Sequence[int]] = None,
    ) -> None:
        if seg_mode not in ("cityscapes", "ade20k", "binary"):
            raise ValueError(
                f"Unknown seg_mode '{seg_mode}'. Choose from "
                "'cityscapes', 'ade20k', 'binary'."
            )
        if seg_mode == "binary" and model_id is None:
            raise ValueError(
                "seg_mode='binary' requires an explicit model_id "
                "(local checkpoint directory or HuggingFace repo ID)."
            )

        # Resolve default model ID
        if model_id is None:
            model_id = (
                CITYSCAPES_MODEL_ID if seg_mode == "cityscapes" else ADE20K_MODEL_ID
            )

        self.model_id = model_id
        self.seg_mode = seg_mode
        self.resize_long_side = resize_long_side
        self.road_dilate_px = road_dilate_px
        self.min_area_px = min_area_px
        self.include_sidewalk = include_sidewalk

        self._road_class_ids: List[int] = self._build_class_ids(
            seg_mode, include_sidewalk, custom_road_class_ids
        )

        self._device = self._resolve_device(device)
        self._fp16 = fp16 and self._device == "cuda"
        self._model = None
        self._processor = None
        self._loaded = False

    # ── Lazy loading ──────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import (
                SegformerForSemanticSegmentation,
                SegformerImageProcessor,
            )
        except ImportError as e:
            raise ImportError(
                "SegFormerRoadSegmenter requires 'transformers' and 'torch'. "
                "Install them with: pip install transformers torch"
            ) from e

        _logger.info(
            "Loading SegFormer from '%s' (mode=%s) …", self.model_id, self.seg_mode
        )

        dtype = torch.float16 if self._fp16 else torch.float32

        self._processor = SegformerImageProcessor.from_pretrained(self.model_id)
        self._model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            ignore_mismatched_sizes=(self.seg_mode == "binary"),
        ).to(self._device).eval()

        _logger.info(
            "SegFormer ready — device=%s  fp16=%s  road_classes=%s",
            self._device, self._fp16, self._road_class_ids,
        )
        self._loaded = True

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Run SegFormer and return a binary road/floor mask.

        Parameters
        ----------
        frame : (H, W, 3) uint8 BGR image.

        Returns
        -------
        mask : (H, W) boolean ndarray — True where road / floor.
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
        inputs = {
            k: (v.to(self._device, dtype=torch.float16)
                if self._fp16 and v.dtype == torch.float32
                else v.to(self._device))
            for k, v in inputs.items()
        }

        # ── Inference ─────────────────────────────────────────────────────────
        with torch.inference_mode():
            outputs = self._model(**inputs)

        # logits: (1, num_classes, H/4, W/4)
        logits: torch.Tensor = outputs.logits.float()

        # ── Upsample to original resolution ──────────────────────────────────
        upsampled = F.interpolate(
            logits,
            size=original_hw,
            mode="bilinear",
            align_corners=False,
        )
        pred = upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.int32)

        # ── Map to binary road mask ───────────────────────────────────────────
        mask = np.zeros(pred.shape, dtype=bool)
        for cls_id in self._road_class_ids:
            mask |= pred == cls_id

        # ── Post-processing ───────────────────────────────────────────────────
        mask = _postprocess_mask(mask, self.road_dilate_px, self.min_area_px)

        return mask

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_class_ids(
        seg_mode: str,
        include_sidewalk: bool,
        custom_ids: Optional[Sequence[int]],
    ) -> List[int]:
        """Resolve the list of class IDs to treat as road/floor."""
        if custom_ids is not None:
            return list(custom_ids)
        if seg_mode == "cityscapes":
            ids = list(_CITYSCAPES_ROAD_CLASSES)
            if include_sidewalk:
                ids += _CITYSCAPES_SIDEWALK_CLASSES
            return ids
        if seg_mode == "ade20k":
            return list(_ADE20K_ROAD_CLASSES)
        # binary: class 1 = road
        return [1]

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def set_road_classes(self, class_ids: Sequence[int]) -> None:
        """Override the road class IDs at runtime without reloading the model.

        Useful when switching between road-only and road+sidewalk mapping.

        Parameters
        ----------
        class_ids : sequence of int class indices to treat as road.
        """
        self._road_class_ids = list(class_ids)
        _logger.info("Road class IDs updated to %s", self._road_class_ids)

    def warmup(self, height: int = 480, width: int = 640) -> None:
        """Run one dummy inference to warm up CUDA kernels."""
        self._ensure_loaded()
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        self.predict(dummy)
        _logger.info("SegFormer warmup complete.")

    @property
    def device(self) -> str:
        return self._device

    @property
    def road_class_ids(self) -> List[int]:
        return list(self._road_class_ids)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        return (
            f"SegFormerRoadSegmenter(model_id='{self.model_id}', "
            f"seg_mode='{self.seg_mode}', device='{self._device}', "
            f"fp16={self._fp16}, road_classes={self._road_class_ids})"
        )
