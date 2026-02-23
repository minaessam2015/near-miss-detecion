"""
scene_understanding.models
--------------------------
Model wrappers for road segmentation and monocular depth estimation.

This package supersedes the flat ``models.py`` file. Python 3 resolves
packages (directories with ``__init__.py``) before same-named modules, so
creating this directory is sufficient — no file deletion needed.

Public API
----------
Base ABCs:
    RoadSegmenter, DepthEstimator

Dummy stubs (no dependencies — for tests and smoke-runs):
    DummyRoadSegmenter, DummyDepthEstimator

Generic PyTorch wrappers (plug in any compatible model):
    TorchRoadSegmenter, TorchDepthEstimator

Production models:
    DepthAnythingV2Estimator   — Depth Anything V2 via HuggingFace
    SegFormerRoadSegmenter     — SegFormer via HuggingFace

Cache helpers:
    load_cached, save_cached
"""

from __future__ import annotations

# ── Re-export everything that was previously in the flat models.py ────────────
# (keeps all existing imports working without change)

import abc
import hashlib
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class RoadSegmenter(abc.ABC):
    """Abstract road / floor segmentation model.

    Implement :meth:`predict` to integrate any segmentation backbone.
    Input frame is BGR uint8 (OpenCV convention).
    """

    @abc.abstractmethod
    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Return (H, W) boolean road mask at the original frame resolution."""


class DepthEstimator(abc.ABC):
    """Abstract monocular depth estimation model.

    Depth values are *relative* unless the caller provides calibrated
    intrinsics and a known metric scale.
    """

    @abc.abstractmethod
    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Return (H, W) float32 depth map at the original frame resolution."""


# ---------------------------------------------------------------------------
# Dummy stubs
# ---------------------------------------------------------------------------

class DummyRoadSegmenter(RoadSegmenter):
    """Bottom ``road_fraction`` of the frame is labelled road (no model needed)."""

    def __init__(self, road_fraction: float = 0.60) -> None:
        self.road_fraction = road_fraction

    def predict(self, frame: np.ndarray) -> np.ndarray:
        H = frame.shape[0]
        mask = np.zeros((H, frame.shape[1]), dtype=bool)
        mask[int(H * (1.0 - self.road_fraction)):, :] = True
        return mask


class DummyDepthEstimator(DepthEstimator):
    """Synthetic depth that increases linearly from bottom to top of frame."""

    def __init__(
        self,
        min_depth: float = 0.5,
        max_depth: float = 10.0,
        noise_sigma: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.noise_sigma = noise_sigma
        self._rng = np.random.default_rng(seed)

    def predict(self, frame: np.ndarray) -> np.ndarray:
        H, W = frame.shape[:2]
        t = np.linspace(1.0, 0.0, H, dtype=np.float32).reshape(-1, 1)
        depth = self.min_depth + t * (self.max_depth - self.min_depth)
        depth = np.tile(depth, (1, W))
        if self.noise_sigma > 0:
            noise = self._rng.normal(0, self.noise_sigma, (H, W)).astype(np.float32)
            depth = np.clip(depth + noise, self.min_depth, self.max_depth)
        return depth


# ---------------------------------------------------------------------------
# Generic PyTorch wrappers (kept for backward-compatibility)
# ---------------------------------------------------------------------------

class TorchRoadSegmenter(RoadSegmenter):
    """Thin wrapper for any PyTorch segmentation model.

    Override ``_preprocess`` / ``_postprocess`` for your model's API.
    Defaults assume a ``transformers``-style model with a ``.logits`` output.
    """

    def __init__(
        self,
        model: object,
        road_class_ids: Optional[Sequence[int]] = None,
        device: str = "cpu",
        input_size: Optional[tuple] = (512, 512),
    ) -> None:
        self.model = model
        self.road_class_ids: Sequence[int] = road_class_ids if road_class_ids is not None else [0]
        self.device = device
        self.input_size = input_size

    def _preprocess(self, frame: np.ndarray):
        import torch
        img = frame if self.input_size is None else cv2.resize(frame, self.input_size)
        return (
            torch.from_numpy(img).float().permute(2, 0, 1).div(255.0)
            .unsqueeze(0).to(self.device)
        )

    def _postprocess(self, output: object, original_hw: tuple) -> np.ndarray:
        import torch
        logits = output.logits if hasattr(output, "logits") else output
        pred = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.int32)
        mask = np.zeros(pred.shape, dtype=bool)
        for cls_id in self.road_class_ids:
            mask |= pred == cls_id
        if mask.shape != (original_hw[0], original_hw[1]):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (original_hw[1], original_hw[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return mask

    def predict(self, frame: np.ndarray) -> np.ndarray:
        import torch
        H, W = frame.shape[:2]
        with torch.no_grad():
            output = self.model(self._preprocess(frame))
        return self._postprocess(output, (H, W))


class TorchDepthEstimator(DepthEstimator):
    """Thin wrapper for any PyTorch depth model.

    Override ``_preprocess`` / ``_postprocess`` for your model's API.
    """

    def __init__(
        self,
        model: object,
        device: str = "cpu",
        input_size: Optional[tuple] = (518, 518),
    ) -> None:
        self.model = model
        self.device = device
        self.input_size = input_size

    def _preprocess(self, frame: np.ndarray):
        import torch
        img = frame if self.input_size is None else cv2.resize(frame, self.input_size)
        return (
            torch.from_numpy(img).float().permute(2, 0, 1).div(255.0)
            .unsqueeze(0).to(self.device)
        )

    def _postprocess(self, output: object, original_hw: tuple) -> np.ndarray:
        import torch
        if hasattr(output, "predicted_depth"):
            depth = output.predicted_depth.squeeze().cpu().numpy()
        elif isinstance(output, torch.Tensor):
            depth = output.squeeze().cpu().numpy()
        elif isinstance(output, dict) and "depth" in output:
            depth = output["depth"].squeeze().cpu().numpy()
        else:
            depth = list(output)[0].squeeze().cpu().numpy()
        depth = depth.astype(np.float32)
        if depth.shape != (original_hw[0], original_hw[1]):
            depth = cv2.resize(
                depth, (original_hw[1], original_hw[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return depth

    def predict(self, frame: np.ndarray) -> np.ndarray:
        import torch
        H, W = frame.shape[:2]
        with torch.no_grad():
            output = self.model(self._preprocess(frame))
        return self._postprocess(output, (H, W))


# ---------------------------------------------------------------------------
# Production model imports  (lazy — only fail at instantiation, not import)
# ---------------------------------------------------------------------------

def _lazy_import_depth_anything():
    from .depth_anything_v2 import DepthAnythingV2Estimator
    return DepthAnythingV2Estimator


def _lazy_import_segformer():
    from .segformer_road import SegFormerRoadSegmenter
    return SegFormerRoadSegmenter


# Expose at package level so ``from scene_understanding.models import X`` works
try:
    from .depth_anything_v2 import DepthAnythingV2Estimator
except Exception:  # pragma: no cover
    DepthAnythingV2Estimator = None  # type: ignore[assignment,misc]

try:
    from .segformer_road import SegFormerRoadSegmenter
except Exception:  # pragma: no cover
    SegFormerRoadSegmenter = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_segmenter(kind: str = "dummy", **kwargs) -> RoadSegmenter:
    """Instantiate a road segmenter by name.

    Parameters
    ----------
    kind : ``'dummy'``, ``'torch'``, or ``'segformer'``.
    **kwargs : forwarded to the constructor.

    Examples
    --------
    >>> seg = create_segmenter("dummy", road_fraction=0.55)
    >>> seg = create_segmenter("segformer", device="cuda")
    >>> seg = create_segmenter("segformer",
    ...     model_id="nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    ...     seg_mode="cityscapes", device="cpu")
    """
    kinds: Dict[str, type] = {
        "dummy": DummyRoadSegmenter,
        "torch": TorchRoadSegmenter,
    }
    if SegFormerRoadSegmenter is not None:
        kinds["segformer"] = SegFormerRoadSegmenter

    if kind not in kinds:
        raise ValueError(f"Unknown segmenter kind '{kind}'. Choose from {list(kinds)}")
    return kinds[kind](**kwargs)


def create_depth_estimator(kind: str = "dummy", **kwargs) -> DepthEstimator:
    """Instantiate a depth estimator by name.

    Parameters
    ----------
    kind : ``'dummy'``, ``'torch'``, or ``'depth_anything_v2'``.
    **kwargs : forwarded to the constructor.

    Examples
    --------
    >>> est = create_depth_estimator("dummy", min_depth=1.0, max_depth=15.0)
    >>> est = create_depth_estimator("depth_anything_v2", device="cuda", fp16=True)
    """
    kinds: Dict[str, type] = {
        "dummy": DummyDepthEstimator,
        "torch": TorchDepthEstimator,
    }
    if DepthAnythingV2Estimator is not None:
        kinds["depth_anything_v2"] = DepthAnythingV2Estimator

    if kind not in kinds:
        raise ValueError(f"Unknown depth estimator kind '{kind}'. Choose from {list(kinds)}")
    return kinds[kind](**kwargs)


# ---------------------------------------------------------------------------
# NPZ cache helpers
# ---------------------------------------------------------------------------

def _cache_key(video_path: Union[str, Path], frame_idx: int, suffix: str) -> str:
    abs_path = str(Path(video_path).resolve())
    path_hash = hashlib.md5(abs_path.encode()).hexdigest()[:8]
    return f"{path_hash}_f{frame_idx:06d}_{suffix}.npz"


def load_cached(
    cache_dir: Union[str, Path],
    video_path: Union[str, Path],
    frame_idx: int,
    suffix: str,
) -> Optional[np.ndarray]:
    """Load a cached numpy array from disk; return ``None`` on cache miss."""
    path = Path(cache_dir) / _cache_key(video_path, frame_idx, suffix)
    if not path.exists():
        return None
    try:
        return np.load(str(path))["arr"]
    except Exception:
        return None


def save_cached(
    cache_dir: Union[str, Path],
    video_path: Union[str, Path],
    frame_idx: int,
    suffix: str,
    data: np.ndarray,
) -> None:
    """Save a numpy array to the .npz cache."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = Path(cache_dir) / _cache_key(video_path, frame_idx, suffix)
    np.savez_compressed(str(path), arr=data)


__all__ = [
    "RoadSegmenter", "DepthEstimator",
    "DummyRoadSegmenter", "DummyDepthEstimator",
    "TorchRoadSegmenter", "TorchDepthEstimator",
    "DepthAnythingV2Estimator", "SegFormerRoadSegmenter",
    "create_segmenter", "create_depth_estimator",
    "load_cached", "save_cached",
]
