"""
scene_understanding.models
---------------------------
Model wrappers for road segmentation and monocular depth estimation.

Design
------
Each model type is expressed as an abstract base class with a single
``predict`` method so the rest of the pipeline is model-agnostic.
Two concrete stubs are provided for unit tests and quick-start demos:

* :class:`DummyRoadSegmenter`  — labels the bottom 60 % of the frame as road.
* :class:`DummyDepthEstimator` — synthesises a depth map that increases
  monotonically from the bottom of the frame to the top (simulates a
  forward-looking camera where closer objects are at the bottom).

Two thin PyTorch wrappers let you plug in any compatible model:

* :class:`TorchRoadSegmenter`  — segmentation models (SegFormer, …).
* :class:`TorchDepthEstimator` — depth models (DepthAnything, ZoeDepth, …).

All wrappers optionally resize frames before inference and restore the
original resolution in post-processing.

Cache helpers
-------------
:func:`load_cached` / :func:`save_cached`  provide a simple .npz cache
keyed on (video path, frame index, model suffix).  Re-running the same
video skips all inference.
"""

from __future__ import annotations

import abc
import hashlib
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base classes
# ─────────────────────────────────────────────────────────────────────────────

class RoadSegmenter(abc.ABC):
    """Abstract road / floor segmentation model.

    Subclass this and implement :meth:`predict` to integrate any
    segmentation backbone into the scene-understanding pipeline.
    """

    @abc.abstractmethod
    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Run segmentation on a single BGR (or RGB) frame.

        Parameters
        ----------
        frame : (H, W, 3) uint8 image.

        Returns
        -------
        mask : (H, W) boolean ndarray where ``True`` marks road / floor
               pixels in the *original* frame resolution.
        """


class DepthEstimator(abc.ABC):
    """Abstract monocular depth estimation model.

    Subclass this and implement :meth:`predict` to integrate any depth
    backbone into the scene-understanding pipeline.

    Notes
    -----
    The pipeline treats depth values as *relative* unless the caller
    explicitly provides calibrated intrinsics and a known metric scale.
    The pipeline never claims real-world metres from a relative model.
    """

    @abc.abstractmethod
    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Run depth estimation on a single BGR (or RGB) frame.

        Parameters
        ----------
        frame : (H, W, 3) uint8 image.

        Returns
        -------
        depth : (H, W) float32 ndarray of depth values in the *original*
                frame resolution.  Relative-scale models typically output
                values in the range [0.1, 10] or similar.
        """


# ─────────────────────────────────────────────────────────────────────────────
# Dummy stubs for unit-testing and smoke tests
# ─────────────────────────────────────────────────────────────────────────────

class DummyRoadSegmenter(RoadSegmenter):
    """Stub segmenter: marks the bottom ``road_fraction`` of the image as road.

    Suitable for pipeline smoke-tests without a real model.

    Parameters
    ----------
    road_fraction : fraction of image height (from the bottom) labelled road.
                    Default 0.60 mimics a typical forward-facing traffic cam.
    """

    def __init__(self, road_fraction: float = 0.60) -> None:
        self.road_fraction = road_fraction

    def predict(self, frame: np.ndarray) -> np.ndarray:
        H = frame.shape[0]
        mask = np.zeros((H, frame.shape[1]), dtype=bool)
        mask[int(H * (1.0 - self.road_fraction)):, :] = True
        return mask


class DummyDepthEstimator(DepthEstimator):
    """Stub depth estimator: depth increases linearly from bottom to top.

    Rationale: in a forward-facing traffic camera, objects at the bottom of
    the frame are physically closer to the camera than those at the top.

    The output is perturbed with small Gaussian noise to avoid degenerate
    RANSAC inputs (perfectly co-planar points due to linear depth).

    Parameters
    ----------
    min_depth   : depth value at the bottom of the image (closest).
    max_depth   : depth value at the top of the image (farthest).
    noise_sigma : standard deviation of additive Gaussian noise.
    seed        : random seed for reproducibility (``None`` = random).
    """

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
        # Row index 0 (top) → max_depth; row index H-1 (bottom) → min_depth
        t = np.linspace(1.0, 0.0, H, dtype=np.float32).reshape(-1, 1)
        depth = self.min_depth + t * (self.max_depth - self.min_depth)
        depth = np.tile(depth, (1, W))
        if self.noise_sigma > 0:
            noise = self._rng.normal(0, self.noise_sigma, (H, W)).astype(np.float32)
            depth = np.clip(depth + noise, self.min_depth, self.max_depth)
        return depth


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch wrappers  (import torch lazily so the module loads without it)
# ─────────────────────────────────────────────────────────────────────────────

class TorchRoadSegmenter(RoadSegmenter):
    """Thin wrapper for any PyTorch semantic-segmentation model.

    The default ``_preprocess`` / ``_postprocess`` pipeline assumes a
    ``transformers``-style segmentation model (e.g. SegFormer) that accepts
    a (1, C, H, W) float tensor and returns an object with a ``.logits``
    attribute of shape (1, num_classes, H', W').  Override these methods for
    other APIs.

    Parameters
    ----------
    model          : PyTorch model; called as ``model(tensor)``.
    road_class_ids : class indices in the model's output that correspond to
                     road / floor.  All matching pixels are merged into the
                     output mask.
    device         : torch device string (``'cpu'``, ``'cuda'``, …).
    input_size     : (W, H) tuple to resize frames before inference.
                     Set to ``None`` to pass the original resolution.
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

    # ── Override these for your model ────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray):
        """Convert a uint8 BGR/RGB frame to a model-ready tensor."""
        import torch
        img = frame if self.input_size is None else cv2.resize(frame, self.input_size)
        tensor = (
            torch.from_numpy(img)
            .float()
            .permute(2, 0, 1)
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        return tensor

    def _postprocess(self, output: object, original_hw: tuple) -> np.ndarray:
        """Convert model output to a (H, W) boolean mask at original resolution."""
        # Support transformers-style output with .logits, bare tensors, etc.
        import torch
        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output

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
            tensor = self._preprocess(frame)
            output = self.model(tensor)
        return self._postprocess(output, (H, W))


class TorchDepthEstimator(DepthEstimator):
    """Thin wrapper for any PyTorch monocular depth model.

    Compatible out of the box with:

    * **DepthAnything v1/v2**: output is a (1, H, W) tensor.
    * **ZoeDepth / MiDaS**: output is a (1, H, W) tensor or dict.
    * **Transformers** ``DepthEstimationOutput``: uses ``.predicted_depth``.

    Override ``_preprocess`` / ``_postprocess`` for other APIs.

    Parameters
    ----------
    model      : PyTorch model.
    device     : torch device string.
    input_size : (W, H) tuple to resize frames before inference.
                 ``None`` = pass original resolution.
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
        tensor = (
            torch.from_numpy(img)
            .float()
            .permute(2, 0, 1)
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        return tensor

    def _postprocess(self, output: object, original_hw: tuple) -> np.ndarray:
        import torch
        if hasattr(output, "predicted_depth"):
            depth = output.predicted_depth.squeeze().cpu().numpy()
        elif isinstance(output, torch.Tensor):
            depth = output.squeeze().cpu().numpy()
        elif isinstance(output, dict) and "depth" in output:
            depth = output["depth"].squeeze().cpu().numpy()
        else:
            # best-effort: try first element
            depth = list(output)[0].squeeze().cpu().numpy()

        depth = depth.astype(np.float32)
        if depth.shape != (original_hw[0], original_hw[1]):
            depth = cv2.resize(
                depth,
                (original_hw[1], original_hw[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return depth

    def predict(self, frame: np.ndarray) -> np.ndarray:
        import torch
        H, W = frame.shape[:2]
        with torch.no_grad():
            tensor = self._preprocess(frame)
            output = self.model(tensor)
        return self._postprocess(output, (H, W))


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_segmenter(
    kind: str = "dummy",
    **kwargs,
) -> RoadSegmenter:
    """Instantiate a road segmenter by name.

    Parameters
    ----------
    kind : ``'dummy'`` (default) or ``'torch'``.
    **kwargs : forwarded to the constructor.

    Examples
    --------
    >>> seg = create_segmenter("dummy", road_fraction=0.55)
    >>> seg = create_segmenter("torch", model=my_model, road_class_ids=[13])
    """
    kinds: Dict[str, type] = {
        "dummy": DummyRoadSegmenter,
        "torch": TorchRoadSegmenter,
    }
    if kind not in kinds:
        raise ValueError(f"Unknown segmenter kind '{kind}'. Choose from {list(kinds)}")
    return kinds[kind](**kwargs)


def create_depth_estimator(
    kind: str = "dummy",
    **kwargs,
) -> DepthEstimator:
    """Instantiate a depth estimator by name.

    Parameters
    ----------
    kind : ``'dummy'`` (default) or ``'torch'``.
    **kwargs : forwarded to the constructor.

    Examples
    --------
    >>> depth = create_depth_estimator("dummy", min_depth=1.0, max_depth=15.0)
    >>> depth = create_depth_estimator("torch", model=depth_anything_model)
    """
    kinds: Dict[str, type] = {
        "dummy": DummyDepthEstimator,
        "torch": TorchDepthEstimator,
    }
    if kind not in kinds:
        raise ValueError(f"Unknown depth estimator kind '{kind}'. Choose from {list(kinds)}")
    return kinds[kind](**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# NPZ cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_key(video_path: Union[str, Path], frame_idx: int, suffix: str) -> str:
    """Generate a unique filename for a cached result.

    The key hashes the *absolute* video path to handle collisions when
    processing multiple videos into the same cache directory.
    """
    abs_path = str(Path(video_path).resolve())
    path_hash = hashlib.md5(abs_path.encode()).hexdigest()[:8]
    return f"{path_hash}_f{frame_idx:06d}_{suffix}.npz"


def load_cached(
    cache_dir: Union[str, Path],
    video_path: Union[str, Path],
    frame_idx: int,
    suffix: str,
) -> Optional[np.ndarray]:
    """Load a cached numpy array from disk if it exists.

    Parameters
    ----------
    cache_dir  : directory containing .npz files.
    video_path : source video path (used to build the cache key).
    frame_idx  : frame index.
    suffix     : model identifier string (e.g. ``'seg'``, ``'depth'``).

    Returns
    -------
    data : (H, W) ndarray, or ``None`` if cache miss.
    """
    path = Path(cache_dir) / _cache_key(video_path, frame_idx, suffix)
    if not path.exists():
        return None
    try:
        data = np.load(str(path))["arr"]
        return data
    except Exception:
        return None


def save_cached(
    cache_dir: Union[str, Path],
    video_path: Union[str, Path],
    frame_idx: int,
    suffix: str,
    data: np.ndarray,
) -> None:
    """Save a numpy array to the .npz cache.

    Parameters
    ----------
    cache_dir  : target directory.
    video_path : source video path.
    frame_idx  : frame index.
    suffix     : model identifier string.
    data       : array to save.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = Path(cache_dir) / _cache_key(video_path, frame_idx, suffix)
    np.savez_compressed(str(path), arr=data)
