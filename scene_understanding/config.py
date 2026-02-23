"""
scene_understanding.config
--------------------------
Central configuration dataclass for the scene-understanding pipeline.
All parameters have documented defaults.  Override only the fields you need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SceneConfig:
    """Full configuration for the SceneUnderstanding pipeline.

    Device & precision
    ------------------
    device : ``'auto'``, ``'cuda'``, or ``'cpu'``.
        ``'auto'`` uses CUDA if PyTorch can see a GPU, otherwise CPU.
    fp16 : bool
        Use float16 (half-precision) inference on CUDA.  Roughly halves
        VRAM usage and speeds up inference on Ampere+ GPUs.  Ignored on CPU.

    Depth model (Depth Anything V2)
    --------------------------------
    depth_model_id : HuggingFace Hub ID or local directory path.
        Small  : ``depth-anything/Depth-Anything-V2-Small-hf``
        Base   : ``depth-anything/Depth-Anything-V2-Base-hf``
        Large  : ``depth-anything/Depth-Anything-V2-Large-hf``
    depth_resize_long_side : resize input so max(H,W) = this before inference.
        Smaller → faster but less detail.  768–1024 is a good range.
    local_depth_checkpoint : override ``depth_model_id`` with a local path.

    Segmentation model (SegFormer)
    ------------------------------
    seg_model_id : HuggingFace Hub ID or local directory path.
        Cityscapes: ``nvidia/segformer-b2-finetuned-cityscapes-1024-1024``
        ADE20K    : ``nvidia/segformer-b2-finetuned-ade-512-512``
    seg_mode : ``'cityscapes'``, ``'ade20k'``, or ``'binary'``.
        'binary' requires a fine-tuned 2-class checkpoint (model_id must be set).
    seg_resize_long_side : resize input so max(H,W) = this before inference.
    local_seg_checkpoint : override ``seg_model_id`` with a local path.
    include_sidewalk : (cityscapes mode) also include sidewalk pixels.

    Segmentation post-processing
    ----------------------------
    road_dilate_px : morphological dilation radius (pixels).  Fills small gaps.
    min_area_px    : remove connected road components smaller than this area.

    Camera intrinsics
    -----------------
    fx, fy : focal lengths in pixels.  ``None`` → estimated from image size.
    cx, cy : principal point.  ``None`` → image centre.

    Unknown-intrinsics calibration (k-search)
    ------------------------------------------
    k_candidates : focal-length multiplier values to try.
        fx = fy = k * max(W, H).
    k_lambda1    : penalty weight for std of inter-frame normal angle.
    k_lambda2    : penalty weight for fraction of invalid-plane frames.
    calibration_frames : max bootstrap frames used for k-search.

    Timing / stride
    ---------------
    fps            : source video FPS (timestamp labelling only).
    stride_seg     : run segmenter every N frames; reuse last result otherwise.
    stride_depth   : run depth estimator every N frames.

    Bootstrap
    ---------
    num_bootstrap_frames : frames processed in bootstrap mode.

    Plane fitting (RANSAC)
    ----------------------
    sample_points_per_frame : road pixels subsampled before RANSAC.
    ransac_iters            : RANSAC trial count.
    inlier_thresh           : inlier distance threshold (depth units).
    min_inlier_ratio        : minimum inlier fraction to accept a plane.

    Plane validation / smoothing
    ----------------------------
    max_normal_angle_change_deg : reject a new plane whose normal deviates
                                  more than this angle from the current stable.
    ema_alpha : EMA weight for a newly accepted plane (0 = keep old, 1 = replace).

    Contact-point snapping
    ----------------------
    snap_radius_px : pixel search radius for snapping off-road probes.

    Grid visualisation
    ------------------
    grid_spacing : distance between grid lines (depth units).
    grid_extent  : half-extent of the projected grid.

    Output
    ------
    output_dir          : root directory for all saved outputs.
    cache_dir           : intermediate .npz cache; defaults to output_dir/cache.
    inference_max_dim   : hard cap on largest image dimension before inference.
                          ``None`` = defer to per-model ``resize_long_side``.
    save_overlay_video  : write an annotated MP4.
    bev_max_points      : subsample cap for BEV scatter plots.
    """

    # ── Device & precision ────────────────────────────────────────────────────
    device: str = "auto"
    fp16: bool = True

    # ── Depth model ───────────────────────────────────────────────────────────
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    depth_resize_long_side: int = 768
    local_depth_checkpoint: Optional[str] = None

    # ── Segmentation model ────────────────────────────────────────────────────
    seg_model_id: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
    seg_mode: str = "cityscapes"         # 'cityscapes' | 'ade20k' | 'binary'
    seg_resize_long_side: int = 1024
    local_seg_checkpoint: Optional[str] = None
    include_sidewalk: bool = False

    # ── Segmentation post-processing ──────────────────────────────────────────
    road_dilate_px: int = 5
    min_area_px: int = 500

    # ── Camera intrinsics (None = auto-estimate) ──────────────────────────────
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None

    # ── Unknown-intrinsics calibration ────────────────────────────────────────
    k_candidates: List[float] = field(
        default_factory=lambda: [0.8, 1.0, 1.2, 1.4]
    )
    k_lambda1: float = 0.5   # penalty weight: std of normal angles
    k_lambda2: float = 0.5   # penalty weight: invalid-plane rate
    calibration_frames: int = 15   # bootstrap frames used for k-search

    # ── Timing ────────────────────────────────────────────────────────────────
    fps: float = 30.0
    stride_seg: int = 1
    stride_depth: int = 1

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    num_bootstrap_frames: int = 30

    # ── Plane fitting ─────────────────────────────────────────────────────────
    sample_points_per_frame: int = 2000
    ransac_iters: int = 150
    inlier_thresh: float = 0.05
    min_inlier_ratio: float = 0.30

    # ── Plane validation / smoothing ──────────────────────────────────────────
    max_normal_angle_change_deg: float = 15.0
    ema_alpha: float = 0.20

    # ── Contact-point snapping ────────────────────────────────────────────────
    snap_radius_px: int = 50

    # ── Grid visualisation ────────────────────────────────────────────────────
    grid_spacing: float = 1.0
    grid_extent: float = 10.0

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "outputs/scene_understanding"
    cache_dir: Optional[str] = None
    inference_max_dim: Optional[int] = None   # hard cap; None = use per-model setting
    save_overlay_video: bool = True

    # ── BEV plot ──────────────────────────────────────────────────────────────
    bev_max_points: int = 5000

    # ── Methods ───────────────────────────────────────────────────────────────

    def resolve_device(self) -> str:
        """Return the concrete device string (``'cuda'`` or ``'cpu'``)."""
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

    def get_torch_dtype(self):
        """Return the appropriate ``torch.dtype`` for model loading.

        Returns ``torch.float16`` when fp16 is enabled and CUDA is active,
        otherwise ``torch.float32``.
        """
        import torch
        if self.fp16 and self.resolve_device() == "cuda":
            return torch.float16
        return torch.float32

    def resolve_cache_dir(self) -> Path:
        """Return the resolved cache directory, creating it if needed."""
        p = Path(self.cache_dir) if self.cache_dir else Path(self.output_dir) / "cache"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def build_intrinsics(self, frame_height: int, frame_width: int):
        """Build :class:`CameraIntrinsics` from config fields.

        Missing values are filled with image-based defaults:
        ``cx = W/2``, ``cy = H/2``, ``fx = fy = 0.85 * max(W, H)``.
        """
        from .geometry import CameraIntrinsics
        cx = self.cx if self.cx is not None else frame_width / 2.0
        cy = self.cy if self.cy is not None else frame_height / 2.0
        default_f = max(frame_width, frame_height) * 0.85
        fx = self.fx if self.fx is not None else default_f
        fy = self.fy if self.fy is not None else fx
        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)

    def effective_depth_model_id(self) -> str:
        """Return local checkpoint path if set, else the Hub model ID."""
        return self.local_depth_checkpoint or self.depth_model_id

    def effective_seg_model_id(self) -> str:
        """Return local checkpoint path if set, else the Hub model ID."""
        return self.local_seg_checkpoint or self.seg_model_id

    def build_depth_estimator(self):
        """Instantiate the configured depth estimator.

        Returns a :class:`DepthAnythingV2Estimator` when transformers +
        torch are available; falls back to :class:`DummyDepthEstimator`
        otherwise (with a warning).
        """
        try:
            from .models.depth_anything_v2 import DepthAnythingV2Estimator
            return DepthAnythingV2Estimator(
                model_id=self.effective_depth_model_id(),
                device=self.device,
                fp16=self.fp16,
                resize_long_side=self.depth_resize_long_side,
            )
        except ImportError:
            import warnings
            warnings.warn(
                "transformers / torch not available — using DummyDepthEstimator.",
                stacklevel=2,
            )
            from .models import DummyDepthEstimator
            return DummyDepthEstimator()

    def build_segmenter(self):
        """Instantiate the configured road segmenter.

        Returns a :class:`SegFormerRoadSegmenter` when transformers + torch
        are available; falls back to :class:`DummyRoadSegmenter` otherwise.
        """
        try:
            from .models.segformer_road import SegFormerRoadSegmenter
            return SegFormerRoadSegmenter(
                model_id=self.effective_seg_model_id(),
                seg_mode=self.seg_mode,
                device=self.device,
                fp16=self.fp16,
                resize_long_side=self.seg_resize_long_side,
                road_dilate_px=self.road_dilate_px,
                min_area_px=self.min_area_px,
                include_sidewalk=self.include_sidewalk,
            )
        except ImportError:
            import warnings
            warnings.warn(
                "transformers / torch not available — using DummyRoadSegmenter.",
                stacklevel=2,
            )
            from .models import DummyRoadSegmenter
            return DummyRoadSegmenter()
