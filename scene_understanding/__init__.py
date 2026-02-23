"""
scene_understanding
===================
Monocular video ground-plane estimation and geometric visualisation.

Runs independently of the detection / tracking / near-miss pipeline.

Quick-start
-----------
>>> from scene_understanding import (
...     SceneConfig, CameraIntrinsics,
...     DummyRoadSegmenter, DummyDepthEstimator,
...     ScenePlaneTracker, ransac_plane_fit,
...     backproject_depth_map, overlay_mask,
... )

Production models (require ``transformers`` + ``torch``):
---------------------------------------------------------
>>> from scene_understanding import (
...     DepthAnythingV2Estimator,
...     SegFormerRoadSegmenter,
... )

Unknown-intrinsics calibration:
--------------------------------
>>> from scene_understanding import calibrate_intrinsics, IntrinsicsCalibrator
"""

from .config import SceneConfig
from .geometry import (
    CameraIntrinsics,
    backproject_pixel,
    backproject_depth_map,
    ray_direction,
    ray_plane_intersection,
    project_3d_to_image,
    angle_between_normals,
)
from .models import (
    RoadSegmenter,
    DepthEstimator,
    DummyRoadSegmenter,
    DummyDepthEstimator,
    TorchRoadSegmenter,
    TorchDepthEstimator,
    create_segmenter,
    create_depth_estimator,
)
from .plane_fit import (
    PlaneParams,
    ransac_plane_fit,
    validate_plane,
    ema_smooth_plane,
    ScenePlaneTracker,
)
from .visualize import (
    overlay_mask,
    depth_viz,
    road_depth_overlay,
    make_side_by_side,
    draw_plane_grid,
    draw_ground_cube,
    bev_scatter_plot,
    plane_diagnostics_plot,
    draw_pixel_probe,
    draw_ground_probe_grid,
)
from .calibration import (
    IntrinsicsCalibrator,
    CalibrationResult,
    KCandidateStats,
    calibrate_intrinsics,
)

# Production model classes — available when transformers + torch are installed.
# Imported lazily so the package remains usable without ML dependencies.
try:
    from .models.depth_anything_v2 import DepthAnythingV2Estimator
except Exception:  # pragma: no cover
    DepthAnythingV2Estimator = None  # type: ignore[assignment,misc]

try:
    from .models.segformer_road import SegFormerRoadSegmenter
except Exception:  # pragma: no cover
    SegFormerRoadSegmenter = None  # type: ignore[assignment,misc]


__all__ = [
    # config
    "SceneConfig",
    # geometry
    "CameraIntrinsics",
    "backproject_pixel",
    "backproject_depth_map",
    "ray_direction",
    "ray_plane_intersection",
    "project_3d_to_image",
    "angle_between_normals",
    # models — base + dummies
    "RoadSegmenter",
    "DepthEstimator",
    "DummyRoadSegmenter",
    "DummyDepthEstimator",
    "TorchRoadSegmenter",
    "TorchDepthEstimator",
    "create_segmenter",
    "create_depth_estimator",
    # models — production
    "DepthAnythingV2Estimator",
    "SegFormerRoadSegmenter",
    # plane fitting
    "PlaneParams",
    "ransac_plane_fit",
    "validate_plane",
    "ema_smooth_plane",
    "ScenePlaneTracker",
    # visualisation
    "overlay_mask",
    "depth_viz",
    "road_depth_overlay",
    "make_side_by_side",
    "draw_plane_grid",
    "draw_ground_cube",
    "bev_scatter_plot",
    "plane_diagnostics_plot",
    "draw_pixel_probe",
    "draw_ground_probe_grid",
    # calibration
    "IntrinsicsCalibrator",
    "CalibrationResult",
    "KCandidateStats",
    "calibrate_intrinsics",
]
