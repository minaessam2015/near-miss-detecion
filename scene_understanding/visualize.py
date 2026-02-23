"""
scene_understanding.visualize
------------------------------
All visual outputs for the scene-understanding stage.

Functions
---------
overlay_mask          : semi-transparent road mask on BGR frame.
depth_viz             : colourised depth map.
road_depth_overlay    : depth shown only on road pixels, frame darkened elsewhere.
make_side_by_side     : horizontal concatenation of multiple images.
draw_plane_grid       : perspective ground-plane grid projected to image.
draw_plane_inliers    : inlier / outlier classification dots on image.
bev_scatter_plot      : bird's-eye-view scatter of road pixels on the ground plane.
plane_diagnostics_plot: 3-panel time-series diagnostics.
draw_pixel_probe      : annotate user-chosen pixels with ray-plane intersections.
draw_ground_probe_grid: grid-sampled distance probe for ground-plane sanity check.
build_scene_frame     : 2x2 composite convenience builder.

All image functions accept and return BGR uint8 images (OpenCV convention).
Matplotlib figures are returned as Figure objects for caller to save/show.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import (
    CameraIntrinsics,
    project_3d_to_image,
    ray_direction,
    ray_plane_intersection,
)
from .plane_fit import PlaneParams


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Convert greyscale to 3-channel BGR if needed."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _plane_forward_basis(plane_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two orthonormal basis vectors in the ground plane.

    u_axis is the projection of the camera forward direction [0,0,1] onto the
    plane (so the grid extends toward the horizon).  v_axis is perpendicular
    to both n and u_axis.  Falls back to world-X if forward is near-parallel
    to the normal.
    """
    n = plane_n / (np.linalg.norm(plane_n) + 1e-12)
    candidates = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    for c in candidates:
        u = c - n * float(n @ c)
        mag = np.linalg.norm(u)
        if mag > 0.1:
            u = u / mag
            v = np.cross(n, u)
            v = v / (np.linalg.norm(v) + 1e-12)
            return u, v
    return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])


def _optical_axis_plane_intersection(
    plane_n: np.ndarray,
    plane_d: float,
) -> Optional[np.ndarray]:
    """Where the camera optical axis [0,0,1] hits the plane (t > 0)."""
    denom = float(np.dot(plane_n, np.array([0.0, 0.0, 1.0])))
    if abs(denom) < 1e-8:
        return None
    t = -plane_d / denom
    if t <= 0:
        return None
    return np.array([0.0, 0.0, t])


# ---------------------------------------------------------------------------
# Core image-overlay functions
# ---------------------------------------------------------------------------

def overlay_mask(
    frame: np.ndarray,
    road_mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 200, 80),
    alpha: float = 0.40,
    edge_color: Optional[Tuple[int, int, int]] = (0, 255, 100),
    edge_thickness: int = 2,
) -> np.ndarray:
    """Semi-transparent road-mask overlay on a BGR frame.

    Parameters
    ----------
    frame         : (H, W, 3) uint8 BGR image.
    road_mask     : (H, W) boolean mask — True = road.
    color         : BGR fill colour.
    alpha         : fill opacity (0 transparent → 1 opaque).
    edge_color    : BGR colour for mask contour; None to skip.
    edge_thickness: contour line width in pixels.

    Returns
    -------
    (H, W, 3) uint8 BGR image.
    """
    out = frame.copy()
    mask_3ch = road_mask[:, :, np.newaxis].astype(bool)
    colored = np.zeros_like(frame)
    colored[road_mask] = color
    blended = np.clip(frame * (1.0 - alpha) + colored * alpha, 0, 255).astype(np.uint8)
    out = np.where(mask_3ch, blended, frame)

    if edge_color is not None:
        mask_u8 = road_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, edge_color, edge_thickness)
    return out


def depth_viz(
    depth: np.ndarray,
    colormap: int = cv2.COLORMAP_INFERNO,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Convert a float depth map to a colour image.

    Normalises using the 2nd–98th percentile of valid pixels when vmin/vmax
    are not supplied, which handles outlier depth values gracefully.

    Parameters
    ----------
    depth   : (H, W) float depth map.
    colormap: OpenCV colormap constant.
    vmin, vmax : optional manual clip range.

    Returns
    -------
    (H, W, 3) uint8 BGR colour image.
    """
    d = np.asarray(depth, dtype=np.float32)
    valid = (d > 0) & np.isfinite(d)
    if valid.any():
        lo = float(vmin) if vmin is not None else float(np.percentile(d[valid], 2))
        hi = float(vmax) if vmax is not None else float(np.percentile(d[valid], 98))
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    d_norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    return cv2.applyColorMap((d_norm * 255).astype(np.uint8), colormap)


def road_depth_overlay(
    frame: np.ndarray,
    depth: np.ndarray,
    road_mask: np.ndarray,
    colormap: int = cv2.COLORMAP_INFERNO,
    dim_factor: float = 0.25,
) -> np.ndarray:
    """Show depth colourmap only at road pixels; darken everything else.

    Useful for inspecting the exact depth values fed into RANSAC.

    Parameters
    ----------
    frame      : (H, W, 3) uint8 BGR source image.
    depth      : (H, W) float depth map.
    road_mask  : (H, W) boolean road mask.
    colormap   : OpenCV colormap for depth.
    dim_factor : brightness multiplier for non-road pixels (0–1).

    Returns
    -------
    (H, W, 3) uint8 BGR image.
    """
    depth_color = depth_viz(depth, colormap=colormap)
    bg = np.clip(frame.astype(np.float32) * dim_factor, 0, 255).astype(np.uint8)
    mask_3ch = np.stack([road_mask] * 3, axis=2)
    return np.where(mask_3ch, depth_color, bg)


def make_side_by_side(
    *images: np.ndarray,
    pad: int = 4,
    pad_color: int = 128,
    labels: Optional[Sequence[str]] = None,
    label_scale: float = 0.55,
) -> np.ndarray:
    """Concatenate images horizontally, scaling all to a common height.

    Parameters
    ----------
    *images    : (H, W, 3) or greyscale images — any number.
    pad        : separator width in pixels.
    pad_color  : greyscale fill value of the separator.
    labels     : optional text label for each panel (top-left corner).
    label_scale: OpenCV font scale for labels.

    Returns
    -------
    (H_max, W_total, 3) uint8 BGR image.
    """
    if not images:
        raise ValueError("At least one image required.")
    target_h = max(img.shape[0] for img in images)
    resized: List[np.ndarray] = []
    for img in images:
        img = _ensure_bgr(img)
        if img.shape[0] != target_h:
            scale = target_h / img.shape[0]
            new_w = max(1, int(img.shape[1] * scale))
            img = cv2.resize(img, (new_w, target_h))
        resized.append(img)

    if labels:
        for img, lbl in zip(resized, labels):
            bg_w = min(len(lbl) * 12 + 10, img.shape[1])
            cv2.rectangle(img, (0, 0), (bg_w, 24), (0, 0, 0), -1)
            cv2.putText(
                img, lbl, (4, 17),
                cv2.FONT_HERSHEY_SIMPLEX, label_scale,
                (220, 220, 220), 1, cv2.LINE_AA,
            )

    sep = np.full((target_h, pad, 3), pad_color, dtype=np.uint8)
    parts: List[np.ndarray] = []
    for i, img in enumerate(resized):
        parts.append(img)
        if i < len(resized) - 1:
            parts.append(sep)
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Ground-plane grid overlay
# ---------------------------------------------------------------------------

def draw_plane_grid(
    frame: np.ndarray,
    intrinsics: CameraIntrinsics,
    plane: PlaneParams,
    grid_spacing: float = 1.0,
    grid_extent: float = 10.0,
    color: Tuple[int, int, int] = (30, 220, 90),
    line_thickness: int = 1,
    overlay_alpha: float = 0.80,
    axis_color: Optional[Tuple[int, int, int]] = (50, 50, 255),
) -> np.ndarray:
    """Project a regular ground-plane grid onto the image.

    Builds a coordinate frame in the fitted ground plane centred on the
    optical-axis / plane intersection point, then projects the grid vertices
    using the pinhole model.  Lines are clipped to image bounds.

    Parameters
    ----------
    frame         : (H, W, 3) uint8 BGR image.
    intrinsics    : camera intrinsics.
    plane         : estimated ground plane.
    grid_spacing  : distance between adjacent grid lines (depth units).
    grid_extent   : half-size of the grid in each direction.
    color         : BGR colour of regular grid lines.
    line_thickness: pixel width.
    overlay_alpha : blend weight of the grid overlay (1 = fully opaque).
    axis_color    : BGR colour of the two central axis lines; None to skip.

    Returns
    -------
    (H, W, 3) uint8 annotated image.
    """
    H, W = frame.shape[:2]
    overlay = frame.copy()

    centre = _optical_axis_plane_intersection(plane.n, plane.d)
    if centre is None:
        centre = plane.closest_point_to_origin()

    u_axis, v_axis = _plane_forward_basis(plane.n)
    n_steps = max(1, int(math.ceil(grid_extent / grid_spacing)))
    coords = [i * grid_spacing for i in range(-n_steps, n_steps + 1)]

    def _to_px(p3d: np.ndarray) -> Optional[Tuple[int, int]]:
        if p3d[2] <= 0:
            return None
        pu = intrinsics.fx * p3d[0] / p3d[2] + intrinsics.cx
        pv = intrinsics.fy * p3d[1] / p3d[2] + intrinsics.cy
        if 0 <= pu < W and 0 <= pv < H:
            return (int(round(pu)), int(round(pv)))
        return None

    def _draw_strip(
        fixed_val: float,
        vary_axis: np.ndarray,
        fixed_axis: np.ndarray,
        lcolor: Tuple[int, int, int],
    ) -> None:
        prev = None
        for coord in coords:
            p3d = centre + fixed_val * fixed_axis + coord * vary_axis
            px = _to_px(p3d)
            if px is not None and prev is not None:
                cv2.line(overlay, prev, px, lcolor, line_thickness, cv2.LINE_AA)
            prev = px

    for v_val in coords:
        c = axis_color if (v_val == 0 and axis_color) else color
        _draw_strip(v_val, u_axis, v_axis, c)
    for u_val in coords:
        c = axis_color if (u_val == 0 and axis_color) else color
        _draw_strip(u_val, v_axis, u_axis, c)

    return cv2.addWeighted(overlay, overlay_alpha, frame, 1.0 - overlay_alpha, 0)


def draw_plane_inliers(
    frame: np.ndarray,
    pixel_coords: np.ndarray,
    is_inlier: np.ndarray,
    inlier_color: Tuple[int, int, int] = (0, 200, 80),
    outlier_color: Tuple[int, int, int] = (30, 30, 220),
    radius: int = 2,
    max_draw: int = 3000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Show RANSAC inlier / outlier road pixels as coloured dots on the frame.

    Parameters
    ----------
    frame        : (H, W, 3) uint8 BGR image.
    pixel_coords : (N, 2) array of (u, v) road-pixel coordinates.
    is_inlier    : (N,) boolean — True = within inlier threshold.
    inlier_color : BGR dot colour for inliers.
    outlier_color: BGR dot colour for outliers.
    radius       : dot radius in pixels.
    max_draw     : subsample cap (random) to keep rendering fast.
    rng          : numpy random generator for subsampling.

    Returns
    -------
    (H, W, 3) annotated image.
    """
    out = frame.copy()
    rng = rng or np.random.default_rng()
    N = len(pixel_coords)
    idx = np.arange(N)
    if N > max_draw:
        idx = rng.choice(N, max_draw, replace=False)
    for i in idx:
        u, v = int(round(pixel_coords[i, 0])), int(round(pixel_coords[i, 1]))
        c = inlier_color if is_inlier[i] else outlier_color
        cv2.circle(out, (u, v), radius, c, -1)
    return out


# ---------------------------------------------------------------------------
# 3-D ground-plane cube (geometry verification)
# ---------------------------------------------------------------------------

def draw_ground_cube(
    frame: np.ndarray,
    intrinsics: CameraIntrinsics,
    plane: PlaneParams,
    center_uv: Optional[Tuple[float, float]] = None,
    width: float = 1.5,
    length: float = 3.0,
    height: float = 2.0,
    color_base: Tuple[int, int, int] = (0, 220, 255),
    color_top: Tuple[int, int, int] = (255, 180, 0),
    color_vert: Tuple[int, int, int] = (160, 255, 120),
    thickness: int = 2,
    _debug_out: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Project a 3-D box seated on the estimated ground plane.

    This is the primary geometry sanity-check: if the box sits flat on the
    road, tilts with the road, and its posts are vertical, the plane normal
    and camera intrinsics are consistent with the scene.

    Interpretation guide
    --------------------
    - Box floats above road  → plane d is too small (plane too close to camera)
    - Box sinks into road    → plane d is too large
    - Box is slanted sideways→ plane normal is wrong (bad RANSAC or wrong k)
    - Box looks too wide/narrow → intrinsics (fx/fy) are off

    Parameters
    ----------
    frame      : (H, W, 3) uint8 BGR image.
    intrinsics : CameraIntrinsics.
    plane      : estimated ground plane.
    center_uv  : (u, v) pixel at which to anchor the cube's base centre.
                 Defaults to the principal point (cx, cy).
    width      : extent along the lateral axis (v_axis) in depth units.
    length     : extent along the forward axis (u_axis) in depth units.
    height     : vertical extent above the ground in depth units.
    color_base : BGR colour for base-rectangle edges.
    color_top  : BGR colour for top-rectangle edges.
    color_vert : BGR colour for vertical-post edges.
    thickness  : line width in pixels.
    _debug_out : optional dict; if provided it is populated with every
                 intermediate value so callers can diagnose failures.

    Returns
    -------
    Annotated (H, W, 3) BGR image, or the original frame if the cube
    cannot be placed (ray misses the plane or corners are behind camera).
    """
    H, W = frame.shape[:2]

    # ── Centre on the ground plane ─────────────────────────────────────────
    if center_uv is None:
        center_uv = (intrinsics.cx, intrinsics.cy)

    r = ray_direction(float(center_uv[0]), float(center_uv[1]), intrinsics)

    # Pre-compute intersection denominator for debug (dot of plane normal and ray)
    _denom = float(np.dot(plane.n, r))
    _t = (-float(plane.d) / _denom) if abs(_denom) > 1e-8 else float("nan")

    if _debug_out is not None:
        _debug_out["frame_H"] = H
        _debug_out["frame_W"] = W
        _debug_out["center_uv"] = [float(center_uv[0]), float(center_uv[1])]
        _debug_out["ray_dir"] = r.tolist()
        _debug_out["plane_n"] = plane.n.tolist()
        _debug_out["plane_d"] = float(plane.d)
        _debug_out["plane_inlier_ratio"] = float(plane.inlier_ratio)
        _debug_out["dot_n_ray"] = _denom          # near 0 → ray parallel to plane
        _debug_out["t_intersection"] = _t         # negative → plane behind camera

    center3d = ray_plane_intersection(r, plane.n, plane.d)
    if center3d is None:
        if _debug_out is not None:
            _debug_out["ray_plane_miss"] = True
            _debug_out["center3d"] = None
            _debug_out["failure_reason"] = (
                "ray parallel to plane" if abs(_denom) < 1e-8
                else "plane behind camera (t={:.3f})".format(_t)
            )
        return frame

    if _debug_out is not None:
        _debug_out["ray_plane_miss"] = False
        _debug_out["center3d"] = center3d.tolist()

    # ── Plane basis ────────────────────────────────────────────────────────
    u_axis, v_axis = _plane_forward_basis(plane.n)
    n_norm = plane.n / (np.linalg.norm(plane.n) + 1e-12)

    # "up" = normal side that faces the camera (camera sits at origin)
    # If n · (-center3d) > 0, n points from center toward origin = toward camera
    up = n_norm if float(np.dot(n_norm, -center3d)) > 0.0 else -n_norm

    if _debug_out is not None:
        _debug_out["u_axis"] = u_axis.tolist()
        _debug_out["v_axis"] = v_axis.tolist()
        _debug_out["up"] = up.tolist()
        _debug_out["cube_width"] = width
        _debug_out["cube_length"] = length
        _debug_out["cube_height"] = height

    # ── 8 corners: indices 0-3 = base (on plane), 4-7 = top ───────────────
    hl, hw = length / 2.0, width / 2.0
    base = [
        center3d - hl * u_axis - hw * v_axis,  # 0 back-left
        center3d - hl * u_axis + hw * v_axis,  # 1 back-right
        center3d + hl * u_axis - hw * v_axis,  # 2 front-left
        center3d + hl * u_axis + hw * v_axis,  # 3 front-right
    ]
    top = [c + height * up for c in base]
    corners = base + top

    # ── Project to image ───────────────────────────────────────────────────
    def _proj(p: np.ndarray) -> Optional[Tuple[int, int]]:
        if p[2] <= 1e-3:
            return None
        pu = intrinsics.fx * p[0] / p[2] + intrinsics.cx
        pv = intrinsics.fy * p[1] / p[2] + intrinsics.cy
        # Clamp so cv2.line can still draw the visible segment even when one
        # endpoint is off-screen.  Without clamping, very large coordinates
        # cause int32 overflow and silent draw failures.
        pu = int(round(max(-32000.0, min(32000.0, pu))))
        pv = int(round(max(-32000.0, min(32000.0, pv))))
        return (pu, pv)

    px = [_proj(c) for c in corners]

    if _debug_out is not None:
        _debug_out["corners_3d"] = [c.tolist() for c in corners]
        _debug_out["corners_px"] = [list(p) if p is not None else None for p in px]
        _debug_out["corners_behind_camera"] = sum(
            1 for c in corners if c[2] <= 1e-3
        )
        _debug_out["corners_in_image"] = sum(
            1 for p in px
            if p is not None and 0 <= p[0] < W and 0 <= p[1] < H
        )

    # ── Draw 12 edges ──────────────────────────────────────────────────────
    out = frame.copy()
    lines_drawn = 0

    def _line(i: int, j: int, color: Tuple[int, int, int]) -> None:
        nonlocal lines_drawn
        a, b = px[i], px[j]
        if a is not None and b is not None:
            cv2.line(out, a, b, color, thickness, cv2.LINE_AA)
            lines_drawn += 1

    # Base rectangle
    _line(0, 1, color_base); _line(1, 3, color_base)
    _line(3, 2, color_base); _line(2, 0, color_base)
    # Top rectangle
    _line(4, 5, color_top);  _line(5, 7, color_top)
    _line(7, 6, color_top);  _line(6, 4, color_top)
    # Vertical posts
    _line(0, 4, color_vert); _line(1, 5, color_vert)
    _line(2, 6, color_vert); _line(3, 7, color_vert)

    if _debug_out is not None:
        _debug_out["lines_drawn"] = lines_drawn  # 0 → all corner pairs were None

    # Ground contact dot
    cp = _proj(center3d)
    if cp is not None:
        cv2.circle(out, cp, 5, (0, 0, 200), -1)
        cv2.circle(out, cp, 6, (255, 255, 255), 1)

    return out


# ---------------------------------------------------------------------------
# Bird's-Eye View (BEV) scatter plot
# ---------------------------------------------------------------------------

def bev_scatter_plot(
    projected_points: np.ndarray,
    camera_origin: Optional[np.ndarray] = None,
    plane_normal: Optional[np.ndarray] = None,
    title: str = "BEV — Road Pixels on Ground Plane",
    max_points: int = 5000,
    rng: Optional[np.random.Generator] = None,
    xlim: Optional[Tuple[float, float]] = None,
    zlim: Optional[Tuple[float, float]] = None,
) -> Any:
    """Bird's-eye-view scatter of road pixels projected onto the ground plane.

    How it works
    ------------
    Each road pixel (u, v) emits a ray from the camera.  That ray is
    intersected with the RANSAC-fitted ground plane.  The resulting 3-D
    point's X (lateral) and Z (forward) coordinates are plotted here.
    This is **independent of per-pixel depth** — it only depends on the
    plane geometry and the intrinsics.

    A healthy BEV looks like a fan / wedge spreading forward from the
    camera (red triangle at origin).  If the fan is narrow or distorted,
    the plane normal is probably wrong.

    Why the normal arrow is not shown
    ----------------------------------
    For a ground plane the normal points almost entirely in the camera-Y
    direction (up/down).  Its X and Z components are near zero, so an
    arrow in X-Z space would be invisible.  The normal vector is printed
    as text instead.

    Axis stability
    --------------
    Pass ``xlim`` and ``zlim`` computed once from the first valid BEV frame
    and reuse them for every subsequent frame.  Without fixed limits,
    ``set_aspect('equal')`` + auto-scaling cause the plot to jump in size.

    Parameters
    ----------
    projected_points : (N, 3) 3-D intersection points in camera space.
    camera_origin    : (3,) camera position; defaults to [0, 0, 0].
    plane_normal     : (3,) plane normal — shown as text annotation.
    title            : plot title string.
    max_points       : random subsample cap.
    rng              : numpy random generator.
    xlim             : (x_min, x_max) for the lateral axis.  ``None`` =
                       auto-fit with 20 % padding, rounded outward.
    zlim             : (z_min, z_max) for the forward axis.  ``None`` =
                       auto-fit with 20 % padding, starting at 0.

    Returns
    -------
    ``matplotlib.figure.Figure``
    """
    import matplotlib.pyplot as plt

    rng = rng or np.random.default_rng()
    pts = np.asarray(projected_points, dtype=np.float64)
    if len(pts) > max_points:
        idx = rng.choice(len(pts), max_points, replace=False)
        pts = pts[idx]

    fig, ax = plt.subplots(figsize=(7, 6))

    if len(pts) > 0:
        sc = ax.scatter(
            pts[:, 0], pts[:, 2],
            c=pts[:, 2], cmap="plasma",
            s=3, alpha=0.5, label="road pixels",
        )
        fig.colorbar(sc, ax=ax, label="Z (forward, depth units)")

        # Auto-compute limits if not provided
        if xlim is None:
            x_ext = max(float(np.abs(pts[:, 0]).max()) * 1.2, 2.0)
            xlim = (-x_ext, x_ext)
        if zlim is None:
            z_max = max(float(pts[:, 2].max()) * 1.2, 5.0)
            zlim = (0.0, z_max)
    else:
        if xlim is None:
            xlim = (-10.0, 10.0)
        if zlim is None:
            zlim = (0.0, 20.0)

    # Camera marker
    cam = np.zeros(3) if camera_origin is None else np.asarray(camera_origin)
    ax.scatter([cam[0]], [cam[2]], s=140, c="red", marker="^",
               zorder=5, label="camera (origin)")
    ax.annotate("cam", (cam[0], cam[2]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)

    # Normal as text (arrow is invisible for ground planes — n is mostly Y)
    if plane_normal is not None:
        n = np.asarray(plane_normal, dtype=np.float64)
        n_str = f"n=[{n[0]:.2f}, {n[1]:.2f}, {n[2]:.2f}]"
        ax.text(
            0.02, 0.97, n_str,
            transform=ax.transAxes, fontsize=8,
            verticalalignment="top", color="darkorange",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_xlabel("X  (lateral ←→)")
    ax.set_ylabel("Z  (forward ↑)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plane diagnostics time-series plot
# ---------------------------------------------------------------------------

def plane_diagnostics_plot(
    history: List[Dict[str, Any]],
    min_inlier_ratio: float = 0.30,
    max_angle_change: float = 15.0,
    title_prefix: str = "Scene Understanding",
) -> Any:
    """Three-panel diagnostics plot for the plane tracker over time.

    Panels
    ------
    1. Inlier ratio vs frame index (threshold line shown).
    2. Normal angle change vs frame (threshold line shown).
    3. Plane *d* (offset) vs frame.

    Parameters
    ----------
    history          : list of dicts from ScenePlaneTracker.get_history_dicts().
    min_inlier_ratio : threshold drawn on panel 1.
    max_angle_change : threshold drawn on panel 2.
    title_prefix     : figure suptitle prefix.

    Returns
    -------
    ``matplotlib.figure.Figure``
    """
    import matplotlib.pyplot as plt

    frames = [h["frame_idx"] for h in history]
    inlier_ratios = [h["inlier_ratio"] for h in history]
    angle_changes = [h.get("normal_angle_change") or 0.0 for h in history]
    plane_ds_raw = [h.get("plane_d") for h in history]
    valid_flags = [h.get("plane_valid", False) for h in history]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(f"{title_prefix} — Plane Diagnostics", fontsize=13)

    # Panel 1 — inlier ratio
    ax = axes[0]
    ax.plot(frames, inlier_ratios, color="steelblue", lw=1.5, label="inlier ratio")
    vf = [f for f, v in zip(frames, valid_flags) if v]
    vr = [r for r, v in zip(inlier_ratios, valid_flags) if v]
    ax.scatter(vf, vr, s=15, c="green", zorder=3, label="accepted frame")
    ax.axhline(min_inlier_ratio, color="red", ls="--", lw=1,
               label=f"threshold ({min_inlier_ratio})")
    ax.set_ylabel("Inlier ratio")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel 2 — normal angle change
    ax = axes[1]
    ax.plot(frames, angle_changes, color="darkorange", lw=1.5,
            label="normal Δ angle (°)")
    ax.axhline(max_angle_change, color="red", ls="--", lw=1,
               label=f"max ({max_angle_change}°)")
    ax.set_ylabel("Normal Δ angle (°)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 3 — plane d
    ax = axes[2]
    d_items = [(f, d) for f, d in zip(frames, plane_ds_raw) if d is not None]
    if d_items:
        fv, dv = zip(*d_items)
        ax.plot(fv, dv, color="purple", lw=1.5, label="plane d")
        ax.scatter(fv, dv, s=6, c="purple", zorder=3)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Plane d (depth units)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Pixel probe (ray–plane intersection annotation)
# ---------------------------------------------------------------------------

def draw_pixel_probe(
    frame: np.ndarray,
    intrinsics: CameraIntrinsics,
    plane: PlaneParams,
    test_pixels: Sequence[Tuple[float, float]],
    road_mask: Optional[np.ndarray] = None,
    snap_radius_px: int = 50,
    dot_color: Tuple[int, int, int] = (0, 230, 230),
    snapped_color: Tuple[int, int, int] = (0, 165, 255),
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Annotate test pixels with ray-plane intersection distances.

    For each (u, v):
    1. If not on road mask, snap to nearest road pixel within snap_radius_px.
    2. Compute unit ray through the (possibly snapped) pixel.
    3. Intersect with the ground plane.
    4. Draw a coloured dot and annotate the estimated 3-D distance.

    Parameters
    ----------
    frame         : (H, W, 3) uint8 BGR image.
    intrinsics    : camera intrinsics.
    plane         : current stable ground plane.
    test_pixels   : list of (u, v) pixel coordinates to probe.
    road_mask     : optional (H, W) boolean road mask for snapping.
    snap_radius_px: search radius for off-road pixel snapping.
    dot_color     : BGR colour for on-road dots.
    snapped_color : BGR colour for snapped dots.

    Returns
    -------
    annotated_frame : annotated BGR image.
    probe_results   : list of dicts with keys:
        input_pixel, used_pixel, snapped, intersection_3d, ground_distance.
    """
    H, W = frame.shape[:2]
    out = frame.copy()
    results: List[Dict[str, Any]] = []

    for u, v in test_pixels:
        u_use, v_use = float(u), float(v)
        snapped = False

        # Snap to nearest road pixel if not on road
        if road_mask is not None:
            ui, vi = int(round(u)), int(round(v))
            in_bounds = 0 <= vi < H and 0 <= ui < W
            if in_bounds and not road_mask[vi, ui]:
                r = snap_radius_px
                y1 = max(0, vi - r)
                y2 = min(H, vi + r + 1)
                x1 = max(0, ui - r)
                x2 = min(W, ui + r + 1)
                ys, xs = np.where(road_mask[y1:y2, x1:x2])
                if len(ys) > 0:
                    ys = ys + y1
                    xs = xs + x1
                    dists2 = (ys - vi) ** 2 + (xs - ui) ** 2
                    best = int(np.argmin(dists2))
                    u_use, v_use = float(xs[best]), float(ys[best])
                    snapped = True

        # Ray–plane intersection
        r_dir = ray_direction(u_use, v_use, intrinsics)
        p_intersect = ray_plane_intersection(r_dir, plane.n, plane.d)

        result: Dict[str, Any] = {
            "input_pixel": (float(u), float(v)),
            "used_pixel": (u_use, v_use),
            "snapped": snapped,
            "intersection_3d": p_intersect.tolist() if p_intersect is not None else None,
            "ground_distance": float(np.linalg.norm(p_intersect))
            if p_intersect is not None
            else None,
        }
        results.append(result)

        # Draw
        pu, pv = int(round(u_use)), int(round(v_use))
        c = snapped_color if snapped else dot_color
        cv2.circle(out, (pu, pv), 7, c, -1)
        cv2.circle(out, (pu, pv), 9, (0, 0, 0), 1)

        if snapped:
            ou, ov = int(round(u)), int(round(v))
            cv2.circle(out, (ou, ov), 4, (180, 180, 180), 1)
            cv2.line(out, (ou, ov), (pu, pv), (180, 180, 180), 1, cv2.LINE_AA)

        if p_intersect is not None:
            dist = float(np.linalg.norm(p_intersect))
            label = f"d={dist:.2f}"
            lw = len(label) * 9 + 8
            tx = pu + 10 if pu + lw + 10 < W else pu - lw - 10
            ty = max(20, pv - 6)
            cv2.rectangle(out, (tx - 2, ty - 16), (tx + lw, ty + 2), (0, 0, 0), -1)
            cv2.putText(
                out, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, c, 1, cv2.LINE_AA,
            )
        else:
            cv2.putText(
                out, "no hit", (pu + 10, pv),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1, cv2.LINE_AA,
            )

    return out, results


def draw_ground_probe_grid(
    frame: np.ndarray,
    intrinsics: CameraIntrinsics,
    plane: PlaneParams,
    road_mask: np.ndarray,
    n_rows: int = 4,
    n_cols: int = 5,
    show_stats: bool = True,
) -> Tuple[np.ndarray, List[float]]:
    """Sample a grid of road pixels and annotate with ray-plane distances.

    This is the primary qualitative test for ground-plane geometry.  For a
    correct plane the annotated distances should:

    * **Increase from bottom to top** — pixels close to the camera are near;
      pixels near the horizon are far.
    * **Be roughly left-right symmetric** for a straight road.
    * Fall in a physically plausible range (5–20 depth units after the
      default p95 = 10 normalisation).

    Color coding: red/warm = close, green/cool = far (hue 0→120 as distance
    increases).

    Parameters
    ----------
    frame       : (H, W, 3) uint8 BGR image.
    intrinsics  : camera intrinsics.
    plane       : current stable ground plane.
    road_mask   : (H, W) boolean road mask.
    n_rows      : number of grid rows (vertical bands across the road).
    n_cols      : number of grid columns (horizontal bands across the road).
    show_stats  : draw a stats box (plane normal, cam-to-plane, inlier ratio).

    Returns
    -------
    annotated_frame : BGR image with probe dots and distance labels.
    distances       : list of valid finite distances (one per probe point).
    """
    H, W = frame.shape[:2]
    out = frame.copy()
    distances: List[float] = []

    if not road_mask.any():
        return out, distances

    ys, xs = np.where(road_mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    # ── Collect one probe point per grid cell ─────────────────────────────
    row_edges = np.linspace(y_min, y_max + 1, n_rows + 1)
    col_edges = np.linspace(x_min, x_max + 1, n_cols + 1)

    probe_uvs: List[Tuple[float, float]] = []
    probe_dists: List[Optional[float]] = []

    for r in range(n_rows):
        for c in range(n_cols):
            in_cell = (
                (ys >= row_edges[r]) & (ys < row_edges[r + 1]) &
                (xs >= col_edges[c]) & (xs < col_edges[c + 1])
            )
            if not in_cell.any():
                continue
            u_use = float(xs[in_cell].mean())
            v_use = float(ys[in_cell].mean())
            probe_uvs.append((u_use, v_use))

            r_dir = ray_direction(u_use, v_use, intrinsics)
            p3d = ray_plane_intersection(r_dir, plane.n, plane.d)
            if p3d is not None:
                dist = float(np.linalg.norm(p3d))
                probe_dists.append(dist)
                distances.append(dist)
            else:
                probe_dists.append(None)

    # ── Colour map: hue 0 (red=near) → 120 (green=far) ───────────────────
    valid = [d for d in probe_dists if d is not None]
    d_lo = min(valid) if valid else 0.0
    d_hi = max(valid) if valid else 1.0
    d_range = max(d_hi - d_lo, 1e-3)

    for (u_use, v_use), dist in zip(probe_uvs, probe_dists):
        pu, pv = int(round(u_use)), int(round(v_use))

        if dist is not None:
            t = (dist - d_lo) / d_range        # 0 = near, 1 = far
            hue = int(t * 120)                 # 0=red, 60=yellow, 120=green
            hsv_px = np.array([[[hue, 210, 210]]], dtype=np.uint8)
            bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0, 0]
            color: Tuple[int, int, int] = (int(bgr_px[0]), int(bgr_px[1]), int(bgr_px[2]))
        else:
            color = (80, 80, 200)

        cv2.circle(out, (pu, pv), 9, color, -1)
        cv2.circle(out, (pu, pv), 11, (0, 0, 0), 1)

        if dist is not None:
            label = f"{dist:.1f}"
            lw = len(label) * 9 + 6
            tx = pu + 13 if pu + lw + 13 < W else pu - lw - 13
            ty = max(20, pv - 5)
            cv2.rectangle(out, (tx - 2, ty - 15), (tx + lw, ty + 3), (0, 0, 0), -1)
            cv2.putText(
                out, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA,
            )
        else:
            cv2.putText(
                out, "?", (pu + 13, pv),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 200), 1, cv2.LINE_AA,
            )

    # ── Stats box ─────────────────────────────────────────────────────────
    if show_stats and valid:
        n = plane.n
        cam_dist = abs(plane.d) / (float(np.linalg.norm(n)) + 1e-12)
        lines = [
            f"n=({n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f})",
            f"cam-plane={cam_dist:.2f}  inlier={plane.inlier_ratio:.2f}",
            f"d=[{d_lo:.1f}, {d_hi:.1f}]  pts={len(valid)}/{len(probe_dists)}",
        ]
        box_h = len(lines) * 20 + 10
        box_w = 310
        cv2.rectangle(out, (5, 5), (box_w, box_h), (0, 0, 0), -1)
        for li, txt in enumerate(lines):
            cv2.putText(
                out, txt, (10, 22 + li * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 230, 255), 1, cv2.LINE_AA,
            )

    return out, distances


# ---------------------------------------------------------------------------
# Compound frame builder
# ---------------------------------------------------------------------------

def build_scene_frame(
    frame: np.ndarray,
    road_mask: np.ndarray,
    depth: np.ndarray,
    plane: Optional[PlaneParams],
    intrinsics: CameraIntrinsics,
    grid_spacing: float = 1.0,
    grid_extent: float = 10.0,
    show_grid: bool = True,
) -> np.ndarray:
    """Build a 2×2 composite visualisation for a single processed frame.

    Layout
    ------
    ┌───────────────────┬──────────────────┐
    │ Mask + grid       │ Depth colourmap  │
    ├───────────────────┼──────────────────┤
    │ Road-only depth   │ Depth + contour  │
    └───────────────────┴──────────────────┘

    Parameters
    ----------
    frame                 : (H, W, 3) uint8 BGR image.
    road_mask             : (H, W) boolean road mask.
    depth                 : (H, W) float depth map.
    plane                 : current stable ground plane, or None.
    intrinsics            : camera intrinsics.
    grid_spacing, grid_extent : forwarded to draw_plane_grid.
    show_grid             : if False, skip the grid overlay.

    Returns
    -------
    (2H, 2W, 3) uint8 composite BGR image.
    """
    panel_a = overlay_mask(frame, road_mask)
    if show_grid and plane is not None:
        panel_a = draw_plane_grid(
            panel_a, intrinsics, plane,
            grid_spacing=grid_spacing, grid_extent=grid_extent,
        )
        panel_a, _ = draw_ground_probe_grid(
            panel_a, intrinsics, plane, road_mask,
            n_rows=3, n_cols=4,
        )

    panel_b = depth_viz(depth)

    panel_c = road_depth_overlay(frame, depth, road_mask)

    panel_d = depth_viz(depth)
    mask_u8 = road_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(panel_d, contours, -1, (0, 255, 100), 2)

    top = np.concatenate([panel_a, panel_b], axis=1)
    bot = np.concatenate([panel_c, panel_d], axis=1)
    return np.concatenate([top, bot], axis=0)
