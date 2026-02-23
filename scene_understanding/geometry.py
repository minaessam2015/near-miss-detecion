"""
scene_understanding.geometry
-----------------------------
Camera geometry primitives: intrinsics, backprojection, ray–plane
intersection, and 3-D → image projection.

All coordinates are in *camera space*:
    +X  → right
    +Y  → down   (standard OpenCV / pinhole convention)
    +Z  → forward (optical axis)

Depth values are assumed to be the Z-component of the 3-D point (i.e.
the distance along the optical axis, **not** the Euclidean ray length).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Camera intrinsics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters.

    Attributes
    ----------
    fx, fy : focal lengths in pixels.
    cx, cy : principal point (image centre in pixels).

    All units are pixels unless otherwise noted.
    """

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_frame(
        cls,
        frame_or_shape: "np.ndarray | Tuple[int, int]",
        fx: Optional[float] = None,
        fy: Optional[float] = None,
    ) -> "CameraIntrinsics":
        """Build intrinsics from a frame array (or (H, W) tuple).

        If *fx* / *fy* are ``None`` a crude approximation is used:
        ``fx = fy = max(H, W) * 0.85``, which corresponds to a ~50° FoV.

        Parameters
        ----------
        frame_or_shape : (H, W, C) ndarray or (H, W) tuple.
        fx, fy         : override focal lengths (pixels).  Both default to
                         the estimated value when omitted.
        """
        if hasattr(frame_or_shape, "shape"):
            H, W = frame_or_shape.shape[:2]
        else:
            H, W = frame_or_shape[:2]

        cx_c = W / 2.0
        cy_c = H / 2.0
        default_f = max(H, W) * 0.85
        fx_c = fx if fx is not None else default_f
        fy_c = fy if fy is not None else fx_c
        return cls(fx=fx_c, fy=fy_c, cx=cx_c, cy=cy_c)

    def scale(self, sx: float, sy: float) -> "CameraIntrinsics":
        """Return a new :class:`CameraIntrinsics` scaled by (sx, sy).

        Useful when the image is resized before inference.

        Parameters
        ----------
        sx : horizontal scale factor (new_W / original_W).
        sy : vertical scale factor (new_H / original_H).
        """
        return CameraIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
        )

    def as_matrix(self) -> np.ndarray:
        """Return the 3×3 intrinsic matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0,    1.0]],
            dtype=np.float64,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Backprojection utilities
# ─────────────────────────────────────────────────────────────────────────────

def backproject_pixel(
    u: float,
    v: float,
    depth: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Backproject a single image pixel with known depth to 3-D camera space.

    Parameters
    ----------
    u, v     : pixel coordinates (column, row).
    depth    : Z-component in camera space (along the optical axis).
    intrinsics : pinhole intrinsics.

    Returns
    -------
    P : (3,) float64 array [X, Y, Z] in camera coordinates.

    Notes
    -----
    The formula follows the standard pinhole model::

        X = (u - cx) / fx * Z
        Y = (v - cy) / fy * Z
        Z = depth
    """
    Z = float(depth)
    X = (u - intrinsics.cx) / intrinsics.fx * Z
    Y = (v - intrinsics.cy) / intrinsics.fy * Z
    return np.array([X, Y, Z], dtype=np.float64)


def backproject_depth_map(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask: Optional[np.ndarray] = None,
    max_points: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backproject an entire depth map (or a masked subset) to 3-D points.

    Parameters
    ----------
    depth      : (H, W) float32/64 depth map.  Non-positive and NaN pixels
                 are excluded automatically.
    intrinsics : camera intrinsics.
    mask       : (H, W) boolean array; only *True* pixels are backprojected.
                 ``None`` → backproject all valid depth pixels.
    max_points : if set, randomly subsample down to this many points before
                 returning (reduces memory and downstream computation cost).
    rng        : numpy random generator for reproducible subsampling.

    Returns
    -------
    points_3d   : (N, 3) float64 array [X, Y, Z].
    pixel_coords: (N, 2) float64 array [(u, v), ...] for each point.
    """
    H, W = depth.shape[:2]
    us, vs = np.meshgrid(
        np.arange(W, dtype=np.float64),
        np.arange(H, dtype=np.float64),
    )

    valid: np.ndarray = (depth > 0) & np.isfinite(depth)
    if mask is not None:
        valid = valid & mask.astype(bool)

    us_v = us[valid]
    vs_v = vs[valid]
    zs_v = depth[valid].astype(np.float64)

    if len(zs_v) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    # Optional subsampling
    if max_points is not None and len(zs_v) > max_points:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(len(zs_v), size=max_points, replace=False)
        us_v = us_v[idx]
        vs_v = vs_v[idx]
        zs_v = zs_v[idx]

    xs = (us_v - intrinsics.cx) / intrinsics.fx * zs_v
    ys = (vs_v - intrinsics.cy) / intrinsics.fy * zs_v

    points_3d = np.stack([xs, ys, zs_v], axis=1)     # (N, 3)
    pixel_coords = np.stack([us_v, vs_v], axis=1)     # (N, 2)
    return points_3d, pixel_coords


# ─────────────────────────────────────────────────────────────────────────────
# Ray utilities
# ─────────────────────────────────────────────────────────────────────────────

def ray_direction(
    u: float,
    v: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Compute the unit-length ray direction for pixel (u, v).

    The ray originates at the camera centre (origin) and passes through
    the backprojected point at unit depth::

        r = normalize([(u - cx)/fx,  (v - cy)/fy,  1])

    Parameters
    ----------
    u, v       : pixel coordinates (column, row).
    intrinsics : camera intrinsics.

    Returns
    -------
    r : (3,) unit vector in camera coordinates.
    """
    r = np.array(
        [(u - intrinsics.cx) / intrinsics.fx,
         (v - intrinsics.cy) / intrinsics.fy,
         1.0],
        dtype=np.float64,
    )
    norm = np.linalg.norm(r)
    return r / norm if norm > 1e-12 else r


def ray_plane_intersection(
    ray_dir: np.ndarray,
    plane_n: np.ndarray,
    plane_d: float,
    eps: float = 1e-8,
) -> Optional[np.ndarray]:
    """Intersect a ray from the camera origin with a plane.

    The ray is parameterised as ``P(t) = t * ray_dir``  (t ≥ 0).
    The plane is ``n · P + d = 0``.

    Solving::

        t * (n · ray_dir) + d = 0
        t = -d / (n · ray_dir)

    Parameters
    ----------
    ray_dir  : (3,) unit or non-unit ray direction (camera space).
    plane_n  : (3,) unit plane normal.
    plane_d  : scalar plane offset.  The plane passes through the point
               ``-d * n`` (closest point on the plane to the origin).
    eps      : denominator threshold; returns ``None`` when the ray is
               nearly parallel to the plane.

    Returns
    -------
    intersection : (3,) 3-D point in camera coordinates, or ``None`` if
                   the ray is parallel to the plane or the intersection is
                   behind the camera (t < 0).
    """
    denom = float(np.dot(plane_n, ray_dir))
    if abs(denom) < eps:
        return None          # ray parallel to plane (or plane_n ≈ 0)

    t = -plane_d / denom
    if t < 0.0:
        return None          # intersection is behind the camera

    return t * np.asarray(ray_dir, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 3-D → image projection
# ─────────────────────────────────────────────────────────────────────────────

def project_3d_to_image(
    points_3d: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3-D camera-space points to image pixel coordinates.

    Parameters
    ----------
    points_3d  : (N, 3) array [X, Y, Z] in camera coordinates.
    intrinsics : camera intrinsics.

    Returns
    -------
    us, vs : (N,) float64 arrays of pixel coordinates (column, row).
             Points with Z ≤ 0 are assigned NaN.
    """
    pts = np.asarray(points_3d, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]

    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    valid = Z > 0

    us = np.full(len(Z), np.nan, dtype=np.float64)
    vs = np.full(len(Z), np.nan, dtype=np.float64)
    us[valid] = intrinsics.fx * X[valid] / Z[valid] + intrinsics.cx
    vs[valid] = intrinsics.fy * Y[valid] / Z[valid] + intrinsics.cy
    return us, vs


# ─────────────────────────────────────────────────────────────────────────────
# Angular utilities
# ─────────────────────────────────────────────────────────────────────────────

def angle_between_normals(n1: np.ndarray, n2: np.ndarray) -> float:
    """Angle in degrees between two (possibly flipped) unit normals.

    Uses the absolute value of the cosine so that antiparallel normals
    (which describe the same plane) report 0° instead of 180°.

    Parameters
    ----------
    n1, n2 : (3,) normal vectors (need not be unit length).

    Returns
    -------
    angle : float in [0°, 90°].
    """
    norm1 = np.linalg.norm(n1)
    norm2 = np.linalg.norm(n2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    # Compute cos(θ) via explicit division — avoids the (norm+ε)² deflation
    # that occurs when adding ε to the norm before squaring.
    cos_theta = float(abs(np.dot(n1, n2) / (norm1 * norm2)))
    cos_theta = float(np.clip(cos_theta, 0.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def ground_distance(point_3d: np.ndarray) -> float:
    """Euclidean distance from the camera origin to a 3-D point.

    In camera coordinates the camera is at the origin, so this is simply
    ``||point_3d||``.

    Parameters
    ----------
    point_3d : (3,) array in camera coordinates.

    Returns
    -------
    distance : scalar float.
    """
    return float(np.linalg.norm(point_3d))
