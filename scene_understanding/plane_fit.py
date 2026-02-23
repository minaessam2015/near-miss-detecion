"""
scene_understanding.plane_fit
------------------------------
Ground-plane estimation from 3-D point clouds using RANSAC, followed by
EMA smoothing and per-frame validation.

Plane convention
----------------
A plane is expressed as::

    n · P + d = 0

where ``n`` is a **unit** normal vector and ``d`` is the scalar offset.
The closest point on the plane to the camera origin (0, 0, 0) is::

    P₀ = -d * n

Key classes / functions
-----------------------
:class:`PlaneParams`      — immutable container for a fitted plane.
:func:`ransac_plane_fit`  — RANSAC + optional SVD refinement.
:func:`validate_plane`    — inlier-ratio and angle-change checks.
:func:`ema_smooth_plane`  — EMA blend of two :class:`PlaneParams`.
:class:`ScenePlaneTracker` — stateful tracker; call ``update()`` per frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .geometry import angle_between_normals


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlaneParams:
    """Container for a fitted (or smoothed) plane estimate.

    Attributes
    ----------
    n            : (3,) unit normal vector.
    d            : scalar plane offset  (n · P + d = 0).
    inlier_ratio : fraction of input points within ``inlier_thresh`` of the
                   fitted plane  (0–1).
    frame_idx    : source frame index (−1 when not associated with a frame).
    valid        : whether this plane passed the validation checks.
    n_inliers    : absolute count of inlier points.
    n_points     : total input point count.
    """

    n: np.ndarray            # (3,) float64 unit normal
    d: float
    inlier_ratio: float
    frame_idx: int
    valid: bool
    n_inliers: int = 0
    n_points: int = 0

    # ── Convenience methods ───────────────────────────────────────────────────

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from each point to the plane: ``n · P + d``.

        Parameters
        ----------
        points : (N, 3) float array of 3-D points.

        Returns
        -------
        (N,) float array.
        """
        return points @ self.n + self.d

    def abs_distance(self, points: np.ndarray) -> np.ndarray:
        """Absolute distance from each point to the plane."""
        return np.abs(self.signed_distance(points))

    def closest_point_to_origin(self) -> np.ndarray:
        """Return the point on the plane closest to the camera origin.

        This is ``-d * n``.
        """
        return -self.d * self.n

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "n": self.n.tolist(),
            "d": float(self.d),
            "inlier_ratio": float(self.inlier_ratio),
            "frame_idx": int(self.frame_idx),
            "valid": bool(self.valid),
            "n_inliers": int(self.n_inliers),
            "n_points": int(self.n_points),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlaneParams":
        """Deserialise from a dict produced by :meth:`to_dict`."""
        return cls(
            n=np.array(data["n"], dtype=np.float64),
            d=float(data["d"]),
            inlier_ratio=float(data["inlier_ratio"]),
            frame_idx=int(data.get("frame_idx", -1)),
            valid=bool(data["valid"]),
            n_inliers=int(data.get("n_inliers", 0)),
            n_points=int(data.get("n_points", 0)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Low-level plane geometry
# ─────────────────────────────────────────────────────────────────────────────

def _fit_plane_3pts(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> Optional[Tuple[np.ndarray, float]]:
    """Fit a plane through three 3-D points.

    Returns ``(n, d)`` where *n* is the unit normal and *d* is the offset
    so that ``n · P + d = 0`` for all points on the plane.
    Returns ``None`` if the points are (near-) collinear.

    Parameters
    ----------
    p0, p1, p2 : (3,) arrays.
    """
    v1 = p1 - p0
    v2 = p2 - p0
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-10:
        return None
    n = n / norm
    d = -float(n @ p0)
    return n, d


def _fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit a plane to a set of 3-D points using SVD (PCA).

    The normal is the left-singular vector corresponding to the smallest
    singular value (minimum variance direction).

    Parameters
    ----------
    points : (N, 3) float array, N ≥ 3.

    Returns
    -------
    n : (3,) unit normal.
    d : scalar offset.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    # SVD of (N, 3): rows are observations, columns are xyz
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    n = Vt[-1]  # row with smallest singular value
    n = n / (np.linalg.norm(n) + 1e-12)
    d = -float(n @ centroid)
    return n, d


# ─────────────────────────────────────────────────────────────────────────────
# RANSAC plane fitting
# ─────────────────────────────────────────────────────────────────────────────

def ransac_plane_fit(
    points: np.ndarray,
    ransac_iters: int = 150,
    inlier_thresh: float = 0.05,
    min_inlier_ratio: float = 0.30,
    refine_with_svd: bool = True,
    rng: Optional[np.random.Generator] = None,
    early_stop_ratio: float = 0.80,
) -> Optional[PlaneParams]:
    """Fit a plane to 3-D points with RANSAC, optionally refined by SVD.

    Algorithm
    ---------
    1.  Randomly sample 3 points and compute their plane.
    2.  Count inliers: points with ``|n · P + d| < inlier_thresh``.
    3.  Keep the plane with the most inliers.
    4.  (Optional) Refit to all inliers using SVD (least-squares).

    Parameters
    ----------
    points          : (N, 3) float array of 3-D points.
    ransac_iters    : maximum number of RANSAC trials.
    inlier_thresh   : inlier distance threshold, same units as *points*.
    min_inlier_ratio: minimum fraction of inliers required for a valid plane.
    refine_with_svd : if ``True``, refit to all inliers after RANSAC.
    rng             : numpy random generator for reproducibility.
    early_stop_ratio: stop early if inlier fraction exceeds this threshold
                      (avoids wasting iterations when a good plane is found).

    Returns
    -------
    :class:`PlaneParams` or ``None`` if fewer than 3 points or all trials fail.
    """
    if rng is None:
        rng = np.random.default_rng()

    points = np.asarray(points, dtype=np.float64)
    N = len(points)
    if N < 3:
        return None

    best_n: Optional[np.ndarray] = None
    best_d: float = 0.0
    best_count: int = 0
    best_mask: Optional[np.ndarray] = None

    for _ in range(ransac_iters):
        idx = rng.choice(N, size=3, replace=False)
        result = _fit_plane_3pts(points[idx[0]], points[idx[1]], points[idx[2]])
        if result is None:
            continue
        n_cand, d_cand = result

        dists = np.abs(points @ n_cand + d_cand)
        inlier_mask = dists < inlier_thresh
        count = int(inlier_mask.sum())

        if count > best_count:
            best_count = count
            best_n = n_cand.copy()
            best_d = d_cand
            best_mask = inlier_mask.copy()

            if count / N >= early_stop_ratio:
                break

    if best_n is None or best_count == 0:
        return None

    # ── SVD refinement ────────────────────────────────────────────────────────
    if refine_with_svd and best_count >= 3 and best_mask is not None:
        best_n, best_d = _fit_plane_svd(points[best_mask])
        # Recount inliers with refined plane
        dists = np.abs(points @ best_n + best_d)
        best_mask = dists < inlier_thresh
        best_count = int(best_mask.sum())

    inlier_ratio = best_count / N

    return PlaneParams(
        n=best_n,
        d=best_d,
        inlier_ratio=inlier_ratio,
        frame_idx=-1,
        valid=inlier_ratio >= min_inlier_ratio,
        n_inliers=best_count,
        n_points=N,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation and smoothing
# ─────────────────────────────────────────────────────────────────────────────

def validate_plane(
    new_plane: PlaneParams,
    prev_plane: Optional[PlaneParams],
    min_inlier_ratio: float = 0.30,
    max_normal_angle_change_deg: float = 15.0,
) -> bool:
    """Check whether a newly fitted plane should be accepted.

    Two criteria must both pass:

    1.  ``inlier_ratio >= min_inlier_ratio``
    2.  If a previous valid plane exists: the angle between normals must
        be ≤ ``max_normal_angle_change_deg``.

    Parameters
    ----------
    new_plane                   : newly fitted plane to validate.
    prev_plane                  : last accepted stable plane, or ``None``.
    min_inlier_ratio            : minimum acceptable inlier fraction.
    max_normal_angle_change_deg : maximum acceptable normal-angle change.

    Returns
    -------
    ``True`` if the plane passes all checks.
    """
    if new_plane.inlier_ratio < min_inlier_ratio:
        return False

    if prev_plane is not None and prev_plane.valid:
        angle = angle_between_normals(new_plane.n, prev_plane.n)
        if angle > max_normal_angle_change_deg:
            return False

    return True


def ema_smooth_plane(
    prev_plane: PlaneParams,
    new_plane: PlaneParams,
    alpha: float = 0.20,
) -> PlaneParams:
    """Blend two planes with exponential moving average (EMA).

    Blending is done in (n, d) space::

        n_blend = normalize((1 - α) * n_prev + α * n_new)
        d_blend = (1 - α) * d_prev + α * d_new

    Renormalising *n* after blending keeps it a unit vector regardless of
    how far the normals diverge.

    Parameters
    ----------
    prev_plane : current stable plane (the "old" state).
    new_plane  : newly validated plane (the "new" observation).
    alpha      : blending weight for the new observation; 0 = keep old,
                 1 = replace immediately.

    Returns
    -------
    New :class:`PlaneParams` with blended (n, d) and the metadata of
    *new_plane*.
    """
    n_blend = (1.0 - alpha) * prev_plane.n + alpha * new_plane.n
    norm = np.linalg.norm(n_blend)
    n_smooth = n_blend / norm if norm > 1e-12 else new_plane.n.copy()
    d_smooth = (1.0 - alpha) * prev_plane.d + alpha * new_plane.d

    return PlaneParams(
        n=n_smooth,
        d=d_smooth,
        inlier_ratio=new_plane.inlier_ratio,
        frame_idx=new_plane.frame_idx,
        valid=True,
        n_inliers=new_plane.n_inliers,
        n_points=new_plane.n_points,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stateful plane tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _HistoryEntry:
    """Single-frame entry in the plane tracker history log."""
    frame_idx: int
    plane_valid: bool
    inlier_ratio: float
    normal_angle_change: Optional[float]
    plane_d: Optional[float]
    n_inliers: int
    n_points: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "plane_valid": self.plane_valid,
            "inlier_ratio": self.inlier_ratio,
            "normal_angle_change": self.normal_angle_change,
            "plane_d": self.plane_d,
            "n_inliers": self.n_inliers,
            "n_points": self.n_points,
        }


class ScenePlaneTracker:
    """Stateful ground-plane tracker with EMA smoothing and history logging.

    Call :meth:`update` once per processed frame with the freshly fitted
    :class:`PlaneParams` (or ``None`` when no plane could be fitted).
    The tracker maintains a *stable plane* that is updated only when the new
    fit is valid, and freezes otherwise.

    History of per-frame quality metrics is stored in :attr:`history` and
    can be exported via :meth:`get_history_dicts` for JSON serialisation.

    Parameters
    ----------
    min_inlier_ratio            : validation threshold forwarded to
                                  :func:`validate_plane`.
    max_normal_angle_change_deg : validation threshold forwarded to
                                  :func:`validate_plane`.
    ema_alpha                   : EMA weight forwarded to
                                  :func:`ema_smooth_plane`.
    """

    def __init__(
        self,
        min_inlier_ratio: float = 0.30,
        max_normal_angle_change_deg: float = 15.0,
        ema_alpha: float = 0.20,
    ) -> None:
        self.min_inlier_ratio = min_inlier_ratio
        self.max_normal_angle_change_deg = max_normal_angle_change_deg
        self.ema_alpha = ema_alpha

        self.stable_plane: Optional[PlaneParams] = None
        self._history: List[_HistoryEntry] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        new_plane: Optional[PlaneParams],
        frame_idx: int,
    ) -> Optional[PlaneParams]:
        """Update the stable plane estimate with a new observation.

        Parameters
        ----------
        new_plane : freshly fitted plane, or ``None`` when fitting failed.
        frame_idx : current frame index.

        Returns
        -------
        The current *stable* plane (may be from an earlier frame), or
        ``None`` if no valid plane has been established yet.
        """
        angle_change: Optional[float] = None
        plane_valid = False
        inlier_ratio = 0.0
        n_inliers = 0
        n_points = 0

        if new_plane is not None:
            inlier_ratio = new_plane.inlier_ratio
            n_inliers = new_plane.n_inliers
            n_points = new_plane.n_points

            if self.stable_plane is not None:
                angle_change = angle_between_normals(
                    new_plane.n, self.stable_plane.n
                )
            else:
                angle_change = 0.0

            plane_valid = validate_plane(
                new_plane,
                self.stable_plane,
                self.min_inlier_ratio,
                self.max_normal_angle_change_deg,
            )

            if plane_valid:
                new_plane.frame_idx = frame_idx
                if self.stable_plane is None:
                    self.stable_plane = new_plane
                else:
                    self.stable_plane = ema_smooth_plane(
                        self.stable_plane, new_plane, self.ema_alpha
                    )
                    self.stable_plane.frame_idx = frame_idx

        self._history.append(_HistoryEntry(
            frame_idx=frame_idx,
            plane_valid=plane_valid,
            inlier_ratio=inlier_ratio,
            normal_angle_change=angle_change,
            plane_d=float(self.stable_plane.d) if self.stable_plane else None,
            n_inliers=n_inliers,
            n_points=n_points,
        ))
        return self.stable_plane

    def reset(self) -> None:
        """Clear the stable plane and history (useful for mode switching)."""
        self.stable_plane = None
        self._history.clear()

    @property
    def history(self) -> List[_HistoryEntry]:
        """Read-only list of :class:`_HistoryEntry` per processed frame."""
        return list(self._history)

    def get_history_dicts(self) -> List[Dict[str, Any]]:
        """Return the history as a list of JSON-serialisable dicts."""
        return [e.to_dict() for e in self._history]

    def get_summary_dict(self) -> Dict[str, Any]:
        """Return a summary dict for JSON report export."""
        valid_frames = [e for e in self._history if e.plane_valid]
        return {
            "total_frames": len(self._history),
            "valid_plane_frames": len(valid_frames),
            "valid_ratio": len(valid_frames) / len(self._history) if self._history else 0.0,
            "final_plane": self.stable_plane.to_dict() if self.stable_plane else None,
            "per_frame": self.get_history_dicts(),
        }
