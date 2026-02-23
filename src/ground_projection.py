"""
src.ground_projection
---------------------
Bridges scene_understanding geometry to the tracking pipeline.

GroundProjector converts each detection's bounding-box bottom-center (the
ground-contact proxy) from pixel coordinates to a 3-D point on the fitted
ground plane, using ray–plane intersection.  The resulting ``ground_pt`` field
is then consumed by CentroidTrackerV2 (for 3-D distance matching) and
NearMissDetectorV20 (for physical-scale proximity and TTC).

Why bottom-center?
    The bottom edge of a bounding box is the pixel where the object meets the
    road surface (for upright objects viewed from a slightly elevated camera).
    Casting a ray through this point to the ground plane gives the physical
    ground-contact location in 3-D camera space.

Coordinate convention
    All returned points are in **camera space** (X right, Y down, Z forward).
    Units match the depth scale produced by DepthAnythingV2 with the p95=10
    normalisation — roughly 1 unit ≈ 1 metre for typical traffic scenes.

Usage
-----
    from src.ground_projection import GroundProjector

    projector = GroundProjector(intrinsics, stable_plane)

    # Enrich a list of detection dicts before passing to the tracker:
    dets = projector.enrich_detections(dets)   # adds "ground_pt" field

    # Or project a single pixel:
    pt3d = projector.pixel_to_ground(u=640.0, v=480.0)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scene_understanding.geometry import CameraIntrinsics, ray_direction
from scene_understanding.plane_fit import PlaneParams

_log = logging.getLogger("ground_projection")


class GroundProjector:
    """Convert bounding-box pixels to 3-D ground-plane points.

    Parameters
    ----------
    intrinsics : CameraIntrinsics
        Pinhole camera intrinsics (fx, fy, cx, cy).
    plane : PlaneParams
        Fitted ground plane (n, d).  May be ``None`` or ``plane.valid=False``
        — in that case all projections return ``None`` gracefully.

    Thread safety
    -------------
    Read-only after construction; safe to share across threads.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        plane: Optional[PlaneParams],
        t_max: Optional[float] = 20.0,
    ) -> None:
        """
        Parameters
        ----------
        intrinsics : camera intrinsics.
        plane      : fitted ground plane.  May be None / invalid.
        t_max      : maximum ray-parameter t accepted as a valid intersection.
                     Rays that intersect the plane very far from the camera
                     (near the visual horizon) give t → ∞ and produce
                     astronomically large 3-D coordinates.  Any intersection
                     with t > t_max is treated as "no hit" and returns None.
                     Set to ``None`` (or <=0) to disable this cap.
        """
        self.intrinsics = intrinsics
        self.plane = plane
        self.t_max = float(t_max) if (t_max is not None and t_max > 0.0) else None
        if not self.is_valid:
            _log.warning(
                "GroundProjector created with invalid/None plane — "
                "all projections will return None until plane is updated."
            )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_valid(self) -> bool:
        """True when the plane is set and marked valid."""
        return self.plane is not None and self.plane.valid

    def update_plane(self, plane: Optional[PlaneParams]) -> None:
        """Hot-swap the ground plane (e.g. each frame in a live pipeline)."""
        self.plane = plane
        if not self.is_valid:
            _log.debug("update_plane: plane invalid or None — projections disabled")

    # ── Core projection ───────────────────────────────────────────────────────

    def pixel_to_ground(
        self,
        u: float,
        v: float,
    ) -> Optional[Tuple[float, float, float]]:
        """Cast a ray through pixel (u, v) and intersect with the ground plane.

        Parameters
        ----------
        u, v : pixel column and row.

        Returns
        -------
        (X, Y, Z) in camera space, or ``None`` if the plane is invalid,
        the ray is parallel to / behind the plane, or t > t_max (near-horizon).
        """
        if not self.is_valid:
            return None
        r = ray_direction(u, v, self.intrinsics)
        n     = self.plane.n
        d     = float(self.plane.d)
        denom = float(np.dot(n, r))
        if abs(denom) < 1e-8:
            _log.debug("pixel_to_ground: ray parallel to plane at (%.1f,%.1f)", u, v)
            return None
        t = -d / denom
        if t < 0.0:
            _log.debug(
                "pixel_to_ground: t=%.3f<0 (behind camera) at (%.1f,%.1f)", t, u, v
            )
            return None
        if self.t_max is not None and t > self.t_max:
            _log.debug(
                "pixel_to_ground: t=%.3f>t_max=%.3f (near horizon) at (%.1f,%.1f)",
                t, self.t_max, u, v,
            )
            return None
        pt = t * np.asarray(r, dtype=np.float64)
        return (float(pt[0]), float(pt[1]), float(pt[2]))

    def bbox_footpoint_to_ground(
        self,
        bbox: List[float],
    ) -> Optional[Tuple[float, float, float]]:
        """Project the bbox bottom-center to the ground plane.

        Parameters
        ----------
        bbox : [x1, y1, x2, y2] in pixels.

        Returns
        -------
        (X, Y, Z) in camera space, or ``None``.
        """
        x1, _y1, x2, y2 = bbox
        u = (x1 + x2) / 2.0
        v = float(y2)
        return self.pixel_to_ground(u, v)

    # ── Batch enrichment ──────────────────────────────────────────────────────

    def enrich_detections(
        self,
        dets: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add ``'ground_pt'`` to every detection dict.

        Adds ``ground_pt: Optional[Tuple[float, float, float]]`` by projecting
        each detection's bounding-box bottom-center to the ground plane.
        Modifies the dicts **in place** and returns the same list.

        Parameters
        ----------
        dets : list of detection dicts.  Each must have a ``'bbox'`` field.

        Returns
        -------
        The same ``dets`` list with ``'ground_pt'`` populated.
        """
        n_hit = n_miss = 0
        for det in dets:
            gp = self.bbox_footpoint_to_ground(det["bbox"])
            det["ground_pt"] = gp
            if gp is not None:
                n_hit += 1
                _log.debug(
                    "enrich: bbox=[%.0f,%.0f,%.0f,%.0f] → gp=(%.3f,%.3f,%.3f)",
                    *det["bbox"], *gp,
                )
            else:
                n_miss += 1
        if dets:
            _log.debug(
                "enrich_detections: %d/%d projected (plane_valid=%s)",
                n_hit, len(dets), self.is_valid,
            )
        return dets

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def ground_distance(
        pt1: Tuple[float, float, float],
        pt2: Tuple[float, float, float],
    ) -> float:
        """Euclidean distance between two ground-plane 3-D points."""
        return float(np.linalg.norm(np.array(pt2, dtype=np.float64)
                                    - np.array(pt1, dtype=np.float64)))

    @staticmethod
    def dedup_detections(
        dets: List[Dict[str, Any]],
        gp_dist:  float = 0.02,
        px_dist:  float = 60.0,
    ) -> List[Dict[str, Any]]:
        """Remove duplicate detections of the same physical object.

        The detector's within-class NMS (IoU threshold ≈ 0.45) does not
        suppress two partial detections of the same large vehicle (e.g. a bus
        detected as front + rear halves) when their IoU < 0.45.  Both survive
        NMS, enter the centroid tracker as separate tracks, and can trigger a
        false near-miss event between them.

        This function applies a second-pass deduplication using ground-plane
        distance (when both detections have a valid ``ground_pt``) or 2-D
        centroid distance (fallback).  Only detections of the same class are
        compared.  The higher-confidence detection of each duplicate pair is
        kept; the lower-confidence one is removed.

        Parameters
        ----------
        dets     : list of detection dicts, already enriched with
                   ``ground_pt`` by :meth:`enrich_detections`.
        gp_dist  : 3-D ground-plane distance threshold (depth units).
                   Two detections whose foot-points are within this distance
                   are treated as the same physical object.
                   Default 0.02 ≈ ~80 px at typical depth (t≈0.25, fx≈1088).
        px_dist  : 2-D centroid distance threshold (pixels) used when either
                   detection lacks a ground_pt.  Default 60 px.

        Returns
        -------
        Deduplicated list (same dicts, not copies; lower-confidence
        duplicates are simply excluded).
        """
        # Sort descending by confidence so we always keep the better detection
        ordered = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        kept: List[Dict[str, Any]] = []

        for cand in ordered:
            duplicate = False
            cgp = cand.get("ground_pt")
            ccx, ccy = cand["center"]

            for ref in kept:
                # Only compare same-class detections
                if ref["class"] != cand["class"]:
                    continue

                rgp = ref.get("ground_pt")

                if cgp is not None and rgp is not None:
                    dist = float(np.linalg.norm(
                        np.array(cgp, dtype=np.float64)
                        - np.array(rgp, dtype=np.float64)
                    ))
                    if dist < gp_dist:
                        duplicate = True
                        break
                else:
                    # Fallback: 2-D centroid distance
                    rcx, rcy = ref["center"]
                    if (abs(ccx - rcx) < px_dist and abs(ccy - rcy) < px_dist):
                        duplicate = True
                        break

            if not duplicate:
                kept.append(cand)

        if len(kept) < len(dets):
            _log.debug(
                "dedup_detections: removed %d duplicate(s) (%d → %d)",
                len(dets) - len(kept), len(dets), len(kept),
            )
        return kept

    def __repr__(self) -> str:
        valid = self.is_valid
        plane_str = (
            f"n=[{self.plane.n[0]:+.3f},{self.plane.n[1]:+.3f},{self.plane.n[2]:+.3f}] "
            f"d={self.plane.d:.4f} inlier={self.plane.inlier_ratio:.2f}"
            if valid and self.plane is not None
            else "invalid"
        )
        return (
            f"GroundProjector(valid={valid}, "
            f"fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}, "
            f"t_max={self.t_max}, plane={plane_str})"
        )
