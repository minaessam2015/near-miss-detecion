"""
scene_understanding.calibration
--------------------------------
Unknown-intrinsics calibration via focal-length multiplier search.

Problem
-------
CCTV cameras rarely come with documented intrinsics.  A fully uncalibrated
camera has unknown focal length (fx, fy) and principal point (cx, cy).
We make two practical simplifying assumptions:

    cx = W / 2,   cy = H / 2          (principal point at image centre)
    fx = fy = k * max(W, H)           (square pixels, k is the unknown)

The task is then to find the scalar *k* that makes the ground-plane fit as
consistent as possible across a small set of bootstrap frames.

Algorithm — k-search
--------------------
For each candidate k in ``k_candidates = [0.8, 1.0, 1.2, 1.4]``:

1.  Build ``CameraIntrinsics(fx = fy = k * max(W, H), cx = W/2, cy = H/2)``.
2.  For every bootstrap frame (road_mask + depth):
      a.  Backproject road pixels to 3-D points.
      b.  Run RANSAC plane fitting.
      c.  Record: inlier_ratio, normal_angle_change (vs previous frame).
3.  Score k with::

        score(k) = median(inlier_ratios)
                   - lambda1 * std(normal_angle_changes)
                   - lambda2 * (invalid_frames / total_frames)

    Higher score → better k.

4.  Select k* = argmax score(k).

Outputs
-------
``IntrinsicsCalibrator.calibrate()`` returns a :class:`CalibrationResult`
containing the chosen k, the corresponding :class:`CameraIntrinsics`, and
per-k diagnostics useful for inspection.

Usage
-----
::

    from scene_understanding.calibration import IntrinsicsCalibrator

    calibrator = IntrinsicsCalibrator(cfg)
    result = calibrator.calibrate(frames, road_masks, depths)

    print(result.best_k, result.intrinsics)
    result.save(output_dir / "calibration.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .geometry import CameraIntrinsics, angle_between_normals, backproject_depth_map
from .plane_fit import PlaneParams, ransac_plane_fit

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class KCandidateStats:
    """Per-k statistics collected during calibration."""
    k: float
    score: float
    median_inlier_ratio: float
    std_normal_angle: float
    invalid_rate: float
    n_valid_frames: int
    n_total_frames: int
    per_frame_inlier_ratios: List[float] = field(default_factory=list)
    per_frame_normal_angles: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert numpy types to plain Python for JSON serialisation
        for key, val in d.items():
            if isinstance(val, (np.floating, np.integer)):
                d[key] = float(val)
            elif isinstance(val, list):
                d[key] = [float(x) for x in val]
        return d


@dataclass
class CalibrationResult:
    """Output of :class:`IntrinsicsCalibrator.calibrate`.

    Attributes
    ----------
    best_k          : chosen focal-length multiplier.
    intrinsics      : :class:`CameraIntrinsics` built from ``best_k``.
    frame_height    : source frame height (pixels).
    frame_width     : source frame width (pixels).
    all_stats       : per-k diagnostic statistics.
    method          : always ``'k_search'`` for traceability.
    """
    best_k: float
    intrinsics: CameraIntrinsics
    frame_height: int
    frame_width: int
    all_stats: List[KCandidateStats] = field(default_factory=list)
    method: str = "k_search"

    def best_stats(self) -> Optional[KCandidateStats]:
        for s in self.all_stats:
            if s.k == self.best_k:
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "best_k": float(self.best_k),
            "frame_height": self.frame_height,
            "frame_width": self.frame_width,
            "intrinsics": {
                "fx": float(self.intrinsics.fx),
                "fy": float(self.intrinsics.fy),
                "cx": float(self.intrinsics.cx),
                "cy": float(self.intrinsics.cy),
            },
            "all_k_stats": [s.to_dict() for s in self.all_stats],
        }

    def save(self, path: "str | Path") -> None:
        """Write the calibration result to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        _logger.info("Calibration result saved → %s", path)

    @classmethod
    def load(cls, path: "str | Path") -> "CalibrationResult":
        """Load a previously saved calibration result from JSON."""
        with open(path) as f:
            data = json.load(f)
        intr = data["intrinsics"]
        return cls(
            best_k=float(data["best_k"]),
            intrinsics=CameraIntrinsics(
                fx=float(intr["fx"]),
                fy=float(intr["fy"]),
                cx=float(intr["cx"]),
                cy=float(intr["cy"]),
            ),
            frame_height=int(data["frame_height"]),
            frame_width=int(data["frame_width"]),
            method=data.get("method", "k_search"),
        )


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class IntrinsicsCalibrator:
    """Calibrate focal-length multiplier k via ground-plane quality scoring.

    Parameters
    ----------
    k_candidates : sequence of k values to try.
    lambda1      : penalty weight for std of inter-frame normal angle.
    lambda2      : penalty weight for fraction of invalid-plane frames.
    ransac_iters : RANSAC iteration count used during calibration.
    inlier_thresh: RANSAC inlier distance threshold.
    min_inlier_ratio : minimum inlier fraction for a plane to be "valid".
    sample_points_per_frame : road pixels subsampled per frame for speed.
    rng          : numpy random generator.
    """

    def __init__(
        self,
        k_candidates: Sequence[float] = (0.8, 1.0, 1.2, 1.4),
        lambda1: float = 0.5,
        lambda2: float = 0.5,
        ransac_iters: int = 150,
        inlier_thresh: float = 0.05,
        min_inlier_ratio: float = 0.25,
        sample_points_per_frame: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.k_candidates = list(k_candidates)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.ransac_iters = ransac_iters
        self.inlier_thresh = inlier_thresh
        self.min_inlier_ratio = min_inlier_ratio
        self.sample_points_per_frame = sample_points_per_frame
        self._rng = rng or np.random.default_rng(42)

    # ── Public API ────────────────────────────────────────────────────────────

    def calibrate(
        self,
        frames: Sequence[np.ndarray],
        road_masks: Sequence[np.ndarray],
        depths: Sequence[np.ndarray],
    ) -> CalibrationResult:
        """Search for the best focal-length multiplier k.

        Parameters
        ----------
        frames     : sequence of (H, W, 3) BGR frames (used for shape only).
        road_masks : sequence of (H, W) boolean masks aligned to frames.
        depths     : sequence of (H, W) float32 depth maps aligned to frames.

        Returns
        -------
        :class:`CalibrationResult` with ``best_k``, ``intrinsics``, and
        per-k diagnostics.

        Notes
        -----
        All three sequences must have the same length.  Pass only the
        *bootstrap* frames (typically 10–30) for speed.
        """
        if not (len(frames) == len(road_masks) == len(depths)):
            raise ValueError(
                "frames, road_masks, and depths must have the same length."
            )
        if len(frames) == 0:
            raise ValueError("At least one frame is required for calibration.")

        H, W = frames[0].shape[:2]
        _logger.info(
            "Calibrating k over %d frames  (H=%d, W=%d)  candidates=%s",
            len(frames), H, W, self.k_candidates,
        )

        all_stats: List[KCandidateStats] = []

        for k in self.k_candidates:
            stats = self._score_k(k, H, W, road_masks, depths)
            all_stats.append(stats)
            _logger.info(
                "  k=%.2f  score=%.4f  median_ir=%.3f  std_angle=%.2f°  invalid=%.0f%%",
                k, stats.score, stats.median_inlier_ratio,
                stats.std_normal_angle, stats.invalid_rate * 100,
            )

        # Choose k with highest score; break ties with index (smaller k preferred)
        best_stats = max(all_stats, key=lambda s: s.score)
        best_k = best_stats.k
        intrinsics = self._build_intrinsics(best_k, H, W)

        _logger.info("Selected k* = %.2f  (score=%.4f)", best_k, best_stats.score)

        return CalibrationResult(
            best_k=best_k,
            intrinsics=intrinsics,
            frame_height=H,
            frame_width=W,
            all_stats=all_stats,
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _score_k(
        self,
        k: float,
        H: int,
        W: int,
        road_masks: Sequence[np.ndarray],
        depths: Sequence[np.ndarray],
    ) -> KCandidateStats:
        """Fit planes on all frames with this k and compute the quality score."""
        intrinsics = self._build_intrinsics(k, H, W)

        inlier_ratios: List[float] = []
        angle_changes: List[float] = []
        n_invalid = 0
        prev_plane: Optional[PlaneParams] = None

        for mask, depth in zip(road_masks, depths):
            pts3d, _ = backproject_depth_map(
                depth, intrinsics,
                mask=mask,
                max_points=self.sample_points_per_frame,
                rng=self._rng,
            )
            if len(pts3d) < 3:
                n_invalid += 1
                continue

            plane = ransac_plane_fit(
                pts3d,
                ransac_iters=self.ransac_iters,
                inlier_thresh=self._adaptive_thresh(pts3d),
                min_inlier_ratio=self.min_inlier_ratio,
                rng=self._rng,
            )

            if plane is None or not plane.valid:
                n_invalid += 1
                continue

            inlier_ratios.append(plane.inlier_ratio)

            if prev_plane is not None:
                angle = angle_between_normals(plane.n, prev_plane.n)
                angle_changes.append(angle)

            prev_plane = plane

        n_total = len(road_masks)
        n_valid = n_total - n_invalid

        med_ir = float(np.median(inlier_ratios)) if inlier_ratios else 0.0
        std_ang = float(np.std(angle_changes)) if angle_changes else 0.0
        inv_rate = n_invalid / n_total if n_total > 0 else 1.0

        score = med_ir - self.lambda1 * std_ang - self.lambda2 * inv_rate

        return KCandidateStats(
            k=k,
            score=score,
            median_inlier_ratio=med_ir,
            std_normal_angle=std_ang,
            invalid_rate=inv_rate,
            n_valid_frames=n_valid,
            n_total_frames=n_total,
            per_frame_inlier_ratios=inlier_ratios,
            per_frame_normal_angles=angle_changes,
        )

    def _adaptive_thresh(self, pts3d: np.ndarray) -> float:
        """Adapt the inlier threshold to the point-cloud scale.

        Uses a fraction of the median Z depth so the threshold is meaningful
        regardless of the model's absolute depth range.
        """
        median_z = float(np.median(pts3d[:, 2]))
        if median_z <= 0:
            return self.inlier_thresh
        # 1% of median Z, but at least the configured absolute threshold
        adaptive = 0.01 * median_z
        return max(adaptive, self.inlier_thresh)

    @staticmethod
    def _build_intrinsics(k: float, H: int, W: int) -> CameraIntrinsics:
        f = k * max(H, W)
        return CameraIntrinsics(fx=f, fy=f, cx=W / 2.0, cy=H / 2.0)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def calibrate_intrinsics(
    frames: Sequence[np.ndarray],
    road_masks: Sequence[np.ndarray],
    depths: Sequence[np.ndarray],
    k_candidates: Sequence[float] = (0.8, 1.0, 1.2, 1.4),
    lambda1: float = 0.5,
    lambda2: float = 0.5,
    ransac_iters: int = 150,
    inlier_thresh: float = 0.05,
    min_inlier_ratio: float = 0.25,
    sample_points_per_frame: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> CalibrationResult:
    """One-shot convenience wrapper around :class:`IntrinsicsCalibrator`.

    Parameters
    ----------
    frames            : list of BGR frames (shape only; pixel values ignored).
    road_masks        : list of (H, W) boolean road masks.
    depths            : list of (H, W) float32 depth maps.
    k_candidates      : focal-length multiplier values to search over.
    lambda1, lambda2  : penalty weights in the score function.
    ransac_iters      : RANSAC iteration count.
    inlier_thresh     : minimum absolute inlier distance threshold.
    min_inlier_ratio  : minimum fraction of inliers for a valid plane.
    sample_points_per_frame : max road pixels subsampled per frame.
    rng               : numpy random generator.

    Returns
    -------
    :class:`CalibrationResult`

    Example
    -------
    >>> result = calibrate_intrinsics(frames, masks, depths)
    >>> print(result.best_k, result.intrinsics)
    >>> result.save("outputs/scene_understanding/calibration.json")
    """
    calibrator = IntrinsicsCalibrator(
        k_candidates=k_candidates,
        lambda1=lambda1,
        lambda2=lambda2,
        ransac_iters=ransac_iters,
        inlier_thresh=inlier_thresh,
        min_inlier_ratio=min_inlier_ratio,
        sample_points_per_frame=sample_points_per_frame,
        rng=rng,
    )
    return calibrator.calibrate(frames, road_masks, depths)
