"""
Unit tests for scene_understanding.plane_fit
---------------------------------------------
All tests use synthetic 3-D point clouds constructed from known ground-truth
planes so that correctness can be verified analytically.

Run with:
    python -m pytest tests/test_scene_understanding_plane_fit.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scene_understanding.plane_fit import (
    PlaneParams,
    ScenePlaneTracker,
    _fit_plane_3pts,
    _fit_plane_svd,
    ema_smooth_plane,
    ransac_plane_fit,
    validate_plane,
)
from scene_understanding.geometry import angle_between_normals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plane_points(
    n: np.ndarray,
    d: float,
    n_pts: int = 500,
    noise_sigma: float = 0.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate random 3-D points on the plane n·P + d = 0.

    The normal is first normalised, then d is re-derived from the centroid so
    the points genuinely satisfy the equation.  Optional Gaussian noise is
    added perpendicular to the plane.
    """
    rng = rng or np.random.default_rng(0)
    n = n / np.linalg.norm(n)

    # Two orthogonal vectors spanning the plane
    candidates = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])]
    for c in candidates:
        u = c - n * float(n @ c)
        if np.linalg.norm(u) > 0.1:
            u = u / np.linalg.norm(u)
            v = np.cross(n, u)
            v = v / np.linalg.norm(v)
            break

    # Random coefficients in [−5, 5]
    alphas = rng.uniform(-5, 5, n_pts)
    betas = rng.uniform(-5, 5, n_pts)

    # Point on plane closest to origin
    origin_on_plane = -d * n
    points = (
        origin_on_plane[np.newaxis, :]
        + alphas[:, np.newaxis] * u
        + betas[:, np.newaxis] * v
    )

    if noise_sigma > 0:
        noise = rng.normal(0, noise_sigma, n_pts)
        points += noise[:, np.newaxis] * n

    return points


def _make_noisy_plane_points(
    n: np.ndarray,
    d: float,
    n_inliers: int = 400,
    n_outliers: int = 100,
    noise_sigma: float = 0.01,
    outlier_range: float = 5.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Inlier points near the plane + random outlier noise."""
    rng = rng or np.random.default_rng(42)
    inliers = _make_plane_points(n, d, n_inliers, noise_sigma, rng)
    outliers = rng.uniform(-outlier_range, outlier_range, (n_outliers, 3))
    return np.vstack([inliers, outliers])


# ---------------------------------------------------------------------------
# _fit_plane_3pts
# ---------------------------------------------------------------------------

class TestFitPlane3pts:

    def test_xy_plane(self):
        p0 = np.array([0., 0., 0.])
        p1 = np.array([1., 0., 0.])
        p2 = np.array([0., 1., 0.])
        result = _fit_plane_3pts(p0, p1, p2)
        assert result is not None
        n, d = result
        assert abs(np.dot(n, [0., 0., 1.])) == pytest.approx(1.0, abs=1e-6)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_points_on_plane(self):
        """Each input point should satisfy n·P + d = 0."""
        p0 = np.array([1., 2., 0.])
        p1 = np.array([3., 0., 1.])
        p2 = np.array([0., 1., 4.])
        result = _fit_plane_3pts(p0, p1, p2)
        assert result is not None
        n, d = result
        for p in [p0, p1, p2]:
            assert abs(float(n @ p) + d) == pytest.approx(0.0, abs=1e-6)

    def test_collinear_points_return_none(self):
        """Collinear points have no unique plane → None."""
        p0 = np.array([0., 0., 0.])
        p1 = np.array([1., 1., 1.])
        p2 = np.array([2., 2., 2.])
        result = _fit_plane_3pts(p0, p1, p2)
        assert result is None

    def test_normal_is_unit_length(self):
        p0 = np.array([0., 0., 1.])
        p1 = np.array([1., 0., 1.])
        p2 = np.array([0., 1., 1.])
        result = _fit_plane_3pts(p0, p1, p2)
        assert result is not None
        n, _ = result
        assert np.linalg.norm(n) == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# _fit_plane_svd
# ---------------------------------------------------------------------------

class TestFitPlaneSvd:

    def test_recovers_known_plane(self):
        """SVD should recover the ground-truth normal within a small angle."""
        n_gt = np.array([0., -1., 0.])
        d_gt = 3.0
        pts = _make_plane_points(n_gt, d_gt, n_pts=1000, noise_sigma=0.001)
        n_est, d_est = _fit_plane_svd(pts)
        angle = angle_between_normals(n_est, n_gt)
        assert angle < 1.0, f"Normal angle error {angle:.2f}° too large"

    def test_d_satisfies_centroid(self):
        """n·centroid + d should be ≈ 0 for any perfectly coplanar point set."""
        n_gt = np.array([1., 1., 1.]) / np.sqrt(3)
        d_gt = -2.0
        pts = _make_plane_points(n_gt, d_gt, n_pts=500, noise_sigma=0.0)
        n_est, d_est = _fit_plane_svd(pts)
        centroid = pts.mean(axis=0)
        assert abs(float(n_est @ centroid) + d_est) == pytest.approx(0.0, abs=1e-6)

    def test_minimum_3_points(self):
        """SVD should work with exactly 3 non-collinear points."""
        pts = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        n, d = _fit_plane_svd(pts)
        assert np.linalg.norm(n) == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# ransac_plane_fit
# ---------------------------------------------------------------------------

class TestRansacPlaneFit:

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(7)

    def test_returns_none_for_fewer_than_3_points(self, rng):
        pts = np.random.default_rng(0).uniform(0, 1, (2, 3))
        result = ransac_plane_fit(pts, rng=rng)
        assert result is None

    def test_perfect_plane_high_inlier_ratio(self, rng):
        """Points exactly on a plane should yield inlier_ratio ≈ 1."""
        n_gt = np.array([0., 1., 0.])
        d_gt = -2.0
        pts = _make_plane_points(n_gt, d_gt, n_pts=300, noise_sigma=0.0, rng=rng)
        result = ransac_plane_fit(pts, inlier_thresh=0.05, rng=rng)
        assert result is not None
        assert result.inlier_ratio == pytest.approx(1.0, abs=0.02)

    def test_noisy_plane_still_found(self, rng):
        """With 80% inliers (20% outliers), RANSAC should find the plane."""
        n_gt = np.array([0., -1., 0.])
        d_gt = 3.0
        pts = _make_noisy_plane_points(
            n_gt, d_gt, n_inliers=400, n_outliers=100,
            noise_sigma=0.01, rng=rng
        )
        result = ransac_plane_fit(
            pts,
            ransac_iters=200,
            inlier_thresh=0.05,
            min_inlier_ratio=0.3,
            rng=rng,
        )
        assert result is not None
        assert result.valid
        # Normal should be close to ground truth
        angle = angle_between_normals(result.n, n_gt)
        assert angle < 5.0, f"Normal angle error {angle:.2f}° too large"

    def test_inlier_ratio_is_in_0_1(self, rng):
        pts = _make_noisy_plane_points(
            np.array([0., 0., 1.]), -1.0,
            n_inliers=300, n_outliers=100, rng=rng
        )
        result = ransac_plane_fit(pts, rng=rng)
        assert result is not None
        assert 0.0 <= result.inlier_ratio <= 1.0

    def test_n_inliers_consistent_with_ratio(self, rng):
        """n_inliers / n_points should equal inlier_ratio."""
        pts = _make_plane_points(
            np.array([1., 1., 0.]), -1.0, n_pts=200, noise_sigma=0.005, rng=rng
        )
        result = ransac_plane_fit(pts, rng=rng)
        assert result is not None
        assert result.n_points == len(pts)
        assert result.n_inliers / result.n_points == pytest.approx(
            result.inlier_ratio, rel=1e-4
        )

    def test_fitted_plane_is_unit_normal(self, rng):
        pts = _make_plane_points(
            np.array([0., 1., 0.5]), -2.0, n_pts=200, rng=rng
        )
        result = ransac_plane_fit(pts, rng=rng)
        assert result is not None
        assert np.linalg.norm(result.n) == pytest.approx(1.0, rel=1e-9)

    def test_plane_params_to_dict_round_trip(self, rng):
        pts = _make_plane_points(
            np.array([0., 1., 0.]), -1.0, n_pts=100, rng=rng
        )
        result = ransac_plane_fit(pts, rng=rng)
        assert result is not None
        d = result.to_dict()
        recovered = PlaneParams.from_dict(d)
        np.testing.assert_allclose(recovered.n, result.n, rtol=1e-9)
        assert recovered.d == pytest.approx(result.d)
        assert recovered.valid == result.valid


# ---------------------------------------------------------------------------
# validate_plane
# ---------------------------------------------------------------------------

class TestValidatePlane:

    def _make_plane(self, n, d, inlier_ratio=0.8, valid=True):
        n = np.array(n, dtype=np.float64)
        n = n / np.linalg.norm(n)
        return PlaneParams(
            n=n, d=float(d),
            inlier_ratio=inlier_ratio,
            frame_idx=0, valid=valid,
        )

    def test_accepts_good_plane_without_prev(self):
        p = self._make_plane([0, 1, 0], -1.0, inlier_ratio=0.6)
        assert validate_plane(p, None, min_inlier_ratio=0.3) is True

    def test_rejects_low_inlier_ratio(self):
        p = self._make_plane([0, 1, 0], -1.0, inlier_ratio=0.1)
        assert validate_plane(p, None, min_inlier_ratio=0.3) is False

    def test_rejects_large_angle_change(self):
        prev = self._make_plane([0, 1, 0], -1.0, inlier_ratio=0.7)
        new  = self._make_plane([1, 0, 0], -1.0, inlier_ratio=0.7)
        # Angle between [0,1,0] and [1,0,0] is 90°, max is 15°
        assert validate_plane(new, prev,
                               min_inlier_ratio=0.3,
                               max_normal_angle_change_deg=15.0) is False

    def test_accepts_small_angle_change(self):
        prev = self._make_plane([0, 1, 0], -1.0, inlier_ratio=0.7)
        # Perturb normal by ~5°
        eps = np.sin(np.radians(5))
        n_new = np.array([eps, np.cos(np.radians(5)), 0.0])
        new = PlaneParams(
            n=n_new / np.linalg.norm(n_new), d=-1.0,
            inlier_ratio=0.7, frame_idx=1, valid=True,
        )
        assert validate_plane(new, prev,
                               min_inlier_ratio=0.3,
                               max_normal_angle_change_deg=15.0) is True

    def test_no_previous_plane_skips_angle_check(self):
        """Without a previous plane, only the inlier ratio is checked."""
        # Even a large angle change is fine when prev=None
        p = self._make_plane([1, 0, 0], 0.0, inlier_ratio=0.5)
        assert validate_plane(p, None, min_inlier_ratio=0.3,
                               max_normal_angle_change_deg=1.0) is True


# ---------------------------------------------------------------------------
# ema_smooth_plane
# ---------------------------------------------------------------------------

class TestEmaSmoothing:

    def _plane(self, n, d, inlier_ratio=0.8):
        n = np.array(n, dtype=np.float64)
        n = n / np.linalg.norm(n)
        return PlaneParams(n=n, d=float(d), inlier_ratio=inlier_ratio,
                           frame_idx=0, valid=True)

    def test_alpha_zero_keeps_prev(self):
        """alpha=0 → result identical to prev_plane."""
        prev = self._plane([0, 1, 0], -1.0)
        new = self._plane([1, 0, 0], -2.0)
        blended = ema_smooth_plane(prev, new, alpha=0.0)
        np.testing.assert_allclose(blended.n, prev.n, atol=1e-9)
        assert blended.d == pytest.approx(prev.d)

    def test_alpha_one_replaces_with_new(self):
        """alpha=1 → result identical to new_plane."""
        prev = self._plane([0, 1, 0], -1.0)
        new = self._plane([1, 0, 0], -2.0)
        blended = ema_smooth_plane(prev, new, alpha=1.0)
        np.testing.assert_allclose(blended.n, new.n, atol=1e-9)
        assert blended.d == pytest.approx(new.d)

    def test_result_is_unit_normal(self):
        """Blended normal should always be unit length."""
        prev = self._plane([0, 1, 0], -1.0)
        new = self._plane([1, 0.1, 0.1], -2.0)
        blended = ema_smooth_plane(prev, new, alpha=0.3)
        assert np.linalg.norm(blended.n) == pytest.approx(1.0, rel=1e-9)

    def test_d_is_linear_blend(self):
        """Scalar d should be blended as (1-alpha)*d_prev + alpha*d_new."""
        prev = self._plane([0, 1, 0], -1.0)
        new = self._plane([0, 1, 0], -3.0)  # same n, different d
        alpha = 0.4
        blended = ema_smooth_plane(prev, new, alpha=alpha)
        expected_d = (1 - alpha) * (-1.0) + alpha * (-3.0)
        assert blended.d == pytest.approx(expected_d)

    def test_convergence_after_many_steps(self):
        """With repeated updates, the blended value should converge to new."""
        prev = self._plane([0, 1, 0], -1.0)
        new = self._plane([0, 1, 0], -5.0)
        state = prev
        for _ in range(100):
            state = ema_smooth_plane(state, new, alpha=0.3)
        assert state.d == pytest.approx(-5.0, abs=0.01)


# ---------------------------------------------------------------------------
# ScenePlaneTracker
# ---------------------------------------------------------------------------

class TestScenePlaneTracker:

    def _plane(self, n, d, inlier_ratio=0.8, valid=True):
        n_arr = np.array(n, dtype=np.float64)
        n_arr = n_arr / np.linalg.norm(n_arr)
        return PlaneParams(n=n_arr, d=float(d), inlier_ratio=inlier_ratio,
                           frame_idx=-1, valid=valid)

    def test_initial_state_no_plane(self):
        tracker = ScenePlaneTracker()
        assert tracker.stable_plane is None
        assert len(tracker.history) == 0

    def test_first_valid_update_sets_stable(self):
        tracker = ScenePlaneTracker()
        plane = self._plane([0, 1, 0], -1.0, inlier_ratio=0.6)
        result = tracker.update(plane, frame_idx=0)
        assert result is not None
        assert tracker.stable_plane is not None

    def test_none_update_preserves_stable(self):
        tracker = ScenePlaneTracker()
        plane = self._plane([0, 1, 0], -1.0)
        tracker.update(plane, frame_idx=0)
        before = tracker.stable_plane
        tracker.update(None, frame_idx=1)
        assert tracker.stable_plane is not None
        np.testing.assert_allclose(tracker.stable_plane.n, before.n, atol=1e-9)

    def test_invalid_plane_does_not_replace_stable(self):
        tracker = ScenePlaneTracker(min_inlier_ratio=0.5)
        good = self._plane([0, 1, 0], -1.0, inlier_ratio=0.8)
        tracker.update(good, frame_idx=0)
        stable_before = tracker.stable_plane

        bad = self._plane([0, 1, 0], -1.0, inlier_ratio=0.1)  # below threshold
        tracker.update(bad, frame_idx=1)
        # Stable plane should be unchanged
        np.testing.assert_allclose(tracker.stable_plane.n, stable_before.n, atol=1e-9)

    def test_history_grows_with_each_update(self):
        tracker = ScenePlaneTracker()
        plane = self._plane([0, 1, 0], -1.0)
        for i in range(5):
            tracker.update(plane, frame_idx=i)
        assert len(tracker.history) == 5

    def test_history_dicts_serialisable(self):
        import json
        tracker = ScenePlaneTracker()
        plane = self._plane([0, 1, 0], -1.0)
        tracker.update(plane, frame_idx=0)
        tracker.update(None, frame_idx=1)
        dicts = tracker.get_history_dicts()
        # Should not raise
        json.dumps(dicts)

    def test_summary_dict_has_required_keys(self):
        tracker = ScenePlaneTracker()
        plane = self._plane([0, 1, 0], -1.0)
        tracker.update(plane, frame_idx=0)
        summary = tracker.get_summary_dict()
        for key in ("total_frames", "valid_plane_frames", "valid_ratio",
                    "final_plane", "per_frame"):
            assert key in summary, f"Missing key: {key}"

    def test_reset_clears_state(self):
        tracker = ScenePlaneTracker()
        plane = self._plane([0, 1, 0], -1.0)
        tracker.update(plane, frame_idx=0)
        tracker.reset()
        assert tracker.stable_plane is None
        assert len(tracker.history) == 0

    def test_ema_smoothing_applied(self):
        """With alpha=1.0, second valid plane should completely replace first."""
        tracker = ScenePlaneTracker(ema_alpha=1.0, max_normal_angle_change_deg=180.0)
        p1 = self._plane([0, 1, 0], -1.0, inlier_ratio=0.8)
        p2 = self._plane([0, 1, 0], -5.0, inlier_ratio=0.8)  # same n, different d
        tracker.update(p1, frame_idx=0)
        tracker.update(p2, frame_idx=1)
        assert tracker.stable_plane.d == pytest.approx(-5.0, abs=1e-6)

    def test_large_angle_change_rejected(self):
        tracker = ScenePlaneTracker(max_normal_angle_change_deg=10.0)
        p1 = self._plane([0, 1, 0], -1.0, inlier_ratio=0.8)
        # 90° different
        p2 = self._plane([1, 0, 0], -1.0, inlier_ratio=0.8)
        tracker.update(p1, frame_idx=0)
        stable_before = tracker.stable_plane
        tracker.update(p2, frame_idx=1)
        # p2 should be rejected; stable_plane should not change
        np.testing.assert_allclose(
            tracker.stable_plane.n, stable_before.n, atol=1e-9
        )
