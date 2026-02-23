"""
Unit tests for scene_understanding.geometry
--------------------------------------------
All tests use synthetic data with known analytic ground truths.
No GPU, no real models, no video files required.

Run with:
    python -m pytest tests/test_scene_understanding_geometry.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from scene_understanding.geometry import (
    CameraIntrinsics,
    angle_between_normals,
    backproject_depth_map,
    backproject_pixel,
    ground_distance,
    project_3d_to_image,
    ray_direction,
    ray_plane_intersection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def standard_intrinsics() -> CameraIntrinsics:
    """800×600 image with fx=fy=800, principal point at image centre."""
    return CameraIntrinsics(fx=800.0, fy=800.0, cx=400.0, cy=300.0)


# ---------------------------------------------------------------------------
# CameraIntrinsics
# ---------------------------------------------------------------------------

class TestCameraIntrinsics:

    def test_from_frame_array(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        intr = CameraIntrinsics.from_frame(frame)
        assert intr.cx == pytest.approx(320.0)
        assert intr.cy == pytest.approx(240.0)
        assert intr.fx > 0
        assert intr.fy > 0

    def test_from_frame_tuple(self):
        intr = CameraIntrinsics.from_frame((480, 640))
        assert intr.cx == pytest.approx(320.0)
        assert intr.cy == pytest.approx(240.0)

    def test_from_frame_explicit_fx(self):
        intr = CameraIntrinsics.from_frame((480, 640), fx=1000.0)
        assert intr.fx == pytest.approx(1000.0)
        assert intr.fy == pytest.approx(1000.0)  # fy defaults to fx

    def test_scale(self, standard_intrinsics):
        scaled = standard_intrinsics.scale(0.5, 0.5)
        assert scaled.fx == pytest.approx(400.0)
        assert scaled.fy == pytest.approx(400.0)
        assert scaled.cx == pytest.approx(200.0)
        assert scaled.cy == pytest.approx(150.0)

    def test_as_matrix_shape(self, standard_intrinsics):
        K = standard_intrinsics.as_matrix()
        assert K.shape == (3, 3)
        assert K[0, 0] == pytest.approx(800.0)
        assert K[1, 1] == pytest.approx(800.0)
        assert K[0, 2] == pytest.approx(400.0)
        assert K[2, 2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# backproject_pixel
# ---------------------------------------------------------------------------

class TestBackprojectPixel:

    def test_principal_point_projects_forward(self, standard_intrinsics):
        """The principal point (cx, cy) at depth Z should give X=Y=0."""
        P = backproject_pixel(400.0, 300.0, 5.0, standard_intrinsics)
        assert P[0] == pytest.approx(0.0, abs=1e-9)
        assert P[1] == pytest.approx(0.0, abs=1e-9)
        assert P[2] == pytest.approx(5.0)

    def test_unit_depth_one_pixel_right(self, standard_intrinsics):
        """One pixel right of principal point at depth 1 → X = 1/fx."""
        P = backproject_pixel(401.0, 300.0, 1.0, standard_intrinsics)
        assert P[0] == pytest.approx(1.0 / 800.0, rel=1e-6)
        assert P[1] == pytest.approx(0.0, abs=1e-9)
        assert P[2] == pytest.approx(1.0)

    def test_unit_depth_one_pixel_down(self, standard_intrinsics):
        """One pixel below principal point at depth 1 → Y = 1/fy."""
        P = backproject_pixel(400.0, 301.0, 1.0, standard_intrinsics)
        assert P[0] == pytest.approx(0.0, abs=1e-9)
        assert P[1] == pytest.approx(1.0 / 800.0, rel=1e-6)
        assert P[2] == pytest.approx(1.0)

    def test_depth_scales_linearly(self, standard_intrinsics):
        """Doubling depth should double X, Y, Z."""
        P1 = backproject_pixel(450.0, 320.0, 2.0, standard_intrinsics)
        P2 = backproject_pixel(450.0, 320.0, 4.0, standard_intrinsics)
        np.testing.assert_allclose(P2, 2.0 * P1, rtol=1e-9)

    def test_return_dtype(self, standard_intrinsics):
        P = backproject_pixel(400.0, 300.0, 3.0, standard_intrinsics)
        assert P.dtype == np.float64
        assert P.shape == (3,)


# ---------------------------------------------------------------------------
# backproject_depth_map
# ---------------------------------------------------------------------------

class TestBackprojectDepthMap:

    def test_empty_mask(self, standard_intrinsics):
        depth = np.ones((100, 100), dtype=np.float32) * 5.0
        mask = np.zeros((100, 100), dtype=bool)
        pts, pxcoords = backproject_depth_map(depth, standard_intrinsics, mask=mask)
        assert len(pts) == 0
        assert len(pxcoords) == 0

    def test_full_mask_count(self, standard_intrinsics):
        H, W = 10, 10
        depth = np.ones((H, W), dtype=np.float32) * 3.0
        pts, pxcoords = backproject_depth_map(depth, standard_intrinsics)
        assert len(pts) == H * W
        assert len(pxcoords) == H * W

    def test_principal_point_backprojects_correctly(self):
        """For a pixel at the principal point, X=Y=0."""
        fx, fy, cx, cy = 500.0, 500.0, 50.0, 50.0
        intr = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
        H, W = 101, 101
        depth = np.ones((H, W), dtype=np.float32) * 2.0
        mask = np.zeros((H, W), dtype=bool)
        mask[50, 50] = True  # pixel at (u=50, v=50) = principal point
        pts, pxcoords = backproject_depth_map(depth, intr, mask=mask)
        assert len(pts) == 1
        assert pts[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert pts[0, 1] == pytest.approx(0.0, abs=1e-6)
        assert pts[0, 2] == pytest.approx(2.0)

    def test_zero_depth_excluded(self, standard_intrinsics):
        depth = np.zeros((10, 10), dtype=np.float32)
        pts, _ = backproject_depth_map(depth, standard_intrinsics)
        assert len(pts) == 0

    def test_max_points_subsampling(self, standard_intrinsics):
        depth = np.ones((100, 100), dtype=np.float32) * 1.0
        pts, _ = backproject_depth_map(
            depth, standard_intrinsics, max_points=50, rng=np.random.default_rng(0)
        )
        assert len(pts) == 50


# ---------------------------------------------------------------------------
# ray_direction
# ---------------------------------------------------------------------------

class TestRayDirection:

    def test_unit_length(self, standard_intrinsics):
        for u, v in [(400, 300), (0, 0), (800, 600), (200, 150)]:
            r = ray_direction(float(u), float(v), standard_intrinsics)
            assert np.linalg.norm(r) == pytest.approx(1.0, rel=1e-9)

    def test_principal_point_is_forward(self, standard_intrinsics):
        """Ray through the principal point should be [0, 0, 1]."""
        r = ray_direction(400.0, 300.0, standard_intrinsics)
        np.testing.assert_allclose(r, [0.0, 0.0, 1.0], atol=1e-9)

    def test_positive_z_component(self, standard_intrinsics):
        """All rays should have Z > 0 (pointing forward)."""
        for u in [0, 200, 400, 600, 800]:
            for v in [0, 150, 300, 450, 600]:
                r = ray_direction(float(u), float(v), standard_intrinsics)
                assert r[2] > 0, f"Ray at ({u},{v}) has Z<=0: {r}"


# ---------------------------------------------------------------------------
# ray_plane_intersection
# ---------------------------------------------------------------------------

class TestRayPlaneIntersection:

    def test_forward_ray_hits_floor(self):
        """A straight-down ray from the optical axis should hit the floor plane."""
        # Floor plane: n = [0, -1, 0], d = 2.0  → plane at Y = 2
        # (n·P + d = 0  ↔  -Y + 2 = 0  ↔  Y = 2)
        n = np.array([0.0, -1.0, 0.0])
        d = 2.0
        # Ray pointing slightly down: direction (0, 1, 1) normalised
        r = np.array([0.0, 1.0, 1.0])
        r = r / np.linalg.norm(r)
        p = ray_plane_intersection(r, n, d)
        assert p is not None
        # n·p + d should be 0
        assert float(n @ p + d) == pytest.approx(0.0, abs=1e-9)

    def test_parallel_ray_returns_none(self):
        """A ray parallel to the plane should return None."""
        n = np.array([0.0, 1.0, 0.0])
        d = -1.0
        # Ray in XZ plane (no Y component) — parallel to the plane
        r = np.array([0.0, 0.0, 1.0])
        result = ray_plane_intersection(r, n, d)
        assert result is None

    def test_behind_camera_returns_none(self):
        """Intersection behind the camera (t < 0) should return None.

        Setup: n=[0,1,0], d=2.0  →  plane at Y = -2 (above camera, since +Y is down).
        Ray direction [0,1,1] has positive Y component, so denom = n·r = 1/√2 > 0.
        t = -d / denom = -2 / (1/√2) = -2√2 < 0  →  behind camera.
        """
        n = np.array([0.0, 1.0, 0.0])
        d = 2.0   # plane at Y = -2
        r = np.array([0.0, 1.0, 1.0])
        r = r / np.linalg.norm(r)
        result = ray_plane_intersection(r, n, d)
        assert result is None

    def test_known_intersection(self):
        """Verify the intersection formula against a hand-calculated result."""
        # Plane: z = 5  →  n = [0,0,1], d = -5  (n·P + d = Z - 5 = 0)
        n = np.array([0.0, 0.0, 1.0])
        d = -5.0
        # Ray along z-axis: direction = [0, 0, 1]
        r = np.array([0.0, 0.0, 1.0])
        p = ray_plane_intersection(r, n, d)
        assert p is not None
        np.testing.assert_allclose(p, [0.0, 0.0, 5.0], atol=1e-9)

    def test_intersection_lies_on_plane(self, standard_intrinsics):
        """For any pixel, the intersection point should satisfy n·P + d = 0."""
        n = np.array([0.0, -1.0, 0.5])
        n = n / np.linalg.norm(n)
        d = 3.0
        for u, v in [(200, 400), (100, 500), (600, 450)]:
            r = ray_direction(float(u), float(v), standard_intrinsics)
            p = ray_plane_intersection(r, n, d)
            if p is not None:
                residual = float(n @ p + d)
                assert residual == pytest.approx(0.0, abs=1e-8), \
                    f"Point {p} does not lie on plane (residual={residual})"


# ---------------------------------------------------------------------------
# project_3d_to_image
# ---------------------------------------------------------------------------

class TestProject3dToImage:

    def test_round_trip_principal_point(self, standard_intrinsics):
        """Backproject then re-project the principal point → same pixel."""
        P = backproject_pixel(400.0, 300.0, 7.0, standard_intrinsics)
        us, vs = project_3d_to_image(P[np.newaxis, :], standard_intrinsics)
        assert us[0] == pytest.approx(400.0, abs=1e-6)
        assert vs[0] == pytest.approx(300.0, abs=1e-6)

    def test_round_trip_arbitrary_pixel(self, standard_intrinsics):
        """Backproject then re-project an arbitrary pixel → original coordinates."""
        u_orig, v_orig, depth = 350.0, 420.0, 4.0
        P = backproject_pixel(u_orig, v_orig, depth, standard_intrinsics)
        us, vs = project_3d_to_image(P[np.newaxis, :], standard_intrinsics)
        assert us[0] == pytest.approx(u_orig, abs=1e-6)
        assert vs[0] == pytest.approx(v_orig, abs=1e-6)

    def test_negative_z_gives_nan(self, standard_intrinsics):
        """Points behind the camera (Z ≤ 0) should project to NaN."""
        P_neg = np.array([[1.0, 0.0, -1.0]])
        P_zero = np.array([[0.0, 0.0, 0.0]])
        us_neg, _ = project_3d_to_image(P_neg, standard_intrinsics)
        us_zero, _ = project_3d_to_image(P_zero, standard_intrinsics)
        assert np.isnan(us_neg[0])
        assert np.isnan(us_zero[0])

    def test_batch_projection(self, standard_intrinsics):
        """Batch of N points → N output coordinates."""
        pts = np.random.default_rng(0).uniform(-1, 1, (50, 3))
        pts[:, 2] = np.abs(pts[:, 2]) + 0.1  # ensure Z > 0
        us, vs = project_3d_to_image(pts, standard_intrinsics)
        assert us.shape == (50,)
        assert vs.shape == (50,)


# ---------------------------------------------------------------------------
# angle_between_normals
# ---------------------------------------------------------------------------

class TestAngleBetweenNormals:

    def test_same_normal_is_zero(self):
        n = np.array([0.0, 1.0, 0.0])
        # arccos(1.0) has O(1e-7) floating-point error; use 1e-6 tolerance
        assert angle_between_normals(n, n) == pytest.approx(0.0, abs=1e-6)

    def test_antiparallel_is_zero(self):
        """Flipped normals describe the same plane — should report 0°."""
        n1 = np.array([0.0, 1.0, 0.0])
        n2 = -n1
        assert angle_between_normals(n1, n2) == pytest.approx(0.0, abs=1e-6)

    def test_perpendicular_is_90(self):
        n1 = np.array([1.0, 0.0, 0.0])
        n2 = np.array([0.0, 1.0, 0.0])
        assert angle_between_normals(n1, n2) == pytest.approx(90.0, abs=1e-6)

    def test_45_degrees(self):
        n1 = np.array([1.0, 0.0, 0.0])
        n2 = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
        assert angle_between_normals(n1, n2) == pytest.approx(45.0, abs=1e-6)

    def test_unnormalised_inputs(self):
        """Function should handle unnormalised normals."""
        n1 = np.array([2.0, 0.0, 0.0])   # length 2
        n2 = np.array([0.0, 3.0, 0.0])   # length 3
        assert angle_between_normals(n1, n2) == pytest.approx(90.0, abs=1e-6)


# ---------------------------------------------------------------------------
# ground_distance
# ---------------------------------------------------------------------------

class TestGroundDistance:

    def test_origin(self):
        assert ground_distance(np.array([0.0, 0.0, 0.0])) == pytest.approx(0.0)

    def test_unit_vector(self):
        assert ground_distance(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)

    def test_pythagorean_triple(self):
        # 3-4-5 right triangle in 3D
        assert ground_distance(np.array([3.0, 4.0, 0.0])) == pytest.approx(5.0)
        assert ground_distance(np.array([0.0, 3.0, 4.0])) == pytest.approx(5.0)
