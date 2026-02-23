"""
scene_understanding.run_scene_understanding
--------------------------------------------
CLI entry point for the SceneUnderstanding pipeline.

Modes
-----
bootstrap
    Process the first ``num_bootstrap_frames`` frames, fit an initial ground
    plane, and write a diagnostics report.  Use this to verify that the
    segmenter and depth estimator are working and to tune thresholds before
    processing a full video.

run
    Process the entire video with periodic plane updates.  The stable plane
    is frozen whenever a new estimate fails validation and unfrozen as soon as
    a valid estimate is obtained again.

Usage examples
--------------
Bootstrap with dummy models (no PyTorch required)::

    python -m scene_understanding.run_scene_understanding \\
        --video_path data/traffic_video.mp4 \\
        --mode bootstrap \\
        --output_dir outputs/scene_understanding

Full run with known intrinsics::

    python -m scene_understanding.run_scene_understanding \\
        --video_path data/traffic_video.mp4 \\
        --intrinsics 800 800 640 360 \\
        --mode run \\
        --output_dir outputs/scene_understanding \\
        --stride_seg 2 --stride_depth 2

Using a frames directory instead of a video file::

    python -m scene_understanding.run_scene_understanding \\
        --frames_dir data/frames/ \\
        --mode run \\
        --output_dir outputs/scene_understanding

Plugging in real models
-----------------------
Edit the ``_build_models`` function near the bottom of this file and replace
the DummyRoadSegmenter / DummyDepthEstimator instances with your own
TorchRoadSegmenter / TorchDepthEstimator subclasses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .config import SceneConfig
from .geometry import CameraIntrinsics, backproject_depth_map, ray_plane_intersection, ray_direction
from .models import (
    DummyDepthEstimator,
    DummyRoadSegmenter,
    RoadSegmenter,
    DepthEstimator,
    load_cached,
    save_cached,
)
from .plane_fit import PlaneParams, ScenePlaneTracker, ransac_plane_fit
from .visualize import (
    bev_scatter_plot,
    build_scene_frame,
    depth_viz,
    draw_ground_cube,
    draw_pixel_probe,
    draw_plane_grid,
    draw_plane_inliers,
    make_side_by_side,
    overlay_mask,
    plane_diagnostics_plot,
    road_depth_overlay,
)


# ---------------------------------------------------------------------------
# Video / frame-dir iterators
# ---------------------------------------------------------------------------

def _iter_video(
    video_path: str,
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_idx, bgr_frame) for every *stride*-th frame in a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frame_idx = 0
    emitted = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                yield frame_idx, frame
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()


def _iter_frames_dir(
    frames_dir: str,
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_idx, bgr_frame) from a directory of image files."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = sorted(
        p for p in Path(frames_dir).iterdir()
        if p.suffix.lower() in exts
    )
    emitted = 0
    for i, p in enumerate(paths):
        if i % stride != 0:
            continue
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        yield i, frame
        emitted += 1
        if max_frames is not None and emitted >= max_frames:
            break


# ---------------------------------------------------------------------------
# Per-frame inference with cache
# ---------------------------------------------------------------------------

def _get_mask(
    segmenter: RoadSegmenter,
    frame: np.ndarray,
    cache_dir: Path,
    source_path: str,
    frame_idx: int,
    dilate_px: int,
    inference_max_dim: Optional[int],
) -> np.ndarray:
    """Return the road mask for a frame, using cache if available."""
    cached = load_cached(cache_dir, source_path, frame_idx, "seg")
    if cached is not None:
        return cached.astype(bool)

    infer_frame = _maybe_resize(frame, inference_max_dim)
    mask = segmenter.predict(infer_frame)
    # Resize mask back if inference was on a smaller frame
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

    save_cached(cache_dir, source_path, frame_idx, "seg", mask.astype(np.uint8))
    return mask


def _get_depth(
    estimator: DepthEstimator,
    frame: np.ndarray,
    cache_dir: Path,
    source_path: str,
    frame_idx: int,
    inference_max_dim: Optional[int],
) -> np.ndarray:
    """Return the depth map for a frame, using cache if available."""
    cached = load_cached(cache_dir, source_path, frame_idx, "depth")
    if cached is not None:
        return cached.astype(np.float32)

    infer_frame = _maybe_resize(frame, inference_max_dim)
    depth = estimator.predict(infer_frame)
    if depth.shape[:2] != frame.shape[:2]:
        depth = cv2.resize(
            depth,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    save_cached(cache_dir, source_path, frame_idx, "depth", depth.astype(np.float32))
    return depth.astype(np.float32)


def _maybe_resize(frame: np.ndarray, max_dim: Optional[int]) -> np.ndarray:
    """Downscale frame so that max(H, W) ≤ max_dim, preserving aspect ratio."""
    if max_dim is None:
        return frame
    H, W = frame.shape[:2]
    if max(H, W) <= max_dim:
        return frame
    scale = max_dim / max(H, W)
    new_w = max(1, int(W * scale))
    new_h = max(1, int(H * scale))
    return cv2.resize(frame, (new_w, new_h))


# ---------------------------------------------------------------------------
# BEV helper — project road pixels to ground plane
# ---------------------------------------------------------------------------

def _project_road_to_bev(
    depth: np.ndarray,
    road_mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    plane: PlaneParams,
    max_points: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample road pixels, intersect with ground plane, return (N, 3) points."""
    rng = rng or np.random.default_rng()
    _, pixel_coords = backproject_depth_map(
        depth, intrinsics, mask=road_mask,
        max_points=max_points, rng=rng,
    )
    pts_3d: List[np.ndarray] = []
    for u, v in pixel_coords:
        r = ray_direction(u, v, intrinsics)
        p = ray_plane_intersection(r, plane.n, plane.d)
        if p is not None:
            pts_3d.append(p)
    return np.array(pts_3d) if pts_3d else np.empty((0, 3))


# ---------------------------------------------------------------------------
# Summary-report writer
# ---------------------------------------------------------------------------

def _write_markdown_report(
    out_dir: Path,
    mode: str,
    source: str,
    cfg: SceneConfig,
    tracker: ScenePlaneTracker,
    elapsed_sec: float,
) -> Path:
    """Write a short Markdown summary report."""
    summary = tracker.get_summary_dict()
    plane = tracker.stable_plane

    lines = [
        "# Scene Understanding — Summary Report",
        "",
        f"**Mode:** {mode}  ",
        f"**Source:** {source}  ",
        f"**Processing time:** {elapsed_sec:.1f} s  ",
        "",
        "## Plane tracker summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total frames processed | {summary['total_frames']} |",
        f"| Frames with valid plane | {summary['valid_plane_frames']} |",
        f"| Valid ratio | {summary['valid_ratio']:.2%} |",
    ]

    if plane is not None:
        lines += [
            "",
            "## Final stable plane",
            "",
            f"```",
            f"n = [{plane.n[0]:.4f},  {plane.n[1]:.4f},  {plane.n[2]:.4f}]",
            f"d = {plane.d:.4f}",
            f"inlier_ratio = {plane.inlier_ratio:.3f}",
            f"```",
        ]
    else:
        lines += ["", "> No valid plane was established."]

    lines += [
        "",
        "## Configuration",
        "",
        "```",
        f"proximity_px      (not applicable — scene_understanding module)",
        f"ransac_iters      = {cfg.ransac_iters}",
        f"inlier_thresh     = {cfg.inlier_thresh}",
        f"min_inlier_ratio  = {cfg.min_inlier_ratio}",
        f"max_normal_angle  = {cfg.max_normal_angle_change_deg}°",
        f"ema_alpha         = {cfg.ema_alpha}",
        f"stride_seg        = {cfg.stride_seg}",
        f"stride_depth      = {cfg.stride_depth}",
        "```",
        "",
        "## Output files",
        "",
        "| Directory | Contents |",
        "|---|---|",
        "| `frames_overlay/` | Mask + grid overlay per frame |",
        "| `frames_depth/`   | Depth colourmap per frame |",
        "| `frames_grid/`    | 2×2 composite per frame |",
        "| `bev_plots/`      | BEV scatter plots |",
        "| `diagnostics/`    | Inlier / outlier visualisations |",
        "| `plane_params.json` | Per-frame plane parameters |",
        "| `summary_report.md` | This report |",
        "| `output_video_overlay.mp4` | Annotated video (if enabled) |",
    ]

    path = out_dir / "summary_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Core pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    cfg: SceneConfig,
    source_path: str,
    frame_iter: Iterator[Tuple[int, np.ndarray]],
    segmenter: RoadSegmenter,
    estimator: DepthEstimator,
    mode: str,
    test_pixels: Optional[List[Tuple[float, float]]] = None,
    rng: Optional[np.random.Generator] = None,
) -> ScenePlaneTracker:
    """Execute the full scene-understanding pipeline.

    Parameters
    ----------
    cfg         : SceneConfig instance.
    source_path : path string used as cache key.
    frame_iter  : iterator of (frame_idx, bgr_frame) tuples.
    segmenter   : road segmenter.
    estimator   : depth estimator.
    mode        : ``'bootstrap'`` or ``'run'``.
    test_pixels : optional list of (u, v) pixels for probe visualisation.
    rng         : numpy random generator for reproducibility.

    Returns
    -------
    :class:`ScenePlaneTracker` with full history and stable plane.
    """
    rng = rng or np.random.default_rng(42)

    out_dir = Path(cfg.output_dir)
    cache_dir = cfg.resolve_cache_dir()

    # Output sub-directories
    dirs = {
        "overlay":     out_dir / "frames_overlay",
        "depth":       out_dir / "frames_depth",
        "grid":        out_dir / "frames_grid",
        "bev":         out_dir / "bev_plots",
        "diagnostics": out_dir / "diagnostics",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    tracker = ScenePlaneTracker(
        min_inlier_ratio=cfg.min_inlier_ratio,
        max_normal_angle_change_deg=cfg.max_normal_angle_change_deg,
        ema_alpha=cfg.ema_alpha,
    )

    # Video writer (lazily initialised on first frame)
    video_writer: Optional[cv2.VideoWriter] = None
    first_frame_shape: Optional[Tuple[int, int]] = None

    last_mask:  Optional[np.ndarray] = None
    last_depth: Optional[np.ndarray] = None
    intrinsics: Optional[CameraIntrinsics] = None
    n_processed = 0

    # BEV axis limits — set from the first valid BEV frame and reused for all
    # subsequent frames so the plot scale stays consistent.
    _bev_xlim: Optional[tuple] = None
    _bev_zlim: Optional[tuple] = None

    # Cube debug records — collected for the first 15 frames that reach the
    # cube-drawing step and saved to debug_cube.json for post-run inspection.
    _cube_debug_records: List[dict] = []
    _CUBE_DEBUG_MAX = 15

    max_frames = cfg.num_bootstrap_frames if mode == "bootstrap" else None

    print(f"[SceneUnderstanding] mode={mode}  source={source_path}")
    print(f"[SceneUnderstanding] output → {out_dir}")
    t0 = time.perf_counter()

    for frame_idx, frame in frame_iter:
        # Initialise intrinsics from first frame
        if intrinsics is None:
            H, W = frame.shape[:2]
            intrinsics = cfg.build_intrinsics(H, W)
            first_frame_shape = (H, W)
            print(f"[SceneUnderstanding] intrinsics: fx={intrinsics.fx:.1f} "
                  f"fy={intrinsics.fy:.1f} cx={intrinsics.cx:.1f} cy={intrinsics.cy:.1f}")

        # ── Segmentation ───────────────────────────────────────────────────────
        if frame_idx % cfg.stride_seg == 0:
            last_mask = _get_mask(
                segmenter, frame, cache_dir, source_path, frame_idx,
                cfg.road_dilate_px, cfg.inference_max_dim,
            )
        road_mask = last_mask if last_mask is not None else np.zeros(frame.shape[:2], bool)

        # ── Depth ──────────────────────────────────────────────────────────────
        if frame_idx % cfg.stride_depth == 0:
            last_depth = _get_depth(
                estimator, frame, cache_dir, source_path, frame_idx,
                cfg.inference_max_dim,
            )
        depth = last_depth if last_depth is not None else np.zeros(frame.shape[:2], np.float32)

        # ── Backproject road pixels ────────────────────────────────────────────
        pts3d, pxcoords = backproject_depth_map(
            depth, intrinsics, mask=road_mask,
            max_points=cfg.sample_points_per_frame, rng=rng,
        )

        # ── RANSAC plane fit ───────────────────────────────────────────────────
        fitted: Optional[PlaneParams] = None
        if len(pts3d) >= 3:
            fitted = ransac_plane_fit(
                pts3d,
                ransac_iters=cfg.ransac_iters,
                inlier_thresh=cfg.inlier_thresh,
                min_inlier_ratio=cfg.min_inlier_ratio,
                rng=rng,
            )
            if fitted is not None:
                fitted.frame_idx = frame_idx

        # ── Update stable plane ───────────────────────────────────────────────
        stable_plane = tracker.update(fitted, frame_idx)

        # ── Save per-frame visualisations ─────────────────────────────────────
        fname = f"frame_{frame_idx:06d}.jpg"

        # Overlay (mask + grid + verification cube)
        overlay = overlay_mask(frame, road_mask)
        if stable_plane is not None:
            overlay = draw_plane_grid(
                overlay, intrinsics, stable_plane,
                grid_spacing=cfg.grid_spacing,
                grid_extent=cfg.grid_extent,
            )
            # Anchor the cube at the road-mask centroid rather than the
            # principal point.  Road pixels were used to fit the plane, so
            # the ray through their centroid is guaranteed to intersect it
            # within the image — unlike the centre pixel which may look at
            # the sky, horizon, or a point far beyond the visible road.
            cube_uv = None
            road_px_count = int(road_mask.sum())
            if road_mask.any():
                ys, xs = np.where(road_mask)
                cube_uv = (float(xs.mean()), float(ys.mean()))

            # Collect debug info for the first _CUBE_DEBUG_MAX frames
            _dbg: Optional[dict] = None
            # if len(_cube_debug_records) < _CUBE_DEBUG_MAX:
            #     _dbg = {
            #         "frame_idx": frame_idx,
            #         "n_processed": n_processed,
            #         "road_px_count": road_px_count,
            #         "intrinsics": {
            #             "fx": intrinsics.fx, "fy": intrinsics.fy,
            #             "cx": intrinsics.cx, "cy": intrinsics.cy,
            #         },
            #         "cfg_grid_spacing": cfg.grid_spacing,
            #     }

            # overlay = draw_ground_cube(
            #     overlay, intrinsics, stable_plane,
            #     center_uv=cube_uv,
            #     width=cfg.grid_spacing * 1.5,
            #     length=cfg.grid_spacing * 3.0,
            #     height=cfg.grid_spacing * 2.0,
            #     _debug_out=_dbg,
            # )

            # if _dbg is not None:
            #     _cube_debug_records.append(_dbg)
            #     # Live one-liner so failures are visible immediately in stdout
            #     miss   = _dbg.get("ray_plane_miss")
            #     c3d    = _dbg.get("center3d")
            #     in_img = _dbg.get("corners_in_image", "?")
            #     lines  = _dbg.get("lines_drawn", "?")
            #     reason = _dbg.get("failure_reason", "")
            #     z_str  = f"Z={c3d[2]:.3f}" if c3d else "NO_HIT"
            #     uv_str = (f"({cube_uv[0]:.0f},{cube_uv[1]:.0f})"
            #               if cube_uv else "None")
            #     print(
            #         f"  [cube dbg] fr={frame_idx:5d}  "
            #         f"road_px={road_px_count:6d}  uv={uv_str}  "
            #         f"miss={miss}  {z_str}  "
            #         f"in_img={in_img}/8  lines={lines}"
            #         + (f"  REASON: {reason}" if reason else "")
            #     )
        cv2.imwrite(str(dirs["overlay"] / fname), overlay)

        # Depth colourmap
        depth_img = depth_viz(depth)
        cv2.imwrite(str(dirs["depth"] / fname), depth_img)

        # 2×2 composite
        grid_frame = build_scene_frame(
            frame, road_mask, depth, stable_plane, intrinsics,
            grid_spacing=cfg.grid_spacing, grid_extent=cfg.grid_extent,
        )
        cv2.imwrite(str(dirs["grid"] / fname), grid_frame)

        # Inlier / outlier diagnostics (every 10th processed frame)
        if n_processed % 10 == 0 and fitted is not None and len(pxcoords) > 0:
            dists = fitted.abs_distance(pts3d)
            is_inlier = dists < cfg.inlier_thresh
            diag_img = draw_plane_inliers(frame, pxcoords, is_inlier, rng=rng)
            cv2.imwrite(str(dirs["diagnostics"] / fname), diag_img)

        # BEV plot (every 30th processed frame)
        if n_processed % 30 == 0 and stable_plane is not None:
            bev_pts = _project_road_to_bev(
                depth, road_mask, intrinsics, stable_plane,
                max_points=cfg.bev_max_points, rng=rng,
            )
            # Lock axis range from the first BEV frame so all plots share
            # the same scale — this prevents the "jumping size" artefact.
            if _bev_xlim is None and len(bev_pts) > 0:
                x_ext = max(float(np.abs(bev_pts[:, 0]).max()) * 1.5, 3.0)
                z_ext = max(float(bev_pts[:, 2].max()) * 1.3, 5.0)
                _bev_xlim = (-x_ext, x_ext)
                _bev_zlim = (0.0, z_ext)

            import matplotlib.pyplot as plt
            fig = bev_scatter_plot(
                bev_pts,
                plane_normal=stable_plane.n,
                title=f"BEV frame {frame_idx}",
                rng=rng,
                xlim=_bev_xlim,
                zlim=_bev_zlim,
            )
            fig.savefig(str(dirs["bev"] / f"bev_{frame_idx:06d}.png"), dpi=90, bbox_inches="tight")
            plt.close(fig)

        # Pixel probe (first frame that has a valid plane, then every 60 frames)
        if test_pixels and stable_plane is not None and n_processed % 60 == 0:
            probe_img, probe_res = draw_pixel_probe(
                frame, intrinsics, stable_plane,
                test_pixels, road_mask=road_mask,
                snap_radius_px=cfg.snap_radius_px,
            )
            probe_fname = f"probe_{frame_idx:06d}.jpg"
            cv2.imwrite(str(dirs["diagnostics"] / probe_fname), probe_img)

        # Video writer
        if cfg.save_overlay_video:
            if video_writer is None and first_frame_shape is not None:
                vpath = str(out_dir / "output_video_overlay.mp4")
                H_v, W_v = first_frame_shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    vpath, fourcc, cfg.fps / max(cfg.stride_seg, 1), (W_v, H_v)
                )
            if video_writer is not None:
                video_writer.write(overlay)

        n_processed += 1
        if n_processed % 20 == 0:
            elapsed = time.perf_counter() - t0
            fps_proc = n_processed / elapsed
            plane_str = (
                f"d={stable_plane.d:.3f} ir={stable_plane.inlier_ratio:.2f}"
                if stable_plane else "no plane"
            )
            print(f"  frame {frame_idx:5d}  processed={n_processed}  "
                  f"{fps_proc:.1f} fr/s  plane: {plane_str}")

        if mode == "bootstrap" and n_processed >= cfg.num_bootstrap_frames:
            break

    if video_writer is not None:
        video_writer.release()

    elapsed_total = time.perf_counter() - t0
    print(f"\n[SceneUnderstanding] done — {n_processed} frames in {elapsed_total:.1f}s")

    # ── Save cube debug records ────────────────────────────────────────────────
    if _cube_debug_records:
        debug_path = out_dir / "debug_cube.json"
        with open(debug_path, "w") as f:
            json.dump(_cube_debug_records, f, indent=2)
        print(f"[SceneUnderstanding] debug_cube.json → {debug_path}")
        # Print a compact human-readable summary table
        print("\n  === Cube debug summary ===")
        print(f"  {'frame':>6}  {'road_px':>8}  {'cube_uv':>18}  "
              f"{'miss':>5}  {'center3d_Z':>11}  {'in_img':>7}  {'lines':>6}  note")
        for rec in _cube_debug_records:
            uv   = rec.get("center_uv")
            uv_s = f"({uv[0]:.0f},{uv[1]:.0f})" if uv else "None"
            c3d  = rec.get("center3d")
            z_s  = f"{c3d[2]:.3f}" if c3d else "None"
            note = rec.get("failure_reason", "")
            print(f"  {rec['frame_idx']:6d}  {rec.get('road_px_count',0):8d}  "
                  f"{uv_s:>18}  {str(rec.get('ray_plane_miss','?')):>5}  "
                  f"{z_s:>11}  {str(rec.get('corners_in_image','?')):>7}  "
                  f"{str(rec.get('lines_drawn','?')):>6}  {note}")
        print()
    elif tracker.stable_plane is None:
        print("[SceneUnderstanding] WARNING: stable_plane was never set — "
              "no cube debug records collected.  Check RANSAC / road mask.")
    else:
        print("[SceneUnderstanding] cube debug records not collected "
              "(cube overlay debug is currently disabled).")

    # ── Save JSON plane parameters ─────────────────────────────────────────────
    json_path = out_dir / "plane_params.json"
    with open(json_path, "w") as f:
        json.dump(tracker.get_summary_dict(), f, indent=2)
    print(f"[SceneUnderstanding] plane_params.json → {json_path}")

    # ── Save diagnostics plot ──────────────────────────────────────────────────
    if tracker.history:
        import matplotlib.pyplot as plt
        fig = plane_diagnostics_plot(
            tracker.get_history_dicts(),
            min_inlier_ratio=cfg.min_inlier_ratio,
            max_angle_change=cfg.max_normal_angle_change_deg,
        )
        diag_png = out_dir / "plane_diagnostics.png"
        fig.savefig(str(diag_png), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[SceneUnderstanding] plane_diagnostics.png → {diag_png}")

    # ── Summary report ─────────────────────────────────────────────────────────
    report_path = _write_markdown_report(
        out_dir, mode, source_path, cfg, tracker, elapsed_total
    )
    print(f"[SceneUnderstanding] summary_report.md → {report_path}")

    return tracker


# ---------------------------------------------------------------------------
# Model builder (edit here to plug in real models)
# ---------------------------------------------------------------------------

def _build_models(args: argparse.Namespace) -> Tuple[RoadSegmenter, DepthEstimator]:
    """Instantiate segmenter and depth estimator.

    To use real models, replace the Dummy instances below with your own
    TorchRoadSegmenter / TorchDepthEstimator subclasses.

    Example (DepthAnything)::

        from transformers import pipeline as hf_pipeline
        from scene_understanding.models import TorchDepthEstimator

        class DepthAnythingWrapper(TorchDepthEstimator):
            def __init__(self):
                pipe = hf_pipeline(
                    task="depth-estimation",
                    model="LiheYoung/depth-anything-small-hf",
                )
                super().__init__(model=pipe, device="cpu")

            def predict(self, frame):
                import PIL.Image, numpy as np
                img_pil = PIL.Image.fromarray(frame[:, :, ::-1])  # BGR→RGB
                out = self.model(img_pil)["depth"]
                return np.array(out, dtype=np.float32)

        estimator = DepthAnythingWrapper()
    """
    segmenter = DummyRoadSegmenter(road_fraction=0.60)
    estimator = DummyDepthEstimator(min_depth=0.5, max_depth=10.0, seed=42)
    return segmenter, estimator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m scene_understanding.run_scene_understanding",
        description="Monocular ground-plane estimation and visualisation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video_path",  type=str, default=None,
                     help="Path to the input video file.")
    src.add_argument("--frames_dir",  type=str, default=None,
                     help="Directory of image files (sorted by filename).")

    # Intrinsics
    p.add_argument("--intrinsics", type=float, nargs=4,
                   metavar=("FX", "FY", "CX", "CY"), default=None,
                   help="Camera intrinsics in pixels.  Omit to auto-estimate.")

    # Mode
    p.add_argument("--mode", choices=["bootstrap", "run"], default="bootstrap",
                   help="bootstrap: first N frames only.  run: full video.")

    # Output
    p.add_argument("--output_dir", type=str, default="outputs/scene_understanding",
                   help="Root output directory.")

    # Stride / performance
    p.add_argument("--stride_seg",   type=int, default=1,
                   help="Run segmenter every N frames.")
    p.add_argument("--stride_depth", type=int, default=1,
                   help="Run depth estimator every N frames.")
    p.add_argument("--max_dim", type=int, default=None,
                   help="Resize frames to max(H,W)=max_dim before inference.")
    p.add_argument("--num_bootstrap_frames", type=int, default=30,
                   help="Number of frames to process in bootstrap mode.")

    # Plane fitting
    p.add_argument("--ransac_iters",       type=int,   default=150)
    p.add_argument("--inlier_thresh",      type=float, default=0.05)
    p.add_argument("--min_inlier_ratio",   type=float, default=0.30)
    p.add_argument("--max_normal_angle",   type=float, default=15.0,
                   help="Max normal angle change (degrees) between plane updates.")
    p.add_argument("--ema_alpha",          type=float, default=0.20)

    # Grid
    p.add_argument("--grid_spacing", type=float, default=1.0)
    p.add_argument("--grid_extent",  type=float, default=10.0)

    # Optional probe pixels
    p.add_argument("--probe_pixels", type=float, nargs="+", default=None,
                   metavar="U_V",
                   help="Test pixels for probe visualisation: u1 v1 u2 v2 ...")

    # Misc
    p.add_argument("--fps",          type=float, default=30.0)
    p.add_argument("--no_video",     action="store_true",
                   help="Skip writing the annotated MP4.")
    p.add_argument("--seed",         type=int,   default=42)

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)

    # Build config
    cfg = SceneConfig(
        fx=args.intrinsics[0] if args.intrinsics else None,
        fy=args.intrinsics[1] if args.intrinsics else None,
        cx=args.intrinsics[2] if args.intrinsics else None,
        cy=args.intrinsics[3] if args.intrinsics else None,
        fps=args.fps,
        stride_seg=args.stride_seg,
        stride_depth=args.stride_depth,
        num_bootstrap_frames=args.num_bootstrap_frames,
        ransac_iters=args.ransac_iters,
        inlier_thresh=args.inlier_thresh,
        min_inlier_ratio=args.min_inlier_ratio,
        max_normal_angle_change_deg=args.max_normal_angle,
        ema_alpha=args.ema_alpha,
        grid_spacing=args.grid_spacing,
        grid_extent=args.grid_extent,
        output_dir=args.output_dir,
        inference_max_dim=args.max_dim,
        save_overlay_video=not args.no_video,
    )

    # Probe pixels
    test_pixels = None
    if args.probe_pixels:
        flat = args.probe_pixels
        if len(flat) % 2 != 0:
            print("[WARNING] --probe_pixels must have an even number of values "
                  "(u v pairs). Last value ignored.")
            flat = flat[:-1]
        test_pixels = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]

    # Frame iterator
    max_frames_bootstrap = cfg.num_bootstrap_frames if args.mode == "bootstrap" else None
    if args.video_path:
        source_path = args.video_path
        frame_iter = _iter_video(
            args.video_path,
            stride=max(args.stride_seg, args.stride_depth),
            max_frames=max_frames_bootstrap,
        )
    else:
        source_path = args.frames_dir
        frame_iter = _iter_frames_dir(
            args.frames_dir,
            stride=max(args.stride_seg, args.stride_depth),
            max_frames=max_frames_bootstrap,
        )

    # Build models
    segmenter, estimator = _build_models(args)

    rng = np.random.default_rng(args.seed)

    run_pipeline(
        cfg=cfg,
        source_path=source_path,
        frame_iter=frame_iter,
        segmenter=segmenter,
        estimator=estimator,
        mode=args.mode,
        test_pixels=test_pixels,
        rng=rng,
    )


if __name__ == "__main__":
    main()
