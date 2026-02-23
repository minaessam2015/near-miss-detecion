"""
Task 1.5 — Visualization & Analysis
Annotated video, dashboard charts, and HTML summary report.
"""

import base64
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Frame annotation — helpers
# ---------------------------------------------------------------------------

_COLOR_VEHICLE    = (255, 100,  0)   # BGR blue-ish
_COLOR_PEDESTRIAN = (  0, 200, 50)   # BGR green
_COLOR_NEARMISS   = ( 30,  30, 220)  # BGR red (fallback)

# Risk-level colors for near-miss highlights (BGR)
_RISK_COLORS: Dict[str, Tuple[int, int, int]] = {
    "High":   ( 30,  30, 220),   # red
    "Medium": ( 20, 130, 245),   # orange
    "Low":    ( 30, 200, 220),   # yellow
}

# Trajectory trail — fixed gray regardless of near-miss status so the
# historical path is never confused with connector lines or heading arrows.
# Full trail (30 frames) = dim gray; recent tail (8 frames) = bright white.
_TRAIL_COLOR        = (160, 160, 160)   # BGR dim gray  — full history
_TRAIL_RECENT_COLOR = (240, 240, 240)   # BGR near-white — recent 8 frames
_PRED_PATH_COLOR    = (220,  60, 255)   # BGR magenta   — debug future path
_PRED_END_COLOR     = (255, 180, 255)   # BGR light magenta

# Maximum frame-to-frame centroid jump (px) considered normal motion.
# Larger jumps indicate a tracker ID re-use (new vehicle assigned an old ID),
# which would corrupt both the trail direction and the heading arrow.
_TRAIL_JUMP_PX: float = 150.0


def _last_clean_segment(
    traj: List[Tuple],
    max_jump_px: float = _TRAIL_JUMP_PX,
) -> List[Tuple]:
    """
    Return the longest suffix of traj that contains no large positional jump.

    ByteTrack re-uses numeric IDs when it thinks a lost track has reappeared.
    If the old and new vehicles are far apart, the trajectory jumps by hundreds
    of pixels in one frame.  Using the full traj would:
      - Draw the trail *ahead* of the vehicle (old positions were in front).
      - Point the heading arrow in the wrong direction.

    Walking backward from the newest point and stopping at the first jump
    returns only the clean, physically-continuous segment for this vehicle.
    """
    if len(traj) < 2:
        return traj
    max_jump_sq = max_jump_px ** 2
    cutoff = 0
    for k in range(len(traj) - 1, 0, -1):
        _, cx1, cy1 = traj[k]
        _, cx0, cy0 = traj[k - 1]
        if (cx1 - cx0) ** 2 + (cy1 - cy0) ** 2 > max_jump_sq:
            cutoff = k   # traj[k] is the first point after the jump
            break
    return traj[cutoff:]


def _velocity_px_per_processed_frame(
    traj: List[Tuple],
    window: int = 5,
) -> Optional[Tuple[float, float]]:
    """
    Estimate (vx, vy) in px/processed-frame from trajectory history.

    Uses trajectory frame indices to normalize tracker gaps while remaining
    stride-agnostic (works for stride=1 and >1).
    """
    pts = traj[-window:]
    if len(pts) < 2:
        return None

    deltas = []
    for i in range(len(pts) - 1):
        try:
            dt = int(pts[i + 1][0]) - int(pts[i][0])
        except Exception:
            dt = 1
        if dt > 0:
            deltas.append(dt)
    nominal_dt = min(deltas) if deltas else 1

    try:
        f0, x0, y0 = pts[0]
        f1, x1, y1 = pts[-1]
        dt_raw = int(f1) - int(f0)
    except Exception:
        return None
    if dt_raw <= 0:
        return None

    dt_norm = dt_raw / float(nominal_dt)
    if dt_norm <= 0:
        return None
    return ((x1 - x0) / dt_norm, (y1 - y0) / dt_norm)


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def _fmt_score(score: Any, ndigits: int = 2) -> str:
    """Format risk score for overlays; returns 'N/A' when unavailable."""
    try:
        return f"{float(score):.{ndigits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _label_bg(
    img: np.ndarray,
    x: int,
    y: int,
    text: str,
    font_scale: float = 0.62,
    thickness: int = 1,
    fg: Tuple[int, int, int] = (255, 255, 255),
    bg: Tuple[int, int, int] = (30, 30, 30),
) -> int:
    """
    Draw text with a filled background rectangle.

    Places the text baseline at y, fills a rect behind it, and returns the
    total pixel height consumed (useful for stacking multiple lines upward).
    """
    (tw, th), bl = cv2.getTextSize(text, _FONT, font_scale, thickness)
    pad = 4
    cv2.rectangle(img,
                  (x - pad,      y - th - pad),
                  (x + tw + pad, y + bl + pad // 2),
                  bg, -1)
    cv2.putText(img, text, (x, y), _FONT, font_scale, fg, thickness, cv2.LINE_AA)
    return th + bl + pad + pad // 2   # total height consumed


def _draw_connector(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash: int = 10,
    gap: int = 5,
) -> None:
    """Draw a dashed line between two points."""
    x1, y1 = pt1
    x2, y2 = pt2
    total = math.hypot(x2 - x1, y2 - y1)
    if total < 1.0:
        return
    dx, dy = (x2 - x1) / total, (y2 - y1) / total
    pos = 0.0
    while pos < total:
        xs = int(x1 + pos * dx)
        ys = int(y1 + pos * dy)
        end = min(pos + dash, total)
        xe = int(x1 + end * dx)
        ye = int(y1 + end * dy)
        cv2.line(img, (xs, ys), (xe, ye), color, thickness, cv2.LINE_AA)
        pos += dash + gap


# ---------------------------------------------------------------------------
# Frame annotation — main function
# ---------------------------------------------------------------------------

def annotate_frame(
    frame: np.ndarray,
    tracked_objects: Dict[int, Dict[str, Any]],
    active_pair_ids: List[Tuple[int, int]],
    frame_idx: int,
    fps: float,
    pair_data: Optional[Dict[Tuple[int, int], Dict[str, Any]]] = None,
    debug_pred_paths: bool = False,
    pred_horizon_sec: float = 1.5,
    pred_step_sec: float = 0.25,
    pred_min_speed_px: float = 1.0,
) -> np.ndarray:
    """
    Draw bounding boxes, IDs, and near-miss overlays on a copy of the frame.

    Args:
        frame:            BGR frame as numpy array.
        tracked_objects:  Dict from any BaseTracker (id → state dict).
        active_pair_ids:  List of (id1, id2) pairs with active near-miss events.
                          Used only when pair_data is None (basic fallback mode).
        frame_idx:        Current frame index (for timestamp overlay).
        fps:              Effective FPS (for timestamp conversion).
        pair_data:        Optional dict from NearMissDetector.active_pair_data().
                          When supplied, enables rich per-object labels with
                          target-ID, TTC, and risk-level info, plus connector
                          lines between colliding pairs.
                          Schema: {(id1, id2): event_dict}
        debug_pred_paths: Draw debug predicted motion path for each object.
        pred_horizon_sec: Prediction horizon in seconds.
        pred_step_sec:    Sampling step (sec) along the future path.
        pred_min_speed_px:Minimum speed (px/processed-frame) to draw prediction.

    Returns:
        Annotated BGR frame.
    """
    out = frame.copy()
    h_frame, w_frame = out.shape[:2]

    # ── Build per-object near-miss info ─────────────────────────────────────
    # obj_info[id] = {partner, ttc, risk, score, color}
    # When an object appears in multiple pairs, keep the worst (highest score).
    obj_info: Dict[int, Dict[str, Any]] = {}

    if pair_data:
        for (id1, id2), ev in pair_data.items():
            risk  = ev.get("risk_level", "Low")
            score = ev.get("risk_score", 0.0)
            ttc   = ev.get("ttc_sec", None)
            dist  = ev.get("distance_px", None)
            rcolor = _RISK_COLORS.get(risk, _COLOR_NEARMISS)
            for my_id, partner_id in ((id1, id2), (id2, id1)):
                existing = obj_info.get(my_id)
                if existing is None or score > existing["score"]:
                    obj_info[my_id] = {
                        "partner": partner_id,
                        "ttc":     ttc,
                        "dist":    dist,
                        "risk":    risk,
                        "score":   score,
                        "color":   rcolor,
                    }
    else:
        # Fallback: mark active IDs with no extra metrics
        for id1, id2 in active_pair_ids:
            for my_id, partner_id in ((id1, id2), (id2, id1)):
                if my_id not in obj_info:
                    obj_info[my_id] = {
                        "partner": partner_id,
                        "ttc": None, "dist": None,
                        "risk": "Low", "score": 0.0,
                        "color": _COLOR_NEARMISS,
                    }

    # ── Phase 1: connector lines (drawn first, behind boxes) ────────────────
    src = pair_data if pair_data else {}
    for (id1, id2), ev in src.items():
        if id1 not in tracked_objects or id2 not in tracked_objects:
            continue
        c1 = tracked_objects[id1]["center"]
        c2 = tracked_objects[id2]["center"]
        risk   = ev.get("risk_level", "Low")
        score  = ev.get("risk_score", None)
        rcolor = _RISK_COLORS.get(risk, _COLOR_NEARMISS)
        _draw_connector(out, c1, c2, rcolor, thickness=3, dash=20, gap=10)

        # Risk badge at midpoint
        mx, my = (c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2
        badge = f"{risk.upper()} S:{_fmt_score(score)}"
        (bw, bh), _ = cv2.getTextSize(badge, _FONT, 0.5, 1)
        cv2.rectangle(out, (mx - 4, my - bh - 4), (mx + bw + 4, my + 4), rcolor, -1)
        cv2.putText(out, badge, (mx, my), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Phase 2: boxes and labels ────────────────────────────────────────────
    for obj_id, obj in tracked_objects.items():
        x1, y1, x2, y2 = obj["bbox"]
        label  = obj.get("label", obj["class"])
        is_nm  = obj_id in obj_info
        info   = obj_info.get(obj_id)

        # Box
        if is_nm:
            box_color = info["color"]
            box_thick = 3
        elif obj["class"] == "vehicle":
            box_color = _COLOR_VEHICLE
            box_thick = 2
        else:
            box_color = _COLOR_PEDESTRIAN
            box_thick = 2
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, box_thick)

        # ── Trajectory trail ──────────────────────────────────────────────────
        # Keep only trajectory points available at this frame.
        # This protects against shallow frame snapshots that share a live list
        # with the tracker and later accumulate future points.
        traj    = [p for p in obj.get("trajectory", []) if p[0] <= frame_idx]
        cx_now  = int(obj["center"][0])
        cy_now  = int(obj["center"][1])

        # Strip trajectory contamination from tracker ID re-use: use only the
        # physically-continuous suffix after the last large positional jump.
        clean_traj = _last_clean_segment(traj)

        trail_pts = [(int(cx), int(cy)) for _, cx, cy in clean_traj[-30:]]
        # Bridge any gap: ensure trail ends exactly at the current centroid.
        if not trail_pts or trail_pts[-1] != (cx_now, cy_now):
            trail_pts.append((cx_now, cy_now))
        if len(trail_pts) >= 2:
            # Full trail — dim gray, independent of box/risk color so it is
            # never confused with the colored near-miss connector lines.
            pts_arr = np.array(trail_pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts_arr], isClosed=False,
                          color=_TRAIL_COLOR, thickness=1, lineType=cv2.LINE_AA)
            # Recent 8 points — bright white, clearly the "wake" just behind
            # the vehicle and distinct from the yellow heading arrow ahead.
            recent = trail_pts[-8:]
            if len(recent) >= 2:
                rec_arr = np.array(recent, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [rec_arr], isClosed=False,
                              color=_TRAIL_RECENT_COLOR, thickness=2, lineType=cv2.LINE_AA)

        # ── Debug: predicted path (1–2 s ahead) ──────────────────────────────
        if debug_pred_paths and fps > 0:
            vel = _velocity_px_per_processed_frame(clean_traj, window=5)
            if vel is not None:
                vx_pf, vy_pf = vel
                spd_pf = math.hypot(vx_pf, vy_pf)
                if spd_pf >= pred_min_speed_px:
                    horizon = max(0.1, float(pred_horizon_sec))
                    step_s  = max(0.05, float(pred_step_sec))
                    n_steps = max(1, int(math.ceil(horizon / step_s)))
                    pred_pts = [(cx_now, cy_now)]
                    for s in range(1, n_steps + 1):
                        t_sec = min(s * step_s, horizon)
                        t_pf  = t_sec * fps
                        px = int(round(cx_now + vx_pf * t_pf))
                        py = int(round(cy_now + vy_pf * t_pf))
                        pred_pts.append((px, py))

                    # Dashed style: draw every other segment.
                    for k in range(len(pred_pts) - 1):
                        if k % 2 == 0:
                            cv2.line(out, pred_pts[k], pred_pts[k + 1],
                                     _PRED_PATH_COLOR, 2, cv2.LINE_AA)
                    cv2.circle(out, pred_pts[-1], 3, _PRED_END_COLOR, -1, cv2.LINE_AA)

        # ── Heading arrow ─────────────────────────────────────────────────────
        # Uses the same clean_traj segment so ID-switch contamination cannot
        # reverse the arrow direction.  Requires at least 5 clean points so
        # brand-new tracks (1-2 frames old) don't show a jitter-driven arrow.
        _HDG_COLOR = (0, 230, 255)   # BGR bright yellow — distinct from trail
        if len(clean_traj) >= 5:
            hdg_pts = [(float(cx), float(cy)) for _, cx, cy in clean_traj[-5:]]
            hdg_pts.append((float(cx_now), float(cy_now)))
            dx  = hdg_pts[-1][0] - hdg_pts[0][0]
            dy  = hdg_pts[-1][1] - hdg_pts[0][1]
            mag = math.hypot(dx, dy)
            if mag > 3.0:
                nx, ny    = dx / mag, dy / mag
                arrow_len = 45
                end_pt    = (int(cx_now + nx * arrow_len), int(cy_now + ny * arrow_len))
                cv2.arrowedLine(out, (cx_now, cy_now), end_pt, _HDG_COLOR,
                                thickness=3, line_type=cv2.LINE_AA, tipLength=0.40)
                cv2.circle(out, (cx_now, cy_now), 4, _HDG_COLOR, -1, cv2.LINE_AA)

        # Labels — stack upward from just above the top-left corner
        ly = max(y1 - 4, 18)   # baseline of the lowest label line

        if is_nm and info:
            # Line 2 (lower): target + TTC + risk
            partner = info["partner"]
            ttc     = info["ttc"]
            ttc_str = f"{ttc:.1f}s" if ttc is not None else "N/A"
            line2   = (
                f"-> #{partner}  TTC:{ttc_str}  "
                f"{info['risk'].upper()}  S:{_fmt_score(info.get('score'))}"
            )
            h2 = _label_bg(out, x1, ly, line2,
                           font_scale=0.55, thickness=1, bg=info["color"])
            ly -= (h2 + 2)

        # Line 1 (top): object ID + sub-type
        line1  = f"#{obj_id}  {label}"
        id_bg  = info["color"] if is_nm else (40, 40, 40)
        _label_bg(out, x1, ly, line1,
                  font_scale=0.65, thickness=1, bg=id_bg)

    # ── Near-miss banner ─────────────────────────────────────────────────────
    if obj_info:
        # Sort pairs by risk score descending, cap at 4 rows in banner
        sorted_pairs = sorted(
            src.items(),
            key=lambda kv: kv[1].get("risk_score", 0.0),
            reverse=True,
        )[:4]
        n_rows    = len(sorted_pairs)
        banner_h  = 42 + n_rows * 24
        overlay   = out.copy()
        cv2.rectangle(overlay, (0, 0), (w_frame, banner_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.70, out, 0.30, 0, out)

        cv2.putText(out, "! NEAR-MISS DETECTED !", (10, 28),
                    _FONT_BOLD, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        py = 50
        for (id1, id2), ev in sorted_pairs:
            risk   = ev.get("risk_level", "Low")
            score  = ev.get("risk_score", None)
            ttc    = ev.get("ttc_sec", None)
            dist   = ev.get("distance_px", None)
            ttc_s  = f"TTC:{ttc:.1f}s" if ttc is not None else "TTC:N/A"
            dist_s = f"  dist:{dist:.0f}px" if dist is not None else ""
            rcolor = _RISK_COLORS.get(risk, (255, 255, 255))
            row    = (
                f"  #{id1} <-> #{id2}   {ttc_s}{dist_s}   "
                f"[{risk.upper()} S:{_fmt_score(score)}]"
            )
            cv2.putText(out, row, (10, py), _FONT, 0.55, rcolor, 1, cv2.LINE_AA)
            py += 22

    # ── Timestamp (bottom-left) ───────────────────────────────────────────────
    ts  = frame_idx / fps if fps > 0 else 0
    mm  = int(ts // 60)
    ss  = ts % 60
    _label_bg(out, 8, h_frame - 10,
              f"Frame {frame_idx}   {mm:02d}:{ss:05.2f}",
              font_scale=0.52, thickness=1, bg=(0, 0, 0))

    return out


# ---------------------------------------------------------------------------
# Annotated video writer
# ---------------------------------------------------------------------------

def create_annotated_video(
    video_path: str,
    output_path: str,
    frame_data: Dict[int, Tuple[Dict, List]],
    fps: float,
    frame_pair_data: Optional[Dict[int, Dict]] = None,
    debug_pred_paths: bool = False,
    pred_horizon_sec: float = 1.5,
    pred_step_sec: float = 0.25,
    pred_min_speed_px: float = 1.0,
) -> None:
    """
    Write an annotated video to disk.

    Args:
        video_path:       Path to the original video.
        output_path:      Path for the output MP4.
        frame_data:       Dict mapping frame_idx → (tracked_objects, active_pairs).
        fps:              Effective FPS for timestamp labeling.
        frame_pair_data:  Optional dict mapping frame_idx → pair_data dict from
                          NearMissDetector.active_pair_data(). When provided,
                          each frame shows TTC, target-ID, and risk labels.
        debug_pred_paths: Draw per-object future path overlay.
        pred_horizon_sec: Prediction horizon (sec) for debug path.
        pred_step_sec:    Time step (sec) between predicted samples.
        pred_min_speed_px:Minimum speed (px/processed-frame) to draw prediction.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, orig_fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_data:
            tracked_objs, active_pairs = frame_data[frame_idx]
            pd = frame_pair_data.get(frame_idx) if frame_pair_data else None
            annotated = annotate_frame(
                frame, tracked_objs, active_pairs, frame_idx, fps,
                pair_data=pd,
                debug_pred_paths=debug_pred_paths,
                pred_horizon_sec=pred_horizon_sec,
                pred_step_sec=pred_step_sec,
                pred_min_speed_px=pred_min_speed_px,
            )
        else:
            annotated = frame

        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Annotated video saved: {output_path}")


# ---------------------------------------------------------------------------
# Dashboard visualizations
# ---------------------------------------------------------------------------

def plot_timeline(
    events_df: pd.DataFrame,
    duration_sec: float,
    output_path: str,
) -> None:
    """Event timeline scatter plot."""
    fig, ax = plt.subplots(figsize=(12, 3))

    if events_df.empty:
        ax.text(0.5, 0.5, "No near-miss events detected", ha="center", va="center",
                transform=ax.transAxes, fontsize=13, color="gray")
    else:
        level_colors = {"High": "#e74c3c", "Medium": "#e67e22", "Low": "#f1c40f"}
        level_order = ["High", "Medium", "Low"]
        y_map = {"High": 2, "Medium": 1, "Low": 0}

        for level in level_order:
            sub = events_df[events_df["risk_level"] == level]
            if sub.empty:
                continue
            ax.scatter(
                sub["timestamp_sec"],
                [y_map[level]] * len(sub),
                color=level_colors[level],
                s=80, zorder=3, label=level, alpha=0.85,
            )

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Low", "Medium", "High"])
        ax.set_xlim(0, duration_sec)
        ax.set_xlabel("Time (seconds)")
        ax.set_title("Near-Miss Events Timeline")
        ax.grid(axis="x", alpha=0.3)
        ax.legend(loc="upper right", title="Risk Level")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_risk_distribution(
    events_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Pie chart of risk level distribution."""
    fig, ax = plt.subplots(figsize=(5, 5))

    if events_df.empty:
        ax.text(0.5, 0.5, "No events", ha="center", va="center",
                transform=ax.transAxes, fontsize=13, color="gray")
    else:
        counts = events_df["risk_level"].value_counts()
        colors = {"High": "#e74c3c", "Medium": "#e67e22", "Low": "#f1c40f"}
        c = [colors.get(l, "#aaa") for l in counts.index]
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            colors=c,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.8,
        )
        for t in autotexts:
            t.set_fontsize(10)

    ax.set_title("Risk Level Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_heatmap(
    all_trajectories: Dict[int, List[Tuple]],
    first_frame: np.ndarray,
    output_path: str,
) -> None:
    """
    2D heatmap of all tracked object centroid positions, overlaid on the first frame.
    """
    h, w = first_frame.shape[:2]

    xs, ys = [], []
    for traj in all_trajectories.values():
        for _, cx, cy in traj:
            xs.append(cx)
            ys.append(cy)

    fig, ax = plt.subplots(figsize=(10, 6))
    rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb, alpha=0.6)

    if xs:
        H, xedges, yedges = np.histogram2d(
            xs, ys,
            bins=[max(w // 20, 20), max(h // 20, 20)],
            range=[[0, w], [0, h]],
        )
        H = H.T  # transpose for correct orientation
        ax.imshow(
            H,
            origin="upper",
            extent=[0, w, h, 0],
            cmap="hot",
            alpha=0.5,
            aspect="auto",
        )

    ax.set_title("Object Activity Heatmap")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_frequency(
    events_df: pd.DataFrame,
    duration_sec: float,
    output_path: str,
    bin_sec: float = 15.0,
) -> None:
    """Bar chart of near-miss frequency per time bin."""
    n_bins = max(int(math.ceil(duration_sec / bin_sec)), 1)
    bins = [i * bin_sec for i in range(n_bins + 1)]
    labels = [f"{int(bins[i])}-{int(bins[i+1])}s" for i in range(n_bins)]

    counts = {lbl: 0 for lbl in labels}
    if not events_df.empty:
        for ts in events_df["timestamp_sec"]:
            idx = min(int(ts // bin_sec), n_bins - 1)
            counts[labels[idx]] += 1

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, [counts[l] for l in labels], color="#3498db", edgecolor="white")
    ax.set_xlabel("Time Interval")
    ax.set_ylabel("Near-Miss Count")
    ax.set_title("Near-Miss Frequency Over Time")
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def _img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_html_report(
    metadata: Dict[str, Any],
    events_df: pd.DataFrame,
    tracker_stats: Dict[str, Any],
    img_paths: Dict[str, str],
    output_path: str,
) -> None:
    """
    Generate a self-contained HTML report with embedded images.

    Args:
        metadata: Dict from get_video_metadata().
        events_df: DataFrame of near-miss events.
        tracker_stats: Dict with total_ids, avg_track_length, peak_active.
        img_paths: Dict mapping name → file path for each PNG chart.
        output_path: Where to save the HTML file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    summary = {
        "total": len(events_df),
        "high": int((events_df["risk_level"] == "High").sum()) if not events_df.empty else 0,
        "medium": int((events_df["risk_level"] == "Medium").sum()) if not events_df.empty else 0,
        "low": int((events_df["risk_level"] == "Low").sum()) if not events_df.empty else 0,
    }

    # Top 5 events
    top5_html = "<p><em>No events detected.</em></p>"
    if not events_df.empty:
        top5 = events_df.nlargest(5, "risk_score")[
            ["timestamp_sec", "class_1", "class_2", "distance_px", "ttc_sec", "risk_score", "risk_level"]
        ]
        top5_html = top5.to_html(index=False, border=0, classes="table")

    # Embed images as base64
    img_tags = ""
    for name, path in img_paths.items():
        if os.path.exists(path):
            b64 = _img_to_base64(path)
            img_tags += f"""
            <div class="chart">
              <h3>{name.replace('_', ' ').title()}</h3>
              <img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:6px;" />
            </div>"""

    dur_mm = int(metadata['duration_sec'] // 60)
    dur_ss = metadata['duration_sec'] % 60

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Near-Miss Detection Report</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; max-width: 960px; margin: 40px auto; color: #222; }}
  h1 {{ color: #c0392b; }}
  h2 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 6px; }}
  table.table {{ border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 0.9em; }}
  table.table th {{ background: #2c3e50; color: white; padding: 8px 12px; text-align: left; }}
  table.table td {{ padding: 7px 12px; border-bottom: 1px solid #eee; }}
  table.table tr:nth-child(even) {{ background: #f9f9f9; }}
  .stat-grid {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 16px 0; }}
  .stat-box {{ background: #f0f4f8; border-radius: 8px; padding: 16px 24px; min-width: 140px; text-align: center; }}
  .stat-box .value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
  .stat-box .label {{ font-size: 0.85em; color: #666; margin-top: 4px; }}
  .chart {{ margin: 24px 0; }}
  ul.limits {{ line-height: 1.9; }}
</style>
</head>
<body>
<h1>Near-Miss Incident Detection Report</h1>
<p><strong>Generated by:</strong> Near-Miss Detection Pipeline &nbsp;|&nbsp; <strong>Video duration:</strong> {dur_mm:02d}:{dur_ss:05.2f}</p>

<h2>Video Metadata</h2>
<table class="table">
  <tr><th>Property</th><th>Value</th></tr>
  <tr><td>Duration</td><td>{dur_mm:02d}:{dur_ss:05.2f} ({metadata['duration_sec']:.1f}s)</td></tr>
  <tr><td>Resolution</td><td>{metadata['width']} × {metadata['height']} px</td></tr>
  <tr><td>Frame Rate</td><td>{metadata['fps']:.2f} FPS</td></tr>
  <tr><td>Total Frames</td><td>{metadata['frame_count']}</td></tr>
</table>

<h2>Tracking Summary</h2>
<div class="stat-grid">
  <div class="stat-box"><div class="value">{tracker_stats.get('total_ids', '—')}</div><div class="label">Unique Objects Tracked</div></div>
  <div class="stat-box"><div class="value">{tracker_stats.get('avg_track_length', '—')}</div><div class="label">Avg Track Length (frames)</div></div>
  <div class="stat-box"><div class="value">{tracker_stats.get('peak_active', '—')}</div><div class="label">Peak Simultaneous Objects</div></div>
</div>

<h2>Near-Miss Summary</h2>
<div class="stat-grid">
  <div class="stat-box"><div class="value">{summary['total']}</div><div class="label">Total Events</div></div>
  <div class="stat-box" style="background:#fdecea"><div class="value" style="color:#c0392b">{summary['high']}</div><div class="label">High Risk</div></div>
  <div class="stat-box" style="background:#fef5e7"><div class="value" style="color:#e67e22">{summary['medium']}</div><div class="label">Medium Risk</div></div>
  <div class="stat-box" style="background:#fefde7"><div class="value" style="color:#d4ac0d">{summary['low']}</div><div class="label">Low Risk</div></div>
</div>

<h2>Top 5 Highest-Risk Events</h2>
{top5_html}

<h2>Visualizations</h2>
{img_tags}

<h2>Limitations &amp; Assumptions</h2>
<ul class="limits">
  <li><strong>Pixel-space distances only:</strong> Proximity thresholds are in screen pixels and do not account for perspective distortion. Objects far from the camera appear closer in pixel space than they actually are in real-world coordinates.</li>
  <li><strong>No depth estimation:</strong> The system cannot distinguish between objects that pass behind each other vs. objects that are genuinely close in 3D space. A bird flying low over traffic would be treated as a vehicle near-miss.</li>
  <li><strong>Static camera assumption:</strong> The TTC and speed estimation assume the camera is stationary. A pan/tilt camera would introduce spurious motion vectors into all tracked objects simultaneously.</li>
  <li><strong>2D TTC approximation:</strong> Time-to-collision is estimated from centroid-to-centroid closing speed in the image plane, not from actual vehicle kinematics. This is a rough proxy.</li>
  <li><strong>Occlusion handling:</strong> The centroid tracker uses simple nearest-centroid matching. Under heavy occlusion, ID switches can create false trajectory continuations, affecting speed and heading estimates.</li>
  <li><strong>Fixed thresholds:</strong> Proximity and TTC thresholds are global constants. A scene-adaptive approach (e.g., calibrated from lane markings) would be more robust.</li>
</ul>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved: {output_path}")
