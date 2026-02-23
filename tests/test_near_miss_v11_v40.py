"""
Unit tests for near-miss v1.1 / v4.0 false-positive suppression.

Focus:
    Adjacent-lane side-by-side traffic should not trigger near-miss events
    when predicted closest approach remains wider than object footprint width.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.near_miss import NearMissDetectorV11, NearMissDetectorV40


def _make_obj(
    obj_id: int,
    center_x: float,
    center_y: float,
    traj: list,
    w: float = 40.0,
    h: float = 40.0,
    conf: float = 0.95,
):
    x1 = center_x - w / 2.0
    x2 = center_x + w / 2.0
    y1 = center_y - h / 2.0
    y2 = center_y + h / 2.0
    return {
        "id": obj_id,
        "class": "vehicle",
        "label": "car",
        "bbox": [x1, y1, x2, y2],
        "confidence": conf,
        "center": (center_x, center_y),
        "trajectory": traj,
    }


def test_v11_suppresses_adjacent_lane_false_positive():
    det = NearMissDetectorV11(
        proximity_px=100.0,
        ttc_threshold=2.0,
        fps=20.0,
        debounce_frames=1,
        filters_enabled=True,
        confirm_frames=1,
        moving_speed_px=4.0,
        clearance_scale=1.1,
    )

    # Different directions and high speed, but still lane-separated.
    # Without clearance gating this pair can satisfy the legacy 2-of-3 gate.
    obj1 = _make_obj(
        1, 100.0, 280.0,
        traj=[(0, 100.0, 270.0), (1, 100.0, 275.0), (2, 100.0, 280.0)],
    )
    obj2 = _make_obj(
        2, 178.0, 280.0,
        traj=[(0, 180.0, 290.0), (1, 179.0, 285.0), (2, 178.0, 280.0)],
    )

    events = det.process_frame(2, {1: obj1, 2: obj2})
    assert events == []


def test_v11_keeps_true_converging_case():
    det = NearMissDetectorV11(
        proximity_px=100.0,
        ttc_threshold=2.0,
        fps=20.0,
        debounce_frames=1,
        filters_enabled=True,
        confirm_frames=1,
        moving_speed_px=4.0,
        clearance_scale=1.1,
    )

    # Opposing directions, same path, predicted closest approach ~0 px.
    obj1 = _make_obj(
        1, 110.0, 280.0,
        traj=[(0, 100.0, 280.0), (1, 105.0, 280.0), (2, 110.0, 280.0)],
    )
    obj2 = _make_obj(
        2, 150.0, 280.0,
        traj=[(0, 160.0, 280.0), (1, 155.0, 280.0), (2, 150.0, 280.0)],
    )

    events = det.process_frame(2, {1: obj1, 2: obj2})
    assert len(events) == 1
    assert events[0]["object_id_1"] == 1
    assert events[0]["object_id_2"] == 2


def test_v40_inherits_adjacent_lane_suppression_without_flow():
    det = NearMissDetectorV40(
        proximity_px=100.0,
        ttc_threshold=2.0,
        fps=20.0,
        debounce_frames=1,
        filters_enabled=True,
        confirm_frames=1,
        moving_speed_px=4.0,
        clearance_scale=1.1,
    )

    obj1 = _make_obj(
        1, 100.0, 280.0,
        traj=[(0, 100.0, 270.0), (1, 100.0, 275.0), (2, 100.0, 280.0)],
    )
    obj2 = _make_obj(
        2, 178.0, 280.0,
        traj=[(0, 180.0, 290.0), (1, 179.0, 285.0), (2, 178.0, 280.0)],
    )

    # frame=None -> v4.0 degrades to trajectory-only velocity path.
    events = det.process_frame(2, {1: obj1, 2: obj2}, frame=None)
    assert events == []
