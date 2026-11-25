"""
tests/test_vision_stopline_exits.py
===================================
Unit tests for camera-side stop-line exit counting in
``src/vision/state_extractor.py`` (schema 1.3.0 throughput counterpart).

Covers:
- Crossing the stop line in the exit direction with >= min_track_frames counts 1.
- Disappearance inside the ROI before the stop line counts 0.
- Jittering back and forth across the stop line counts at most 1 (same track ID).
- A 2-frame track crossing the stop line counts 0 (< default min_track_frames=3).
- Reference point uses bbox bottom-centre (not geometric centre).
- Reverse-direction crossings and crossings outside the finite segment count 0.
- Per-decision-step accumulation and reset via ``get_step_exits()`` and ``reset()``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from src.vision.state_extractor import LaneROI, StateExtractor


def _make_extractor(
    *,
    stop_line: Tuple[Tuple[float, float], Tuple[float, float]] | None = (
        (100.0, 200.0),
        (200.0, 200.0),
    ),
    min_track_frames: int = 3,
    extra_rois: Dict[str, LaneROI] | None = None,
) -> StateExtractor:
    rois: Dict[str, LaneROI] = {
        "approach_0": LaneROI(
            lane_id="approach_0",
            tls_id="J0",
            polygon=[(100.0, 50.0), (200.0, 50.0), (200.0, 200.0), (100.0, 200.0)],
            cam_id="cam0",
            lane_length_px=150.0,
            lane_width_px=100.0,
            stop_line=stop_line,
        )
    }
    if extra_rois:
        rois.update(extra_rois)
    controlled = {"J0": list(rois.keys())}
    return StateExtractor(
        tls_ids=["J0"],
        controlled_lanes_dict=controlled,
        lane_rois=rois,
        max_lanes_per_tls=max(4, len(rois)),
        min_track_frames=min_track_frames,
    )


def _track_with_bottom_center(
    track_id: int,
    bx: float,
    by: float,
    *,
    width: float = 20.0,
    height: float = 40.0,
    cls_id: int = 2,
    cam_id: str = "cam0",
    speed_px_s: float = 40.0,
) -> Dict[str, Any]:
    """Build a track dict whose bbox bottom-centre is (bx, by)."""
    return {
        "track_id": track_id,
        "cls_id": cls_id,
        "bbox": [bx - width / 2.0, by - height, bx + width / 2.0, by],
        "speed_px_s": speed_px_s,
        "cam_id": cam_id,
    }


def _step_frame(
    extractor: StateExtractor,
    tracks: List[Dict[str, Any]],
    timestamp: float = 0.0,
) -> Dict[str, Any]:
    frame = {"cam_id": "cam0", "timestamp": timestamp, "tracks": tracks, "accident_found": False}
    _, info = extractor.build_state(
        frame, phase_map={"J0": 0}, green_timers={"J0": 10.0}, return_info=True
    )
    return info


@pytest.mark.unit
class TestStopLineExitCounting:
    def test_crossing_counts_one(self):
        extractor = _make_extractor()
        # 3-frame track crossing y=200 in the exit direction (+y)
        _step_frame(extractor, [_track_with_bottom_center(1, 150.0, 170.0)], 0.0)
        _step_frame(extractor, [_track_with_bottom_center(1, 150.0, 190.0)], 0.1)
        info = _step_frame(extractor, [_track_with_bottom_center(1, 150.0, 215.0)], 0.2)

        assert info["approach_exits"]["approach_0"] == 1
        assert extractor.get_step_exits() == {"approach_0": 1}
        # Consuming step exits resets counter for next decision step
        assert extractor.get_step_exits() == {"approach_0": 0}

    def test_disappearance_before_line_counts_zero(self):
        extractor = _make_extractor()
        # Track exists for 4 frames inside ROI approaching the stop line, then disappears
        for idx, by in enumerate([120.0, 150.0, 175.0, 195.0]):
            _step_frame(extractor, [_track_with_bottom_center(2, 150.0, by)], idx * 0.1)
        # Disappears (occlusion / tracker drop)
        _step_frame(extractor, [], 0.4)
        _step_frame(extractor, [], 0.5)

        assert extractor.get_step_exits() == {"approach_0": 0}

    def test_jitter_back_and_forth_across_line_counts_one(self):
        extractor = _make_extractor()
        # Crosses forward, jitters back, crosses forward repeatedly
        ys = [180.0, 195.0, 205.0, 195.0, 208.0, 192.0, 212.0]
        for idx, by in enumerate(ys):
            _step_frame(extractor, [_track_with_bottom_center(3, 150.0, by)], idx * 0.1)

        assert extractor.get_step_exits() == {"approach_0": 1}

        # Even in the next decision step, the same track ID jittering across must not recount
        _step_frame(extractor, [_track_with_bottom_center(3, 194.0, 194.0)], 0.8)
        _step_frame(extractor, [_track_with_bottom_center(3, 150.0, 194.0)], 0.9)
        _step_frame(extractor, [_track_with_bottom_center(3, 150.0, 210.0)], 1.0)
        assert extractor.get_step_exits() == {"approach_0": 0}

    def test_two_frame_track_crossing_counts_zero(self):
        extractor = _make_extractor(min_track_frames=3)
        # Only 2 frames: before line -> past line
        _step_frame(extractor, [_track_with_bottom_center(4, 150.0, 190.0)], 0.0)
        _step_frame(extractor, [_track_with_bottom_center(4, 150.0, 215.0)], 0.1)
        assert extractor.get_step_exits() == {"approach_0": 0}

        # Staying on the exit side on a 3rd frame does not retroactively count
        _step_frame(extractor, [_track_with_bottom_center(4, 150.0, 235.0)], 0.2)
        assert extractor.get_step_exits() == {"approach_0": 0}

    def test_reverse_direction_crossing_counts_zero(self):
        extractor = _make_extractor()
        # Moving from exit side (+y) back into approach (-y)
        for idx, by in enumerate([230.0, 210.0, 185.0]):
            _step_frame(extractor, [_track_with_bottom_center(5, 150.0, by)], idx * 0.1)

        assert extractor.get_step_exits() == {"approach_0": 0}

    def test_crossing_outside_finite_segment_counts_zero(self):
        extractor = _make_extractor()
        # Stop line spans x in [100, 200] at y=200; track crosses y=200 at x=250
        for idx, by in enumerate([170.0, 190.0, 215.0]):
            _step_frame(extractor, [_track_with_bottom_center(6, 250.0, by)], idx * 0.1)

        assert extractor.get_step_exits() == {"approach_0": 0}

    def test_uses_bbox_bottom_center_not_geometric_center(self):
        extractor = _make_extractor()
        # Tall bbox (height=60): at by=205, geometric center y=175 is still < 200,
        # while bottom-centre y=205 has crossed stop line y=200.
        for idx, by in enumerate([175.0, 195.0, 205.0]):
            _step_frame(
                extractor,
                [_track_with_bottom_center(7, 150.0, by, height=60.0)],
                idx * 0.1,
            )

        assert extractor.get_step_exits() == {"approach_0": 1}

    def test_stepping_exactly_onto_line_then_exiting_counts_one(self):
        extractor = _make_extractor()
        for idx, by in enumerate([185.0, 200.0, 215.0]):
            _step_frame(extractor, [_track_with_bottom_center(8, 150.0, by)], idx * 0.1)

        assert extractor.get_step_exits() == {"approach_0": 1}

    def test_multiple_approaches_and_full_reset(self):
        approach_1 = LaneROI(
            lane_id="approach_1",
            tls_id="J0",
            polygon=[(300.0, 50.0), (400.0, 50.0), (400.0, 200.0), (300.0, 200.0)],
            cam_id="cam0",
            lane_length_px=150.0,
            lane_width_px=100.0,
            # Reversed segment ((400, 50) -> (300, 50)): exit direction is -y (upward)
            stop_line=((400.0, 50.0), (300.0, 50.0)),
        )
        approach_no_line = LaneROI(
            lane_id="approach_2",
            tls_id="J0",
            polygon=[(500.0, 50.0), (600.0, 50.0), (600.0, 200.0), (500.0, 200.0)],
            cam_id="cam0",
            lane_length_px=150.0,
            lane_width_px=100.0,
            stop_line=None,
        )
        extractor = _make_extractor(
            extra_rois={"approach_1": approach_1, "approach_2": approach_no_line}
        )

        # Track 10 exits approach_0 downward; tracks 11 & 12 exit approach_1 upward
        _step_frame(
            extractor,
            [
                _track_with_bottom_center(10, 150.0, 170.0),
                _track_with_bottom_center(11, 340.0, 80.0),
                _track_with_bottom_center(12, 370.0, 75.0),
            ],
            0.0,
        )
        _step_frame(
            extractor,
            [
                _track_with_bottom_center(10, 150.0, 190.0),
                _track_with_bottom_center(11, 340.0, 60.0),
                _track_with_bottom_center(12, 370.0, 55.0),
            ],
            0.1,
        )
        _step_frame(
            extractor,
            [
                _track_with_bottom_center(10, 150.0, 215.0),
                _track_with_bottom_center(11, 340.0, 40.0),
                _track_with_bottom_center(12, 370.0, 35.0),
            ],
            0.2,
        )

        assert extractor.get_step_exits() == {
            "approach_0": 1,
            "approach_1": 2,
            "approach_2": 0,
        }

        # After episode reset(), the same track IDs can be counted again
        extractor.reset()
        for idx, by in enumerate([170.0, 190.0, 215.0]):
            _step_frame(extractor, [_track_with_bottom_center(10, 150.0, by)], idx * 0.1)
        assert extractor.get_step_exits() == {
            "approach_0": 1,
            "approach_1": 0,
            "approach_2": 0,
        }
