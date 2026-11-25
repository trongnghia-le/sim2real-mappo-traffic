"""
tests/test_reward_approach_level.py
===================================
Unit tests for schema 1.3.0 approach-level reward calculation (no SUMO).

Covers:
1. A vehicle changing lanes inside the same approach edge yields 0 exits.
2. A vehicle leaving the approach edge yields 1 exit (plus global fallback).
3. Queue penalty with 1 saturated approach + 3 empty equals 0.25 (mean of
   squares) and 0.0625 with queue_mean_of_squares=False, and aggregates lanes
   per approach before squaring.
4. Pressure sign (red > green positive/penalised, green > red negative/rewarded
   when pressure_signed=True and 0.0 when pressure_signed=False).
5. Vehicle-count-weighted approach speed, linear queue_occupancy, backward-
   compatible legacy "waiting_time" weight key, and schema version 1.3.0.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import pytest

from src.traffic_env.components.rewards import RewardCalculator
from src.traffic_env.config import DEFAULT_REWARD_WEIGHTS, build_default_config

TLS = ("J0",)
GROUP_A = ["E0_0", "E0_1", "E1_0", "E1_1"]
GROUP_B = ["E2_0", "E2_1", "E3_0", "E3_1"]
ALL_LANES = GROUP_A + GROUP_B


def _make_cfg(**kwargs: Any):
    return build_default_config(
        sumo_cfg_path="synthetic.sumocfg",
        tls_ids=TLS,
        manual_lane_groups={"J0": (list(GROUP_A), list(GROUP_B))},
        **kwargs,
    )


def _lane(
    q: float = 0.0,
    speed: float = 1.0,
    count: float = 1.0,
    vehicle_ids: Iterable[str] | None = None,
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "effective_queue_norm": q,
        "avg_speed_norm": speed,
        "raw_vehicle_count": count,
    }
    if vehicle_ids is not None:
        d["vehicle_ids"] = tuple(vehicle_ids)
    return d


@pytest.mark.unit
class TestApproachLevelThroughput:
    def test_intra_approach_lane_change_yields_zero_exits(self):
        cfg = _make_cfg()
        calc = RewardCalculator(config=cfg)
        info = {"lane_ids_by_tls": {"J0": ALL_LANES}}

        # Step 1: vehicle v1 is on E0_0; E0_1 is empty.
        cache_t0 = {lid: _lane(vehicle_ids=()) for lid in ALL_LANES}
        cache_t0["E0_0"] = _lane(vehicle_ids=("v1",))
        calc.calculate_rewards(TLS, cache_t0, info)

        # Step 2: v1 changes lane from E0_0 to E0_1 within approach E0.
        cache_t1 = {lid: _lane(vehicle_ids=()) for lid in ALL_LANES}
        cache_t1["E0_1"] = _lane(vehicle_ids=("v1",))
        calc.calculate_rewards(TLS, cache_t1, info)

        assert calc.last_reward_details["J0"]["throughput_reward"] == pytest.approx(0.0)

    def test_vehicle_leaving_approach_yields_one_exit(self):
        cfg = _make_cfg()
        calc = RewardCalculator(config=cfg)
        info = {"lane_ids_by_tls": {"J0": ALL_LANES}}

        # Step 1: v1 on E0_0.
        cache_t0 = {lid: _lane(vehicle_ids=()) for lid in ALL_LANES}
        cache_t0["E0_0"] = _lane(vehicle_ids=("v1",))
        calc.calculate_rewards(TLS, cache_t0, info)

        # Step 2: v1 exits approach E0 completely.
        cache_t1 = {lid: _lane(vehicle_ids=()) for lid in ALL_LANES}
        calc.calculate_rewards(TLS, cache_t1, info)

        expected = 1.0 / cfg.reward.throughput_norm_divisor
        assert calc.last_reward_details["J0"]["throughput_reward"] == pytest.approx(expected)

    def test_global_arrived_fallback_untouched_when_no_vehicle_ids(self):
        cfg = _make_cfg()
        calc = RewardCalculator(config=cfg)
        cache = {lid: _lane() for lid in ALL_LANES}  # no "vehicle_ids" key
        info = {"lane_ids_by_tls": {"J0": ALL_LANES}, "delta_passed": 10.0}

        calc.calculate_rewards(TLS, cache, info)

        expected = 10.0 / cfg.reward.throughput_norm_divisor
        assert calc.last_reward_details["J0"]["throughput_reward"] == pytest.approx(expected)


@pytest.mark.unit
class TestApproachLevelQueuePenalty:
    def test_one_saturated_approach_mean_of_squares_and_flag_off(self):
        # Approach E0 saturated (both lanes q=1.0), approaches E1..E3 empty.
        cache = {lid: _lane(q=0.0) for lid in ALL_LANES}
        cache["E0_0"] = _lane(q=1.0)
        cache["E0_1"] = _lane(q=1.0)
        info = {"lane_ids_by_tls": {"J0": ALL_LANES}}

        calc_sq = RewardCalculator(config=_make_cfg(queue_mean_of_squares=True))
        calc_sq.calculate_rewards(TLS, cache, info)
        assert calc_sq.last_reward_details["J0"]["queue_penalty"] == pytest.approx(0.25)

        calc_ms = RewardCalculator(config=_make_cfg(queue_mean_of_squares=False))
        calc_ms.calculate_rewards(TLS, cache, info)
        assert calc_ms.last_reward_details["J0"]["queue_penalty"] == pytest.approx(0.0625)

    def test_queue_aggregates_approach_before_squaring(self):
        # Within approach E0: one lane saturated (1.0), sibling lane empty (0.0)
        # -> approach q_E0 = 0.5 -> mean_a(q_a^2) = 0.25 / 4 = 0.0625
        # (per-lane mean(q_l^2) would wrongly yield 1.0 / 8 = 0.125).
        cache = {lid: _lane(q=0.0) for lid in ALL_LANES}
        cache["E0_0"] = _lane(q=1.0)
        info = {"lane_ids_by_tls": {"J0": ALL_LANES}}

        calc = RewardCalculator(config=_make_cfg(queue_mean_of_squares=True))
        calc.calculate_rewards(TLS, cache, info)
        assert calc.last_reward_details["J0"]["queue_penalty"] == pytest.approx(0.0625)


@pytest.mark.unit
class TestApproachLevelPressureAndOtherTerms:
    def test_pressure_sign_and_unsigned_flag(self):
        # Group A (E0, E1) green; Group B (E2, E3) red.
        info = {
            "lane_ids_by_tls": {"J0": ALL_LANES},
            "green_lanes_by_tls": {"J0": GROUP_A},
        }

        # Red approach E2 congested -> pressure > 0 (penalised by negative weight).
        cache_red_heavy = {lid: _lane(q=0.0) for lid in ALL_LANES}
        cache_red_heavy["E2_0"] = _lane(q=1.0)
        cache_red_heavy["E2_1"] = _lane(q=1.0)

        calc_signed = RewardCalculator(config=_make_cfg(pressure_signed=True))
        calc_signed.calculate_rewards(TLS, cache_red_heavy, info)
        assert calc_signed.last_reward_details["J0"]["pressure_penalty"] == pytest.approx(0.25)

        # Green approach E0 congested -> pressure < 0 when signed, 0.0 when unsigned.
        cache_green_heavy = {lid: _lane(q=0.0) for lid in ALL_LANES}
        cache_green_heavy["E0_0"] = _lane(q=1.0)
        cache_green_heavy["E0_1"] = _lane(q=1.0)

        calc_signed.calculate_rewards(TLS, cache_green_heavy, info)
        assert calc_signed.last_reward_details["J0"]["pressure_penalty"] == pytest.approx(-0.25)

        calc_unsigned = RewardCalculator(config=_make_cfg(pressure_signed=False))
        calc_unsigned.calculate_rewards(TLS, cache_green_heavy, info)
        assert calc_unsigned.last_reward_details["J0"]["pressure_penalty"] == pytest.approx(0.0)

    def test_low_speed_uses_count_weighted_approach_mean(self):
        # Single approach E0: lane 0 has 4 vehicles at speed 0.4, lane 1 empty at 0.0.
        # Count-weighted approach speed = 0.4 (>= threshold 0.20) -> penalty 0.0.
        # (Unweighted per-lane mean would be 0.2, or with speed 0.3 -> 0.15 < 0.20.)
        cache = {
            "E0_0": _lane(q=0.0, speed=0.3, count=4.0),
            "E0_1": _lane(q=0.0, speed=0.0, count=0.0),
        }
        info = {"lane_ids_by_tls": {"J0": ["E0_0", "E0_1"]}}

        calc = RewardCalculator(config=_make_cfg())
        calc.calculate_rewards(TLS, cache, info)
        assert calc.last_reward_details["J0"]["low_speed_penalty"] == pytest.approx(0.0)

    def test_queue_occupancy_and_legacy_waiting_weight_compat(self):
        cache = {lid: _lane(q=0.0) for lid in ALL_LANES}
        cache["E0_0"] = _lane(q=1.0)
        cache["E0_1"] = _lane(q=1.0)
        info = {"lane_ids_by_tls": {"J0": ALL_LANES}}

        cfg_new = _make_cfg()
        assert cfg_new.version == "1.3.0"
        assert "queue_occupancy" in DEFAULT_REWARD_WEIGHTS

        legacy_weights = {
            k: v for k, v in DEFAULT_REWARD_WEIGHTS.items() if k != "queue_occupancy"
        }
        legacy_weights["waiting_time"] = DEFAULT_REWARD_WEIGHTS["queue_occupancy"]
        cfg_legacy = _make_cfg(reward_weights=legacy_weights)

        calc_new = RewardCalculator(config=cfg_new)
        calc_legacy = RewardCalculator(config=cfg_legacy)
        r_new = calc_new.calculate_rewards(TLS, cache, info)["J0"]
        r_legacy = calc_legacy.calculate_rewards(TLS, cache, info)["J0"]

        assert calc_new.last_reward_details["J0"]["queue_occupancy_penalty"] == pytest.approx(0.25)
        assert r_legacy == pytest.approx(r_new)
