"""
tests/test_revision_a4_a7.py
============================
Unit tests for revision items A4, A5, A6, and A7 (no live SUMO required).

Covers:
- A4: env-enforced max_green_time forced switch via yellow transition,
      realized green duration recording + episode-end flush (idempotent),
      SOTL default phi_min=3 (15 s) and Webster default min_green=15.0.
- A5: --reward-preset privileged keeps 26-dim proxy obs (no privileged leak),
      populates privileged_* metrics in lane_cache, and computes exact per-lane
      queue/pressure/waiting + global throughput reward differing from 'full'.
- A6: 'maxpressure' deprecated CLI alias maps to 'lqi'; canonical Varaiya
      Max-Pressure ('mp') subtracts downstream halting counts via getLinks
      and is classified as an instrumented reference.
- A7: eval_compare.write_outputs emits raw_samples.csv and green_durations.csv
      with the required columns for both learned checkpoints and baselines.
"""

from __future__ import annotations

import csv
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from experiment.baselines.max_pressure import CanonicalMaxPressurePolicy, MaxPressurePolicy
from experiment.baselines.sotl import DEFAULT_PHI_MIN, SOTLPolicy
from experiment.baselines.webster import WebsterPolicy
from experiment.plots import campaign_figures
from experiment.runners import eval_compare, make_tables
from experiment.runners.train_ppo import REWARD_PRESETS, _reward_preset_kwargs
from src.traffic_env.components.observations import ObservationBuilder
from src.traffic_env.components.rewards import RewardCalculator
from src.traffic_env.config import (
    DEFAULT_LANE_FEATURE_NAMES,
    PRIVILEGED_EXTRA_LANE_FEATURE_NAMES,
    ObservationConfig,
    build_default_config,
)
from src.traffic_env.envs.multi_agent import MappoTrafficEnv


# ---------------------------------------------------------------------------
# Lightweight fake SUMO connection (for unit tests without SUMO)
# ---------------------------------------------------------------------------

class _FakePhase:
    def __init__(self, state: str):
        self.state = state


class _FakeLogic:
    def __init__(self, states: List[str]):
        self._phases = [_FakePhase(s) for s in states]

    def getPhases(self):
        return self._phases


class _FakeTL:
    def __init__(self, lanes_by_tls: Dict[str, List[str]]):
        self._lanes_by_tls = lanes_by_tls
        self._phases = {t: 0 for t in lanes_by_tls}
        self.phase_history: Dict[str, List[int]] = {t: [] for t in lanes_by_tls}

    def getAllProgramLogics(self, tls_id: str):
        return [_FakeLogic(["GGrr", "yyrr", "rrGG", "rryy"])]

    def getPhase(self, tls_id: str) -> int:
        return self._phases[tls_id]

    def setPhase(self, tls_id: str, phase: int) -> None:
        self._phases[tls_id] = int(phase)
        self.phase_history[tls_id].append(int(phase))

    def getControlledLanes(self, tls_id: str) -> List[str]:
        return list(self._lanes_by_tls.get(tls_id, []))


class _FakeLane:
    def __init__(
        self,
        links: Dict[str, List[Tuple[str, ...]]] | None = None,
        halting: Dict[str, int] | None = None,
    ):
        self._links = links or {}
        self._halting = halting or {}

    def getLength(self, lane_id: str) -> float:
        return 100.0

    def getLastStepVehicleIDs(self, lane_id: str):
        return ()

    def getLastStepOccupancy(self, lane_id: str) -> float:
        return 0.0

    def getLastStepHaltingNumber(self, lane_id: str) -> int:
        return int(self._halting.get(lane_id, 0))

    def getWaitingTime(self, lane_id: str) -> float:
        return 0.0

    def getLinks(self, lane_id: str):
        return list(self._links.get(lane_id, []))


class _FakeVehicle:
    def getAllSubscriptionResults(self):
        return {}


class _FakeSimulation:
    def getArrivedNumber(self) -> int:
        return 0


class _FakeConn:
    def __init__(
        self,
        lanes_by_tls: Dict[str, List[str]],
        links: Dict[str, List[Tuple[str, ...]]] | None = None,
        halting: Dict[str, int] | None = None,
    ):
        self.trafficlight = _FakeTL(lanes_by_tls)
        self.lane = _FakeLane(links=links, halting=halting)
        self.vehicle = _FakeVehicle()
        self.simulation = _FakeSimulation()

    def simulationStep(self) -> None:
        pass

    def close(self) -> None:
        pass


class _MockMappoEnv(MappoTrafficEnv):
    """MappoTrafficEnv wired to an in-memory fake SUMO connection."""

    def __init__(self, config, fake_conn: _FakeConn):
        # Bypass BaseSumoEnv.__init__ (which imports traci/libsumo) cleanly.
        self.sumo_cfg_path = config.sim.sumo_cfg_path
        self.gui = False
        self.use_libsumo = False
        self.sumo_conn = fake_conn
        self.sim_started = True
        self._arrived_cumulative = 0.0
        self._is_libsumo = False
        self.extra_sumo_args = []

        self.config = config
        self.tls_ids = list(config.tls_ids)
        self.obs_builder = ObservationBuilder(config=config, sumo_conn=fake_conn)
        self.reward_calculator = RewardCalculator(config=config)

        self.green_timers = {t: 0.0 for t in self.tls_ids}
        self.last_actions = {t: 0 for t in self.tls_ids}
        self.green_durations = []
        self._phase_elapsed_s = {t: 0.0 for t in self.tls_ids}
        self._current_green_phase = {t: 0 for t in self.tls_ids}
        self._green_durations_flushed = False
        self._green_phase_map = {}
        self._yellow_phase_map = {}
        self.action_dim = 2
        self._step_count = 0

    def start_simulation(self, seed=None, extra_args=None):
        self._arrived_cumulative = 0.0
        self.sim_started = True
        return self.sumo_conn


# ---------------------------------------------------------------------------
# A4: Identical timing constraints & realized green durations
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestA4TimingConstraints:
    def test_max_green_forces_switch_through_yellow_and_records_durations(self):
        group_a = ["E0_0", "E0_1", "E1_0", "E1_1"]
        group_b = ["E2_0", "E2_1", "E3_0", "E3_1"]
        cfg = build_default_config(
            sumo_cfg_path="synthetic.sumocfg",
            tls_ids=("J0",),
            manual_lane_groups={"J0": (group_a, group_b)},
            step_length=5,
            yellow_time=3,
            min_green_time=15,
            max_green_time=60,
            max_steps=35,
        )
        conn = _FakeConn({"J0": group_a + group_b})
        env = _MockMappoEnv(cfg, conn)
        env.reset(seed=42)

        # Constantly request action 0 (keep phase 0) for the full episode.
        for _ in range(cfg.sim.max_steps):
            _, _, _, truncated, _ = env.step({"J0": 0})
            if truncated["J0"]:
                break

        # Must have visited yellow phase 1 and opposing green phase 2.
        assert 1 in conn.trafficlight.phase_history["J0"]
        assert 2 in conn.trafficlight.phase_history["J0"]

        # Realized green durations recorded and flushed on episode end.
        recorded_after_done = list(env.green_durations)
        assert len(recorded_after_done) >= 3
        # First phase 0 held 12 steps = 60.0 s before forced switch.
        assert recorded_after_done[0] == ("J0", 0, pytest.approx(60.0))
        # Every realized green duration must stay <= max_green_time (never 62 s).
        max_dur = max(d for _, _, d in recorded_after_done)
        assert max_dur <= cfg.sim.max_green_time

        # close() must not double-flush running durations.
        env.close()
        assert env.green_durations == recorded_after_done

    def test_sotl_and_webster_defaults_match_min_green_15s(self):
        assert DEFAULT_PHI_MIN == 3
        sotl = SOTLPolicy(lane_groups={"J0": (["E0_0"], ["E1_0"])})
        assert sotl.phi_min == 3

        webster = WebsterPolicy(
            lane_groups={"J0": (["E0_0"], ["E1_0"])},
            lane_flows_pcu_h={"E0_0": 200.0, "E1_0": 200.0},
        )
        assert webster.min_green == pytest.approx(15.0)
        assert webster.timings["J0"].green_a >= 15.0
        assert webster.timings["J0"].green_b >= 15.0


# ---------------------------------------------------------------------------
# A5: Privileged-REWARD training arm (26-dim proxy obs, exact simulator reward)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestA5PrivilegedRewardPreset:
    def test_preset_keeps_26_dim_proxy_obs_and_differs_from_full_reward(self):
        assert "privileged" in REWARD_PRESETS

        group_a = ["E0_0", "E0_1", "E1_0", "E1_1"]
        group_b = ["E2_0", "E2_1", "E3_0", "E3_1"]
        all_lanes = group_a + group_b
        groups = {"J0": (list(group_a), list(group_b))}

        cfg_priv = build_default_config(
            sumo_cfg_path="synthetic.sumocfg",
            tls_ids=("J0",),
            manual_lane_groups=groups,
            obs_mode="proxy",
            **_reward_preset_kwargs("privileged"),
        )
        cfg_full = build_default_config(
            sumo_cfg_path="synthetic.sumocfg",
            tls_ids=("J0",),
            manual_lane_groups=groups,
            obs_mode="proxy",
            **_reward_preset_kwargs("full"),
        )

        # Obs stays 26-dim camera-proxy; config guard against privileged obs features intact.
        assert cfg_priv.local_obs_dim == 26
        assert cfg_priv.observation.obs_mode == "proxy"
        assert cfg_priv.reward.privileged_reward is True
        with pytest.raises(ValueError):
            ObservationConfig(
                lane_feature_names=DEFAULT_LANE_FEATURE_NAMES + PRIVILEGED_EXTRA_LANE_FEATURE_NAMES,
                obs_mode="proxy",
            ).validate()

        # ObservationBuilder collects privileged metrics for reward without leaking into obs.
        conn = _FakeConn({"J0": all_lanes})
        builder = ObservationBuilder(config=cfg_priv, sumo_conn=conn)
        assert builder._privileged is False
        assert builder._collect_privileged_metrics is True

        # Fixture where exact and proxy queues, waiting times, and throughput paths differ.
        def _mk(q_proxy: float, h_exact: float, w_exact: float, vids=()) -> Dict[str, Any]:
            return {
                "effective_queue_norm": q_proxy,
                "occupancy_norm": q_proxy,
                "avg_speed_norm": 0.5,
                "motorbike_share": 0.7,
                "heavy_vehicle_share": 0.1,
                "raw_vehicle_count": 3.0,
                "vehicle_ids": tuple(vids),
                "privileged_halt_count_norm": h_exact,
                "privileged_waiting_norm": w_exact,
                "privileged_vehicle_count_norm": h_exact,
            }

        cache_with_priv = {lid: _mk(0.10, 0.00, 0.00) for lid in all_lanes}
        # Exact queues & waiting heavily concentrated on red lane E2_0, while proxy is small.
        cache_with_priv["E2_0"] = _mk(0.15, 0.90, 0.80, vids=("v1",))
        cache_with_priv["E0_0"] = _mk(0.60, 0.10, 0.05, vids=("v2",))
        cache_proxy_only = {
            lid: {k: v for k, v in m.items() if not k.startswith("privileged_")}
            for lid, m in cache_with_priv.items()
        }

        obs_a = builder.build_local_obs("J0", cache_with_priv, green_timers={"J0": 10.0})
        obs_b = builder.build_local_obs("J0", cache_proxy_only, green_timers={"J0": 10.0})
        assert obs_a.shape == (26,)
        assert np.array_equal(obs_a, obs_b)

        action_info = {
            "lane_ids_by_tls": {"J0": all_lanes},
            "green_lanes_by_tls": {"J0": group_a},
            "delta_passed": 4.0,
            "switches_by_tls": {"J0": False},
        }
        calc_full = RewardCalculator(config=cfg_full)
        calc_priv = RewardCalculator(config=cfg_priv)

        r_full = calc_full.calculate_rewards(("J0",), cache_with_priv, action_info)["J0"]
        r_priv = calc_priv.calculate_rewards(("J0",), cache_with_priv, action_info)["J0"]
        assert r_priv != pytest.approx(r_full)

        # Verify exact privileged components: per-lane queue/pressure/waiting + global throughput.
        det = calc_priv.last_reward_details["J0"]
        assert det["queue_penalty"] == pytest.approx((0.90 ** 2 + 0.10 ** 2) / 8.0)
        assert det["pressure_penalty"] == pytest.approx((0.90 - 0.10) / 8.0)
        assert det["queue_occupancy_penalty"] == pytest.approx((0.80 + 0.05) / 8.0)
        assert det["throughput_reward"] == pytest.approx(4.0 / cfg_priv.reward.throughput_norm_divisor)


# ---------------------------------------------------------------------------
# A6: LQI rename, deprecated alias, and canonical Varaiya Max-Pressure
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestA6MaxPressureAndNaming:
    def test_canonical_mp_accounts_for_downstream_halting(self):
        groups = {"J0": (["A_in_0"], ["B_in_0"])}
        links = {
            "A_in_0": [("A_out_0", True, True, False, ":J0_0", "G", "s", 10.0)],
            "B_in_0": [("B_out_0", True, True, False, ":J0_1", "r", "s", 10.0)],
        }
        # Incoming queue is larger on A (10 vs 6), but A's downstream lane is jammed (9 vs 0):
        #   pressure(A) = 10 - 9 = 1
        #   pressure(B) =  6 - 0 = 6  -> canonical MP must select group B (action 1),
        # while LQI (incoming only) selects group A (action 0).
        halting = {"A_in_0": 10, "A_out_0": 9, "B_in_0": 6, "B_out_0": 0}
        conn = _FakeConn({"J0": ["A_in_0", "B_in_0"]}, links=links, halting=halting)

        mp = CanonicalMaxPressurePolicy(lane_groups=groups)
        assert mp.select_actions(conn) == {"J0": 1}

        lqi = MaxPressurePolicy(lane_groups=groups)
        lane_cache = {
            "A_in_0": {"effective_queue_norm": 0.8},
            "B_in_0": {"effective_queue_norm": 0.4},
        }
        assert lqi.select_actions(lane_cache) == {"J0": 0}

    def test_deprecated_alias_and_instrumented_classification(self):
        assert eval_compare._resolve_methods(["maxpressure", "mp", "sotl"]) == ["lqi", "mp", "sotl"]
        assert eval_compare.SENSING_CLASS["mp"] == "instrumented"
        assert eval_compare.SENSING_CLASS["actuated"] == "instrumented"
        assert eval_compare.SENSING_CLASS["lqi"] == "camera"
        assert "mp" in make_tables.INSTRUMENTED_METHODS
        assert "mp" in campaign_figures.INSTRUMENTED_METHODS

        dummy_metric = {"mean": 100.0, "ci95": 5.0, "n": 5, "values": [100.0]}
        summary = {
            "reference": "lqi",
            "aggregates": {
                "lqi": {k: dummy_metric for k in make_tables.METRIC_KEYS},
                "mp": {k: dummy_metric for k in make_tables.METRIC_KEYS},
            },
            "stats": {},
        }
        tex = make_tables.main_table_tex(summary, "n3_grid")
        assert "Camera-deployable" in tex
        assert "Instrumented references" in tex
        assert "LQI" in tex
        assert "Max-pressure" in tex


# ---------------------------------------------------------------------------
# A7: Raw samples + green durations CSV output
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestA7RawSamplesAndGreenCSV:
    def test_write_outputs_emits_raw_samples_and_green_durations(self, tmp_path):
        args = SimpleNamespace(
            split="eval",
            seeds=[42, 123],
            max_steps=100,
            step_length=5,
            learned_step_length=None,
            min_green_time=15,
            max_green_time=60,
        )
        m1 = {
            "mean_travel_time_s": 200.0, "mean_waiting_time_s": 50.0,
            "mean_time_loss_s": 80.0, "p95_waiting_time_s": 120.0,
            "n_trips": 500, "total_co2_mg": float("nan"), "total_fuel_mg": float("nan"),
        }
        agg = {"mappo": eval_compare.aggregate([m1]), "lqi": eval_compare.aggregate([m1])}
        report = eval_compare.build_report(agg, "mappo")

        raw_rows: List[Dict[str, Any]] = []
        green_rows: List[Dict[str, Any]] = []
        eval_compare._append_raw_sample(raw_rows, "mappo", 0, 42, 42, m1)
        eval_compare._append_raw_sample(raw_rows, "lqi", -1, None, 42, m1)
        eval_compare._append_green_durations(green_rows, "lqi", -1, 42, [("A0", 0, 35.0)])

        eval_compare.write_outputs(
            tmp_path, "n3_grid", agg, report, args,
            raw_rows=raw_rows, green_rows=green_rows,
        )

        raw_path = tmp_path / "raw_samples.csv"
        green_path = tmp_path / "green_durations.csv"
        assert raw_path.exists()
        assert green_path.exists()

        with open(raw_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["method"] == "mappo"
        assert rows[0]["ckpt_idx"] == "0"
        assert rows[0]["train_seed"] == "42"
        assert rows[0]["route_seed"] == "42"
        assert rows[1]["method"] == "lqi"
        assert rows[1]["ckpt_idx"] == "-1"
        assert rows[1]["train_seed"] == ""

        with open(green_path, encoding="utf-8") as f:
            grows = list(csv.DictReader(f))
        assert len(grows) == 1
        assert grows[0] == {
            "method": "lqi",
            "ckpt_idx": "-1",
            "route_seed": "42",
            "tls_id": "A0",
            "phase": "0",
            "duration_s": "35.00",
            "completed": "True",
        }
