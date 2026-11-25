"""
tests/test_eval_fairness.py
===========================
Verification tests for evaluation-fairness fixes E1-E8:
- E1: Shared tripinfo SUMO args across all evaluation paths + explicit separation
      of completed vs unfinished/vaporized trips under SUMO 1.24 semantics.
- E2: Canonical checkpoint resolution (exactly one best_model.pt per job dir via
      bench.json; never returns directories; raises on 0 or >1 completed runs).
- E3: Per-method buffering of raw_rows / green_rows (no partial rows on failure).
- E4: Actuated loop order matches SUMO phase time exactly (no +1 s overcount),
      writes tls_actuated.add.xml to tmp dir, and propagates min/max green args.
- E5: green_durations carries a boolean `completed` flag (False for right-censored
      episode-end phases) in multi_agent, actuated, and green_durations.csv.
- E6: MappoTrafficEnv.reset keeps sim time and green_timers consistent (t=0.0,
      green_timers=0.0; initial phase runs 60 s sim time and logs 60 s).
- E7: WebsterPolicy clamps green splits g_a, g_b to [min_green, max_green] and
      eval_compare forwards args.max_green_time.
- E8: n4_irregular registered in --network CLI choices across baselines/runners.
"""
from __future__ import annotations

import csv
import inspect
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

from experiment.baselines import actuated as actuated_mod
from experiment.baselines import webster as webster_mod
from experiment.baselines.webster import WebsterPolicy
from experiment.common import tripinfo as tripinfo_mod
from experiment.common.tripinfo import (
    TRIPINFO_EXTRA_ARGS,
    parse_tripinfo_means,
    tripinfo_sumo_args,
)
from experiment.runners import eval_compare, eval_noise_robustness
from scripts import run_pilot
from src.traffic_env.config import build_default_config, load_lane_groups_json
from src.traffic_env.envs.multi_agent import GreenDurationRecord, MappoTrafficEnv
from tests.test_revision_a4_a7 import _FakeConn, _MockMappoEnv

ROOT = Path(__file__).resolve().parents[1]
N2_DIR = ROOT / "sumo_configs" / "networks" / "n2_corridor"


# ---------------------------------------------------------------------------
# E1: Shared tripinfo args & completed vs unfinished separation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestE1TripinfoFairness:
    def test_shared_tripinfo_args_used_across_eval_paths(self, tmp_path, monkeypatch):
        assert "--tripinfo-output.write-unfinished" in TRIPINFO_EXTRA_ARGS
        tri = tmp_path / "sample.xml"
        expected = ["--tripinfo-output", str(tri), "--tripinfo-output.write-unfinished"]
        assert tripinfo_sumo_args(tri) == expected

        # Verify eval_compare.run_env_episode passes the exact shared args to MappoTrafficEnv.
        captured_env_args: List[List[str]] = []

        class _StubEnv:
            def __init__(self, *, config, gui, use_libsumo, extra_sumo_args):
                captured_env_args.append(list(extra_sumo_args))
                self.tls_ids = ["J0"]
                self.green_durations = []

            def reset(self, seed=None):
                return {"J0": np.zeros(26, dtype=np.float32)}, {}

            def step(self, action):
                return (
                    {"J0": np.zeros(26, dtype=np.float32)},
                    {"J0": 0.0},
                    {"J0": False},
                    {"J0": True},
                    {},
                )

            def close(self):
                pass

        tri.write_text("<tripinfos/>", encoding="utf-8")
        monkeypatch.setattr(eval_compare, "MappoTrafficEnv", _StubEnv)
        env_cfg = SimpleNamespace(sim=SimpleNamespace(step_length=5))
        eval_compare.run_env_episode(
            env_cfg, lambda env, obs, step, dt: {"J0": 0}, seed=42, max_steps=1, tripinfo_path=tri
        )
        assert captured_env_args == [expected]

        # Verify actuated.run_actuated builds its SUMO command with the exact same shared args.
        captured_act_cmd: List[List[str]] = []

        class _StubConn:
            class trafficlight:
                @staticmethod
                def setProgram(tid, prog):
                    pass

                @staticmethod
                def getAllProgramLogics(tid):
                    return []

                @staticmethod
                def getPhase(tid):
                    return 0

            class simulation:
                @staticmethod
                def getDeltaT():
                    return 1.0

            def simulationStep(self):
                pass

            def close(self):
                pass

        class _StubTraci:
            @staticmethod
            def start(cmd, label=None):
                captured_act_cmd.append(list(cmd))

            @staticmethod
            def getConnection(label):
                return _StubConn()

        monkeypatch.setitem(sys.modules, "traci", _StubTraci)
        actuated_mod.run_actuated(
            sumo_cfg=tmp_path / "c.sumocfg",
            add_path=tmp_path / "a.add.xml",
            vtypes_path=tmp_path / "v.add.xml",
            tls_ids=["J0"],
            seed=42,
            duration_s=1,
            tripinfo_path=tri,
        )
        assert len(captured_act_cmd) == 1
        cmd = captured_act_cmd[0]
        idx = cmd.index("--tripinfo-output")
        assert cmd[idx : idx + len(expected)] == expected

        # Verify other eval paths also reference tripinfo_sumo_args (no hand-rolled mismatch).
        for fn in (
            webster_mod.evaluate_policy,
            eval_noise_robustness.main,
            run_pilot.evaluate_checkpoint,
        ):
            assert "tripinfo_sumo_args" in inspect.getsource(fn)

    def test_parser_separates_completed_and_unfinished_sumo124(self, tmp_path):
        # Synthetic XML mirroring SUMO 1.24 --tripinfo-output.write-unfinished:
        #   * 2 completed trips: arrival >= 0 and vaporized=""
        #   * 1 unfinished trip with arrival="-1.00" and vaporized="end"
        #   * 1 unfinished trip with arrival="-1.00" and vaporized="" (observed in SUMO 1.24)
        #   * 1 vaporized trip with vaporized="teleport"
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<tripinfos>
    <tripinfo id="c1" depart="0.00" arrival="100.00" duration="100.00" waitingTime="10.00" timeLoss="20.00" vaporized="">
        <emissions CO2_abs="1000.0" fuel_abs="300.0"/>
    </tripinfo>
    <tripinfo id="c2" depart="5.00" arrival="205.00" duration="200.00" waitingTime="30.00" timeLoss="40.00" vaporized="">
        <emissions CO2_abs="2000.0" fuel_abs="700.0"/>
    </tripinfo>
    <tripinfo id="u_end" depart="10.00" arrival="-1.00" duration="500.00" waitingTime="400.00" timeLoss="450.00" vaporized="end">
        <emissions CO2_abs="9999.0" fuel_abs="9999.0"/>
    </tripinfo>
    <tripinfo id="u_empty_vap" depart="20.00" arrival="-1.00" duration="300.00" waitingTime="200.00" timeLoss="240.00" vaporized="">
        <emissions CO2_abs="8888.0" fuel_abs="8888.0"/>
    </tripinfo>
    <tripinfo id="u_tele" depart="30.00" arrival="180.00" duration="150.00" waitingTime="120.00" timeLoss="150.00" vaporized="teleport"/>
</tripinfos>
"""
        tri = tmp_path / "synthetic_tripinfo.xml"
        tri.write_text(xml, encoding="utf-8")

        m = parse_tripinfo_means(tri)
        # Primary completed-only metrics (only c1 and c2)
        assert m["n_trips"] == 2
        assert m["mean_travel_time_s"] == pytest.approx(150.0)
        assert m["mean_waiting_time_s"] == pytest.approx(20.0)
        assert m["mean_time_loss_s"] == pytest.approx(30.0)
        assert m["p95_waiting_time_s"] == pytest.approx(np.percentile([10.0, 30.0], 95))
        assert m["total_co2_mg"] == pytest.approx(3000.0)
        assert m["total_fuel_mg"] == pytest.approx(1000.0)

        # Unfinished-inclusive columns account for stuck / vaporized vehicles
        assert m["n_unfinished"] == 3
        assert m["mean_time_loss_incl_unfinished"] == pytest.approx(
            np.mean([20.0, 40.0, 450.0, 240.0, 150.0])
        )
        assert m["mean_travel_time_incl_unfinished"] == pytest.approx(
            np.mean([100.0, 200.0, 500.0, 300.0, 150.0])
        )

    def test_parser_handles_all_unfinished_xml(self, tmp_path):
        xml = """<tripinfos>
    <tripinfo id="u1" depart="0.00" arrival="-1.00" duration="60.00" waitingTime="45.00" timeLoss="55.00" vaporized="end"/>
</tripinfos>"""
        tri = tmp_path / "all_unfin.xml"
        tri.write_text(xml, encoding="utf-8")
        m = parse_tripinfo_means(tri)
        assert m["n_trips"] == 0
        assert np.isnan(m["mean_travel_time_s"])
        assert np.isnan(m["mean_time_loss_s"])
        assert m["n_unfinished"] == 1
        assert m["mean_time_loss_incl_unfinished"] == pytest.approx(55.0)
        assert m["mean_travel_time_incl_unfinished"] == pytest.approx(60.0)

    def test_new_unfinished_keys_exported_in_summary_and_raw_samples(self, tmp_path):
        args = SimpleNamespace(
            split="eval", seeds=[42], max_steps=10, step_length=5,
            learned_step_length=None, min_green_time=15, max_green_time=60,
        )
        sample = {
            "mean_travel_time_s": 100.0, "mean_waiting_time_s": 20.0,
            "mean_time_loss_s": 30.0, "p95_waiting_time_s": 50.0,
            "n_trips": 10, "total_co2_mg": float("nan"), "total_fuel_mg": float("nan"),
            "n_unfinished": 4, "mean_time_loss_incl_unfinished": 75.5,
            "mean_travel_time_incl_unfinished": 140.0,
        }
        agg = {"fixed": eval_compare.aggregate([sample])}
        report = eval_compare.build_report(agg, "fixed")
        raw_rows: List[Dict[str, Any]] = []
        eval_compare._append_raw_sample(raw_rows, "fixed", -1, None, 42, sample)
        eval_compare.write_outputs(tmp_path, "n2_corridor", agg, report, args, raw_rows=raw_rows)

        summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        assert summary["aggregates"]["fixed"]["n_unfinished"]["mean"] == pytest.approx(4.0)
        assert summary["aggregates"]["fixed"]["mean_time_loss_incl_unfinished"]["mean"] == pytest.approx(75.5)

        with open(tmp_path / "raw_samples.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["n_unfinished"] == "4"
        assert float(rows[0]["mean_time_loss_incl_unfinished"]) == pytest.approx(75.5)


# ---------------------------------------------------------------------------
# E2: Canonical checkpoint resolution
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestE2CheckpointResolution:
    def _mk_run(self, run_dir: Path, *, with_best: bool = True, with_bench: bool = False) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = run_dir / "best_model.pt"
        if with_best:
            ckpt.write_bytes(b"pt")
        if with_bench:
            (run_dir / "bench.json").write_text("{}", encoding="utf-8")
        return ckpt

    def test_one_checkpoint_per_job_and_glob_never_returns_dirs(self, tmp_path):
        models = tmp_path / "models"
        # Job 1: preempted incomplete run + completed resumed run -> must select ONLY completed
        j1 = models / "n3_grid_mappo_proxy_seed42"
        self._mk_run(j1 / "20261124_010000", with_best=True, with_bench=False)
        j1_good = self._mk_run(j1 / "20261124_020000", with_best=True, with_bench=True)

        # Job 2: single completed run
        j2 = models / "n3_grid_mappo_proxy_seed123"
        j2_good = self._mk_run(j2 / "20261124_030000", with_best=True, with_bench=True)

        # 1. Single job dir -> 1 canonical best_model.pt
        assert eval_compare.resolve_checkpoints(str(j1)) == [j1_good]

        # 2. Multi-job models dir -> 1 per job, never double-counted
        res_dir = eval_compare.resolve_checkpoints(str(models))
        assert res_dir == [j2_good, j1_good] or set(res_dir) == {j1_good, j2_good}
        assert len(res_dir) == 2
        assert all(p.is_file() for p in res_dir)

        # 3. Glob matching job directories -> resolves to files, NEVER directories
        glob_dirs = str(models / "n3_grid_mappo_proxy_seed*")
        res_glob_dirs = eval_compare.resolve_checkpoints(glob_dirs)
        assert len(res_glob_dirs) == 2
        assert all(p.is_file() for p in res_glob_dirs)
        assert set(res_glob_dirs) == {j1_good, j2_good}

        # 4. Glob matching **/best_model.pt -> deduplicated to canonical bench.json run per job
        glob_pts = str(models / "n3_grid_mappo_proxy_seed*" / "**" / "best_model.pt")
        res_glob_pts = eval_compare.resolve_checkpoints(glob_pts)
        assert len(res_glob_pts) == 2
        assert set(res_glob_pts) == {j1_good, j2_good}

    def test_two_completed_run_dirs_raises_error(self, tmp_path):
        job = tmp_path / "models" / "job_two_complete"
        self._mk_run(job / "run_1", with_best=True, with_bench=True)
        self._mk_run(job / "run_2", with_best=True, with_bench=True)

        with pytest.raises(ValueError, match="multiple completed run"):
            eval_compare.resolve_checkpoints(str(job))

        # Also fails when resolving the parent models dir or globbing
        with pytest.raises(ValueError, match="multiple completed run"):
            eval_compare.resolve_checkpoints(str(tmp_path / "models"))
        with pytest.raises(ValueError, match="multiple completed run"):
            eval_compare.resolve_checkpoints(str(tmp_path / "models" / "job_*"))

    def test_zero_completed_run_dirs_raises_error(self, tmp_path):
        job_incomp = tmp_path / "models" / "job_incomplete"
        self._mk_run(job_incomp / "run_1", with_best=True, with_bench=False)

        with pytest.raises(ValueError, match="no completed run"):
            eval_compare.resolve_checkpoints(str(job_incomp))

        job_empty = tmp_path / "models" / "job_empty"
        job_empty.mkdir(parents=True)
        with pytest.raises(ValueError, match="no completed run"):
            eval_compare.resolve_checkpoints(str(job_empty))


# ---------------------------------------------------------------------------
# E3: Per-method buffering of raw_rows and green_rows
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestE3RowBuffering:
    def test_failed_method_emits_no_partial_rows(self, tmp_path, monkeypatch):
        ok_metric = {k: 10.0 for k in eval_compare.METRIC_KEYS}
        ok_metric["n_trips"] = 5
        ok_metric["n_unfinished"] = 1

        def _fake_fixed(scenario, args, seeds, tmp, raw_rows=None, green_rows=None):
            for s in seeds:
                eval_compare._append_raw_sample(raw_rows, "fixed", -1, None, s, ok_metric)
                eval_compare._append_green_durations(
                    green_rows, "fixed", -1, s, [("J0", 0, 30.0, True)]
                )
            return [ok_metric for _ in seeds]

        def _failing_lqi(scenario, args, seeds, tmp, raw_rows=None, green_rows=None):
            # Succeed on seed 1 (writing partial rows to the provided buffer), then fail on seed 2
            eval_compare._append_raw_sample(raw_rows, "lqi", -1, None, seeds[0], ok_metric)
            eval_compare._append_green_durations(
                green_rows, "lqi", -1, seeds[0], [("J0", 0, 20.0, True)]
            )
            raise RuntimeError("boom on seed 2")

        monkeypatch.setattr(eval_compare, "eval_fixed", _fake_fixed)
        monkeypatch.setattr(eval_compare, "eval_lqi", _failing_lqi)

        out_dir = tmp_path / "out"
        args = SimpleNamespace(
            network=None,
            split="eval",
            sumo_cfg="synthetic.sumocfg",
            lane_groups=None,
            methods=["lqi", "fixed"],
            seeds=[42, 123],
            reference="fixed",
            checkpoint=None,
            ippo_checkpoint=None,
            privileged_checkpoint=None,
            allow_legacy_schema=False,
            emissions=False,
            max_steps=10,
            step_length=5,
            learned_step_length=None,
            min_green_time=15,
            max_green_time=60,
            fixed_green_s=30.0,
            phi_min=3,
            phi_max=12,
            kappa=5,
            sat_flow=1800.0,
            tmp_dir=str(tmp_path / "tmp"),
            out_dir=str(out_dir),
        )
        res = eval_compare.run(args)
        assert list(res["aggregates"].keys()) == ["fixed"]

        with open(out_dir / "raw_samples.csv", encoding="utf-8") as f:
            raw = list(csv.DictReader(f))
        with open(out_dir / "green_durations.csv", encoding="utf-8") as f:
            greens = list(csv.DictReader(f))

        assert len(raw) == 2
        assert {r["method"] for r in raw} == {"fixed"}
        assert len(greens) == 2
        assert {r["method"] for r in greens} == {"fixed"}


# ---------------------------------------------------------------------------
# E4 & E5: Actuated timing precision, tmp XML placement, completed flag
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestE4AndE5ActuatedAndCompletedFlag:
    def test_actuated_first_phase_duration_matches_configured_min_green_and_flags_completed(
        self, tmp_path
    ):
        net_path = N2_DIR / "intersections.net.xml"
        sumo_cfg = N2_DIR / "sumo_config_eval.sumocfg"
        vtypes_path = N2_DIR / "vtypes.add.xml"
        add_path = tmp_path / "tls_actuated.add.xml"
        tri_path = tmp_path / "actuated_tiny.xml"

        # Configure custom min_green=12, max_green=40 to verify arg propagation too.
        tls_ids = actuated_mod.write_actuated_add(
            net_path, add_path, min_green=12, max_green=40
        )
        assert add_path.exists()
        # Repo tree must remain clean (no tls_actuated.add.xml created under sumo_configs/)
        assert not (N2_DIR / "tls_actuated.add.xml").exists()

        root = ET.parse(add_path).getroot()
        green_phases = [p for p in root.iter("phase") if p.get("minDur") is not None]
        assert green_phases
        assert all(p.get("minDur") == "12" and p.get("maxDur") == "40" for p in green_phases)

        # Tiny 25-second SUMO run: on n2_corridor seed=42, A0 starts empty and gaps out
        # right at minDur=12 s (previously logged 13 s due to sampling before simulationStep).
        greens = actuated_mod.run_actuated(
            sumo_cfg=sumo_cfg,
            add_path=add_path,
            vtypes_path=vtypes_path,
            tls_ids=tls_ids,
            seed=42,
            duration_s=25,
            tripinfo_path=tri_path,
        )
        assert tri_path.exists()

        a0_greens = [rec for rec in greens if rec[0] == "A0"]
        assert len(a0_greens) >= 2
        # First phase completed at exactly configured min_green (12.0 s, NOT 13.0 s)
        tls_id, phase, dur, completed = a0_greens[0]
        assert tls_id == "A0"
        assert phase == 0
        assert dur == pytest.approx(12.0)
        assert completed is True

        # Last phase still active at t=25 s is right-censored -> completed=False
        # (12 s green phase 0 + 3 s yellow phase 1 + 10 s green phase 2 = 25 s)
        _, phase_last, dur_last, completed_last = a0_greens[-1]
        assert phase_last == 2
        assert dur_last == pytest.approx(10.0)
        assert completed_last is False

    def test_multi_agent_and_csv_completed_flag(self, tmp_path):
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
            max_steps=14,  # 12 steps = 60s (switch on step 13 -> 2s new green, step 14 -> 7s right-censored)
        )
        conn = _FakeConn({"J0": group_a + group_b})
        env = _MockMappoEnv(cfg, conn)
        env.reset(seed=42)
        for _ in range(cfg.sim.max_steps):
            _, _, _, truncated, _ = env.step({"J0": 0})
            if truncated["J0"]:
                break
        env.close()

        assert len(env.green_durations) == 2
        first, last = env.green_durations
        assert isinstance(first, GreenDurationRecord)
        assert isinstance(last, GreenDurationRecord)
        assert first == ("J0", 0, pytest.approx(60.0))
        assert first.completed is True
        assert last == ("J0", 2, pytest.approx(7.0))
        assert last.completed is False

        # Verify write_outputs emits the `completed` boolean column in green_durations.csv
        args = SimpleNamespace(
            split="eval", seeds=[42], max_steps=14, step_length=5,
            learned_step_length=None, min_green_time=15, max_green_time=60,
        )
        m = {k: 10.0 for k in eval_compare.METRIC_KEYS}
        agg = {"fixed": eval_compare.aggregate([m])}
        report = eval_compare.build_report(agg, "fixed")
        green_rows: List[Dict[str, Any]] = []
        eval_compare._append_green_durations(green_rows, "fixed", -1, 42, env.green_durations)
        eval_compare._append_green_durations(
            green_rows, "actuated", -1, 42, [("J0", 0, 15.0, True), ("J0", 2, 8.0, False)]
        )
        eval_compare.write_outputs(tmp_path, "n2_corridor", agg, report, args, green_rows=green_rows)

        with open(tmp_path / "green_durations.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [r["completed"] for r in rows] == ["True", "False", "True", "False"]
        assert [r["duration_s"] for r in rows] == ["60.00", "7.00", "15.00", "8.00"]


# ---------------------------------------------------------------------------
# E6: MappoTrafficEnv.reset sim-time consistency
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestE6ResetSimTimeConsistency:
    def test_reset_does_not_step_extra_second_and_timer_matches_sim_time(self):
        tls_ids, lane_groups = load_lane_groups_json(str(N2_DIR / "lane_groups.json"))
        cfg = build_default_config(
            sumo_cfg_path=str(N2_DIR / "sumo_config_eval.sumocfg"),
            gui=False,
            tls_ids=tls_ids,
            manual_lane_groups=lane_groups,
            step_length=5,
            yellow_time=3,
            min_green_time=15,
            max_green_time=60,
            max_steps=13,
        )
        env = MappoTrafficEnv(config=cfg, gui=False, use_libsumo=False)
        try:
            obs, _ = env.reset(seed=42)
            # Sim time is 0.0 s at reset, matching green_timers == 0.0 and green_timer_norm == 0.0
            assert env.sumo_conn.simulation.getTime() == pytest.approx(0.0)
            for t in tls_ids:
                assert env.green_timers[t] == pytest.approx(0.0)
                assert obs[t][24] == pytest.approx(0.0)

            # Hold action 0 for 12 steps -> 60.0 s sim time, timer == 60.0, green_timer_norm == 1.0
            for k in range(1, 13):
                obs, _, _, _, _ = env.step({t: 0 for t in tls_ids})
                assert env.sumo_conn.simulation.getTime() == pytest.approx(5.0 * k)
                assert env.green_timers[tls_ids[0]] == pytest.approx(5.0 * k)
                assert obs[tls_ids[0]][24] == pytest.approx((5.0 * k) / 60.0)

            # Step 13 forces switch through 3 s yellow + 2 s new green -> sim time 65.0 s,
            # initial phase ran EXACTLY 60.0 s of sim time and logs 60.0 s (completed=True).
            env.step({t: 0 for t in tls_ids})
            assert env.sumo_conn.simulation.getTime() == pytest.approx(65.0)
        finally:
            env.close()

        t0_recs = [r for r in env.green_durations if r[0] == tls_ids[0]]
        assert len(t0_recs) == 2
        assert t0_recs[0] == (tls_ids[0], 0, pytest.approx(60.0))
        assert t0_recs[0].completed is True
        assert t0_recs[1] == (tls_ids[0], 2, pytest.approx(2.0))
        assert t0_recs[1].completed is False


# ---------------------------------------------------------------------------
# E7: WebsterPolicy max_green clamp
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestE7WebsterClamp:
    def test_webster_clamps_greens_to_min_and_max_green(self):
        # Extreme asymmetric flows would yield g_a >> 60 s and g_b << 15 s without clamping.
        pol = WebsterPolicy(
            lane_groups={"J0": (["A_0"], ["B_0"])},
            lane_flows_pcu_h={"A_0": 1600.0, "B_0": 20.0},
            sat_flow_pcu_h=1800.0,
            min_green=15.0,
            max_green=60.0,
        )
        t = pol.timings["J0"]
        assert 15.0 <= t.green_a <= 60.0
        assert 15.0 <= t.green_b <= 60.0
        assert t.green_a == pytest.approx(60.0)
        assert t.green_b == pytest.approx(15.0)
        assert t.cycle == pytest.approx(t.green_a + t.green_b + 2.0 * pol.yellow)

        # Custom max_green bound is respected
        pol_custom = WebsterPolicy(
            lane_groups={"J0": (["A_0"], ["B_0"])},
            lane_flows_pcu_h={"A_0": 1600.0, "B_0": 20.0},
            min_green=10.0,
            max_green=45.0,
        )
        tc = pol_custom.timings["J0"]
        assert tc.green_a == pytest.approx(45.0)
        assert tc.green_b == pytest.approx(10.0)

        # Verify eval_compare.eval_webster passes max_green_time to WebsterPolicy
        assert "max_green=" in inspect.getsource(eval_compare.eval_webster)


# ---------------------------------------------------------------------------
# E8: n4_irregular CLI registration
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestE8N4IrregularNetworkChoices:
    def test_n4_irregular_accepted_by_all_cli_parsers(self):
        for mod_path in (
            ROOT / "experiment" / "baselines" / "webster.py",
            ROOT / "experiment" / "baselines" / "actuated.py",
            ROOT / "scripts" / "run_pilot.py",
            ROOT / "experiment" / "runners" / "eval_noise_robustness.py",
            ROOT / "experiment" / "runners" / "eval_compare.py",
        ):
            text = mod_path.read_text(encoding="utf-8")
            assert '"n4_irregular"' in text, f"{mod_path} missing n4_irregular choice"

    def test_no_tls_actuated_left_in_sumo_configs(self):
        stray = list((ROOT / "sumo_configs").rglob("*tls_actuated*"))
        assert stray == [], f"Unexpected tls_actuated file(s) under sumo_configs/: {stray}"
