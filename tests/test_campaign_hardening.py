"""
tests/test_campaign_hardening.py
================================
Unit and integration tests for campaign-hardening blockers (B1-B5) and
should-fix items (S1-S7). Uses pytest ``tmp_path`` (under %TEMP%) with no
leftover repository artifacts and no live SUMO required.
"""
from __future__ import annotations

import csv
import json
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from experiment.plots import campaign_figures
from experiment.runners import train_ppo
from scripts import build_paper_artifacts, generate_summary_stats, parallel_launcher
from src.traffic_env.config import build_default_config


# ---------------------------------------------------------------------------
# Fake fast env for B3/S1 in-place preemption-resume tests (no SUMO needed)
# ---------------------------------------------------------------------------

class _TinyFakeEnv:
    """Minimal multi-agent env with deterministic episode length & rewards."""

    def __init__(
        self,
        env_cfg,
        *,
        ep_len: int = 4,
        episode_rewards: List[float] | None = None,
        on_step_hook=None,
    ) -> None:
        self.config = env_cfg
        self.tls_ids = list(env_cfg.tls_ids)
        self.obs_dim = env_cfg.local_obs_dim
        self.state_dim = env_cfg.global_state_dim
        self.action_dim = 2
        self._ep_len = ep_len
        self._episode_rewards = list(episode_rewards or [5.0, 1.0, 1.0, 1.0])
        self._on_step_hook = on_step_hook
        self._step_in_ep = 0
        self._ep_idx = 0
        self._total_steps_called = 0
        self.closed = False

    def _obs(self) -> Dict[str, np.ndarray]:
        val = float((self._step_in_ep + 1) % 5) * 0.1
        return {t: np.full((self.obs_dim,), val, dtype=np.float32) for t in self.tls_ids}

    def _info(self) -> Dict[str, Dict[str, Any]]:
        gs = np.concatenate([self._obs()[t] for t in self.tls_ids], axis=0)
        return {
            t: {
                "global_state": gs,
                "queue_total_proxy": 0.2,
                "co2_mg_per_s": 1.0,
                "fuel_ml_per_s": 0.5,
                "avg_waiting_proxy": 0.1,
                "current_passed": float(self._step_in_ep),
            }
            for t in self.tls_ids
        }

    def reset(self, seed: int = 42):
        self._step_in_ep = 0
        return self._obs(), self._info()

    def step(self, action_dict: Dict[str, int]):
        self._total_steps_called += 1
        if self._on_step_hook is not None:
            self._on_step_hook(self._total_steps_called)
        self._step_in_ep += 1
        done = self._step_in_ep >= self._ep_len
        r_idx = min(self._ep_idx, len(self._episode_rewards) - 1)
        per_step_r = self._episode_rewards[r_idx] / float(self._ep_len)
        if done:
            self._ep_idx += 1
        rewards = {t: per_step_r for t in self.tls_ids}
        terminated = {t: False for t in self.tls_ids}
        truncated = {t: done for t in self.tls_ids}
        return self._obs(), rewards, terminated, truncated, self._info()

    def close(self) -> None:
        self.closed = True


def _make_train_args(ckpt_dir: Path, log_dir: Path, total_timesteps: int = 24) -> SimpleNamespace:
    return SimpleNamespace(
        mode="train",
        algo="mappo",
        lane_groups=None,
        sumo_cfg="sumo_configs/networks/n3_grid/sumo_config.sumocfg",
        checkpoint="",
        seed=42,
        total_timesteps=total_timesteps,
        rollout_horizon=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        target_kl=0.015,
        policy_lr=3e-4,
        value_lr=3e-4,
        update_epochs=2,
        minibatch_size=8,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        log_dir=str(log_dir),
        ckpt_dir=str(ckpt_dir),
        save_every_steps=16,
        save_latest_every_updates=1,
        eval_episodes=1,
        eval_max_steps=8,
        device="cpu",
        gui=False,
        no_libsumo=True,
        step_length=5,
        yellow_time=3,
        max_steps=4,
        min_green_time=15,
        max_green_time=60,
        queue_cap=50.0,
        waiting_cap=300.0,
        speed_cap=15.0,
        max_lanes_per_tls=4,
        obs_mode="proxy",
        reward_preset="full",
        obs_ablation="none",
        shared_reward=False,
        upstream_phase_obs=False,
        no_n2_coordination=False,
        reward_scale=1.0,
        allow_legacy_schema=False,
        debug=False,
        tls_ids=[],
    )


# ---------------------------------------------------------------------------
# B1: Network + demand file validation fails fast
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestB1DemandValidation:
    def test_validate_network_files_requires_train_and_eval_demand(self, tmp_path, monkeypatch):
        net_dir = tmp_path / "n3_grid"
        net_dir.mkdir()
        (net_dir / "sumo_config.sumocfg").write_text("<configuration/>", encoding="utf-8")
        (net_dir / "lane_groups.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(parallel_launcher, "NETWORKS_DIR", tmp_path)

        job = parallel_launcher.Job(
            name="n3_grid_mappo_proxy_seed42",
            network="n3_grid",
            algo="mappo",
            seed=42,
            obs_mode="proxy",
            cmd=["python"],
            log_path=tmp_path / "job.log",
            ckpt_dir=tmp_path / "ckpt",
        )

        # Missing both demand_train.rou.xml and demand_eval.rou.xml -> fails fast
        missing = parallel_launcher._validate_network_files([job])
        missing_names = {p.name for p in missing}
        assert "demand_train.rou.xml" in missing_names
        assert "demand_eval.rou.xml" in missing_names

        # Add train only -> still missing eval
        (net_dir / "demand_train.rou.xml").write_text("<routes/>", encoding="utf-8")
        missing2 = parallel_launcher._validate_network_files([job])
        assert [p.name for p in missing2] == ["demand_eval.rou.xml"]

        # Add eval -> clean
        (net_dir / "demand_eval.rou.xml").write_text("<routes/>", encoding="utf-8")
        assert parallel_launcher._validate_network_files([job]) == []

    def test_build_jobs_fails_fast_when_demand_missing(self, tmp_path, monkeypatch):
        net_dir = tmp_path / "n3_grid"
        net_dir.mkdir()
        (net_dir / "sumo_config.sumocfg").write_text("<configuration/>", encoding="utf-8")
        (net_dir / "lane_groups.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(parallel_launcher, "NETWORKS_DIR", tmp_path)

        args = SimpleNamespace(
            preset=None,
            tier="all",
            shard=None,
            dry_run=False,
            networks=["n3_grid"],
            algos=["mappo"],
            obs_modes=["proxy"],
            seeds=[42],
            reward_presets=["full"],
            obs_ablations=["none"],
            total_timesteps=300_000,
            rollout_horizon=128,
        )
        with pytest.raises(SystemExit, match=r"demand_train\.rou\.xml"):
            parallel_launcher.build_jobs(args, tmp_path / "camp", passthrough=[])


# ---------------------------------------------------------------------------
# B3 + S1: In-place Spot-preemption resume, atomic writes, SIGTERM, discovery
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestB3AndS1InPlaceResume:
    @pytest.mark.parametrize("interrupt_mode", ["sigterm", "exception"])
    def test_preemption_resume_continues_in_place_and_restores_best_reward(
        self, tmp_path, monkeypatch, interrupt_mode
    ):
        ckpt_parent = tmp_path / "models" / "job1"
        log_parent = tmp_path / "tblogs" / "job1"
        args = _make_train_args(ckpt_parent, log_parent, total_timesteps=24)

        def _fake_cfg():
            return build_default_config(
                sumo_cfg_path=args.sumo_cfg,
                tls_ids=("J0", "J2"),
                max_steps=4,
            )

        # Run 1: episode 1 gets reward +12.0 (sets best_mean_reward=12.0), episode 2 gets +2.0;
        # interrupt on step 9 (right after the first 8-step rollout + PPO update saved last_model.pt).
        def _hook_run1(step_n: int) -> None:
            if step_n == 9:
                if interrupt_mode == "sigterm":
                    signal.raise_signal(signal.SIGTERM)
                else:
                    raise RuntimeError("simulated Spot preemption crash")

        monkeypatch.setattr(
            train_ppo,
            "make_env",
            lambda _a: (
                _TinyFakeEnv(
                    _fake_cfg(),
                    ep_len=4,
                    episode_rewards=[12.0, 2.0, 1.0, 1.0],
                    on_step_hook=_hook_run1,
                ),
                _fake_cfg(),
            ),
        )

        if interrupt_mode == "sigterm":
            train_ppo.train(args)  # exits cleanly without raising
        else:
            with pytest.raises(RuntimeError, match="simulated Spot preemption crash"):
                train_ppo.train(args)

        run_dirs_after_1 = [p for p in ckpt_parent.iterdir() if p.is_dir()]
        assert len(run_dirs_after_1) == 1
        run_dir = run_dirs_after_1[0]
        assert (run_dir / "last_model.pt").exists()
        assert (run_dir / "best_model.pt").exists()
        assert (run_dir / "run_config.json").exists()
        assert not (run_dir / "bench.json").exists()
        assert list(run_dir.glob("*.tmp")) == []

        ck1 = torch.load(run_dir / "last_model.pt", map_location="cpu", weights_only=False)
        step_after_1 = int(ck1["global_step"])
        assert step_after_1 >= 8
        assert step_after_1 < 24
        assert ck1["best_mean_reward"] == pytest.approx(12.0)
        assert "actor_optim_state_dict" in ck1
        assert "actor_scheduler_state_dict" in ck1
        assert "obs_rms" in ck1 and "state_rms" in ck1
        assert "entropy_coef" in ck1
        assert "python_rng_state" in ck1 and "numpy_rng_state" in ck1 and "torch_rng_state" in ck1

        # Run 2 (restart exact same command): give strictly LOWER rewards (+1.5)
        # so best_model.pt MUST NOT be overwritten if best_mean_reward=12.0 was restored.
        best_mtime_before = (run_dir / "best_model.pt").stat().st_mtime_ns
        monkeypatch.setattr(
            train_ppo,
            "make_env",
            lambda _a: (
                _TinyFakeEnv(
                    _fake_cfg(),
                    ep_len=4,
                    episode_rewards=[1.5, 1.5, 1.5, 1.5, 1.5],
                ),
                _fake_cfg(),
            ),
        )
        train_ppo.train(args)

        # Invariants: STILL exactly ONE run dir, ONE best_model.pt, ONE bench.json
        run_dirs_after_2 = [p for p in ckpt_parent.iterdir() if p.is_dir()]
        assert run_dirs_after_2 == [run_dir]
        assert len(list(ckpt_parent.glob("**/best_model.pt"))) == 1
        assert len(list(ckpt_parent.glob("**/bench.json"))) == 1
        assert list(run_dir.glob("*.tmp")) == []

        bench = json.loads((run_dir / "bench.json").read_text(encoding="utf-8"))
        assert bench["run_id"] == run_dir.name
        assert bench["global_steps"] == 24

        ck2 = torch.load(run_dir / "last_model.pt", map_location="cpu", weights_only=False)
        assert ck2["global_step"] == 24
        assert ck2["best_mean_reward"] == pytest.approx(12.0)
        best_ck = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
        assert best_ck["best_mean_reward"] == pytest.approx(12.0)
        assert (run_dir / "best_model.pt").stat().st_mtime_ns == best_mtime_before

        # Single log dir reused in place and train_updates.csv shows continued steps
        log_dirs = [p for p in log_parent.iterdir() if p.is_dir()]
        assert [p.name for p in log_dirs] == [run_dir.name]
        with open(log_dirs[0] / "train_updates.csv", encoding="utf-8") as f:
            steps = [int(r["global_step"]) for r in csv.DictReader(f)]
        assert steps[0] == 8
        assert steps[-1] == 24
        assert steps == sorted(steps)

    def test_build_paper_artifacts_selects_canonical_checkpoint_and_errors_if_ambiguous(
        self, tmp_path
    ):
        camp = tmp_path / "campaign_test"
        job_dir = camp / "models" / "n3_grid_mappo_proxy_seed42"
        # Earlier incomplete run dir with an older best_model.pt (no bench.json)
        old_run = job_dir / "20261124_010000"
        old_run.mkdir(parents=True)
        (old_run / "best_model.pt").write_bytes(b"old")
        # Completed canonical run dir with bench.json + best_model.pt
        good_run = job_dir / "20261124_020000"
        good_run.mkdir(parents=True)
        (good_run / "best_model.pt").write_bytes(b"canonical")
        (good_run / "bench.json").write_text("{}", encoding="utf-8")

        found = build_paper_artifacts.discover_checkpoints(camp, "n3_grid", "mappo", "proxy")
        assert found == [good_run / "best_model.pt"]

        # Adding a second bench.json makes the job dir ambiguous -> must raise ValueError
        (old_run / "bench.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="Ambiguous completed runs"):
            build_paper_artifacts.discover_checkpoints(camp, "n3_grid", "mappo", "proxy")

    def test_launcher_terminate_timeout_is_at_most_10s(self, tmp_path):
        sched = parallel_launcher.CampaignScheduler(
            jobs=[], campaign_dir=tmp_path, max_workers=2, stagger_s=0.0
        )
        assert sched.terminate_timeout_s <= 10.0


# ---------------------------------------------------------------------------
# B5: revision130 defaults to 300,000 timesteps & dry-run prints per-job steps
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestB5TimestepBudget:
    def test_revision130_defaults_to_300k_and_dry_run_prints_steps(self, monkeypatch, capsys):
        monkeypatch.setattr(
            sys, "argv", ["parallel_launcher.py", "--preset", "revision130", "--dry-run"]
        )
        rc = parallel_launcher.main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "Tier 1: 55 jobs" in out
        assert "Tier 2: 40 jobs" in out
        assert "Total:  95 jobs" in out
        assert out.count("steps=300000") == 95

        # Also check every job's cmd passes --total-timesteps 300000 explicitly
        args = SimpleNamespace(
            preset="revision130", tier="all", shard=None, dry_run=True,
            total_timesteps=None, rollout_horizon=128,
        )
        jobs = parallel_launcher.build_jobs(args, Path("tmp_camp"), passthrough=[])
        assert len(jobs) == 95
        for j in jobs:
            idx = j.cmd.index("--total-timesteps")
            assert j.cmd[idx + 1] == "300000"
            assert j.total_timesteps == 300_000

    def test_revision130_honors_explicit_total_timesteps_override(self, monkeypatch, capsys):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "parallel_launcher.py",
                "--preset", "revision130",
                "--total-timesteps", "500000",
                "--dry-run",
            ],
        )
        assert parallel_launcher.main() == 0
        out = capsys.readouterr().out
        assert out.count("steps=500000") == 95
        assert "steps=300000" not in out


# ---------------------------------------------------------------------------
# S3: Sharded manifest names
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestS3ShardedManifests:
    def test_manifest_filename_per_shard(self, tmp_path):
        assert parallel_launcher.manifest_filename(None) == "campaign_manifest.json"
        assert parallel_launcher.manifest_filename("1/4") == "campaign_manifest_shard1of4.json"
        assert parallel_launcher.manifest_filename("4/4") == "campaign_manifest_shard4of4.json"

        s1 = parallel_launcher.CampaignScheduler(
            jobs=[], campaign_dir=tmp_path, max_workers=1, stagger_s=0.0, shard_spec="1/2"
        )
        s2 = parallel_launcher.CampaignScheduler(
            jobs=[], campaign_dir=tmp_path, max_workers=1, stagger_s=0.0, shard_spec="2/2"
        )
        s1.write_manifest()
        s2.write_manifest()
        assert (tmp_path / "campaign_manifest_shard1of2.json").exists()
        assert (tmp_path / "campaign_manifest_shard2of2.json").exists()
        assert not (tmp_path / "campaign_manifest.json").exists()

    def test_startup_sh_uses_per_shard_manifest_and_logs(self):
        sh_text = (Path(__file__).resolve().parents[1] / "scripts" / "gcp" / "startup.sh").read_text(
            encoding="utf-8"
        )
        assert "campaign_manifest_${SHARD_TAG}.json" in sh_text
        assert "launcher_${SHARD_TAG}.log" in sh_text
        assert "sumo --version | head" not in sh_text
        assert "/snap/bin" in sh_text
        assert "google-cloud-cli" in sh_text
        assert "seq 1 25" in sh_text


# ---------------------------------------------------------------------------
# S4: Fill all free slots in one poll tick + --poll-s
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestS4FillAllSlots:
    def test_while_loop_fills_all_free_slots_in_single_tick(self, tmp_path, monkeypatch):
        jobs = [
            parallel_launcher.Job(
                name=f"job_{i}",
                network="n2_corridor",
                algo="mappo",
                seed=i,
                obs_mode="proxy",
                cmd=[sys.executable, "-c", "import time; time.sleep(0.05)"],
                log_path=tmp_path / f"job_{i}.log",
                ckpt_dir=tmp_path / "models" / f"job_{i}",
            )
            for i in range(4)
        ]

        # Record how many jobs are spawned BEFORE the very first time.sleep() call.
        spawns_before_first_sleep: List[int] = []
        real_sleep = time.sleep

        def _tracking_sleep(sec: float) -> None:
            if not spawns_before_first_sleep:
                spawns_before_first_sleep.append(sum(1 for j in jobs if j.attempts > 0))
            real_sleep(min(sec, 0.02))

        monkeypatch.setattr(parallel_launcher.time, "sleep", _tracking_sleep)

        sched = parallel_launcher.CampaignScheduler(
            jobs, tmp_path, max_workers=3, stagger_s=0.0, poll_s=5.0, max_retries=0
        )
        rc = sched.run()
        assert rc == 0
        assert spawns_before_first_sleep == [3]
        assert all(j.status == "done" for j in jobs)


# ---------------------------------------------------------------------------
# S5 & S6: Cross-host POSIX sumo_cfg normalization + eval schema guard / obs_ablation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestS5AndS6TrainPpoHardening:
    def test_resume_signature_normalizes_sumo_cfg_cross_host_posix(self):
        rel_meta = {
            "env_cfg": {"sim": {"sumo_cfg_path": "sumo_configs/networks/n3_grid/sumo_config.sumocfg"}}
        }
        win_meta = {
            "env_cfg": {"sim": {"sumo_cfg_path": r"C:\Users\Other\repo\sumo_configs\networks\n3_grid\sumo_config.sumocfg"}}
        }
        lin_meta = {
            "env_cfg": {"sim": {"sumo_cfg_path": "/opt/sim2real-mappo/repo/sumo_configs/networks/n3_grid/sumo_config.sumocfg"}}
        }
        s_rel = train_ppo._resume_signature(rel_meta)["sumo_cfg"]
        s_win = train_ppo._resume_signature(win_meta)["sumo_cfg"]
        s_lin = train_ppo._resume_signature(lin_meta)["sumo_cfg"]
        assert s_rel == "sumo_configs/networks/n3_grid/sumo_config.sumocfg"
        assert s_win == s_rel
        assert s_lin == s_rel

    def test_eval_modes_guard_schema_and_restore_obs_ablation(self, tmp_path):
        run_dir = tmp_path / "run_ablation"
        run_dir.mkdir()
        ckpt = run_dir / "best_model.pt"
        torch.save({"actor_state_dict": {}, "env_cfg": {"version": "1.2.0"}}, ckpt)
        (run_dir / "run_config.json").write_text(
            json.dumps({
                "obs_ablation": "no_class_shares",
                "env_cfg": {
                    "version": "1.2.0",
                    "observation": {"obs_mode": "proxy"},
                },
            }),
            encoding="utf-8",
        )

        args = _make_train_args(tmp_path / "c", tmp_path / "l")
        args.checkpoint = str(ckpt)
        args.obs_ablation = "none"
        args.allow_legacy_schema = False

        # sync_obs_mode_from_checkpoint restores obs_ablation
        train_ppo.sync_obs_mode_from_checkpoint(args)
        assert args.obs_ablation == "no_class_shares"

        # evaluate & run_multiseed_eval refuse schema 1.2.0 before touching SUMO
        with pytest.raises(ValueError, match=r"1\.2\.0.*1\.3\.0"):
            train_ppo.evaluate(args)
        with pytest.raises(ValueError, match=r"1\.2\.0.*1\.3\.0"):
            train_ppo.run_multiseed_eval(args, seeds=[42])


# ---------------------------------------------------------------------------
# S7: generate_summary_stats S10 vs S5, campaign_figures fig_losses, build_paper_artifacts
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestS7ReportingAndArtifacts:
    def test_generate_summary_stats_per_arm_seed_completeness_s10_vs_s5(self, tmp_path):
        s10 = list(parallel_launcher.S10_SEEDS)
        s5 = list(parallel_launcher.S5_SEEDS)
        jobs = [
            {"name": f"n3_grid_mappo_proxy_seed{s}", "seed": s, "status": "done"}
            for s in s10
        ] + [
            {"name": f"n3_grid_ippo_proxy_seed{s}", "seed": s, "status": "done"}
            for s in s5
        ] + [
            {"name": f"n2_corridor_mappo_proxy_base_seed{s}", "seed": s, "status": "done"}
            for s in s5
        ]
        campaign = {
            "dir": tmp_path / "campaign_rev130",
            "data": {"updated_at": "2026-11-24T00:00:00", "counts": {"done": len(jobs)}, "jobs": jobs},
        }
        lines, anomalies = generate_summary_stats.analyse_campaign(campaign)
        md = "\n".join(lines)
        assert anomalies == []
        assert "10/10" in md
        assert "5/5" in md

    def test_campaign_figures_fig_losses_facets_and_filters_by_network(self, tmp_path):
        camp = tmp_path / "campaign_figs"
        for net in ("n2_corridor", "n3_grid"):
            run_dir = camp / "tblogs" / f"{net}_mappo_proxy_seed42" / "20261124_000001"
            run_dir.mkdir(parents=True)
            with open(run_dir / "train_updates.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f, fieldnames=["global_step", "policy_loss", "value_loss", "entropy", "approx_kl"]
                )
                w.writeheader()
                for step in (128, 256, 384):
                    w.writerow({
                        "global_step": step,
                        "policy_loss": -0.02,
                        "value_loss": 0.5,
                        "entropy": 0.6,
                        "approx_kl": 0.005,
                    })

        out_multi = tmp_path / "figs_multi"
        campaign_figures.fig_losses(camp, arm="mappo_proxy", out_dir=out_multi)
        assert (out_multi / "training_losses.pdf").exists()
        assert (out_multi / "training_losses_n2_corridor.pdf").exists()
        assert (out_multi / "training_losses_n3_grid.pdf").exists()

        out_filt = tmp_path / "figs_filt"
        campaign_figures.fig_losses(camp, arm="mappo_proxy", network="n3_grid", out_dir=out_filt)
        assert (out_filt / "training_losses_n3_grid.pdf").exists()
        assert not (out_filt / "training_losses_n2_corridor.pdf").exists()

    def test_build_paper_artifacts_discovers_n4_r_privileged_and_n2_base(self, tmp_path):
        assert "n4_irregular" in build_paper_artifacts.DEFAULT_NETWORKS
        assert "n4_irregular" in build_paper_artifacts.SUPPORTED_NETWORKS

        camp = tmp_path / "campaign_rev130"
        for job_name in (
            "n3_grid_mappo_proxy_seed42",
            "n3_grid_mappo_proxy_r-privileged_seed42",
            "n2_corridor_mappo_proxy_seed42",
            "n2_corridor_mappo_proxy_base_seed42",
            "n4_irregular_mappo_proxy_seed42",
        ):
            rdir = camp / "models" / job_name / "20261124_000001"
            rdir.mkdir(parents=True)
            (rdir / "best_model.pt").write_bytes(b"ok")
            (rdir / "bench.json").write_text("{}", encoding="utf-8")

        n3_arms = build_paper_artifacts.discover_network_arms(camp, "n3_grid")
        assert len(n3_arms["mappo"]) == 1
        assert len(n3_arms["r_privileged"]) == 1
        assert "r-privileged" in str(n3_arms["r_privileged"][0])

        n2_arms = build_paper_artifacts.discover_network_arms(camp, "n2_corridor")
        assert len(n2_arms["mappo"]) == 1
        assert len(n2_arms["base"]) == 1
        assert "_base_" in str(n2_arms["base"][0])

        n4_arms = build_paper_artifacts.discover_network_arms(camp, "n4_irregular")
        assert len(n4_arms["mappo"]) == 1
