"""
tests/test_schema130_guards.py
==============================
Unit tests for schema-1.3.0 safety guards (no live SUMO required):

- G1: Safe config-matched auto-resume in experiment/runners/train_ppo.py
      (_find_latest_checkpoint skips mismatched or unlabelled runs, prints why,
      and still resumes crash-restarts of matching runs).
- G2: Realized green durations <= max_green_time for SOTL and all env-driven
      controllers (post-switch residual green 2 s + 11*5 s = 57 s, never 62 s).
- G3: Env schema >= 1.3.0 guard in src/core/policy_loader.py and
      experiment/runners/eval_compare.py (--allow-legacy-schema override).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

from experiment.baselines.sotl import DEFAULT_PHI_MAX, DEFAULT_PHI_MIN, SOTLPolicy
from experiment.runners import eval_compare, train_ppo
from experiment.runners.train_ppo import _find_latest_checkpoint
from scripts import parallel_launcher
from src.core.policy_loader import (
    MIN_SCHEMA_VERSION,
    PolicyLoader,
    validate_checkpoint_schema,
)
from src.traffic_env.components.observations import ObservationBuilder
from src.traffic_env.config import (
    DEFAULT_TLS_FEATURE_NAMES,
    UPSTREAM_PHASE_FEATURE_NAMES,
    build_default_config,
)
from tests.test_revision_a4_a7 import _FakeConn, _MockMappoEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_meta(
    *,
    algo: str = "mappo",
    seed: int = 1,
    reward_preset: str = "full",
    obs_mode: str = "proxy",
    obs_ablation: str = "none",
    sumo_cfg_path: str = "sumo_configs/networks/n3_grid/sumo_config.sumocfg",
    total_timesteps: int = 512,
    version: str = "1.3.0",
) -> Dict[str, Any]:
    return {
        "run_id": "test_run",
        "algo": algo,
        "reward_preset": reward_preset,
        "obs_ablation": obs_ablation,
        "env_cfg": {
            "version": version,
            "sim": {"sumo_cfg_path": sumo_cfg_path},
            "observation": {"obs_mode": obs_mode},
        },
        "train_cfg": {
            "seed": seed,
            "total_timesteps": total_timesteps,
        },
    }


def _write_fake_run(
    parent: Path,
    run_name: str,
    run_meta: Dict[str, Any] | None,
    global_step: int,
) -> Path:
    run_dir = parent / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if run_meta is not None:
        (run_dir / "run_config.json").write_text(
            json.dumps(run_meta, indent=2), encoding="utf-8"
        )
    ckpt_path = run_dir / "last_model.pt"
    torch.save({"global_step": int(global_step)}, ckpt_path)
    return ckpt_path


def _write_actor_ckpt(
    run_dir: Path,
    *,
    schema_version: str | None,
    write_run_config: bool = True,
    obs_dim: int = 26,
    action_dim: int = 2,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    net = nn.Sequential(
        nn.Linear(obs_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, action_dim),
    )
    actor_sd = {f"net.{k}": v for k, v in net.state_dict().items()}
    ckpt_payload: Dict[str, Any] = {
        "actor_state_dict": actor_sd,
        "obs_dim": obs_dim,
        "obs_rms": {"mean": [0.0] * obs_dim, "var": [1.0] * obs_dim, "count": 100.0},
    }
    if schema_version is not None:
        ckpt_payload["env_cfg"] = {"version": schema_version}
    ckpt_path = run_dir / "best_model.pt"
    torch.save(ckpt_payload, ckpt_path)

    if write_run_config:
        rc: Dict[str, Any] = {
            "env_cfg": {"observation": {"obs_mode": "proxy"}},
            "train_cfg": {"seed": 42},
        }
        if schema_version is not None:
            rc["env_cfg"]["version"] = schema_version
        (run_dir / "run_config.json").write_text(json.dumps(rc), encoding="utf-8")
    return ckpt_path


# ---------------------------------------------------------------------------
# G1: Safe auto-resume
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestG1SafeAutoResume:
    def test_two_fake_run_dirs_skips_mismatched_and_resumes_matching(self, tmp_path, capsys):
        # Dir 1: more advanced checkpoint (512 steps) for reward_preset="full"
        _write_fake_run(
            tmp_path,
            "20261124_000001",
            _make_run_meta(reward_preset="full"),
            global_step=512,
        )
        # Dir 2: partial checkpoint (256 steps) for reward_preset="privileged"
        priv_ckpt = _write_fake_run(
            tmp_path,
            "20261124_000002",
            _make_run_meta(reward_preset="privileged"),
            global_step=256,
        )

        # Crash-restart of "privileged" must skip "full" (512) and pick "privileged" (256).
        found = _find_latest_checkpoint(tmp_path, _make_run_meta(reward_preset="privileged"))
        out = capsys.readouterr().out
        assert found == priv_ckpt
        assert "skipping" in out
        assert "reward_preset" in out

        # Brand-new preset ("no_pressure") must skip both and start fresh (None).
        found_fresh = _find_latest_checkpoint(
            tmp_path, _make_run_meta(reward_preset="no_pressure")
        )
        out_fresh = capsys.readouterr().out
        assert found_fresh is None
        assert "starting fresh" in out_fresh

    def test_crash_restart_picks_highest_step_among_matching_runs(self, tmp_path):
        _write_fake_run(tmp_path, "run_a", _make_run_meta(), global_step=128)
        ckpt_b = _write_fake_run(tmp_path, "run_b", _make_run_meta(), global_step=384)
        # Relative vs resolved absolute sumo_cfg_path must still match.
        abs_cfg = str(Path("sumo_configs/networks/n3_grid/sumo_config.sumocfg").resolve())
        found = _find_latest_checkpoint(tmp_path, _make_run_meta(sumo_cfg_path=abs_cfg))
        assert found == ckpt_b

    @pytest.mark.parametrize(
        "field,override",
        [
            ("algo", {"algo": "ippo"}),
            ("seed", {"seed": 99}),
            ("reward_preset", {"reward_preset": "privileged"}),
            ("obs_mode", {"obs_mode": "privileged"}),
            ("obs_ablation", {"obs_ablation": "no_class_shares"}),
            ("sumo_cfg", {"sumo_cfg_path": "sumo_configs/networks/n2_corridor/sumo_config.sumocfg"}),
            ("total_timesteps", {"total_timesteps": 1024}),
            ("schema_version", {"version": "1.2.0"}),
        ],
    )
    def test_each_identity_field_blocks_resume_with_reason(
        self, tmp_path, capsys, field, override
    ):
        _write_fake_run(tmp_path, "run_old", _make_run_meta(**override), global_step=256)
        found = _find_latest_checkpoint(tmp_path, _make_run_meta())
        out = capsys.readouterr().out
        assert found is None
        assert field in out

    def test_missing_run_config_is_skipped(self, tmp_path, capsys):
        _write_fake_run(tmp_path, "run_orphan", None, global_step=256)
        found = _find_latest_checkpoint(tmp_path, _make_run_meta())
        out = capsys.readouterr().out
        assert found is None
        assert "missing sibling run_config.json" in out


# ---------------------------------------------------------------------------
# G2: SOTL and env-driven controllers never exceed max_green_time
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestG2MaxGreenInvariant:
    def _build_env(self, tls_ids=("J0",), max_steps=60, max_green_time=60):
        groups = {}
        lanes_by_tls = {}
        for i, tid in enumerate(tls_ids):
            ga = [f"E{i}a0_0", f"E{i}a0_1", f"E{i}a1_0", f"E{i}a1_1"]
            gb = [f"E{i}b0_0", f"E{i}b0_1", f"E{i}b1_0", f"E{i}b1_1"]
            groups[tid] = (ga, gb)
            lanes_by_tls[tid] = ga + gb
        cfg = build_default_config(
            sumo_cfg_path="synthetic.sumocfg",
            tls_ids=tls_ids,
            manual_lane_groups=groups,
            step_length=5,
            yellow_time=3,
            min_green_time=15,
            max_green_time=max_green_time,
            max_steps=max_steps,
        )
        conn = _FakeConn(lanes_by_tls)
        env = _MockMappoEnv(cfg, conn)
        return env, groups, cfg

    @pytest.mark.parametrize("sync_with_env", [False, True])
    def test_sotl_realized_green_never_exceeds_max_green(self, sync_with_env):
        env, groups, cfg = self._build_env(tls_ids=("J0",), max_steps=60, max_green_time=60)
        env.reset(seed=42)
        policy = SOTLPolicy(
            lane_groups=groups,
            phi_min=DEFAULT_PHI_MIN,
            phi_max=DEFAULT_PHI_MAX,
            kappa=5,
            step_length=cfg.sim.step_length,
            yellow_time=cfg.sim.yellow_time,
        )
        policy.reset()

        # Empty roads (n_red = 0 < kappa): forces SOTL to hit phi_max repeatedly.
        for _ in range(cfg.sim.max_steps):
            lane_cache = env.obs_builder.build_lane_cache(env.tls_ids)
            if sync_with_env:
                cur_actions = {
                    t: env._green_phase_map[t].index(env._current_green_phase[t])
                    for t in env.tls_ids
                }
                actions = policy.select_actions(
                    lane_cache,
                    green_timers=env.green_timers,
                    current_actions=cur_actions,
                )
            else:
                actions = policy.select_actions(lane_cache)
            _, _, _, truncated, _ = env.step(actions)
            if truncated["J0"]:
                break
        env.close()

        durations = [d for _, _, d in env.green_durations]
        assert len(durations) >= 4
        # Initial phase reaches 60.0 s; post-switch phases start at remaining=2 s
        # and must switch at 57.0 s instead of overshooting to 62.0 s.
        assert durations[0] == pytest.approx(60.0)
        assert durations[1] == pytest.approx(57.0)
        assert all(d <= cfg.sim.max_green_time for d in durations), (
            f"SOTL exceeded max_green_time={cfg.sim.max_green_time}: {durations}"
        )

    def test_env_clamps_oversized_sotl_phi_max_and_multi_tls_hold(self):
        # Even if a policy requests holding forever (phi_max=99) or holds one TLS
        # while a sibling TLS switches, the env guarantees every realized green <= max_green_time.
        env, groups, cfg = self._build_env(
            tls_ids=("J0", "J1"), max_steps=50, max_green_time=60
        )
        env.reset(seed=7)
        oversized_sotl = SOTLPolicy(
            lane_groups={"J0": groups["J0"]},
            phi_min=3,
            phi_max=99,
            kappa=999,
            step_length=cfg.sim.step_length,
            yellow_time=cfg.sim.yellow_time,
        )

        for step in range(cfg.sim.max_steps):
            lane_cache = env.obs_builder.build_lane_cache(env.tls_ids)
            a0 = oversized_sotl.select_actions(lane_cache)["J0"]
            # J1 switches periodically every 4 steps while J0 tries to hold forever.
            a1 = (step // 4) % 2
            _, _, _, truncated, _ = env.step({"J0": a0, "J1": a1})
            if any(truncated.values()):
                break
        env.close()

        assert len(env.green_durations) >= 6
        for tls_id, phase, dur in env.green_durations:
            assert 0.0 < dur <= cfg.sim.max_green_time, (
                f"{tls_id} phase {phase} exceeded max_green_time={cfg.sim.max_green_time}: {dur}"
            )


# ---------------------------------------------------------------------------
# G3: Schema >= 1.3.0 guard
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestG3SchemaGuard:
    def test_policy_loader_refuses_legacy_schema_unless_allowed(self, tmp_path):
        ckpt_12 = _write_actor_ckpt(tmp_path / "run_120", schema_version="1.2.0")
        with pytest.raises(ValueError, match=r"1\.2\.0.*1\.3\.0|--allow-legacy-schema"):
            PolicyLoader(checkpoint_path=ckpt_12, obs_dim=26, action_dim=2).load()

        pol = PolicyLoader(
            checkpoint_path=ckpt_12,
            obs_dim=26,
            action_dim=2,
            allow_legacy_schema=True,
        ).load()
        assert pol.obs_dim == 26

    def test_policy_loader_accepts_130_and_higher_versions(self, tmp_path):
        for ver in ("1.3.0", "1.3.1", "1.10.0", "2.0.0"):
            ckpt = _write_actor_ckpt(tmp_path / f"run_{ver}", schema_version=ver)
            pol = PolicyLoader(checkpoint_path=ckpt, obs_dim=26, action_dim=2).load()
            assert pol.obs_dim == 26

    def test_eval_compare_refuses_legacy_and_missing_schema_before_sumo(self, tmp_path):
        ckpt_12 = _write_actor_ckpt(tmp_path / "run_120", schema_version="1.2.0")
        ckpt_missing = _write_actor_ckpt(
            tmp_path / "run_noversion", schema_version=None, write_run_config=False
        )

        # Direct eval_learned refusal
        dummy_scenario = SimpleNamespace(
            sumo_cfg=Path("synthetic.sumocfg"),
            tls_ids=("J0",),
            lane_groups={"J0": (["E0_0"], ["E1_0"])},
        )
        args_default = SimpleNamespace(
            step_length=5,
            max_steps=10,
            learned_step_length=None,
            min_green_time=15,
            max_green_time=60,
            emissions=False,
            allow_legacy_schema=False,
        )
        with pytest.raises(ValueError, match=r"1\.2\.0"):
            eval_compare.eval_learned(
                "mappo", [ckpt_12], dummy_scenario, args_default, [42], tmp_path
            )
        with pytest.raises(ValueError, match=r"missing env schema version"):
            eval_compare.eval_learned(
                "mappo", [ckpt_missing], dummy_scenario, args_default, [42], tmp_path
            )

        # Full eval_compare.run refusal (must NOT be swallowed by per-method try/except)
        run_args = SimpleNamespace(
            network=None,
            split="eval",
            sumo_cfg="synthetic.sumocfg",
            lane_groups=None,
            methods=["mappo", "lqi"],
            seeds=[42],
            reference="mappo",
            checkpoint=[str(ckpt_12)],
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
            out_dir=str(tmp_path / "out"),
        )
        with pytest.raises(ValueError, match=r"1\.2\.0"):
            eval_compare.run(run_args)

        # Override passes schema validation cleanly
        assert validate_checkpoint_schema(
            ckpt_12, min_version=MIN_SCHEMA_VERSION, allow_legacy=True
        ) == "1.2.0"

    def test_eval_compare_cli_parses_allow_legacy_schema_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["eval_compare.py", "--network", "n3_grid"])
        parsed_default = eval_compare.parse_args()
        assert parsed_default.allow_legacy_schema is False

        monkeypatch.setattr(
            sys, "argv", ["eval_compare.py", "--network", "n3_grid", "--allow-legacy-schema"]
        )
        parsed_override = eval_compare.parse_args()
        assert parsed_override.allow_legacy_schema is True


# ---------------------------------------------------------------------------
# N2 Corridor "base" arm (--no-n2-coordination)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestN2CorridorBaseArm:
    def test_cli_no_n2_coordination_flag_skips_auto_enable(self, monkeypatch):
        n2_cfg = "sumo_configs/networks/n2_corridor/sumo_config.sumocfg"
        n2_lg = "sumo_configs/networks/n2_corridor/lane_groups.json"

        # Default n2_corridor behaviour unchanged: both coordination flags auto-enabled.
        monkeypatch.setattr(
            sys, "argv", ["train_ppo.py", "--sumo-cfg", n2_cfg, "--lane-groups", n2_lg]
        )
        args_coord = train_ppo.parse_args()
        assert args_coord.no_n2_coordination is False
        assert args_coord.upstream_phase_obs is True
        assert args_coord.shared_reward is True

        # --no-n2-coordination skips auto-enable, keeping both False.
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_ppo.py",
                "--sumo-cfg", n2_cfg,
                "--lane-groups", n2_lg,
                "--no-n2-coordination",
            ],
        )
        args_base = train_ppo.parse_args()
        assert args_base.no_n2_coordination is True
        assert args_base.upstream_phase_obs is False
        assert args_base.shared_reward is False

    def test_auto_resume_separates_base_and_coordinated_n2_runs(self, tmp_path, capsys):
        n2_cfg = "sumo_configs/networks/n2_corridor/sumo_config.sumocfg"
        coord_meta = _make_run_meta(sumo_cfg_path=n2_cfg)
        coord_meta["upstream_phase_obs"] = True
        coord_meta["shared_reward"] = True
        coord_meta["env_cfg"]["observation"]["tls_feature_names"] = list(
            DEFAULT_TLS_FEATURE_NAMES + UPSTREAM_PHASE_FEATURE_NAMES
        )

        base_meta = _make_run_meta(sumo_cfg_path=n2_cfg)
        base_meta["upstream_phase_obs"] = False
        base_meta["shared_reward"] = False
        base_meta["env_cfg"]["observation"]["tls_feature_names"] = list(
            DEFAULT_TLS_FEATURE_NAMES
        )

        # Also test a legacy coordinated run_config lacking top-level boolean keys
        # (only tls_feature_names indicates upstream_phase_obs).
        legacy_coord_meta = _make_run_meta(sumo_cfg_path=n2_cfg)
        legacy_coord_meta["env_cfg"]["observation"]["tls_feature_names"] = list(
            DEFAULT_TLS_FEATURE_NAMES + UPSTREAM_PHASE_FEATURE_NAMES
        )

        coord_ckpt = _write_fake_run(tmp_path, "run_coord", coord_meta, global_step=512)
        _write_fake_run(tmp_path, "run_legacy_coord", legacy_coord_meta, global_step=640)

        # Base run must never resume either coordinated checkpoint.
        assert _find_latest_checkpoint(tmp_path, base_meta) is None
        out = capsys.readouterr().out
        assert "upstream_phase_obs" in out
        assert "shared_reward" in out

        # Coordinated run resumes only its matching coordinated checkpoint.
        assert _find_latest_checkpoint(tmp_path, coord_meta) == coord_ckpt

        # Mismatch on shared_reward alone also blocks resume.
        shared_only_meta = dict(base_meta, shared_reward=True)
        base_ckpt = _write_fake_run(tmp_path, "run_base", base_meta, global_step=256)
        assert _find_latest_checkpoint(tmp_path, base_meta) == base_ckpt
        assert _find_latest_checkpoint(tmp_path, shared_only_meta) is None
        assert "shared_reward" in capsys.readouterr().out

    def test_sync_obs_mode_resolves_upstream_phase_obs_for_eval(self, tmp_path):
        run_dir = tmp_path / "run_base"
        run_dir.mkdir()
        ckpt_path = run_dir / "best_model.pt"
        ckpt_path.write_bytes(b"")
        rc = {
            "upstream_phase_obs": False,
            "shared_reward": False,
            "env_cfg": {
                "observation": {
                    "obs_mode": "proxy",
                    "tls_feature_names": list(DEFAULT_TLS_FEATURE_NAMES),
                }
            },
        }
        (run_dir / "run_config.json").write_text(json.dumps(rc), encoding="utf-8")

        args = SimpleNamespace(
            checkpoint=str(ckpt_path),
            obs_mode="proxy",
            upstream_phase_obs=True,  # e.g. auto-enabled before checkpoint lookup
        )
        train_ppo.sync_obs_mode_from_checkpoint(args)
        assert args.upstream_phase_obs is False

    def test_eval_compare_builds_matching_obs_dim_for_base_and_coordinated_n2(self, tmp_path):
        scenario = SimpleNamespace(
            sumo_cfg=Path("sumo_configs/networks/n2_corridor/sumo_config.sumocfg"),
            tls_ids=("J0",),
            lane_groups={"J0": (["E0_0", "E1_0"], ["E2_0", "E3_0"])},
        )
        eval_args = SimpleNamespace(
            step_length=5,
            max_steps=10,
            min_green_time=15,
            max_green_time=60,
        )

        # Base arm: 26-dim obs (upstream_phase_obs=False)
        base_dir = tmp_path / "n2_base"
        base_ckpt = _write_actor_ckpt(
            base_dir, schema_version="1.3.0", write_run_config=False, obs_dim=26
        )
        base_env_cfg_train = build_default_config(
            sumo_cfg_path=str(scenario.sumo_cfg),
            tls_ids=scenario.tls_ids,
            manual_lane_groups=scenario.lane_groups,
            upstream_phase_obs=False,
        )
        (base_dir / "run_config.json").write_text(
            json.dumps({
                "upstream_phase_obs": False,
                "shared_reward": False,
                "env_cfg": base_env_cfg_train.to_dict(),
                "train_cfg": {"seed": 42},
            }),
            encoding="utf-8",
        )

        # Coordinated arm: 30-dim obs (upstream_phase_obs=True)
        coord_dir = tmp_path / "n2_coord"
        coord_ckpt = _write_actor_ckpt(
            coord_dir, schema_version="1.3.0", write_run_config=False, obs_dim=30
        )
        coord_env_cfg_train = build_default_config(
            sumo_cfg_path=str(scenario.sumo_cfg),
            tls_ids=scenario.tls_ids,
            manual_lane_groups=scenario.lane_groups,
            upstream_phase_obs=True,
        )
        (coord_dir / "run_config.json").write_text(
            json.dumps({
                "upstream_phase_obs": True,
                "shared_reward": True,
                "env_cfg": coord_env_cfg_train.to_dict(),
                "train_cfg": {"seed": 42},
            }),
            encoding="utf-8",
        )

        base_eval_cfg = eval_compare._build_env_cfg(
            scenario,
            eval_args,
            eval_compare._obs_mode_for_checkpoint(base_ckpt),
            obs_overrides=eval_compare._obs_overrides_for_checkpoint(base_ckpt),
        )
        coord_eval_cfg = eval_compare._build_env_cfg(
            scenario,
            eval_args,
            eval_compare._obs_mode_for_checkpoint(coord_ckpt),
            obs_overrides=eval_compare._obs_overrides_for_checkpoint(coord_ckpt),
        )
        assert base_eval_cfg.local_obs_dim == 26
        assert coord_eval_cfg.local_obs_dim == 30

        conn = _FakeConn({"J0": ["E0_0", "E1_0", "E2_0", "E3_0"]})
        assert ObservationBuilder(config=base_eval_cfg, sumo_conn=conn)._upstream_phase_obs is False
        assert ObservationBuilder(config=coord_eval_cfg, sumo_conn=conn)._upstream_phase_obs is True

        # PolicyLoader loads each checkpoint with its reconstructed obs_dim cleanly.
        assert PolicyLoader(base_ckpt, obs_dim=base_eval_cfg.local_obs_dim).load().obs_dim == 26
        assert PolicyLoader(coord_ckpt, obs_dim=coord_eval_cfg.local_obs_dim).load().obs_dim == 30

        # Fallback when tls_feature_names is omitted but top-level upstream_phase_obs is set.
        minimal_dir = tmp_path / "n2_minimal_coord"
        minimal_dir.mkdir()
        minimal_ckpt = minimal_dir / "best_model.pt"
        minimal_ckpt.write_bytes(b"")
        (minimal_dir / "run_config.json").write_text(
            json.dumps({"upstream_phase_obs": True, "env_cfg": {"observation": {"obs_mode": "proxy"}}}),
            encoding="utf-8",
        )
        min_eval_cfg = eval_compare._build_env_cfg(
            scenario,
            eval_args,
            "proxy",
            obs_overrides=eval_compare._obs_overrides_for_checkpoint(minimal_ckpt),
        )
        assert min_eval_cfg.local_obs_dim == 30

    def test_parallel_launcher_revision130_includes_n2_base_arm(self, tmp_path):
        args = SimpleNamespace(
            preset="revision130",
            tier="all",
            shard=None,
            dry_run=True,
            total_timesteps=2_000_000,
            rollout_horizon=128,
        )
        jobs = parallel_launcher.build_jobs(args, tmp_path, passthrough=[])
        t1 = [j for j in jobs if j.tier == 1]
        t2 = [j for j in jobs if j.tier == 2]
        assert len(t1) == 55
        assert len(t2) == 40
        assert len(jobs) == 95
        assert len({j.name for j in jobs}) == len(jobs)

        expected_seeds = [42, 123, 456, 789, 1337]
        n2_coord = [
            j for j in t1
            if j.network == "n2_corridor" and "--no-n2-coordination" not in j.cmd
        ]
        n2_base = [
            j for j in t1
            if j.network == "n2_corridor" and "--no-n2-coordination" in j.cmd
        ]
        assert [j.seed for j in n2_coord] == expected_seeds
        assert [j.seed for j in n2_base] == expected_seeds
        assert all(j.name == f"n2_corridor_mappo_proxy_seed{j.seed}" for j in n2_coord)
        assert all(j.name == f"n2_corridor_mappo_proxy_base_seed{j.seed}" for j in n2_base)


def test_stale_pre_checkpoint_run_dir_is_reused(tmp_path):
    """A run killed before its first last_model.pt is restarted in the same dir."""
    from experiment.runners.train_ppo import _find_stale_run_dir

    meta = _make_run_meta()
    stale = tmp_path / "20260101_000000"
    stale.mkdir()
    (stale / "run_config.json").write_text(json.dumps(meta), encoding="utf-8")
    (stale / "best_model.pt").write_bytes(b"untrained")
    assert _find_stale_run_dir(tmp_path, meta) == stale
    # other arm, or a dir that already checkpointed / finished, is never reused
    assert _find_stale_run_dir(tmp_path, _make_run_meta(reward_preset="privileged")) is None
    torch.save({"global_step": 128}, stale / "last_model.pt")
    assert _find_stale_run_dir(tmp_path, meta) is None
