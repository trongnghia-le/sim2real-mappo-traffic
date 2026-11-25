from __future__ import annotations

"""Clean MAPPO training script for SUMO traffic control.

PATCH NOTES (audit v3 — schema 1.1.0):
- MAPPO CREDIT ASSIGNMENT: the critic now has one value head per agent and
  GAE/advantages are computed PER AGENT from that agent's own shaped reward
  stream (previously: scalar team-mean reward + a single shared advantage,
  which erased the per-agent reward design in rewards.py). Advantages are
  normalized jointly across agents to keep a common scale for the shared actor.
  The actor remains parameter-shared across agents (standard MAPPO).
- FIX: eval_max_steps default 400 -> 1080. 400 truncated evaluation at 37% of
  the 1080-step episode, biasing all reported metrics toward warm-up traffic.
- FIX: TrainConfig.rollout_horizon default aligned to the CLI default (128).
- BREAKING: checkpoints saved before v3 have a scalar critic head and a
  pre-1.1.0 observation schema (truncated lanes, unsigned pressure, J2 lane
  group bug) — they cannot be resumed or evaluated with this code. See
  models/mappo/20260418_215140/LEGACY_REWARD_NOTICE.md.

PATCH NOTES (audit v2):
- FIX: clip_range_vf missing type annotation (was class var, not a dataclass field).
- FIX: value loss clipping now uses clip_range_vf, not clip_ratio.
- FIX: model save names corrected to best_model.pt / last_model.pt per spec.
- FIX: timestamped run dirs under logs/rl/<run_id>/ and models/rl/<run_id>/
  to prevent overwriting between runs.
- FIX: extract_step_emissions now reads from info keys added in multi_agent v2
  (co2_mg_per_s, fuel_ml_per_s) — these were always 0 before because multi_agent
  never set them in info_dict.
- FIX: throughput tracking uses info['current_passed'] (now set by env).
- FIX: run_config.json saved to BOTH log_dir and ckpt_dir for completeness.
- FIX: value_lr and rollout_horizon CLI defaults aligned with TrainConfig defaults.
- NOTE: action_dim is now set by env.action_dim after reset (set in multi_agent v2).
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# repo convention: scripts add the project root to sys.path
# so `python experiment/runners/train_ppo.py` works without pip install -e.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def _apply_omp_thread_limit() -> None:
    """Respect OMP_NUM_THREADS in PyTorch when set (avoids CPU oversubscription)."""
    omp_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_threads:
        try:
            n_threads = int(omp_threads)
            if n_threads > 0:
                torch.set_num_threads(n_threads)
        except ValueError:
            pass


_apply_omp_thread_limit()

# TensorBoard is optional: on some machines torch.utils.tensorboard pulls in a
# broken TensorFlow build (NumPy 2.x ABI clash). CSV logging always works.
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
    _TB_IMPORT_ERROR: Optional[Exception] = None
except Exception as _tb_exc:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]
    _TB_IMPORT_ERROR = _tb_exc

from src.core.policy_loader import MIN_SCHEMA_VERSION, validate_checkpoint_schema
from src.traffic_env.config import build_default_config, load_lane_groups_json, DEFAULT_REWARD_WEIGHTS
from src.traffic_env.envs.multi_agent import MappoTrafficEnv


# ---------------------------------------------------------------------
# Ablation presets (paper VI-F). Defaults reproduce reward 1.3.0 / schema 1.3.0.
# ---------------------------------------------------------------------
REWARD_PRESETS = ("full", "no_pressure", "no_throughput", "queue_only",
                  "unsigned_pressure", "mean_then_square", "no_delay",
                  "no_low_speed", "privileged")
OBS_ABLATIONS = ("none", "no_class_shares", "no_pressure_feature", "lane_truncated")


def _reward_preset_kwargs(preset: str) -> Dict[str, Any]:
    w = dict(DEFAULT_REWARD_WEIGHTS)
    kw: Dict[str, Any] = {
        "pressure_signed": True,
        "queue_mean_of_squares": True,
        "privileged_reward": False,
    }
    if preset == "full":
        pass
    elif preset == "no_pressure":
        w["pressure"] = 0.0
    elif preset == "no_throughput":
        w["throughput"] = 0.0
    elif preset == "queue_only":
        w = {k: 0.0 for k in w}; w["queue"] = -1.0
    elif preset == "unsigned_pressure":
        kw["pressure_signed"] = False
    elif preset == "mean_then_square":
        kw["queue_mean_of_squares"] = False
    elif preset == "no_delay":
        w["queue_occupancy"] = 0.0       # VI-F: drop the linear halted-queue occupancy term (w_w=0)
    elif preset == "no_low_speed":
        w["low_speed_penalty"] = 0.0     # VI-F: drop the near-inert, vision-noisy low-speed term
    elif preset == "privileged":
        kw["privileged_reward"] = True   # A5: camera-proxy obs (26-dim), exact simulator reward
    else:
        raise ValueError(f"unknown reward preset {preset!r}")
    kw["reward_weights"] = w
    return kw


def _obs_ablation_kwargs(ablation: str) -> Dict[str, Any]:
    if ablation in (None, "none"):
        return {}
    if ablation == "no_class_shares":
        return {"drop_class_shares": True}
    if ablation == "no_pressure_feature":
        return {"drop_pressure_feature": True}
    if ablation == "lane_truncated":
        return {"lane_truncated": True}
    raise ValueError(f"unknown obs ablation {ablation!r}")


class _NoopWriter:
    """Drop-in writer used when TensorBoard cannot be imported."""

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


def make_writer(log_dir: str) -> Any:
    if SummaryWriter is None:
        print(
            f"[WARN] TensorBoard unavailable ({_TB_IMPORT_ERROR!r}) — falling back "
            f"to CSV-only logging. Fix: pip install \"numpy<2\" or remove the "
            f"broken tensorflow install."
        )
        return _NoopWriter()
    return SummaryWriter(log_dir=log_dir)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    total_timesteps: int = 500_000
    rollout_horizon: int = 128           # FIX v3: actually matched to CLI default

    gamma: float = 0.99
    gae_lambda: float = 0.95

    normalize_advantages: bool = True

    clip_range_vf: float = 0.2           # FIX: type annotation added
    clip_ratio: float = 0.2
    target_kl: float = 0.015

    policy_lr: float = 3e-4
    value_lr: float = 3e-4               # FIX: aligned with CLI default

    update_epochs: int = 10
    minibatch_size: int = 256            # FIX: aligned with CLI default

    entropy_coef_start: float = 0.01
    entropy_coef_end: float = 0.001
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    log_dir: str = "logs/rl"
    ckpt_dir: str = "models/mappo"
    save_every_steps: int = 50_000
    save_latest_every_updates: int = 1

    eval_episodes: int = 5
    eval_max_steps: int = 1080           # FIX v3: full episode (was 400 = 37% of episode)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------
class CsvLogger:
    """append-mode CSV logger for per-update and per-episode metrics."""

    STEP_FIELDS = [
        "global_step", "episode", "update_idx",
        "policy_loss", "value_loss", "entropy", "approx_kl",
        "clip_fraction", "explained_variance", "sps",
        "lr", "entropy_coef",
    ]
    EPISODE_FIELDS = [
        "global_step", "episode", "episode_length",
        "mean_reward", "queue_total_proxy",
        "co2_total_mg", "fuel_total_ml", "throughput",
        "avg_waiting_proxy",
    ]

    def __init__(self, log_dir: str) -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._step_path = self._dir / "train_updates.csv"
        self._ep_path = self._dir / "train_episodes.csv"
        self._init_file(self._step_path, self.STEP_FIELDS)
        self._init_file(self._ep_path, self.EPISODE_FIELDS)

    def _init_file(self, path: Path, fields: list) -> None:
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def log_update(self, **kwargs: Any) -> None:
        with open(self._step_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.STEP_FIELDS, extrasaction="ignore").writerow(kwargs)

    def log_episode(self, **kwargs: Any) -> None:
        with open(self._ep_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.EPISODE_FIELDS, extrasaction="ignore").writerow(kwargs)


# ---------------------------------------------------------------------
# Running Mean Std (observation normalization)
# ---------------------------------------------------------------------
class RunningMeanStd:
    """Welford online algorithm for mean and variance."""

    def __init__(self, shape: tuple) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = x.reshape(-1, *self.mean.shape)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip(
            (x - self.mean.astype(np.float32)) / (np.sqrt(self.var).astype(np.float32) + 1e-8),
            -10.0, 10.0,
        ).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.mean = np.array(d["mean"], dtype=np.float64)
        self.var = np.array(d["var"], dtype=np.float64)
        self.count = float(d["count"])


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    """Atomically save a PyTorch checkpoint via sibling .tmp + os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _atomic_json_dump(data: Any, path: Path) -> None:
    """Atomically write a JSON file via sibling .tmp + os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _normalize_sumo_cfg_path(raw_sumo_cfg: Any) -> Optional[str]:
    """Normalize sumo_cfg path relative to repo root as POSIX for cross-host resume."""
    if not raw_sumo_cfg:
        return None
    raw_str = str(raw_sumo_cfg)
    posix_raw = raw_str.replace("\\", "/")
    root_resolved = _ROOT.resolve()
    try:
        p = Path(raw_str)
        resolved = p.resolve() if p.is_absolute() else (root_resolved / p).resolve()
        rel = resolved.relative_to(root_resolved)
        return rel.as_posix()
    except Exception:
        pass
    # Cross-host absolute path from another checkout (e.g. /opt/.../sumo_configs/...
    # or C:\...\sumo_configs\...): anchor on 'sumo_configs/' if present.
    marker = "sumo_configs/"
    idx = posix_raw.find(marker)
    if idx >= 0:
        return posix_raw[idx:]
    # Strip leading "./" for clean POSIX consistency on synthetic/external paths.
    while posix_raw.startswith("./"):
        posix_raw = posix_raw[2:]
    return posix_raw


def _resume_signature(run_meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the config fields that must match before auto-resuming a run."""
    env_cfg = run_meta.get("env_cfg") or {}
    sim_cfg = env_cfg.get("sim") or {}
    obs_cfg = env_cfg.get("observation") or {}
    train_cfg = run_meta.get("train_cfg") or {}

    norm_sumo_cfg = _normalize_sumo_cfg_path(sim_cfg.get("sumo_cfg_path"))

    raw_algo = run_meta.get("algo")
    algo = str(raw_algo).lower() if raw_algo is not None else None

    raw_seed = train_cfg.get("seed")
    try:
        seed: Optional[int] = int(raw_seed) if raw_seed is not None else None
    except (TypeError, ValueError):
        seed = None

    raw_total = train_cfg.get("total_timesteps")
    try:
        total_timesteps: Optional[int] = int(raw_total) if raw_total is not None else None
    except (TypeError, ValueError):
        total_timesteps = None

    raw_upstream = run_meta.get("upstream_phase_obs")
    if raw_upstream is not None:
        upstream_phase_obs = bool(raw_upstream)
    else:
        tls_feats = obs_cfg.get("tls_feature_names") or ()
        upstream_phase_obs = any(str(n).startswith("upstream_phase_") for n in tls_feats)

    shared_reward = bool(run_meta.get("shared_reward", False))

    return {
        "algo": algo,
        "seed": seed,
        "reward_preset": run_meta.get("reward_preset"),
        "obs_mode": obs_cfg.get("obs_mode"),
        "obs_ablation": run_meta.get("obs_ablation"),
        "upstream_phase_obs": upstream_phase_obs,
        "shared_reward": shared_reward,
        "sumo_cfg": norm_sumo_cfg,
        "total_timesteps": total_timesteps,
        "schema_version": env_cfg.get("version"),
    }


def _find_latest_checkpoint(
    ckpt_parent: Path,
    current_run_meta: Mapping[str, Any],
) -> Optional[Path]:
    """Return the last_model.pt with the highest global_step whose sibling
    run_config.json matches current_run_meta on all training-identity fields.
    Candidates with missing, unreadable, or mismatched run_config.json are
    skipped with an explicit reason printed to stdout.
    """
    if not ckpt_parent.exists():
        return None
    expected = _resume_signature(current_run_meta)
    best: Optional[Path] = None
    best_step = -1
    saw_candidates = False

    for p in sorted(ckpt_parent.glob("*/last_model.pt")):
        saw_candidates = True
        cfg_path = p.parent / "run_config.json"
        if not cfg_path.exists():
            print(f"[resume] skipping {p}: missing sibling run_config.json")
            continue
        try:
            cand_meta = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[resume] skipping {p}: unreadable run_config.json ({exc})")
            continue
        if not isinstance(cand_meta, dict):
            print(f"[resume] skipping {p}: invalid run_config.json (expected JSON object)")
            continue

        cand_sig = _resume_signature(cand_meta)
        mismatches = [
            f"{k} (ckpt={cand_sig[k]!r} != current={expected[k]!r})"
            for k in expected
            if cand_sig[k] != expected[k]
        ]
        if mismatches:
            print(f"[resume] skipping {p}: config mismatch — {'; '.join(mismatches)}")
            continue

        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            step = int(ckpt.get("global_step", 0))
            if step > best_step:
                best, best_step = p, step
        except Exception as exc:
            print(f"[resume] skipping {p}: unreadable checkpoint ({exc})")
            continue

    if best is None and saw_candidates:
        print("[resume] no matching checkpoint found; starting fresh")
    return best


def _find_stale_run_dir(
    ckpt_parent: Path,
    current_run_meta: Mapping[str, Any],
) -> Optional[Path]:
    """A matching run dir killed before its first last_model.pt (no bench.json).

    Reusing it keeps the one-run-dir-per-job invariant across preemptions that
    land before the first PPO update; otherwise each such restart leaves an
    orphan dir (run_config.json, maybe an untrained best_model.pt) behind.
    """
    if not ckpt_parent.exists():
        return None
    expected = _resume_signature(current_run_meta)
    for d in sorted((p for p in ckpt_parent.iterdir() if p.is_dir()), reverse=True):
        if (d / "last_model.pt").exists() or (d / "bench.json").exists():
            continue
        try:
            cand = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(cand, dict) and _resume_signature(cand) == expected:
            return d
    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def layer_init(layer: nn.Linear, std: float = math.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def flatten_obs_dict(obs_dict: Dict[str, np.ndarray], tls_ids: Sequence[str]) -> np.ndarray:
    parts: List[np.ndarray] = []
    for tls_id in tls_ids:
        parts.append(np.asarray(obs_dict[tls_id], dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def extract_global_state(info_dict: Dict[str, Dict[str, Any]], fallback: np.ndarray) -> np.ndarray:
    for payload in info_dict.values():
        gs = payload.get("global_state")
        if gs is not None:
            arr = np.asarray(gs, dtype=np.float32).reshape(-1)
            if arr.size > 0:
                return arr
    return np.asarray(fallback, dtype=np.float32).reshape(-1)


def extract_step_emissions(info_dict: Dict[str, Dict[str, Any]]) -> Tuple[float, float]:
    """sum CO2 and fuel across all tls entries in info_dict.
    FIX: these keys are now set by multi_agent.py step() info_dict.
    """
    co2 = 0.0
    fuel = 0.0
    for payload in info_dict.values():
        co2 += float(payload.get("co2_mg_per_s", 0.0))
        fuel += float(payload.get("fuel_ml_per_s", 0.0))
    return co2, fuel


def extract_step_waiting(info_dict: Dict[str, Dict[str, Any]]) -> float:
    """mean avg_waiting_proxy across all tls entries."""
    vals = [float(p.get("avg_waiting_proxy", 0.0)) for p in info_dict.values()]
    return float(np.mean(vals)) if vals else 0.0


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    var_y = np.var(y_true)
    if var_y < 1e-8:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_y)


def get_entropy_coef(step: int, total: int, start: float, end: float) -> float:
    """linearly decay entropy coefficient."""
    frac = min(step / max(total, 1), 1.0)
    return start + frac * (end - start)


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(obs))


class Critic(nn.Module):
    """Centralized critic with one value head per agent (MAPPO v3).

    Output shape (B, num_agents): head i estimates agent i's return from the
    global state, enabling per-agent GAE on per-agent shaped rewards.
    """

    def __init__(self, state_dim: int, num_agents: int = 1, hidden_dim: int = 256) -> None:
        super().__init__()
        self.num_agents = int(num_agents)
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, self.num_agents), std=1.0),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class IndependentCritics(nn.Module):
    """IPPO baseline: one fully independent critic per agent.

    Each head consumes ONLY that agent's local 26-dim approach-aggregated
    observation — no global state, no parameter sharing between critics.
    Output shape (B, num_agents), same contract as the centralized Critic,
    so the PPO update path is identical for both algorithms.
    """

    def __init__(self, obs_dim: int, num_agents: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.num_agents = int(num_agents)
        self.nets = nn.ModuleList(
            nn.Sequential(
                layer_init(nn.Linear(obs_dim, hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, 1), std=1.0),
            )
            for _ in range(self.num_agents)
        )

    def forward(self, obs_per_agent: torch.Tensor) -> torch.Tensor:
        """obs_per_agent: (B, num_agents, obs_dim) -> values (B, num_agents)."""
        outs = [net(obs_per_agent[:, i, :]) for i, net in enumerate(self.nets)]
        return torch.cat(outs, dim=-1)


# ---------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------
@dataclass(slots=True)
class StepRecord:
    obs_dict: Dict[str, np.ndarray]
    global_state: np.ndarray
    action_dict: Dict[str, int]
    logprob_dict: Dict[str, float]
    reward_dict: Dict[str, float]
    terminated_dict: Dict[str, bool]
    truncated_dict: Dict[str, bool]
    value: np.ndarray          # v3: per-agent value vector, shape (num_agents,)
    team_reward: float         # kept for logging only


class RolloutBuffer:
    def __init__(self, tls_ids: Sequence[str]) -> None:
        self.tls_ids = list(tls_ids)
        self.records: List[StepRecord] = []

    def add(self, record: StepRecord) -> None:
        self.records.append(record)

    def clear(self) -> None:
        self.records.clear()

    def __len__(self) -> int:
        return len(self.records)

    def actor_flatten(
        self,
        advantages_by_tls: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        obs_list: List[np.ndarray] = []
        action_list: List[int] = []
        logprob_list: List[float] = []
        adv_list: List[float] = []

        T = len(self.records)
        for tls_id in self.tls_ids:
            adv_seq = advantages_by_tls[tls_id]
            for t in range(T):
                rec = self.records[t]
                obs_list.append(np.asarray(rec.obs_dict[tls_id], dtype=np.float32).reshape(-1))
                action_list.append(int(rec.action_dict[tls_id]))
                logprob_list.append(float(rec.logprob_dict[tls_id]))
                adv_list.append(float(adv_seq[t]))

        return {
            "obs": np.asarray(obs_list, dtype=np.float32),
            "actions": np.asarray(action_list, dtype=np.int64),
            "logprobs": np.asarray(logprob_list, dtype=np.float32),
            "advantages": np.asarray(adv_list, dtype=np.float32),
        }

    def critic_flatten(self, returns_mat: np.ndarray) -> Dict[str, np.ndarray]:
        """returns_mat: per-agent returns, shape (T, num_agents)."""
        states = [np.asarray(rec.global_state, dtype=np.float32).reshape(-1) for rec in self.records]
        values = np.stack(
            [np.asarray(rec.value, dtype=np.float32).reshape(-1) for rec in self.records],
            axis=0,
        )  # (T, num_agents)
        return {
            "states": np.asarray(states, dtype=np.float32),
            "values": values,
            "returns": np.asarray(returns_mat, dtype=np.float32),
        }


# ---------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------
def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------
def sync_obs_mode_from_checkpoint(args: argparse.Namespace) -> None:
    """For eval modes: read obs_mode, obs_ablation, and upstream_phase_obs from run_config.json.

    The privileged arm (38-dim vs 26-dim), VI-F obs ablations (no_class_shares,
    no_pressure_feature, lane_truncated), and n2_corridor coordination arm
    (30-dim vs 26-dim) change the actor's input dimensionality or feature layout,
    so evaluating a checkpoint with mismatched flags would otherwise fail or
    corrupt inference. Resolve all of them from the run_config sitting next to
    the checkpoint (matching eval_compare._obs_overrides_for_checkpoint).
    """
    ckpt = getattr(args, "checkpoint", None)
    if not ckpt:
        return
    cfg_path = Path(ckpt).parent / "run_config.json"
    if not cfg_path.exists():
        return
    try:
        run_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        obs_cfg = run_cfg.get("env_cfg", {}).get("observation", {})
        mode = obs_cfg.get("obs_mode")
    except Exception:
        return
    if mode in ("proxy", "privileged") and mode != getattr(args, "obs_mode", "proxy"):
        print(f"[eval] obs_mode resolved to {mode!r} from {cfg_path}")
        args.obs_mode = mode
    if "upstream_phase_obs" in run_cfg:
        want_upstream = bool(run_cfg["upstream_phase_obs"])
        if want_upstream != bool(getattr(args, "upstream_phase_obs", False)):
            print(f"[eval] upstream_phase_obs resolved to {want_upstream!r} from {cfg_path}")
            args.upstream_phase_obs = want_upstream
    elif isinstance(obs_cfg.get("tls_feature_names"), list):
        want_upstream = any(
            str(n).startswith("upstream_phase_") for n in obs_cfg["tls_feature_names"]
        )
        if want_upstream != bool(getattr(args, "upstream_phase_obs", False)):
            print(f"[eval] upstream_phase_obs resolved to {want_upstream!r} from {cfg_path}")
            args.upstream_phase_obs = want_upstream

    obs_ablation = run_cfg.get("obs_ablation")
    if obs_ablation not in OBS_ABLATIONS:
        if obs_cfg.get("lane_truncated"):
            obs_ablation = "lane_truncated"
        else:
            lane_feats = obs_cfg.get("lane_feature_names")
            tls_feats = obs_cfg.get("tls_feature_names")
            if (
                isinstance(lane_feats, list)
                and "motorbike_share" not in lane_feats
                and "heavy_vehicle_share" not in lane_feats
            ):
                obs_ablation = "no_class_shares"
            elif isinstance(tls_feats, list) and "pressure_norm" not in tls_feats:
                obs_ablation = "no_pressure_feature"
            else:
                obs_ablation = None
    if obs_ablation in OBS_ABLATIONS and obs_ablation != getattr(args, "obs_ablation", "none"):
        print(f"[eval] obs_ablation resolved to {obs_ablation!r} from {cfg_path}")
        args.obs_ablation = obs_ablation


def make_env(args: argparse.Namespace) -> Tuple[MappoTrafficEnv, Any]:
    # scaled networks (n2_corridor, n3_grid, ...) ship a lane_groups.json that
    # provides both the TLS roster and the phase-group partition; explicit
    # --tls-ids still wins if given.
    tls_ids: Optional[Tuple[str, ...]] = tuple(args.tls_ids) if args.tls_ids else None
    manual_lane_groups = None
    lane_groups_path = getattr(args, "lane_groups", None)
    if lane_groups_path:
        json_tls_ids, manual_lane_groups = load_lane_groups_json(lane_groups_path)
        if tls_ids is None:
            tls_ids = json_tls_ids

    ablation_kwargs: Dict[str, Any] = {}
    ablation_kwargs.update(_reward_preset_kwargs(getattr(args, "reward_preset", "full")))
    ablation_kwargs.update(_obs_ablation_kwargs(getattr(args, "obs_ablation", "none")))

    env_cfg = build_default_config(
        sumo_cfg_path=args.sumo_cfg,
        gui=args.gui,
        tls_ids=tls_ids,
        manual_lane_groups=manual_lane_groups,
        max_lanes_per_tls=args.max_lanes_per_tls,
        step_length=args.step_length,
        yellow_time=args.yellow_time,
        max_steps=args.max_steps,
        min_green_time=args.min_green_time,
        max_green_time=args.max_green_time,
        queue_cap=args.queue_cap,
        waiting_cap=args.waiting_cap,
        speed_cap=args.speed_cap,
        use_external_state=False,
        obs_mode=getattr(args, "obs_mode", "proxy"),
        upstream_phase_obs=getattr(args, "upstream_phase_obs", False),
        debug=args.debug,
        reward_scale=args.reward_scale,
        **ablation_kwargs,
    )
    env = MappoTrafficEnv(
        config=env_cfg,
        gui=args.gui,
        use_libsumo=(not args.gui) and (not args.no_libsumo),
        no_step_log=True,
        waiting_time_memory=1000,
        print_warnings=False,
        add_default_flags=True,
    )
    return env, env_cfg


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    train_cfg = TrainConfig(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        rollout_horizon=args.rollout_horizon,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        target_kl=args.target_kl,
        policy_lr=args.policy_lr,
        value_lr=args.value_lr,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        entropy_coef_start=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
        save_every_steps=args.save_every_steps,
        save_latest_every_updates=args.save_latest_every_updates,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
    )

    set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)

    env, env_cfg = make_env(args)
    tls_ids = list(env_cfg.tls_ids)
    obs_dim = env_cfg.local_obs_dim
    state_dim = env_cfg.global_state_dim
    algo = str(getattr(args, "algo", "mappo")).lower()

    # Resolve auto-resume BEFORE creating any timestamped run dir so Spot
    # preemption restarts continue IN PLACE (single run_id, ckpt_dir, log_dir,
    # best_model.pt, and bench.json per job).
    probe_meta = {
        "algo": algo,
        "critic_arch": "independent_local" if algo == "ippo" else "central_multihead",
        "reward_preset": getattr(args, "reward_preset", "full"),
        "obs_ablation": getattr(args, "obs_ablation", "none"),
        "upstream_phase_obs": bool(getattr(args, "upstream_phase_obs", False)),
        "shared_reward": bool(getattr(args, "shared_reward", False)),
        "lane_groups_file": getattr(args, "lane_groups", None),
        "env_cfg": env_cfg.to_dict(),
        "train_cfg": asdict(train_cfg),
    }
    _resume_ckpt = _find_latest_checkpoint(Path(train_cfg.ckpt_dir), probe_meta)
    _resume_payload: Optional[Dict[str, Any]] = None
    resume_episode_count = 0
    if _resume_ckpt is not None:
        run_id = _resume_ckpt.parent.name
        ckpt_dir = str(_resume_ckpt.parent)
        log_dir = str(Path(train_cfg.log_dir) / run_id)
        print(f"[resume] found matching checkpoint at {_resume_ckpt} (reusing run_id={run_id} in place)")
        _resume_payload = torch.load(_resume_ckpt, map_location=device, weights_only=False)
        resume_episode_count = int(_resume_payload.get("episode_count", 0))
    elif (_stale := _find_stale_run_dir(Path(train_cfg.ckpt_dir), probe_meta)) is not None:
        # Killed before the first checkpoint: restart from scratch in the same dir,
        # clearing its partial artefacts (e.g. an untrained best_model.pt, logs).
        run_id = _stale.name
        ckpt_dir = str(_stale)
        log_dir = str(Path(train_cfg.log_dir) / run_id)
        print(f"[resume] reusing pre-checkpoint run dir {_stale} (starting fresh in place)")
        for _d in {Path(ckpt_dir), Path(log_dir)}:
            if _d.exists():
                shutil.rmtree(_d)
    else:
        base_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = base_run_id
        _run_suffix = 1
        while (
            (Path(train_cfg.ckpt_dir) / run_id).exists()
            or (Path(train_cfg.log_dir) / run_id).exists()
        ):
            run_id = f"{base_run_id}_{_run_suffix}"
            _run_suffix += 1
        log_dir = str(Path(train_cfg.log_dir) / run_id)
        ckpt_dir = str(Path(train_cfg.ckpt_dir) / run_id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # action_dim: do a quick reset to let env build phase maps, then read it
    obs_dict, info_dict = env.reset(seed=train_cfg.seed + resume_episode_count)
    action_dim: int = getattr(env, "action_dim", 2)

    actor = Actor(obs_dim, action_dim).to(device)
    if algo == "ippo":
        # IPPO: decentralized critics — each value head sees ONLY its agent's
        # local obs. Actor parameter sharing stays active (same as MAPPO).
        critic: nn.Module = IndependentCritics(obs_dim, num_agents=len(tls_ids)).to(device)
    else:
        critic = Critic(state_dim, num_agents=len(tls_ids)).to(device)

    actor_optim = optim.Adam(actor.parameters(), lr=train_cfg.policy_lr, eps=1e-5)
    critic_optim = optim.Adam(critic.parameters(), lr=train_cfg.value_lr, eps=1e-5)

    total_updates = train_cfg.total_timesteps // train_cfg.rollout_horizon

    def lr_lambda(update: int) -> float:
        return max(1.0 - update / max(total_updates, 1), 0.05)

    actor_scheduler = optim.lr_scheduler.LambdaLR(actor_optim, lr_lambda)
    critic_scheduler = optim.lr_scheduler.LambdaLR(critic_optim, lr_lambda)

    obs_rms = RunningMeanStd((obs_dim,))
    state_rms = RunningMeanStd((state_dim,))

    writer = make_writer(log_dir)
    csv_logger = CsvLogger(log_dir)

    run_meta = {
        "run_id": run_id,
        **probe_meta,
        "tls_ids": tls_ids,
        "obs_dim": obs_dim,
        "state_dim": state_dim,
        "action_dim": action_dim,
    }
    for save_dir in (log_dir, ckpt_dir):
        _atomic_json_dump(run_meta, Path(save_dir) / "run_config.json")

    buffer = RolloutBuffer(tls_ids)
    global_step = 0
    episode_count = 0
    update_idx = 0
    best_mean_reward = -float("inf")
    prior_elapsed_s = 0.0
    start_time = time.time()

    if _resume_payload is not None:
        _c = _resume_payload
        actor.load_state_dict(_c["actor_state_dict"])
        critic.load_state_dict(_c["critic_state_dict"])
        actor_optim.load_state_dict(_c["actor_optim_state_dict"])
        critic_optim.load_state_dict(_c["critic_optim_state_dict"])
        actor_scheduler.load_state_dict(_c["actor_scheduler_state_dict"])
        critic_scheduler.load_state_dict(_c["critic_scheduler_state_dict"])
        if "obs_rms" in _c:
            obs_rms.load_state_dict(_c["obs_rms"])
        if "state_rms" in _c:
            state_rms.load_state_dict(_c["state_rms"])
        global_step = int(_c.get("global_step", 0))
        episode_count = int(_c.get("episode_count", 0))
        update_idx = int(_c.get("update_idx", global_step // max(train_cfg.rollout_horizon, 1)))
        if "best_mean_reward" in _c and _c["best_mean_reward"] is not None:
            best_mean_reward = float(_c["best_mean_reward"])
        prior_elapsed_s = float(_c.get("elapsed_seconds_prior", 0.0))
        try:
            if _c.get("python_rng_state") is not None:
                random.setstate(_c["python_rng_state"])
            if _c.get("numpy_rng_state") is not None:
                np.random.set_state(_c["numpy_rng_state"])
            if _c.get("torch_rng_state") is not None:
                rng_t = _c["torch_rng_state"]
                torch.set_rng_state(rng_t.cpu() if isinstance(rng_t, torch.Tensor) else rng_t)
            if torch.cuda.is_available() and _c.get("torch_cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(_c["torch_cuda_rng_state"])
        except Exception as exc:
            print(f"[resume] warning: could not restore RNG state ({exc})")
        print(
            f"[resume] restored run_id={run_id} global_step={global_step}, "
            f"episode_count={episode_count}, update_idx={update_idx}, "
            f"best_mean_reward={best_mean_reward}"
        )

    def _build_checkpoint_payload() -> Dict[str, Any]:
        cur_entropy_coef = get_entropy_coef(
            global_step, train_cfg.total_timesteps,
            train_cfg.entropy_coef_start, train_cfg.entropy_coef_end,
        )
        return {
            "run_id": run_id,
            "algo": algo,
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "actor_optim_state_dict": actor_optim.state_dict(),
            "critic_optim_state_dict": critic_optim.state_dict(),
            "actor_scheduler_state_dict": actor_scheduler.state_dict(),
            "critic_scheduler_state_dict": critic_scheduler.state_dict(),
            "global_step": global_step,
            "episode_count": episode_count,
            "update_idx": update_idx,
            "best_mean_reward": best_mean_reward,
            "entropy_coef": cur_entropy_coef,
            "entropy_coef_start": train_cfg.entropy_coef_start,
            "entropy_coef_end": train_cfg.entropy_coef_end,
            "elapsed_seconds_prior": prior_elapsed_s + (time.time() - start_time),
            "env_cfg": env_cfg.to_dict(),
            "train_cfg": asdict(train_cfg),
            "obs_dim": obs_dim,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "obs_rms": obs_rms.state_dict(),
            "state_rms": state_rms.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }

    _term_requested = False
    _term_sig_name = ""
    _prev_sigterm = None
    if hasattr(signal, "SIGTERM"):
        def _on_sigterm(signum: int, _frame: Any) -> None:
            nonlocal _term_requested, _term_sig_name
            _term_requested = True
            _term_sig_name = (
                signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
            )

        try:
            _prev_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
        except ValueError:
            _prev_sigterm = None

    global_state = extract_global_state(info_dict, fallback=flatten_obs_dict(obs_dict, tls_ids))

    episode_length = 0
    episode_agent_rewards: Dict[str, float] = {tls_id: 0.0 for tls_id in tls_ids}
    episode_queue_total = 0.0
    episode_co2 = 0.0
    episode_fuel = 0.0
    episode_throughput = 0.0
    episode_waiting = 0.0
    ep_prev_passed = 0.0

    def reset_episode_stats() -> None:
        nonlocal episode_length, episode_agent_rewards, episode_queue_total
        nonlocal episode_co2, episode_fuel, episode_throughput, ep_prev_passed, episode_waiting
        episode_length = 0
        episode_agent_rewards = {tls_id: 0.0 for tls_id in tls_ids}
        episode_queue_total = 0.0
        episode_co2 = 0.0
        episode_fuel = 0.0
        episode_throughput = 0.0
        episode_waiting = 0.0
        ep_prev_passed = 0.0

    try:
        while global_step < train_cfg.total_timesteps and not _term_requested:
            buffer.clear()

            actor.eval()
            critic.eval()

            for _ in range(train_cfg.rollout_horizon):
                if _term_requested:
                    break
                action_dict: Dict[str, int] = {}
                logprob_dict: Dict[str, float] = {}
                norm_obs_rows: List[np.ndarray] = []

                for tls_id in tls_ids:
                    obs = np.asarray(obs_dict[tls_id], dtype=np.float32).reshape(-1)
                    norm_obs = obs_rms.normalize(obs)
                    norm_obs_rows.append(norm_obs)
                    obs_tensor = torch.as_tensor(norm_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        dist = actor.distribution(obs_tensor)
                        action = dist.sample()
                        logprob = dist.log_prob(action)

                    action_dict[tls_id] = int(action.item())
                    logprob_dict[tls_id] = float(logprob.item())

                with torch.no_grad():
                    if algo == "ippo":
                        obs_all_t = torch.as_tensor(
                            np.stack(norm_obs_rows, axis=0), dtype=torch.float32, device=device
                        ).unsqueeze(0)  # (1, N, obs_dim)
                        value = critic(obs_all_t).squeeze(0).cpu().numpy().astype(np.float32)
                    else:
                        norm_state = state_rms.normalize(global_state)
                        state_tensor = torch.as_tensor(
                            norm_state, dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        value = critic(state_tensor).squeeze(0).cpu().numpy().astype(np.float32)

                next_obs_dict, reward_dict, terminated_dict, truncated_dict, next_info_dict = env.step(action_dict)
                next_global_state = extract_global_state(next_info_dict, fallback=global_state)

                team_reward = float(np.mean(list(reward_dict.values()))) if reward_dict else 0.0

                rec = StepRecord(
                    obs_dict={k: np.asarray(v, dtype=np.float32).copy() for k, v in obs_dict.items()},
                    global_state=np.asarray(global_state, dtype=np.float32).copy(),
                    action_dict=dict(action_dict),
                    logprob_dict=dict(logprob_dict),
                    reward_dict={k: float(v) for k, v in reward_dict.items()},
                    terminated_dict={k: bool(v) for k, v in terminated_dict.items()},
                    truncated_dict={k: bool(v) for k, v in truncated_dict.items()},
                    value=value,
                    team_reward=team_reward,
                )
                buffer.add(rec)

                episode_length += 1
                global_step += 1

                for tls_id in tls_ids:
                    episode_agent_rewards[tls_id] += float(reward_dict.get(tls_id, 0.0))
                    episode_queue_total += float(next_info_dict.get(tls_id, {}).get("queue_total_proxy", 0.0))

                step_co2, step_fuel = extract_step_emissions(next_info_dict)
                episode_co2 += step_co2
                episode_fuel += step_fuel
                episode_waiting += extract_step_waiting(next_info_dict)

                # FIX: current_passed now set by env; reliable throughput tracking
                first_info = next(iter(next_info_dict.values()), {})
                current_passed = float(first_info.get("current_passed", ep_prev_passed))
                episode_throughput += max(current_passed - ep_prev_passed, 0.0)
                ep_prev_passed = current_passed

                obs_dict = next_obs_dict
                global_state = next_global_state

                done_now = any(terminated_dict.values()) or any(truncated_dict.values())
                if done_now:
                    episode_count += 1
                    mean_episode_reward = float(np.mean(list(episode_agent_rewards.values())))

                    writer.add_scalar("episode/mean_reward", mean_episode_reward, global_step)
                    writer.add_scalar("episode/length", episode_length, global_step)
                    writer.add_scalar("episode/queue_total_proxy", episode_queue_total, global_step)
                    writer.add_scalar("episode/co2_total_mg", episode_co2, global_step)
                    writer.add_scalar("episode/fuel_total_ml", episode_fuel, global_step)
                    writer.add_scalar("episode/throughput", episode_throughput, global_step)
                    writer.add_scalar("episode/avg_waiting_proxy", episode_waiting, global_step)

                    csv_logger.log_episode(
                        global_step=global_step,
                        episode=episode_count,
                        episode_length=episode_length,
                        mean_reward=mean_episode_reward,
                        queue_total_proxy=episode_queue_total,
                        co2_total_mg=episode_co2,
                        fuel_total_ml=episode_fuel,
                        throughput=episode_throughput,
                        avg_waiting_proxy=episode_waiting,
                    )

                    if mean_episode_reward > best_mean_reward:
                        best_mean_reward = mean_episode_reward
                        _atomic_torch_save(_build_checkpoint_payload(), Path(ckpt_dir) / "best_model.pt")

                    if not _term_requested and global_step < train_cfg.total_timesteps:
                        obs_dict, info_dict = env.reset(seed=train_cfg.seed + episode_count)
                        global_state = extract_global_state(info_dict, fallback=flatten_obs_dict(obs_dict, tls_ids))
                    reset_episode_stats()

                if global_step >= train_cfg.total_timesteps or _term_requested:
                    break

            if _term_requested:
                break

            if len(buffer) == 0:
                continue

            # --------------------------------------------------
            # Update RunningMeanStd with rollout batch
            # --------------------------------------------------
            all_obs = np.stack([
                np.asarray(rec.obs_dict[tls_id], dtype=np.float32).reshape(-1)
                for rec in buffer.records
                for tls_id in tls_ids
            ], axis=0)
            obs_rms.update(all_obs)

            all_states = np.stack([
                np.asarray(rec.global_state, dtype=np.float32).reshape(-1)
                for rec in buffer.records
            ], axis=0)
            state_rms.update(all_states)

            # --------------------------------------------------
            # Bootstrap value for last state.
            # --------------------------------------------------
            actor.eval()
            critic.eval()
            with torch.no_grad():
                if algo == "ippo":
                    last_obs_rows = np.stack(
                        [
                            obs_rms.normalize(np.asarray(obs_dict[t], dtype=np.float32).reshape(-1))
                            for t in tls_ids
                        ],
                        axis=0,
                    )
                    last_in = torch.as_tensor(last_obs_rows, dtype=torch.float32, device=device).unsqueeze(0)
                else:
                    norm_last = state_rms.normalize(global_state)
                    last_in = torch.as_tensor(norm_last, dtype=torch.float32, device=device).unsqueeze(0)
                last_values = critic(last_in).squeeze(0).cpu().numpy().astype(np.float32)

            last_record = buffer.records[-1]
            if any(last_record.terminated_dict.values()) or any(last_record.truncated_dict.values()):
                last_values = np.zeros_like(last_values)

            # --------------------------------------------------
            # GAE — v3: per-agent advantages from per-agent shaped rewards
            # and per-agent value heads (episode boundaries are shared).
            # --------------------------------------------------
            dones = np.asarray([
                float(any(rec.terminated_dict.values()) or any(rec.truncated_dict.values()))
                for rec in buffer.records
            ], dtype=np.float32)

            rewards_mat = np.asarray(
                [[rec.reward_dict.get(tls_id, 0.0) for tls_id in tls_ids] for rec in buffer.records],
                dtype=np.float32,
            )  # (T, N)
            values_mat = np.stack(
                [np.asarray(rec.value, dtype=np.float32).reshape(-1) for rec in buffer.records],
                axis=0,
            )  # (T, N)

            # shared-reward: replace per-agent rewards with team mean so all agents
            # receive the same signal — incentivises coordination (e.g. green-wave
            # on corridor networks) at the cost of per-agent credit assignment.
            if getattr(args, "shared_reward", False):
                team_rewards = rewards_mat.mean(axis=1, keepdims=True)          # (T,1)
                rewards_mat = np.repeat(team_rewards, len(tls_ids), axis=1)     # (T,N)

            adv_cols: List[np.ndarray] = []
            ret_cols: List[np.ndarray] = []
            for agent_idx in range(len(tls_ids)):
                adv_i, ret_i = compute_gae(
                    rewards=rewards_mat[:, agent_idx],
                    values=values_mat[:, agent_idx],
                    dones=dones,
                    last_value=float(last_values[agent_idx]),
                    gamma=train_cfg.gamma,
                    gae_lambda=train_cfg.gae_lambda,
                )
                adv_cols.append(adv_i)
                ret_cols.append(ret_i)

            adv_mat = np.stack(adv_cols, axis=1)        # (T, N)
            returns_mat = np.stack(ret_cols, axis=1)    # (T, N)

            # joint normalization keeps a single advantage scale for the shared actor
            adv_flat = adv_mat.reshape(-1)
            adv_mat_norm = (adv_mat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

            advantages_by_tls: Dict[str, np.ndarray] = {
                tls_id: np.ascontiguousarray(adv_mat_norm[:, agent_idx])
                for agent_idx, tls_id in enumerate(tls_ids)
            }

            actor_flat = buffer.actor_flatten(advantages_by_tls)
            critic_flat = buffer.critic_flatten(returns_mat=returns_mat)

            actor_flat["obs"] = obs_rms.normalize(actor_flat["obs"])
            critic_flat["states"] = state_rms.normalize(critic_flat["states"])

            obs_t = torch.as_tensor(actor_flat["obs"], dtype=torch.float32, device=device)
            action_t = torch.as_tensor(actor_flat["actions"], dtype=torch.int64, device=device)
            old_logprob_t = torch.as_tensor(actor_flat["logprobs"], dtype=torch.float32, device=device)
            adv_t = torch.as_tensor(actor_flat["advantages"], dtype=torch.float32, device=device)

            # critic input: MAPPO -> normalized global state (T, S);
            # IPPO -> per-agent normalized local obs (T, N, obs_dim). actor_flat
            # obs is agent-major [(agent0 t0..T-1), (agent1 ...)], already
            # normalized above — reshape instead of re-normalizing.
            if algo == "ippo":
                T_steps = len(buffer)
                critic_in = (
                    actor_flat["obs"]
                    .reshape(len(tls_ids), T_steps, obs_dim)
                    .transpose(1, 0, 2)
                )  # (T, N, obs_dim)
            else:
                critic_in = critic_flat["states"]

            state_t = torch.as_tensor(critic_in, dtype=torch.float32, device=device)
            old_value_t = torch.as_tensor(critic_flat["values"], dtype=torch.float32, device=device)   # (T, N)
            ret_t = torch.as_tensor(critic_flat["returns"], dtype=torch.float32, device=device)        # (T, N)

            # --------------------------------------------------
            # PPO update
            # --------------------------------------------------
            actor.train()
            critic.train()

            entropy_coef = get_entropy_coef(
                global_step, train_cfg.total_timesteps,
                train_cfg.entropy_coef_start, train_cfg.entropy_coef_end,
            )

            policy_losses, value_losses, entropies, kls, clip_fracs = [], [], [], [], []

            batch_size_actor = obs_t.shape[0]
            batch_size_critic = state_t.shape[0]

            actor_indices = np.arange(batch_size_actor)
            skip_actor = False

            for _ in range(train_cfg.update_epochs):

                if not skip_actor:
                    actor_indices = np.random.permutation(batch_size_actor)

                    for start in range(0, batch_size_actor, train_cfg.minibatch_size):
                        mb_idx = actor_indices[start:start + train_cfg.minibatch_size]

                        dist = actor.distribution(obs_t[mb_idx])
                        new_logprob = dist.log_prob(action_t[mb_idx])
                        entropy = dist.entropy().mean()

                        logratio = new_logprob - old_logprob_t[mb_idx]
                        ratio = torch.exp(logratio)

                        with torch.no_grad():
                            approx_kl = ((ratio - 1.0) - logratio).mean().item()
                            clip_frac = ((ratio - 1.0).abs() > train_cfg.clip_ratio).float().mean().item()

                        surr1 = ratio * adv_t[mb_idx]
                        surr2 = torch.clamp(ratio, 1.0 - train_cfg.clip_ratio, 1.0 + train_cfg.clip_ratio) * adv_t[mb_idx]
                        policy_loss = -torch.min(surr1, surr2).mean()

                        loss_actor = policy_loss - entropy_coef * entropy

                        actor_optim.zero_grad(set_to_none=True)
                        loss_actor.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), train_cfg.max_grad_norm)
                        actor_optim.step()

                        policy_losses.append(policy_loss.item())
                        entropies.append(entropy.item())
                        kls.append(approx_kl)
                        clip_fracs.append(clip_frac)

                    if np.mean(kls) > 1.5 * train_cfg.target_kl:
                        skip_actor = True
                        writer.add_scalar("debug/early_stop_kl", 1.0, global_step)

                critic_indices = np.random.permutation(batch_size_critic)
                critic_mb_size = max(train_cfg.minibatch_size, 1)

                for start in range(0, batch_size_critic, critic_mb_size):
                    mb_idx = critic_indices[start:start + critic_mb_size]

                    value_pred = critic(state_t[mb_idx])

                    # FIX: use clip_range_vf (not clip_ratio) for value loss clipping
                    value_pred_clipped = old_value_t[mb_idx] + torch.clamp(
                        value_pred - old_value_t[mb_idx],
                        -train_cfg.clip_range_vf,
                        train_cfg.clip_range_vf,
                    )

                    value_loss_unclipped = (ret_t[mb_idx] - value_pred).pow(2)
                    value_loss_clipped = (ret_t[mb_idx] - value_pred_clipped).pow(2)

                    value_loss = train_cfg.value_coef * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                    critic_optim.zero_grad(set_to_none=True)
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), train_cfg.max_grad_norm)
                    critic_optim.step()

                    value_losses.append(value_loss.item())

            actor_scheduler.step()
            critic_scheduler.step()
            current_lr = actor_scheduler.get_last_lr()[0]

            # --------------------------------------------------
            # Logging
            # --------------------------------------------------
            total_elapsed = prior_elapsed_s + (time.time() - start_time)
            sps = int(global_step / max(total_elapsed, 1e-8))
            ev = explained_variance(
                critic_flat["values"].reshape(-1), returns_mat.reshape(-1)
            )

            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("loss/policy", float(np.mean(policy_losses)), global_step)
            writer.add_scalar("loss/value", float(np.mean(value_losses)), global_step)
            writer.add_scalar("loss/entropy", float(np.mean(entropies)), global_step)
            writer.add_scalar("loss/approx_kl", float(np.mean(kls)), global_step)
            writer.add_scalar("loss/clip_fraction", float(np.mean(clip_fracs)), global_step)
            writer.add_scalar("loss/explained_variance", ev, global_step)
            writer.add_scalar("train/lr", current_lr, global_step)
            writer.add_scalar("train/entropy_coef", entropy_coef, global_step)
            writer.add_scalar("train/actor_batch_size", batch_size_actor, global_step)
            writer.add_scalar("train/episode_count", episode_count, global_step)
            writer.add_scalar("train/return_mean", float(np.mean(returns_mat)), global_step)
            writer.add_scalar("train/adv_mean", float(np.mean(adv_mat)), global_step)
            for agent_idx, tls_id in enumerate(tls_ids):
                writer.add_scalar(f"train/return_mean_{tls_id}", float(np.mean(returns_mat[:, agent_idx])), global_step)

            csv_logger.log_update(
                global_step=global_step,
                episode=episode_count,
                update_idx=update_idx,
                policy_loss=float(np.mean(policy_losses)),
                value_loss=float(np.mean(value_losses)),
                entropy=float(np.mean(entropies)),
                approx_kl=float(np.mean(kls)),
                clip_fraction=float(np.mean(clip_fracs)),
                explained_variance=ev,
                sps=sps,
                lr=current_lr,
                entropy_coef=entropy_coef,
            )

            should_save_latest = (update_idx % train_cfg.save_latest_every_updates == 0)
            update_idx += 1

            # Periodic checkpoint (atomic)
            if global_step % train_cfg.save_every_steps < train_cfg.rollout_horizon:
                ckpt_path = Path(ckpt_dir) / f"checkpoint_{global_step}.pt"
                _atomic_torch_save(_build_checkpoint_payload(), ckpt_path)

            # Latest checkpoint for Spot preemption resume (atomic)
            if should_save_latest:
                _atomic_torch_save(_build_checkpoint_payload(), Path(ckpt_dir) / "last_model.pt")

            print(
                f"step={global_step:>8} | ep={episode_count:>4} | "
                f"pi={np.mean(policy_losses):>8.4f} | vf={np.mean(value_losses):>8.4f} | "
                f"ent={np.mean(entropies):>7.4f} | kl={np.mean(kls):>7.5f} | "
                f"clip={np.mean(clip_fracs):>6.3f} | lr={current_lr:.2e} | sps={sps:>5}"
            )

        if _term_requested:
            _atomic_torch_save(_build_checkpoint_payload(), Path(ckpt_dir) / "last_model.pt")
            print(
                f"[signal] received {_term_sig_name}; saved last_model.pt at "
                f"global_step={global_step} in {ckpt_dir}; exiting cleanly."
            )
            return

        # Guarantee final last_model.pt and at least one best_model.pt on completion
        final_payload = _build_checkpoint_payload()
        _atomic_torch_save(final_payload, Path(ckpt_dir) / "last_model.pt")
        best_path = Path(ckpt_dir) / "best_model.pt"
        if not best_path.exists():
            _atomic_torch_save(final_payload, best_path)

        # ----- wall-clock benchmark summary (pilot instrumentation) -----
        elapsed_s = prior_elapsed_s + (time.time() - start_time)
        mean_sps = global_step / max(elapsed_s, 1e-9)
        proj_2m_h = 2_000_000 / max(mean_sps, 1e-9) / 3600.0
        bench = {
            "run_id": run_id,
            "algo": algo,
            "num_agents": len(tls_ids),
            "global_steps": global_step,
            "episodes": episode_count,
            "elapsed_seconds": round(elapsed_s, 1),
            "mean_steps_per_second": round(mean_sps, 2),
            "projected_hours_2M_steps": round(proj_2m_h, 2),
            "projected_hours_2M_x5_seeds": round(proj_2m_h * 5, 2),
            "device": train_cfg.device,
            "ckpt_dir": ckpt_dir,
            "log_dir": log_dir,
        }
        for save_dir in (log_dir, ckpt_dir):
            _atomic_json_dump(bench, Path(save_dir) / "bench.json")

        print(f"Training completed. Run dir: {ckpt_dir}")
        print(
            f"[BENCH] wall-clock {elapsed_s:.1f}s ({elapsed_s/3600:.2f}h) | "
            f"steps {global_step} | mean SPS {mean_sps:.1f}"
        )
        print(
            f"[BENCH] projected: 2M steps ~ {proj_2m_h:.1f}h | "
            f"2M x 5 seeds ~ {proj_2m_h*5:.1f}h  (algo={algo}, agents={len(tls_ids)})"
        )
    finally:
        if _prev_sigterm is not None and hasattr(signal, "SIGTERM"):
            try:
                signal.signal(signal.SIGTERM, _prev_sigterm)
            except ValueError:
                pass
        try:
            env.close()
        except Exception:
            pass
        try:
            writer.close()
        except Exception:
            pass


# ---------------------------------------------------------------------
# Multi-seed evaluation helper (paper reproducibility)
# ---------------------------------------------------------------------
@torch.no_grad()
def run_multiseed_eval(
    args: argparse.Namespace,
    seeds: Sequence[int] = (42, 123, 456, 789, 1337),
) -> Dict[str, Any]:
    """
    Run evaluation across multiple seeds and report mean ± std.
    Required for Q2/Q1 paper submission.
    Saves evaluation_results.json next to the checkpoint.
    """
    sync_obs_mode_from_checkpoint(args)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_schema(
        args.checkpoint,
        min_version=MIN_SCHEMA_VERSION,
        allow_legacy=bool(getattr(args, "allow_legacy_schema", False)),
        ckpt=ckpt if isinstance(ckpt, dict) else None,
        require_version=True,
    )

    env, env_cfg = make_env(args)
    tls_ids = list(env_cfg.tls_ids)
    obs_dim = env_cfg.local_obs_dim
    action_dim = 2  # default; overridden after reset

    device = torch.device(args.device if hasattr(args, "device") else "cpu")
    actor = Actor(obs_dim, action_dim).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    obs_rms = RunningMeanStd((obs_dim,))
    if "obs_rms" in ckpt:
        obs_rms.load_state_dict(ckpt["obs_rms"])

    per_seed = []
    for seed in seeds:
        obs_dict, info_dict = env.reset(seed=seed)
        total_reward = 0.0
        total_steps = 0
        total_co2 = 0.0
        total_throughput = 0.0
        ep_prev_passed = 0.0
        done = False

        while not done and total_steps < args.eval_max_steps:
            action_dict: Dict[str, int] = {}
            for tls_id in tls_ids:
                obs = obs_rms.normalize(np.asarray(obs_dict[tls_id], dtype=np.float32).reshape(-1))
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                dist = actor.distribution(obs_t)
                action_dict[tls_id] = int(torch.argmax(dist.probs, dim=-1).item())

            next_obs, reward, terminated, truncated, info = env.step(action_dict)
            total_reward += float(np.mean(list(reward.values()))) if reward else 0.0
            step_co2, _ = extract_step_emissions(info)
            total_co2 += step_co2

            first_info = next(iter(info.values()), {})
            cur_passed = float(first_info.get("current_passed", ep_prev_passed))
            total_throughput += max(cur_passed - ep_prev_passed, 0.0)
            ep_prev_passed = cur_passed

            total_steps += 1
            done = any(terminated.values()) or any(truncated.values())
            obs_dict = next_obs

        per_seed.append({
            "seed": seed,
            "total_reward": total_reward,
            "episode_length": total_steps,
            "co2_total_mg": total_co2,
            "throughput": total_throughput,
        })
        print(f"seed={seed} | reward={total_reward:.3f} | steps={total_steps} | co2={total_co2:.1f}mg | tp={total_throughput:.0f}")

    env.close()

    rewards = [r["total_reward"] for r in per_seed]
    lengths = [r["episode_length"] for r in per_seed]
    co2s = [r["co2_total_mg"] for r in per_seed]

    results = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "mean_co2": float(np.mean(co2s)),
        "std_co2": float(np.std(co2s)),
        "seeds": list(seeds),
        "per_seed_results": per_seed,
    }

    # save evaluation_results.json next to the checkpoint
    ckpt_dir = str(Path(args.checkpoint).parent)
    with open(Path(ckpt_dir) / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(
        f"\nMulti-seed summary ({len(seeds)} seeds):\n"
        f"  reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}\n"
        f"  length: {results['mean_length']:.1f} ± {results['std_length']:.1f}\n"
        f"  CO2:    {results['mean_co2']:.1f} ± {results['std_co2']:.1f} mg\n"
    )
    return results


# ---------------------------------------------------------------------
# Single-run evaluation
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    sync_obs_mode_from_checkpoint(args)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_schema(
        args.checkpoint,
        min_version=MIN_SCHEMA_VERSION,
        allow_legacy=bool(getattr(args, "allow_legacy_schema", False)),
        ckpt=ckpt if isinstance(ckpt, dict) else None,
        require_version=True,
    )

    cfg = TrainConfig(
        seed=args.seed,
        device=args.device,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
    )
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    env, env_cfg = make_env(args)
    tls_ids = list(env_cfg.tls_ids)
    obs_dim = env_cfg.local_obs_dim
    action_dim = 2

    actor = Actor(obs_dim, action_dim).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    obs_rms = RunningMeanStd((obs_dim,))
    if "obs_rms" in ckpt:
        obs_rms.load_state_dict(ckpt["obs_rms"])

    for ep in range(cfg.eval_episodes):
        obs_dict, info_dict = env.reset(seed=cfg.seed + ep)
        global_state = extract_global_state(info_dict, fallback=flatten_obs_dict(obs_dict, tls_ids))

        total_reward = 0.0
        total_steps = 0
        done = False
        while not done and total_steps < cfg.eval_max_steps:
            action_dict: Dict[str, int] = {}
            for tls_id in tls_ids:
                obs = obs_rms.normalize(np.asarray(obs_dict[tls_id], dtype=np.float32).reshape(-1))
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                dist = actor.distribution(obs_t)
                action_dict[tls_id] = int(torch.argmax(dist.probs, dim=-1).item())

            next_obs, reward, terminated, truncated, info = env.step(action_dict)
            total_reward += float(np.mean(list(reward.values()))) if reward else 0.0
            total_steps += 1
            done = any(terminated.values()) or any(truncated.values())
            obs_dict = next_obs
            global_state = extract_global_state(info, fallback=global_state)

        print(f"eval_ep={ep + 1} | reward={total_reward:.3f} | steps={total_steps}")

    env.close()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean MAPPO trainer for SUMO traffic control")

    parser.add_argument("--mode", choices=["train", "eval", "multiseed_eval"], default="train")
    parser.add_argument("--algo", choices=["mappo", "ippo"], default="mappo",
                        help="mappo: centralized multi-head critic on global state; "
                             "ippo: independent per-agent critics on local obs only")
    parser.add_argument("--lane-groups", type=str, default=None,
                        help="path to lane_groups.json (scaled networks); provides "
                             "tls_ids + phase-group partition")
    parser.add_argument("--sumo-cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="models/mappo/20260418_215140/best_model.pt")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--rollout-horizon", type=int, default=128)   # FIX: aligned with TrainConfig
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=3e-4)       # FIX: aligned with TrainConfig
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--log-dir", type=str, default="logs/rl")
    parser.add_argument("--ckpt-dir", type=str, default="models/mappo")
    parser.add_argument("--save-every-steps", type=int, default=50_000)
    parser.add_argument("--save-latest-every-updates", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-max-steps", type=int, default=1080,
                        help="FIX v3: full 1080-step episode (was 400)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-libsumo", action="store_true",
                        help="Force traci subprocess mode (slower but crash-safe on large networks)")
    parser.add_argument("--step-length", type=int, default=5)
    parser.add_argument("--yellow-time", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1080)
    parser.add_argument("--min-green-time", type=int, default=15)
    parser.add_argument("--max-green-time", type=int, default=60)
    parser.add_argument("--queue-cap", type=float, default=50.0)
    parser.add_argument("--waiting-cap", type=float, default=300.0)
    parser.add_argument("--speed-cap", type=float, default=15.0)
    parser.add_argument("--max-lanes-per-tls", type=int, default=4)
    parser.add_argument("--obs-mode", choices=["proxy", "privileged"], default="proxy",
                        help="proxy=camera-computable obs (deployable); "
                             "privileged=proxy + exact-SUMO features (VI-C upper bound)")
    parser.add_argument("--reward-preset", choices=list(REWARD_PRESETS), default="full",
                        help="VI-F reward ablation (default 'full' = reward 1.3.0)")
    parser.add_argument("--obs-ablation", choices=list(OBS_ABLATIONS), default="none",
                        help="VI-F observation ablation (default 'none')")
    parser.add_argument("--shared-reward", action="store_true",
                        help="Replace per-agent rewards with team mean before GAE "
                             "(incentivises coordination; useful for corridor networks)")
    parser.add_argument("--upstream-phase-obs", action="store_true",
                        help="Append the mean phase one-hot of each agent's upstream "
                             "TLS neighbours to its observation (+4 dims/agent) — the "
                             "green-wave coordination signal a local obs lacks; for "
                             "corridor networks (n2_corridor)")
    parser.add_argument("--no-n2-coordination", action="store_true",
                        help="Skip auto-enabling --upstream-phase-obs and --shared-reward "
                             "on n2_corridor (train/eval the uncoordinated corridor base arm)")
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--allow-legacy-schema", action="store_true",
                        help=f"Allow eval checkpoints with env schema < {MIN_SCHEMA_VERSION}")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--tls-ids", nargs="*", default=[])

    args = parser.parse_args()
    _auto_enable_n2_coordination(args)
    return args


def _auto_enable_n2_coordination(args: argparse.Namespace) -> None:
    """Corridor networks need the green-wave coordination signal that local
    obs + local reward lack (myopic agents break the platoon chain and lose
    to Max-Pressure). Force both coordination flags whenever the target
    network is n2_corridor, unless --no-n2-coordination is passed. Applies to
    eval modes too: an n2 checkpoint trained with --upstream-phase-obs has a
    wider obs vector, so the eval env must be built with the flag or
    actor.load_state_dict fails (shared_reward is only read during GAE,
    harmless at eval)."""
    paths = (
        getattr(args, "sumo_cfg", None) or "",
        getattr(args, "lane_groups", None) or "",
    )
    if any("n2_corridor" in p for p in paths):
        if getattr(args, "no_n2_coordination", False):
            print("[INFO] N2 Corridor detected, --no-n2-coordination set: "
                  "skipping auto-enable of --upstream-phase-obs and --shared-reward.")
            return
        args.upstream_phase_obs = True
        args.shared_reward = True
        print("[INFO] N2 Corridor detected: Auto-enabling --upstream-phase-obs "
              "and --shared-reward for green-wave coordination.")


def main() -> None:
    _apply_omp_thread_limit()
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "multiseed_eval":
        sync_obs_mode_from_checkpoint(args)
        run_multiseed_eval(args)
    else:
        sync_obs_mode_from_checkpoint(args)
        evaluate(args)


if __name__ == "__main__":
    main()