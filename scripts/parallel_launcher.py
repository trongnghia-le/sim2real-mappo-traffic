"""
parallel_launcher.py
====================
W5-9 main-campaign launcher: runs the full (network x seed x algo) training
matrix as throttled, staggered, isolated subprocesses.

Why subprocesses (not multiprocessing.Pool): each training run owns a SUMO
engine (libsumo in-process) and a torch session; full process isolation means
one crashed run can never poison the others, and logs/artifacts stay per-run.

Features
--------
- Worker-pool throttle      --max-workers (default 3: laptop-safe against RAM
                            pressure and thermal throttling; raise to ~10 on
                            a 16-core server — each run is ~1-1.5 cores).
- Staggered starts          --stagger seconds (default 45) between spawns so
                            concurrent SUMO inits don't spike CPU/IO together.
- Dynamic per-job args      unique --seed, per-job --ckpt-dir/--log-dir
                            (train_ppo.py creates a timestamped run dir inside;
                            those flags are this codebase's "run dir" control),
                            network-specific --sumo-cfg/--lane-groups, plus
                            arbitrary passthrough overrides after `--`.
- Isolation                 each job's stdout+stderr -> <campaign>/<job>.log.
- Resume                    jobs whose ckpt dir already holds a bench.json are
                            skipped, so a killed campaign restarts cleanly.
- Manifest                  <campaign>/campaign_manifest.json rewritten on
                            every state change (queued/running/done/failed,
                            pid, runtime, exit code) for external monitoring.
- Graceful Ctrl+C           terminates all children, marks them 'killed'.

Usage (from project root):
    # full default campaign (paper VI-A/VI-C): {n2_corridor, n3_grid}
    #   x MAPPO x {proxy, privileged} x 5 seeds, 2M steps
    #   = 20 jobs. Add IPPO (proxy-only baseline) for the canonical 30-job matrix:
    python scripts/parallel_launcher.py --algos mappo ippo

    # MAPPO proxy arm only (skip the privileged upper bound)
    python scripts/parallel_launcher.py --obs-modes proxy

    # laptop smoke: tiny budget, 2 workers
    python scripts/parallel_launcher.py --total-timesteps 2048 --max-workers 2 \
        --seeds 42 123 --networks n2_corridor --obs-modes proxy

    # passthrough overrides for train_ppo.py
    python scripts/parallel_launcher.py -- --rollout-horizon 256 --policy-lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
NETWORKS_DIR = ROOT / "sumo_configs" / "networks"
DEFAULT_SEEDS = (42, 123, 456, 789, 1337)
S5_SEEDS: Tuple[int, ...] = DEFAULT_SEEDS
S10_SEEDS: Tuple[int, ...] = S5_SEEDS + (2024, 3141, 4242, 5555, 7777)
SUPPORTED_NETWORKS: Tuple[str, ...] = ("n1", "n2_corridor", "n3_grid", "n4_irregular")
DEFAULT_TOTAL_TIMESTEPS = 2_000_000
REVISION130_TOTAL_TIMESTEPS = 300_000

# Revision schema-1.3.0 campaign specification:
#   (tier, network, algo, obs_mode, reward_preset, obs_ablation, seeds, extra_flags, name_suffix)
# n2_corridor includes both Tier-1 arms:
#   - coordinated default (`n2_corridor_mappo_proxy_seed*`: auto-enables
#     --upstream-phase-obs and --shared-reward)
#   - uncoordinated base (`n2_corridor_mappo_proxy_base_seed*`: passes
#     --no-n2-coordination to keep local 26-dim obs + per-agent reward)
REVISION130_ARMS: Tuple[
    Tuple[int, str, str, str, str, str, Tuple[int, ...], Tuple[str, ...], str], ...
] = (
    # --- Tier 1 (55 jobs) ---
    (1, "n3_grid",      "mappo", "proxy",      "full",              "none",                S10_SEEDS, (),                        ""),
    (1, "n3_grid",      "ippo",  "proxy",      "full",              "none",                S5_SEEDS,  (),                        ""),
    (1, "n3_grid",      "mappo", "privileged", "full",              "none",                S5_SEEDS,  (),                        ""),
    (1, "n3_grid",      "mappo", "proxy",      "privileged",        "none",                S5_SEEDS,  (),                        ""),
    (1, "n3_grid",      "mappo", "proxy",      "no_pressure",       "none",                S5_SEEDS,  (),                        ""),
    (1, "n3_grid",      "mappo", "proxy",      "no_throughput",     "none",                S5_SEEDS,  (),                        ""),
    (1, "n3_grid",      "mappo", "proxy",      "queue_only",        "none",                S5_SEEDS,  (),                        ""),
    (1, "n2_corridor",  "mappo", "proxy",      "full",              "none",                S5_SEEDS,  (),                        ""),
    (1, "n2_corridor",  "mappo", "proxy",      "full",              "none",                S5_SEEDS,  ("--no-n2-coordination",), "_base"),
    (1, "n4_irregular", "mappo", "proxy",      "full",              "none",                S5_SEEDS,  (),                        ""),
    # --- Tier 2 (40 jobs) ---
    (2, "n3_grid",      "mappo", "proxy",      "unsigned_pressure", "none",                S5_SEEDS,  (),                        ""),
    (2, "n3_grid",      "mappo", "proxy",      "mean_then_square",  "none",                S5_SEEDS,  (),                        ""),
    (2, "n3_grid",      "mappo", "proxy",      "no_delay",          "none",                S5_SEEDS,  (),                        ""),
    (2, "n3_grid",      "mappo", "proxy",      "no_low_speed",      "none",                S5_SEEDS,  (),                        ""),
    (2, "n3_grid",      "mappo", "proxy",      "full",              "no_class_shares",     S5_SEEDS,  (),                        ""),
    (2, "n3_grid",      "mappo", "proxy",      "full",              "no_pressure_feature", S5_SEEDS,  (),                        ""),
    (2, "n3_grid",      "mappo", "proxy",      "full",              "lane_truncated",      S5_SEEDS,  (),                        ""),
    (2, "n4_irregular", "ippo",  "proxy",      "full",              "none",                S5_SEEDS,  (),                        ""),
)


# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------

@dataclass
class Job:
    name: str                    # e.g. n3_grid_mappo_proxy_seed42
    network: str
    algo: str
    seed: int
    obs_mode: str
    cmd: List[str]
    log_path: Path
    ckpt_dir: Path
    tier: int = 1
    reward_preset: str = "full"
    obs_ablation: str = "none"
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS
    extra_flags: Tuple[str, ...] = ()
    status: str = "queued"       # queued | running | done | failed | killed | skipped
    pid: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    runtime_s: Optional[float] = None
    exit_code: Optional[int] = None
    attempts: int = 0            # spawn count (incremented in _spawn; drives retry)
    proc: Optional[subprocess.Popen] = field(default=None, repr=False, compare=False)
    log_handle: object = field(default=None, repr=False, compare=False)

    def manifest_row(self) -> Dict:
        # built explicitly — dataclasses.asdict() would deepcopy the live
        # Popen/file-handle fields and fail on their thread locks
        return {
            "name": self.name,
            "tier": self.tier,
            "network": self.network,
            "algo": self.algo,
            "seed": self.seed,
            "obs_mode": self.obs_mode,
            "reward_preset": self.reward_preset,
            "obs_ablation": self.obs_ablation,
            "total_timesteps": self.total_timesteps,
            "status": self.status,
            "pid": self.pid,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "runtime_s": self.runtime_s,
            "exit_code": self.exit_code,
            "attempts": self.attempts,
            "cmd": " ".join(self.cmd),
            "log_path": str(self.log_path),
            "ckpt_dir": str(self.ckpt_dir),
        }


def _network_paths(network: str) -> Tuple[Path, Path]:
    net_dir = NETWORKS_DIR / network
    return net_dir / "sumo_config.sumocfg", net_dir / "lane_groups.json"


def _network_required_files(network: str) -> Tuple[Path, ...]:
    """Files required on disk before spawning any job for ``network``."""
    net_dir = NETWORKS_DIR / network
    return (
        net_dir / "sumo_config.sumocfg",
        net_dir / "lane_groups.json",
        net_dir / "demand_train.rou.xml",
        net_dir / "demand_eval.rou.xml",
    )


def _resolve_total_timesteps(args: argparse.Namespace) -> int:
    override = getattr(args, "total_timesteps", None)
    if override is not None:
        return int(override)
    if getattr(args, "preset", None) == "revision130":
        return REVISION130_TOTAL_TIMESTEPS
    return DEFAULT_TOTAL_TIMESTEPS


def _make_job(
    *,
    network: str,
    algo: str,
    obs_mode: str,
    reward_preset: str,
    obs_ablation: str,
    seed: int,
    tier: int,
    args: argparse.Namespace,
    campaign_dir: Path,
    passthrough: List[str],
    extra_flags: Sequence[str] = (),
    name_suffix: str = "",
) -> Job:
    sumo_cfg, lane_groups = _network_paths(network)
    total_timesteps = _resolve_total_timesteps(args)
    # main arms (full/none) keep the canonical name so
    # build_paper_artifacts/campaign_figures discover them;
    # ablation/variant arms get a suffix and stay out of the main set.
    suffix = ""
    if reward_preset != "full":
        suffix += f"_r-{reward_preset}"
    if obs_ablation != "none":
        suffix += f"_o-{obs_ablation}"
    if name_suffix:
        suffix += name_suffix
    name = f"{network}_{algo}_{obs_mode}{suffix}_seed{seed}"
    ckpt_dir = campaign_dir / "models" / name
    log_dir = campaign_dir / "tblogs" / name
    extra = tuple(extra_flags)
    cmd = [
        sys.executable, str(ROOT / "experiment" / "runners" / "train_ppo.py"),
        "--mode", "train",
        "--algo", algo,
        "--obs-mode", obs_mode,
        "--reward-preset", reward_preset,
        "--obs-ablation", obs_ablation,
        "--sumo-cfg", str(sumo_cfg),
        "--lane-groups", str(lane_groups),
        "--seed", str(seed),
        "--total-timesteps", str(total_timesteps),
        "--rollout-horizon", str(args.rollout_horizon),
        "--ckpt-dir", str(ckpt_dir),
        "--log-dir", str(log_dir),
    ] + list(extra) + passthrough
    return Job(
        name=name,
        network=network,
        algo=algo,
        seed=seed,
        obs_mode=obs_mode,
        cmd=cmd,
        log_path=campaign_dir / f"{name}.log",
        ckpt_dir=ckpt_dir,
        tier=tier,
        reward_preset=reward_preset,
        obs_ablation=obs_ablation,
        total_timesteps=total_timesteps,
        extra_flags=extra,
    )


def _validate_network_files(jobs: Sequence[Job]) -> List[Path]:
    missing: List[Path] = []
    seen = set()
    for job in jobs:
        if job.network in seen:
            continue
        seen.add(job.network)
        for p in _network_required_files(job.network):
            if not p.exists():
                missing.append(p)
    return missing


def manifest_filename(shard_spec: Optional[str] = None) -> str:
    """Return per-shard manifest filename when sharded, else campaign_manifest.json."""
    if shard_spec:
        idx, total = parse_shard(shard_spec)
        return f"campaign_manifest_shard{idx}of{total}.json"
    return "campaign_manifest.json"


def parse_shard(spec: str) -> Tuple[int, int]:
    """Parse a 1-indexed shard spec 'i/n' (1 <= i <= n)."""
    parts = spec.strip().split("/")
    if len(parts) != 2:
        raise SystemExit(f"invalid --shard {spec!r}: expected 1-indexed i/n (e.g. 1/4 .. 4/4)")
    try:
        idx, total = int(parts[0]), int(parts[1])
    except ValueError:
        raise SystemExit(f"invalid --shard {spec!r}: i and n must be integers")
    if total < 1 or idx < 1 or idx > total:
        raise SystemExit(
            f"invalid --shard {spec!r}: expected 1 <= i <= n with n >= 1 (e.g. 1/4 .. 4/4)"
        )
    return idx, total


def apply_tier_and_shard(
    jobs: List[Job],
    tier: str = "all",
    shard_spec: Optional[str] = None,
) -> List[Job]:
    """Filter by tier, enforce tier-1-before-tier-2 order, then shard deterministically."""
    if tier in ("1", "2"):
        want = int(tier)
        selected = [j for j in jobs if j.tier == want]
    else:
        selected = list(jobs)
    # Stable sort guarantees tier-1 jobs always precede tier-2 when --tier all
    selected.sort(key=lambda j: j.tier)
    if shard_spec is not None:
        shard_idx, shard_total = parse_shard(shard_spec)
        selected = [
            j for pos, j in enumerate(selected)
            if pos % shard_total == (shard_idx - 1)
        ]
    return selected


def build_revision130_jobs(
    args: argparse.Namespace,
    campaign_dir: Path,
    passthrough: List[str],
) -> List[Job]:
    jobs: List[Job] = []
    for (
        tier,
        network,
        algo,
        obs_mode,
        reward_preset,
        obs_ablation,
        seeds,
        extra_flags,
        name_suffix,
    ) in REVISION130_ARMS:
        for seed in seeds:
            jobs.append(
                _make_job(
                    network=network,
                    algo=algo,
                    obs_mode=obs_mode,
                    reward_preset=reward_preset,
                    obs_ablation=obs_ablation,
                    seed=seed,
                    tier=tier,
                    args=args,
                    campaign_dir=campaign_dir,
                    passthrough=passthrough,
                    extra_flags=extra_flags,
                    name_suffix=name_suffix,
                )
            )
    return jobs


def build_jobs(args: argparse.Namespace, campaign_dir: Path, passthrough: List[str]) -> List[Job]:
    if getattr(args, "preset", None) == "revision130":
        jobs = build_revision130_jobs(args, campaign_dir, passthrough)
    else:
        jobs = []
        for network in args.networks:
            for algo in args.algos:
                for obs_mode in args.obs_modes:
                    # privileged is the MAPPO upper-bound arm (paper VI-C); the IPPO
                    # baseline isolates centralized-training value on the deployable
                    # proxy obs, so (ippo, privileged) is not part of the design.
                    if obs_mode == "privileged" and algo != "mappo":
                        print(f"[skip]   {network}_{algo}_{obs_mode}: privileged is mappo-only")
                        continue
                    for preset in args.reward_presets:
                        for ablation in args.obs_ablations:
                            for seed in args.seeds:
                                jobs.append(
                                    _make_job(
                                        network=network,
                                        algo=algo,
                                        obs_mode=obs_mode,
                                        reward_preset=preset,
                                        obs_ablation=ablation,
                                        seed=seed,
                                        tier=1,
                                        args=args,
                                        campaign_dir=campaign_dir,
                                        passthrough=passthrough,
                                    )
                                )

    jobs = apply_tier_and_shard(
        jobs,
        tier=getattr(args, "tier", "all"),
        shard_spec=getattr(args, "shard", None),
    )

    if not getattr(args, "dry_run", False):
        missing = _validate_network_files(jobs)
        if missing:
            raise SystemExit(
                f"missing {missing[0]} — run generate_networks.py / generate_demand.py"
            )
    return jobs


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class CampaignScheduler:
    def __init__(self, jobs: List[Job], campaign_dir: Path,
                 max_workers: int, stagger_s: float, poll_s: float = 5.0,
                 max_retries: int = 3, shard_spec: Optional[str] = None,
                 terminate_timeout_s: float = 10.0) -> None:
        self.jobs = jobs
        self.campaign_dir = campaign_dir
        self.max_workers = max(1, int(max_workers))
        self.stagger_s = max(0.0, float(stagger_s))
        self.poll_s = max(0.01, float(poll_s))
        self.max_retries = max(0, int(max_retries))
        self.shard_spec = shard_spec
        self.manifest_path = self.campaign_dir / manifest_filename(shard_spec)
        # Keep launcher child-termination deadline <= 10s so GCP supervisor
        # (25s wait inside ~30s preemption window) can finish GCS sync cleanly.
        self.terminate_timeout_s = min(10.0, max(0.5, float(terminate_timeout_s)))
        self._last_spawn_t = 0.0
        self._interrupted = False

    # -- manifest ----------------------------------------------------------
    def write_manifest(self) -> None:
        manifest = {
            "campaign_dir": str(self.campaign_dir),
            "shard": self.shard_spec,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "max_workers": self.max_workers,
            "stagger_s": self.stagger_s,
            "poll_s": self.poll_s,
            "counts": {
                status: sum(1 for j in self.jobs if j.status == status)
                for status in ("queued", "running", "done", "failed", "killed", "skipped")
            },
            "tier_counts": {
                str(t): sum(1 for j in self.jobs if j.tier == t)
                for t in sorted({j.tier for j in self.jobs})
            },
            "jobs": [j.manifest_row() for j in self.jobs],
        }
        path = self.manifest_path
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp.replace(path)

    # -- lifecycle ---------------------------------------------------------
    def _already_complete(self, job: Job) -> bool:
        """A finished run leaves bench.json inside its timestamped run dir."""
        return any(job.ckpt_dir.glob("*/bench.json")) if job.ckpt_dir.exists() else False

    def _spawn(self, job: Job) -> None:
        job.attempts += 1
        job.log_handle = open(job.log_path, "a", encoding="utf-8", errors="replace")
        job.log_handle.write(
            f"\n===== launch {datetime.now().isoformat(timespec='seconds')} =====\n"
            f"$ {' '.join(job.cmd)}\n\n"
        )
        job.log_handle.flush()
        # Cap BLAS/OpenMP/PyTorch intra-op threads to 1 per child so many
        # concurrent workers do not oversubscribe CPU cores.
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        job.proc = subprocess.Popen(
            job.cmd, cwd=str(ROOT), env=env,
            stdout=job.log_handle, stderr=subprocess.STDOUT,
        )
        job.status = "running"
        job.pid = job.proc.pid
        job.started_at = datetime.now().isoformat(timespec="seconds")
        self._last_spawn_t = time.time()
        print(f"[launch] [T{job.tier}] {job.name} (pid {job.pid}) -> {job.log_path.name}")

    def _reap(self, job: Job) -> None:
        assert job.proc is not None
        job.exit_code = job.proc.returncode
        job.finished_at = datetime.now().isoformat(timespec="seconds")
        if job.started_at:
            t0 = datetime.fromisoformat(job.started_at)
            job.runtime_s = round((datetime.fromisoformat(job.finished_at) - t0).total_seconds(), 1)
        if job.log_handle:
            try:
                job.log_handle.close()
            except Exception:
                pass
        if job.exit_code == 0:
            job.status = "done"
            print(f"[finish] {job.name}: OK ({job.runtime_s}s)")
        elif job.attempts <= self.max_retries:
            # crash/segfault (e.g. intermittent libsumo SIGSEGV exit=-11): re-queue
            # so the freed core restarts the run instead of utilization decaying.
            job.status = "queued"
            print(f"[retry]  {job.name}: exit={job.exit_code} "
                  f"(attempt {job.attempts}/{self.max_retries + 1}) -> requeued")
        else:
            job.status = "failed"
            print(f"[finish] {job.name}: FAILED exit={job.exit_code} "
                  f"after {job.attempts} attempts")

    def _terminate_all(self) -> None:
        for job in self.jobs:
            if job.status == "running" and job.proc is not None:
                try:
                    job.proc.terminate()
                except Exception:
                    pass
        deadline = time.time() + self.terminate_timeout_s
        for job in self.jobs:
            if job.status == "running" and job.proc is not None:
                try:
                    job.proc.wait(timeout=max(0.1, deadline - time.time()))
                except Exception:
                    try:
                        job.proc.kill()
                    except Exception:
                        pass
                job.status = "killed"
                job.finished_at = datetime.now().isoformat(timespec="seconds")
                if job.log_handle:
                    try:
                        job.log_handle.close()
                    except Exception:
                        pass

    # -- main loop ---------------------------------------------------------
    def run(self) -> int:
        def on_signal(sig, _frame):
            self._interrupted = True
            sig_name = signal.Signals(sig).name if hasattr(signal, "Signals") else str(sig)
            print(f"\n[{sig_name}] terminating running jobs ...")

        signal.signal(signal.SIGINT, on_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, on_signal)

        for job in self.jobs:
            if self._already_complete(job):
                job.status = "skipped"
                print(f"[skip]   {job.name}: bench.json already present (resume)")
        self.write_manifest()

        while not self._interrupted:
            # reap finished
            changed = False
            for job in self.jobs:
                if job.status == "running" and job.proc is not None and job.proc.poll() is not None:
                    self._reap(job)
                    changed = True

            running = [j for j in self.jobs if j.status == "running"]
            queued = [j for j in self.jobs if j.status == "queued"]

            # spawn up to all free slots in one tick (respecting --stagger)
            while queued and len(running) < self.max_workers:
                since_last = time.time() - self._last_spawn_t
                if self._last_spawn_t != 0.0 and since_last < self.stagger_s:
                    break
                next_job = queued.pop(0)
                self._spawn(next_job)
                running.append(next_job)
                changed = True

            if changed:
                self.write_manifest()

            if not queued and not running:
                break
            time.sleep(self.poll_s)

        if self._interrupted:
            self._terminate_all()
        self.write_manifest()

        # summary
        print("\n" + "=" * 66)
        print("CAMPAIGN SUMMARY")
        print("=" * 66)
        print(f"  {'job':<28} {'status':<8} {'runtime':>9}  exit")
        for job in self.jobs:
            rt = f"{job.runtime_s:.0f}s" if job.runtime_s else "-"
            print(f"  {job.name:<28} {job.status:<8} {rt:>9}  {job.exit_code}")
        n_bad = sum(1 for j in self.jobs if j.status in ("failed", "killed"))
        print(f"\n  manifest: {self.manifest_path}")
        return 1 if n_bad else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    argv = sys.argv[1:]
    passthrough: List[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, passthrough = argv[:split], argv[split + 1:]

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preset", choices=["revision130"], default=None,
                        help="Named campaign preset (overrides --networks/--algos/"
                             "--obs-modes/--seeds/--reward-presets/--obs-ablations)")
    parser.add_argument("--tier", choices=["1", "2", "all"], default="all",
                        help="Filter jobs by tier (1, 2, or all; tier-1 jobs are "
                             "always scheduled before tier-2 when --tier all)")
    parser.add_argument("--shard", type=str, default=None,
                        help="1-indexed deterministic shard i/n (1 <= i <= n) to "
                             "partition the ordered job list across machines")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved job list and per-tier counts "
                             "without creating directories or spawning processes")
    parser.add_argument("--networks", nargs="+", default=["n2_corridor", "n3_grid"],
                        choices=list(SUPPORTED_NETWORKS))
    parser.add_argument("--algos", nargs="+", default=["mappo"], choices=["mappo", "ippo"])
    parser.add_argument("--obs-modes", nargs="+", default=["proxy", "privileged"],
                        choices=["proxy", "privileged"],
                        help="obs arms to train; privileged is mappo-only "
                             "(VI-C upper bound). noise is eval-time only, not here.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--reward-presets", nargs="+", default=["full"],
                        choices=["full", "no_pressure", "no_throughput", "queue_only",
                                 "unsigned_pressure", "mean_then_square", "no_delay",
                                 "no_low_speed", "privileged"],
                        help="VI-F reward ablation arms (default: full only)")
    parser.add_argument("--obs-ablations", nargs="+", default=["none"],
                        choices=["none", "no_class_shares", "no_pressure_feature", "lane_truncated"],
                        help="VI-F observation ablation arms (default: none only)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help=f"timesteps per job (default: {REVISION130_TOTAL_TIMESTEPS} "
                             f"for --preset revision130; {DEFAULT_TOTAL_TIMESTEPS} otherwise)")
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument("--max-workers", type=int, default=3,
                        help="concurrent training processes (3 = laptop-safe; "
                             "~10 on a 16-core server)")
    parser.add_argument("--stagger", type=float, default=45.0,
                        help="seconds between spawns (avoids simultaneous "
                             "SUMO-init CPU spikes)")
    parser.add_argument("--poll-s", type=float, default=5.0,
                        help="seconds between scheduler poll ticks (default: 5.0)")
    parser.add_argument("--campaign-id", type=str, default=None,
                        help="reuse an existing campaign dir to resume it")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="re-queue a crashed job (e.g. intermittent libsumo "
                             "SIGSEGV exit=-11) up to N times before marking failed")
    args = parser.parse_args(argv)

    campaign_id = args.campaign_id or datetime.now().strftime("campaign_%Y%m%d_%H%M%S")
    campaign_dir = ROOT / "results" / "paper1_mappo" / campaign_id

    jobs = build_jobs(args, campaign_dir, passthrough)
    total = len(jobs)
    t1_count = sum(1 for j in jobs if j.tier == 1)
    t2_count = sum(1 for j in jobs if j.tier == 2)

    if args.dry_run:
        mode_desc = f"preset={args.preset}" if args.preset else "matrix"
        shard_desc = f", shard={args.shard}" if args.shard else ""
        print(f"[dry-run] campaign {campaign_id} ({mode_desc}, tier={args.tier}{shard_desc}):")
        for idx, job in enumerate(jobs, start=1):
            extra_desc = f" extra={' '.join(job.extra_flags)}" if job.extra_flags else ""
            print(
                f"  {idx:>3}. [tier {job.tier}] {job.name:<52} "
                f"(net={job.network} algo={job.algo} obs={job.obs_mode} "
                f"reward={job.reward_preset} ablation={job.obs_ablation} "
                f"seed={job.seed} steps={job.total_timesteps}{extra_desc})"
            )
        print("-" * 72)
        print(f"Tier 1: {t1_count} jobs")
        print(f"Tier 2: {t2_count} jobs")
        print(f"Total:  {total} jobs (manifest={manifest_filename(args.shard)})")
        missing = _validate_network_files(jobs)
        if missing:
            print(f"[note]  {len(missing)} network file(s) not yet on disk (checked at run time):")
            for p in missing:
                print(f"          - {p}")
        return 0

    campaign_dir.mkdir(parents=True, exist_ok=True)
    if args.preset:
        print(f"campaign {campaign_id}: {total} jobs "
              f"(preset={args.preset}, tier={args.tier}"
              f"{f', shard={args.shard}' if args.shard else ''}, "
              f"tier1={t1_count}, tier2={t2_count}, "
              f"timesteps={_resolve_total_timesteps(args)}), "
              f"max_workers={args.max_workers}, stagger={args.stagger}s, poll={args.poll_s}s")
    else:
        print(f"campaign {campaign_id}: {total} jobs "
              f"({len(args.networks)} nets x {len(args.algos)} algos x "
              f"{len(args.obs_modes)} obs-modes x {len(args.seeds)} seeds, "
              f"minus ippo-privileged, timesteps={_resolve_total_timesteps(args)}), "
              f"max_workers={args.max_workers}, stagger={args.stagger}s, poll={args.poll_s}s")
    print(f"artifacts -> {campaign_dir}")

    scheduler = CampaignScheduler(
        jobs, campaign_dir, max_workers=args.max_workers, stagger_s=args.stagger,
        poll_s=args.poll_s, max_retries=args.max_retries, shard_spec=args.shard,
    )
    return scheduler.run()


if __name__ == "__main__":
    sys.exit(main())
