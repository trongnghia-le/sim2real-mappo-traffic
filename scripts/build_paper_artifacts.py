from __future__ import annotations

"""
build_paper_artifacts.py
========================
One command to turn a finished training campaign into all paper deliverables:
discover per-seed checkpoints -> run the unified tripinfo evaluation per network
-> emit LaTeX tables -> render the multi-seed figures. Optionally runs the
sensing-noise robustness sweep.

Each step is launched as an isolated subprocess (so one network's SUMO failure
cannot abort the rest) and is resumable: pass --skip-eval to only (re)build
tables/figures from existing results/paper1_mappo/eval_tables/.

Pipeline
--------
  1. discover  results/paper1_mappo/<campaign>/models/<net>_<algo>_<obs>_seed*/<run>/best_model.pt
  2. eval      experiment/runners/eval_compare.py  (per network; MAPPO proxy +
               IPPO + privileged when present, vs Webster/LQI/SOTL/fixed +
               instrumented actuated/MP) -> results/paper1_mappo/eval_tables/<net>_<ts>/
  3. tables    experiment/runners/make_tables.py --auto -> figures/tables/*.tex
  4. figures   experiment/plots/campaign_figures.py --all -> figures/*.pdf
  5. (opt)     experiment/runners/eval_noise_robustness.py per network proxy ckpt

Usage (from project root, after parallel_launcher.py finishes):
    python scripts/build_paper_artifacts.py                       # latest campaign
    python scripts/build_paper_artifacts.py --campaign-id campaign_2026...
    python scripts/build_paper_artifacts.py --seeds 42 123 456 789 1337 --noise
    python scripts/build_paper_artifacts.py --skip-eval           # tables+figures only
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "paper1_mappo"
PY = sys.executable

EVAL_COMPARE = ROOT / "experiment" / "runners" / "eval_compare.py"
MAKE_TABLES = ROOT / "experiment" / "runners" / "make_tables.py"
CAMPAIGN_FIGS = ROOT / "experiment" / "plots" / "campaign_figures.py"
NOISE_EVAL = ROOT / "experiment" / "runners" / "eval_noise_robustness.py"

SUPPORTED_NETWORKS = ("n1", "n2_corridor", "n3_grid", "n4_irregular")
DEFAULT_NETWORKS = ["n2_corridor", "n3_grid", "n4_irregular"]
BASELINE_METHODS = ["webster", "lqi", "sotl", "fixed", "actuated", "mp"]


def latest_campaign() -> Optional[Path]:
    cands = sorted(RESULTS.glob("campaign_*"))
    return cands[-1] if cands else None


def _select_canonical_checkpoint(job_dir: Path) -> Optional[Path]:
    """Select the single canonical best_model.pt for one completed job dir.

    A completed job has exactly one run directory containing ``bench.json`` and
    a sibling ``best_model.pt``. Raises ``ValueError`` if multiple run dirs
    contain ``bench.json`` (ambiguous) and ``FileNotFoundError`` if a completed
    run dir lacks ``best_model.pt``. Returns ``None`` if no run has finished.
    """
    bench_dirs = sorted({p.parent for p in job_dir.glob("**/bench.json")})
    if len(bench_dirs) > 1:
        raise ValueError(
            f"Ambiguous completed runs in {job_dir}: found {len(bench_dirs)} "
            f"run dirs with bench.json ({[str(d) for d in bench_dirs]})"
        )
    if not bench_dirs:
        return None
    ckpt = bench_dirs[0] / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Completed run dir {bench_dirs[0]} has bench.json but is missing best_model.pt"
        )
    return ckpt


def discover_checkpoints(
    campaign_dir: Path,
    network: str,
    algo: str,
    obs: str,
    *,
    suffix: str = "",
) -> List[Path]:
    """Canonical best_model.pt for every completed seed of one arm, in numeric seed order."""
    models = campaign_dir / "models"
    if not models.exists():
        return []
    pat = re.compile(
        rf"^{re.escape(network)}_{re.escape(algo)}_{re.escape(obs)}{re.escape(suffix)}_seed(\d+)$"
    )
    matched: List[Tuple[int, Path]] = []
    for job_dir in models.iterdir():
        if not job_dir.is_dir():
            continue
        m = pat.match(job_dir.name)
        if m:
            matched.append((int(m.group(1)), job_dir))
    matched.sort(key=lambda kv: kv[0])

    found: List[Path] = []
    for _seed, job_dir in matched:
        ckpt = _select_canonical_checkpoint(job_dir)
        if ckpt is not None:
            found.append(ckpt)
    return found


def discover_network_arms(campaign_dir: Path, network: str) -> Dict[str, List[Path]]:
    """Discover all paper arms for ``network`` including r-privileged and n2 _base."""
    return {
        "mappo": discover_checkpoints(campaign_dir, network, "mappo", "proxy"),
        "ippo": discover_checkpoints(campaign_dir, network, "ippo", "proxy"),
        "privileged": discover_checkpoints(campaign_dir, network, "mappo", "privileged"),
        "r_privileged": discover_checkpoints(
            campaign_dir, network, "mappo", "proxy", suffix="_r-privileged"
        ),
        "base": discover_checkpoints(
            campaign_dir, network, "mappo", "proxy", suffix="_base"
        ),
    }


def _discover_newest_summaries(eval_dir: Path) -> List[Path]:
    """Newest summary.json per network directly under ``eval_dir``."""
    if not eval_dir.exists():
        return []
    by_net: Dict[str, Path] = {}
    for summ in sorted(eval_dir.glob("*/summary.json")):
        try:
            scen = json.loads(summ.read_text(encoding="utf-8")).get("scenario", summ.parent.name)
        except Exception:
            scen = summ.parent.name
        if scen not in by_net or summ.parent.name > by_net[scen].parent.name:
            by_net[scen] = summ
    return list(by_net.values())


def _run(cmd: List[str], label: str) -> bool:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc != 0:
        print(f"[{label}] FAILED (exit {rc})")
        return False
    return True


def run_eval_for_network(
    campaign_dir: Path,
    network: str,
    args: argparse.Namespace,
    eval_dir: Path,
) -> bool:
    arms = discover_network_arms(campaign_dir, network)
    mappo = arms["mappo"]
    ippo = arms["ippo"]
    priv = arms["privileged"]
    r_priv = arms["r_privileged"]
    base = arms["base"]

    methods: List[str] = []
    if mappo:
        methods.append("mappo")
    if ippo:
        methods.append("ippo")
    if priv:
        methods.append("privileged")
    methods += BASELINE_METHODS

    if not (mappo or ippo or priv):
        print(f"[eval] {network}: no checkpoints found — skipping learned methods, "
              f"running baselines only")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cmd = [PY, str(EVAL_COMPARE),
           "--network", network,
           "--split", "eval",
           "--methods", *methods,
           "--seeds", *[str(s) for s in args.seeds],
           "--reference", "mappo" if mappo else methods[0],
           "--max-steps", str(args.max_steps),
           "--out-dir", str(eval_dir / f"{network}_{ts}")]
    if args.emissions:
        cmd.append("--emissions")
    if mappo:
        cmd += ["--checkpoint", *[str(p) for p in mappo]]
    if ippo:
        cmd += ["--ippo-checkpoint", *[str(p) for p in ippo]]
    if priv:
        cmd += ["--privileged-checkpoint", *[str(p) for p in priv]]

    print(
        f"[eval] {network}: mappo={len(mappo)} ippo={len(ippo)} priv={len(priv)} "
        f"r_priv={len(r_priv)} base={len(base)} ckpts"
    )
    ok = _run(cmd, f"eval/{network}")

    # Evaluate extra revision130 arms (reward-privileged and n2 uncoordinated base)
    # under eval_dir/arms/ so they do not clobber the main network summary.json.
    for arm_tag, ckpts in (("r-privileged", r_priv), ("base", base)):
        if not ckpts:
            continue
        arm_cmd = [
            PY, str(EVAL_COMPARE),
            "--network", network,
            "--split", "eval",
            "--methods", "mappo",
            "--reference", "mappo",
            "--seeds", *[str(s) for s in args.seeds],
            "--max-steps", str(args.max_steps),
            "--checkpoint", *[str(p) for p in ckpts],
            "--out-dir", str(eval_dir / "arms" / f"{network}_{arm_tag}_{ts}"),
        ]
        if args.emissions:
            arm_cmd.append("--emissions")
        ok = _run(arm_cmd, f"eval/{network}/{arm_tag}") and ok

    return ok


def run_noise_for_network(campaign_dir: Path, network: str, args: argparse.Namespace) -> bool:
    proxy = discover_checkpoints(campaign_dir, network, "mappo", "proxy")
    if not proxy:
        print(f"[noise] {network}: no proxy checkpoint — skip")
        return True
    ok = True
    for ckpt in proxy[: args.noise_max_ckpts]:
        cmd = [PY, str(NOISE_EVAL),
               "--checkpoint", str(ckpt),
               "--network", network,
               "--seeds", *[str(s) for s in args.seeds]]
        ok = _run(cmd, f"noise/{network}") and ok
    return ok


def main() -> int:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaign-id", type=str, default=None,
                   help="campaign dir under results/paper1_mappo (default: latest)")
    p.add_argument("--networks", nargs="+", default=list(DEFAULT_NETWORKS),
                   choices=list(SUPPORTED_NETWORKS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1337])
    p.add_argument("--max-steps", type=int, default=1080)
    p.add_argument("--emissions", action="store_true",
                   help="enable SUMO emissions device for post-hoc CO2/fuel (TABLE-10)")
    p.add_argument("--skip-eval", action="store_true",
                   help="skip evaluation; only (re)build tables + figures")
    p.add_argument("--noise", action="store_true",
                   help="also run the sensing-noise robustness sweep (proxy arm)")
    p.add_argument("--noise-max-ckpts", type=int, default=1,
                   help="how many proxy seeds to run noise on (default 1)")
    p.add_argument("--eval-dir", type=str, default=None,
                   help="eval output/input dir (default: results/paper1_mappo/eval_tables)")
    p.add_argument("--tables-dir", type=str, default=None,
                   help="LaTeX tables output dir (default: figures/tables)")
    p.add_argument("--figures-dir", type=str, default=None,
                   help="figures output dir (default: figures)")
    args = p.parse_args()

    eval_dir = Path(args.eval_dir) if args.eval_dir else (RESULTS / "eval_tables")
    tables_dir = Path(args.tables_dir) if args.tables_dir else (ROOT / "figures" / "tables")
    figures_dir = Path(args.figures_dir) if args.figures_dir else (ROOT / "figures")

    campaign_dir = (RESULTS / args.campaign_id) if args.campaign_id else latest_campaign()
    if campaign_dir is None or not campaign_dir.exists():
        if not args.skip_eval:
            sys.exit("[build] no campaign dir found under results/paper1_mappo "
                     "(run scripts/parallel_launcher.py first, or use --skip-eval)")
    else:
        print(f"[build] campaign: {campaign_dir.name}")

    failures = 0

    if not args.skip_eval:
        for net in args.networks:
            if not run_eval_for_network(campaign_dir, net, args, eval_dir):
                failures += 1
        if args.noise:
            for net in args.networks:
                if not run_noise_for_network(campaign_dir, net, args):
                    failures += 1
    else:
        print(f"[build] --skip-eval: using existing {eval_dir}")

    # tables (LaTeX) from the newest summary per network
    table_cmd = [PY, str(MAKE_TABLES)]
    if args.eval_dir:
        summaries = _discover_newest_summaries(eval_dir)
        if summaries:
            table_cmd += ["--summaries", *[str(s) for s in summaries]]
        else:
            table_cmd.append("--auto")
    else:
        table_cmd.append("--auto")
    if args.tables_dir:
        table_cmd += ["--out-dir", str(tables_dir)]
    if not _run(table_cmd, "tables"):
        failures += 1

    # figures (multi-seed). campaign_figures auto-detects the campaign for the
    # convergence/loss panels; comparison/restriction-cost/noise read eval_tables.
    fig_cmd = [PY, str(CAMPAIGN_FIGS), "--all"]
    if campaign_dir is not None and campaign_dir.exists():
        fig_cmd += ["--campaign-id", campaign_dir.name]
    if args.eval_dir:
        fig_cmd += ["--eval-dir", str(eval_dir)]
    if args.figures_dir:
        fig_cmd += ["--figures-dir", str(figures_dir)]
    if not _run(fig_cmd, "figures"):
        failures += 1

    print("\n" + "=" * 60)
    print("PAPER ARTIFACTS BUILD COMPLETE" if failures == 0
          else f"BUILD FINISHED WITH {failures} FAILED STEP(S)")
    print("=" * 60)
    print(f"  tables : {tables_dir}")
    print(f"  figures: {figures_dir}")
    print(f"  evals  : {eval_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
