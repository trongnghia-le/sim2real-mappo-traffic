from __future__ import annotations

"""
PATCH NOTES (audit v3 — schema 1.1.0):
- FIX: action_info now includes green_lanes_by_tls (current green phase group
  from manual lane groups). Without it RewardCalculator._compute_pressure_penalty
  returned 0.0 for every agent on every step — the pressure reward term was
  silently inactive in all earlier training runs.

PATCH NOTES (audit v2):
- FIX: info_dict now includes co2_mg_per_s, fuel_ml_per_s aggregated from lane_cache.
- FIX: info_dict now includes current_passed (throughput counter) so the training
  loop can track episode throughput.
- FIX: info_dict now includes avg_waiting_proxy (mean effective_queue_norm × step_length).
- FIX: green_timers are capped at max_green_time to prevent unbounded growth.
"""

from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from src.traffic_env.envs.base_sumo import BaseSumoEnv
from src.traffic_env.config import TrafficEnvConfig
from src.traffic_env.components.observations import ObservationBuilder
from src.traffic_env.components.rewards import RewardCalculator

ObsDict = Dict[str, np.ndarray]
RewardDict = Dict[str, float]
DoneDict = Dict[str, bool]
InfoDict = Dict[str, Dict[str, Any]]


class GreenDurationRecord(tuple):
    """Realized green-phase record: 3-tuple ``(tls_id, phase, duration_s)`` with ``.completed``.

    Preserves 3-tuple unpacking and equality for backward compatibility while
    tagging whether the phase ended via a normal switch (``completed=True``) or
    was right-censored at episode end (``completed=False``).
    """

    completed: bool

    def __new__(
        cls,
        tls_id: str,
        phase: int,
        duration_s: float,
        completed: bool = True,
    ) -> "GreenDurationRecord":
        obj = super().__new__(cls, (str(tls_id), int(phase), float(duration_s)))
        obj.completed = bool(completed)
        return obj

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GreenDurationRecord):
            return super().__eq__(other) and self.completed == other.completed
        return super().__eq__(other)

    def __repr__(self) -> str:
        return (
            f"GreenDurationRecord({self[0]!r}, {self[1]!r}, "
            f"{self[2]!r}, completed={self.completed!r})"
        )


class MappoTrafficEnv(BaseSumoEnv):

    def __init__(
        self,
        config: TrafficEnvConfig,
        extra_sumo_args: Optional[List[str]] = None,
        **sumo_kwargs: Any,
    ) -> None:
        super().__init__(sumo_cfg_path=config.sim.sumo_cfg_path, **sumo_kwargs)

        # extra SUMO CLI args appended on every (re)start — e.g.
        # ["--tripinfo-output", "out.xml", "--tripinfo-output.write-unfinished"]
        self.extra_sumo_args: List[str] = list(extra_sumo_args) if extra_sumo_args else []

        self.config = config
        self.tls_ids = list(config.tls_ids)

        self.obs_builder = ObservationBuilder(config=config, sumo_conn=None)
        self.reward_calculator = RewardCalculator(config=config)

        self.green_timers = {tls_id: 0.0 for tls_id in self.tls_ids}
        self.last_actions = {tls_id: 0 for tls_id in self.tls_ids}
        # realized green-phase durations: GreenDurationRecord(tls_id, phase, duration_s, completed=...)
        self.green_durations: List[GreenDurationRecord] = []
        self._phase_elapsed_s: Dict[str, float] = {tls_id: 0.0 for tls_id in self.tls_ids}
        self._current_green_phase: Dict[str, int] = {tls_id: 0 for tls_id in self.tls_ids}
        self._green_durations_flushed: bool = False

        # dynamic phase maps (auto-detect on reset)
        self._green_phase_map: Dict[str, List[int]] = {}
        self._yellow_phase_map: Dict[str, Dict[int, int]] = {}

        # action_dim inferred after first reset; default 2 for binary phase control
        self.action_dim: int = 2

        self._step_count = 0

    # =========================================================
    # RESET
    # =========================================================
    def reset(self, seed: Optional[int] = None):
        sumo_conn = self.start_simulation(
            seed=seed, extra_args=self.extra_sumo_args or None
        )
        self.obs_builder.set_sumo_conn(sumo_conn)

        self.green_timers = {tls_id: 0.0 for tls_id in self.tls_ids}
        self.last_actions = {tls_id: 0 for tls_id in self.tls_ids}
        self.green_durations = []
        self._phase_elapsed_s = {tls_id: 0.0 for tls_id in self.tls_ids}
        self._green_durations_flushed = False
        self.reward_calculator.reset()
        self._step_count = 0

        # build phase maps from SUMO program logic
        self._build_phase_maps()

        # update action_dim from detected green phases
        if self._green_phase_map:
            first_tls = self.tls_ids[0]
            self.action_dim = len(self._green_phase_map.get(first_tls, [0, 2]))

        # Keep SUMO at t=0.0 s to match green_timers=0.0 and _phase_elapsed_s=0.0
        # (avoids running 1 extra uncounted sim second on the initial phase and
        # keeps total episode duration equal to max_steps * step_length).
        self._current_green_phase = {}
        for tls_id in self.tls_ids:
            cur = self._get_phase(tls_id)
            phase_map = self._green_phase_map.get(tls_id, [0, 2])
            self._current_green_phase[tls_id] = cur if cur in phase_map else phase_map[0]

        lane_cache = self.obs_builder.build_lane_cache(self.tls_ids)
        obs_dict, global_state = self.obs_builder.get_all_observations(
            lane_cache, green_timers=self.green_timers
        )

        info_dict = {
            tls_id: {"global_state": global_state} for tls_id in self.tls_ids
        }

        return obs_dict, info_dict

    # =========================================================
    # STEP
    # =========================================================
    def step(self, action_dict: Dict[str, int]):

        prev_passed = self._get_total_passed()

        self._step_count += 1
        is_done = self._step_count >= self.config.sim.max_steps

        switches = {}
        target_phases = {}

        # -----------------------------------------
        # 1. ACTION → GREEN PHASE
        # -----------------------------------------
        max_green = float(self.config.sim.max_green_time)
        step_len = float(self.config.sim.step_length)
        for tls_id, action in action_dict.items():
            current = self._get_phase(tls_id)
            phase_map = self._green_phase_map[tls_id]

            action_idx = int(action)
            action_idx = max(0, min(action_idx, len(phase_map) - 1))
            target = phase_map[action_idx]

            # enforce max green time: force switch if holding another full step
            # would push realized green past max_green_time (after a switch step,
            # the new green starts at remaining = step_length - yellow_time = 2 s,
            # so 2 + 11*5 = 57 s is the last legal hold before 62 s > 60 s).
            would_exceed_max = (self.green_timers[tls_id] + step_len) > (max_green + 1e-9)
            if target == current and would_exceed_max and len(phase_map) >= 2:
                next_idx = (phase_map.index(current) + 1) % len(phase_map)
                target = phase_map[next_idx]
                switches[tls_id] = True
            # enforce min green time (never forcing a hold that violates max_green_time)
            elif (
                target != current
                and self.green_timers[tls_id] < self.config.sim.min_green_time
                and not would_exceed_max
            ):
                target = current
                switches[tls_id] = False
            else:
                switches[tls_id] = (target != current)

            target_phases[tls_id] = target

        # -----------------------------------------
        # 2. APPLY YELLOW PER TLS
        # -----------------------------------------
        yellow_duration = int(self.config.sim.yellow_time)

        for tls_id in self.tls_ids:
            if switches.get(tls_id, False):
                yellow = self._yellow_phase_map[tls_id].get(
                    self._get_phase(tls_id),
                    self._get_phase(tls_id)
                )
                self._set_phase(tls_id, yellow)

        if any(switches.values()):
            self.sim_step(yellow_duration)

        # -----------------------------------------
        # 3. APPLY GREEN
        # -----------------------------------------
        for tls_id in self.tls_ids:
            self._set_phase(tls_id, target_phases[tls_id])

        remaining = int(
            self.config.sim.step_length - (yellow_duration if any(switches.values()) else 0)
        )

        if remaining > 0:
            self.sim_step(remaining)

        # -----------------------------------------
        # 4. UPDATE TIMERS + REALIZED GREEN DURATIONS
        # -----------------------------------------
        max_green = float(self.config.sim.max_green_time)
        step_len = float(self.config.sim.step_length)
        for tls_id in self.tls_ids:
            if switches.get(tls_id, False):
                ended_dur = float(self._phase_elapsed_s.get(tls_id, 0.0))
                if ended_dur > 0.0:
                    prev_phase = self._current_green_phase.get(tls_id, 0)
                    self.green_durations.append(
                        GreenDurationRecord(tls_id, int(prev_phase), ended_dur, completed=True)
                    )
                self._current_green_phase[tls_id] = int(target_phases[tls_id])
                self._phase_elapsed_s[tls_id] = float(max(remaining, 0))
                self.green_timers[tls_id] = float(max(remaining, 0))
            else:
                self._phase_elapsed_s[tls_id] = self._phase_elapsed_s.get(tls_id, 0.0) + step_len
                self.green_timers[tls_id] = min(
                    self.green_timers[tls_id] + step_len,
                    max_green,
                )

            self.last_actions[tls_id] = int(action_dict.get(tls_id, 0))

        if is_done:
            self._flush_green_durations()

        # -----------------------------------------
        # 5. OBS
        # -----------------------------------------
        lane_cache = self.obs_builder.build_lane_cache(self.tls_ids)
        obs_dict, global_state = self.obs_builder.get_all_observations(
            lane_cache,
            green_timers=self.green_timers
        )

        current_passed = self._get_total_passed()

        # -----------------------------------------
        # 6. REWARD
        # -----------------------------------------
        # green_lanes_by_tls: map the applied green phase back to its manual
        # lane group so the pressure reward term can partition green vs red.
        lane_groups = self.config.get_lane_groups()
        green_lanes_by_tls: Dict[str, List[str]] = {}
        for tls_id in self.tls_ids:
            groups = lane_groups.get(tls_id)
            phase_map = self._green_phase_map.get(tls_id, [])
            target = target_phases.get(tls_id)
            if groups and target in phase_map:
                group_idx = phase_map.index(target)
                green_lanes_by_tls[tls_id] = list(groups[0] if group_idx == 0 else groups[1])

        action_info = {
            "lane_ids_by_tls": {
                tls_id: self._get_lanes(tls_id) for tls_id in self.tls_ids
            },
            "green_lanes_by_tls": green_lanes_by_tls,
            "current_passed": current_passed,
            "prev_passed": prev_passed,
            "switches_by_tls": switches,
        }

        reward_dict = self.reward_calculator.calculate_rewards(
            self.tls_ids, lane_cache, action_info
        )

        # -----------------------------------------
        # 7. DONE
        # -----------------------------------------
        terminated = {tls_id: False for tls_id in self.tls_ids}
        truncated = {tls_id: is_done for tls_id in self.tls_ids}

        # -----------------------------------------
        # 8. INFO  (FIX: add co2, fuel, throughput, waiting)
        # -----------------------------------------
        info_dict = {}
        for tls_id in self.tls_ids:
            lanes = self._get_lanes(tls_id)

            queue = 0.0
            co2 = 0.0
            fuel = 0.0
            avg_wait = 0.0
            if lanes:
                queue = sum(
                    float(self.sumo_conn.lane.getLastStepHaltingNumber(l))
                    for l in lanes
                )
                # aggregate emissions and waiting proxy from lane_cache
                for l in lanes:
                    m = lane_cache.get(l, {})
                    co2 += float(m.get("co2_mg_per_s", 0.0))
                    fuel += float(m.get("fuel_ml_per_s", 0.0))
                    avg_wait += float(m.get("effective_queue_norm", 0.0))
                avg_wait = (avg_wait / len(lanes)) * self.config.sim.step_length

            info_dict[tls_id] = {
                "global_state": global_state,
                "queue_total_proxy": queue,
                "current_passed": current_passed,          # FIX: throughput counter
                "co2_mg_per_s": co2,                       # FIX: per-tls emission
                "fuel_ml_per_s": fuel,                     # FIX: per-tls fuel
                "avg_waiting_proxy": avg_wait,             # FIX: waiting time proxy
                "reward_components": self.reward_calculator.last_reward_details.get(tls_id, {}),
            }

        return obs_dict, reward_dict, terminated, truncated, info_dict

    # =========================================================
    # PHASE MAP BUILDER (AUTO-DETECT)
    # =========================================================
    def _build_phase_maps(self) -> None:
        for tls_id in self.tls_ids:
            try:
                logic = self.sumo_conn.trafficlight.getAllProgramLogics(tls_id)[0]
                phases = logic.getPhases()

                green_phases = []
                yellow_map = {}

                for i, p in enumerate(phases):
                    state = p.state.lower()

                    if "g" in state:  # green phase
                        green_phases.append(i)

                    if "y" in state:
                        prev = (i - 1) % len(phases)
                        yellow_map[prev] = i

                if len(green_phases) >= 2:
                    self._green_phase_map[tls_id] = green_phases[:2]
                else:
                    self._green_phase_map[tls_id] = [0, 2]

                self._yellow_phase_map[tls_id] = yellow_map

            except Exception:
                # safe fallback for standard 4-phase SUMO signal
                self._green_phase_map[tls_id] = [0, 2]
                self._yellow_phase_map[tls_id] = {0: 1, 2: 3}

    # =========================================================
    # HELPERS
    # =========================================================
    def _get_phase(self, tls_id: str) -> int:
        try:
            return int(self.sumo_conn.trafficlight.getPhase(tls_id))
        except Exception:
            return 0

    def _set_phase(self, tls_id: str, phase: int) -> None:
        try:
            self.sumo_conn.trafficlight.setPhase(tls_id, int(phase))
        except Exception:
            pass

    def _get_lanes(self, tls_id: str) -> List[str]:
        try:
            lanes = self.sumo_conn.trafficlight.getControlledLanes(tls_id)
        except Exception:
            return []
        # getControlledLanes() returns one entry per signalised link, so a lane
        # with several movements repeats. Dedup (order-preserving) and drop
        # internal ':'-lanes so the reward iterates each approach lane exactly
        # once — matching ObservationBuilder._safe_lane_ids_for_tls(). Without
        # this, queue/low-speed/waiting penalties over-weight multi-movement
        # lanes (reward and observation used different lane multisets).
        return [l for l in dict.fromkeys(lanes) if l and not str(l).startswith(":")]

    def _flush_green_durations(self) -> None:
        """Append currently running green phases at episode end (idempotent, completed=False)."""
        if self._green_durations_flushed:
            return
        self._green_durations_flushed = True
        for tls_id in self.tls_ids:
            dur = float(self._phase_elapsed_s.get(tls_id, 0.0))
            if dur > 0.0:
                phase = self._current_green_phase.get(tls_id, 0)
                self.green_durations.append(
                    GreenDurationRecord(tls_id, int(phase), dur, completed=False)
                )

    def close(self) -> None:
        self._flush_green_durations()
        super().close()

    def _get_total_passed(self) -> float:
        # cumulative arrivals since episode start (base_sumo sums the per-step
        # simulation.getArrivedNumber()). Returning the raw per-step value here
        # was a bug: prev/current deltas and episode throughput were meaningless.
        return float(self.arrived_cumulative)


__all__ = ["GreenDurationRecord", "MappoTrafficEnv"]