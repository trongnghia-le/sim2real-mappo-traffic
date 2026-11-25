from __future__ import annotations

"""
rewards.py
==========
Per-agent reward calculator for MAPPO traffic-signal control.

Reward signal (reward revision 1.3.0 — approach-level)
------------------------------------------------------
Every term operates on APPROACH aggregates (controlled lanes grouped by parent
edge via ``group_lanes_by_approach`` and aggregated with
``aggregate_approach_metrics`` — the exact same rule used by
``ObservationBuilder``: unweighted mean for queue/occupancy, vehicle-count-
weighted mean for speed):

R(t) = w_q  · queue_penalty(t)           — mean of SQUARED approach queues   [0, 1]
     + w_pr · pressure_penalty(t)        — SIGNED approach-group imbalance   [-1, 1]
     + w_t  · throughput_reward(t)       — LOCAL approach-exit count         [0, 1]
     + w_s  · switch_penalty(t)          — phase-switching cost              [0, ∞)
     + w_sp · low_speed_penalty(t)       — mean approach speed below thr     [0, thr]
     + w_w  · queue_occupancy_penalty(t) — mean halted-queue occupancy       [0, 1]

With default weights (queue=-1, pressure=-0.5, throughput=+1, switch=-0.1,
low_speed=-0.2, queue_occupancy=-0.3) the raw range is [-1.94, +1.50],
matching the clip range [-2.0, +1.5] in RewardConfig.

1.3.0 changes (approach-level reward alignment)
-----------------------------------------------
- All reward terms now operate on per-approach aggregates matching the
  observation builder (no raw per-lane quantities, which are unrecoverable by
  a roadside camera in motorcycle-dominant mixed traffic).
- throughput_reward: per-lane ID diffs -> per-APPROACH vehicle-ID set diffs
  where each approach's set is the union of IDs across all lanes of that edge.
  A vehicle changing lanes within the same approach edge is no longer counted
  as an exit. On the camera side the counterpart is a stop-line-crossing count
  on tracked IDs (a track disappearing inside the ROI is NOT an exit). The
  global ``getArrived`` delta path is kept unchanged as automatic fallback.
- queue_occupancy (weight key renamed from ``waiting_time`` with backward-
  compatible fallback to ``waiting_time``): documented honestly as mean
  halted-queue occupancy across approaches, not accumulated delay.

1.2.0 changes (W5 campaign revision)
------------------------------------
- queue_penalty: mean(q)^2 -> mean(q^2) (anti-starvation, Jensen).
- pressure_penalty: one-sided max(red-green, 0) -> SIGNED in [-1, 1].
- throughput_reward: global arrival delta -> local per-agent exits (global
  path kept as automatic fallback).

Sim-to-real alignment at approach granularity (SUMO ↔ camera)
-------------------------------------------------------------
queue_penalty           mean_a(q_a^2) over approach edges      ↔ mean_a(q_a^2) over approach ROIs
pressure_penalty        signed group A/B approach q_a diff     ↔ signed group A/B ROI q_a diff
throughput_reward       union-set approach ID diffs            ↔ stop-line crossings on tracked IDs (ROI drop != exit)
switch_penalty          signal-controller action logic         ↔ identical action logic (no vision dependency)
low_speed_penalty       mean_a(v_a) count-weighted per edge    ↔ mean_a(v_a) ByteTrack speed per approach ROI
queue_occupancy_penalty mean_a(q_a) halted-queue occupancy     ↔ mean_a(q_a) halted-vehicle length fraction in ROI
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.traffic_env.components.observations import (
    aggregate_approach_metrics,
    group_lanes_by_approach,
)
from src.traffic_env.config import TrafficEnvConfig

logger = logging.getLogger(__name__)

LaneMetrics = Mapping[str, Any]
LaneCache   = Mapping[str, LaneMetrics]
ActionInfo  = Mapping[str, Any]
RewardDict  = Dict[str, float]

_REWARD_APPROACH_FEATURES: Tuple[str, ...] = (
    "effective_queue_norm",
    "avg_speed_norm",
)


@dataclass(slots=True)
class RewardCalculator:
    """Compute per-agent rewards from approach-aggregated traffic statistics.

    Parameters
    ----------
    config : TrafficEnvConfig
        Full environment configuration.  ``reward.weights``,
        ``reward.reward_scale``, and ``reward.reward_clip_*`` control shaping.

    Internal state
    --------------
    _prev_passed_total : float
        Running cumulative arrived-vehicle count.  Used only when the caller
        does not supply ``prev_passed`` in ``action_info``.  Reset via
        :meth:`reset` at episode boundaries.
    last_reward_details : dict
        Per-step component breakdown, keyed by ``tls_id``.  Written after
        every :meth:`calculate_rewards` call; useful for TensorBoard logging.
    """

    config: TrafficEnvConfig
    _prev_passed_total: float = 0.0
    last_reward_details: Dict[str, Dict[str, float]] = field(
        default_factory=dict, init=False
    )
    # per-approach vehicle-ID sets from the previous step (local throughput, 1.3.0)
    _prev_approach_vehicle_ids: Dict[Tuple[str, str], frozenset] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        self.config.validate()
        self.reset()

    def reset(self, prev_passed_total: float = 0.0) -> None:
        """Reset internal throughput trackers between episodes."""
        self._prev_passed_total = float(prev_passed_total)
        self.last_reward_details = {}
        self._prev_approach_vehicle_ids = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_rewards(
        self,
        tls_ids: Sequence[str],
        lane_cache: LaneCache,
        action_info: ActionInfo,
    ) -> RewardDict:
        """Calculate a scalar reward for each TLS agent.

        Expected ``action_info`` keys
        -----------------------------
        ``lane_ids_by_tls`` / ``controlled_lanes_dict``
            ``{tls_id: [lane_id, ...]}``  — all controlled lanes per agent
            (grouped by parent edge into approaches before computing terms).
        ``green_lanes_by_tls``  *(optional)*
            ``{tls_id: [lane_id, ...]}``  — lanes currently showing green
            (group A or B from ``lane_groups``). Required to enable
            ``pressure_penalty``; omit to disable it.
        ``current_passed`` / ``passed``
            Cumulative arrived-vehicle count at this step (fallback only).
        ``prev_passed``  *(optional)*
            Previous cumulative count.  If absent, internal tracker is used.
        ``delta_passed``  *(optional)*
            Pre-computed throughput delta; overrides current/prev when present.
        ``switches_by_tls``
            ``{tls_id: bool|int}``  — phase switches executed this step.
        ``num_switches``
            Shared integer switch count distributed across all agents.
        ``step_length_seconds``  *(optional, unused — kept for compatibility)*
            Previously used for waiting-time scaling; no longer applied.
        """
        tls_ids = tuple(tls_ids)
        if not tls_ids:
            return {}

        external = bool(self.config.observation.use_external_state)
        privileged_reward = bool(self.config.reward.privileged_reward)
        lane_ids_by_tls = self._resolve_lane_ids_by_tls(tls_ids, action_info)
        approaches_by_tls: Dict[str, List[Tuple[str, List[str]]]] = {
            tls_id: group_lanes_by_approach(
                lane_ids_by_tls.get(tls_id, ()), external_state=external
            )
            for tls_id in tls_ids
        }
        green_red_by_tls = self._resolve_green_red_approaches(
            tls_ids, action_info, approaches_by_tls, external_state=external
        )
        green_red_lanes_by_tls = (
            self._resolve_green_red_lanes(tls_ids, action_info, lane_ids_by_tls)
            if privileged_reward
            else {}
        )
        throughput_delta = self._resolve_throughput_delta(action_info)
        switch_counts    = self._resolve_switch_counts(tls_ids, action_info)

        # 1.3.0: snapshot current per-approach vehicle-ID sets (union across
        # all lanes of each approach edge) so intra-approach lane changes are
        # not exits. has_approach_ids is False when the cache carries no ID
        # data (e.g. unit tests with synthetic metrics) -> global fallback.
        current_ids_by_approach: Dict[Tuple[str, str], frozenset] = {}
        if not privileged_reward:
            for tls_id in tls_ids:
                for approach_key, group_lane_ids in approaches_by_tls[tls_id]:
                    ids: set[str] = set()
                    has_key = False
                    for lane_id in group_lane_ids:
                        raw_ids = lane_cache.get(lane_id, {}).get("vehicle_ids")
                        if raw_ids is not None:
                            has_key = True
                            ids.update(str(v) for v in raw_ids)
                    if has_key:
                        current_ids_by_approach[(tls_id, approach_key)] = frozenset(ids)
        has_approach_ids = bool(current_ids_by_approach)

        rewards: RewardDict = {}
        details: Dict[str, Dict[str, float]] = {}

        for tls_id in tls_ids:
            approaches = approaches_by_tls[tls_id]
            approach_keys = tuple(key for key, _ in approaches)
            approach_metrics: Dict[str, Dict[str, float]] = {
                key: aggregate_approach_metrics(
                    group_lane_ids, lane_cache, _REWARD_APPROACH_FEATURES
                )
                for key, group_lane_ids in approaches
            }

            if privileged_reward:
                lane_ids = lane_ids_by_tls.get(tls_id, ())
                green_lanes, red_lanes = green_red_lanes_by_tls[tls_id]
                queue_p = self._compute_privileged_queue_penalty(
                    tls_id, lane_ids, lane_cache
                )
                pressure_p = self._compute_privileged_pressure_penalty(
                    tls_id, green_lanes, red_lanes, lane_cache
                )
                throughput_r = self._compute_throughput_reward(
                    tls_id, throughput_delta, len(tls_ids)
                )
                queue_occ_p = self._compute_privileged_waiting_penalty(
                    tls_id, lane_ids, lane_cache
                )
            else:
                green_approaches, red_approaches = green_red_by_tls[tls_id]
                queue_p = self._compute_queue_penalty(
                    tls_id, approach_keys, approach_metrics
                )
                pressure_p = self._compute_pressure_penalty(
                    tls_id, green_approaches, red_approaches, approach_metrics
                )
                if has_approach_ids:
                    throughput_r = self._compute_local_throughput(
                        tls_id, approach_keys, current_ids_by_approach
                    )
                else:
                    throughput_r = self._compute_throughput_reward(
                        tls_id, throughput_delta, len(tls_ids)
                    )
                queue_occ_p = self._compute_queue_occupancy_penalty(
                    tls_id, approach_keys, approach_metrics
                )

            switch_p = self._compute_switch_penalty(tls_id, switch_counts)
            low_speed_p = self._compute_low_speed_penalty(
                tls_id, approach_keys, approach_metrics
            )

            w = self.config.reward.weights
            # Backward-compatible lookup: prefer 1.3.0 "queue_occupancy",
            # fall back to legacy "waiting_time" from pre-1.3.0 run configs.
            queue_occ_w = (
                self._weight(w, "queue_occupancy")
                if "queue_occupancy" in w
                else self._weight(w, "waiting_time")
            )
            raw = (
                self._weight(w, "queue")              * queue_p
                + self._weight(w, "pressure")         * pressure_p
                + self._weight(w, "throughput")       * throughput_r
                + self._weight(w, "switch_penalty")   * switch_p
                + self._weight(w, "low_speed_penalty")* low_speed_p
                + queue_occ_w                         * queue_occ_p
            )

            scaled  = float(raw * self.config.reward.reward_scale)
            clipped = float(
                np.clip(scaled,
                        self.config.reward.reward_clip_low,
                        self.config.reward.reward_clip_high)
            )

            rewards[tls_id] = clipped
            details[tls_id] = {
                "queue_penalty":           queue_p,
                "pressure_penalty":        pressure_p,
                "throughput_reward":       throughput_r,
                "switch_penalty":          switch_p,
                "low_speed_penalty":       low_speed_p,
                "queue_occupancy_penalty": queue_occ_p,
                "waiting_penalty":         queue_occ_p,  # legacy alias (pre-1.3.0)
                "raw_reward":              raw,
                "scaled_reward":           scaled,
                "clipped_reward":          clipped,
            }
            logger.debug(
                "[%s] r=%.4f  q=%.3f pr=%.3f tp=%.3f sw=%.3f sp=%.3f qo=%.3f",
                tls_id, clipped, queue_p, pressure_p, throughput_r,
                switch_p, low_speed_p, queue_occ_p,
            )

        if has_approach_ids:
            self._prev_approach_vehicle_ids = current_ids_by_approach

        self.last_reward_details = details
        return rewards

    # ------------------------------------------------------------------
    # Reward components
    # ------------------------------------------------------------------

    def _compute_queue_penalty(
        self,
        tls_id: str,
        approach_keys: Sequence[str],
        approach_metrics: Mapping[str, Mapping[str, float]],
    ) -> float:
        """Nonlinear congestion penalty across all controlled approaches.

        Formula (1.3.0): ``mean_a(q_a ** 2)`` where ``q_a`` is the approach-
        aggregated ``effective_queue_norm`` (unweighted mean over the approach's
        lanes, matching ``ObservationBuilder``). By Jensen's inequality
        mean(q_a^2) >= mean(q_a)^2 with equality only when all approaches are
        equal, so one saturated approach among empty ones is penalised heavily
        instead of being averaged away (anti-starvation guard). Output stays in
        [0, 1].

        Returns
        -------
        float in [0, 1]
        """
        if not approach_keys:
            logger.debug("[%s] queue_penalty: no approaches → 0.0", tls_id)
            return 0.0

        q_vals = [
            self._safe_metric(approach_metrics.get(key, {}), "effective_queue_norm")
            for key in approach_keys
        ]
        # Default: mean_a(q_a^2) (anti-starvation). Ablation
        # queue_mean_of_squares=False reverts to (mean_a(q_a))^2.
        if self.config.reward.queue_mean_of_squares:
            value = float(np.mean(np.square(q_vals)))
        else:
            value = float(np.square(np.mean(q_vals)))
        result = float(np.clip(value, 0.0, 1.0))
        logger.debug(
            "[%s] queue_penalty=%.4f (mean_q=%.4f)",
            tls_id, result, float(np.mean(q_vals)),
        )
        return result

    def _compute_pressure_penalty(
        self,
        tls_id: str,
        green_approaches: Optional[Sequence[str]],
        red_approaches: Optional[Sequence[str]],
        approach_metrics: Mapping[str, Mapping[str, float]],
    ) -> float:
        """SIGNED phase-aware approach queue-imbalance term (1.3.0, PRESSLIGHT-inspired).

        Pressure = (Σ_{a ∈ red} q_a − Σ_{a ∈ green} q_a) / n_approaches  ∈ [-1, 1]

        Green/red membership follows the TLS's group A/B approaches from
        ``lane_groups``.

        Positive  → red approaches carry more queue than green approaches:
                    green is held on the wrong direction. With the negative
                    weight (-0.5) this is a penalty.
        Negative  → green serves the more congested approaches; the negative
                    weight turns this into a positive reward.

        Returns 0.0 (neutral) when phase-group information is absent from
        ``action_info``.

        Returns
        -------
        float in [-1, 1]
        """
        if not green_approaches or not red_approaches:
            logger.debug("[%s] pressure_penalty: phase approaches unavailable → 0.0", tls_id)
            return 0.0

        sum_green = sum(
            self._safe_metric(approach_metrics.get(key, {}), "effective_queue_norm")
            for key in green_approaches
        )
        sum_red = sum(
            self._safe_metric(approach_metrics.get(key, {}), "effective_queue_norm")
            for key in red_approaches
        )
        n_approaches = max(len(green_approaches) + len(red_approaches), 1)
        raw = (sum_red - sum_green) / n_approaches
        # Default: SIGNED in [-1,1] (gradient on correct decisions too).
        # Ablation pressure_signed=False reverts to one-sided max(raw, 0).
        if not self.config.reward.pressure_signed:
            raw = max(raw, 0.0)
        result = float(np.clip(raw, -1.0, 1.0))
        logger.debug(
            "[%s] pressure_penalty=%.4f (red=%.3f green=%.3f n=%d signed=%s)",
            tls_id, result, sum_red, sum_green, n_approaches, self.config.reward.pressure_signed,
        )
        return result

    def _compute_local_throughput(
        self,
        tls_id: str,
        approach_keys: Sequence[str],
        current_ids_by_approach: Mapping[Tuple[str, str], frozenset],
    ) -> float:
        """LOCAL per-agent throughput from approach vehicle-ID exits (1.3.0).

        For each controlled approach edge ``a``, the active vehicle set is the
        union of vehicle IDs across all lanes of ``a``, and

            exits = Σ_a |prev_set_a \\ cur_set_a|.

        Because all lanes of an approach share one set, a vehicle changing
        lanes inside the same approach edge yields 0 exits; only vehicles that
        leave the approach edge are counted. Normalised by
        ``throughput_norm_divisor`` and clipped to [0, 1], with no cross-agent
        splitting.

        Camera counterpart: a stop-line-crossing count on ByteTrack tracked IDs
        leaving each approach ROI — a track that disappears inside the ROI
        (e.g., due to occlusion or tracker drop) is NOT counted as an exit.
        SUMO set-difference exits and camera stop-line crossings therefore
        measure the same approach-discharge quantity via distinct mechanisms
        (not identical implementations).

        Returns
        -------
        float in [0, 1]
        """
        exits = 0
        for approach_key in approach_keys:
            pair = (tls_id, approach_key)
            prev = self._prev_approach_vehicle_ids.get(pair)
            if prev is None:
                continue
            cur = current_ids_by_approach.get(pair, frozenset())
            exits += len(prev - cur)

        norm = exits / max(self.config.reward.throughput_norm_divisor, 1.0)
        result = float(np.clip(norm, 0.0, 1.0))
        logger.debug("[%s] local_throughput=%.4f (exits=%d)", tls_id, result, exits)
        return result

    def _compute_throughput_reward(
        self,
        tls_id: str,
        throughput_delta: float,
        num_tls: int,
    ) -> float:
        """FALLBACK global cleared-vehicles reward (pre-1.2.0 behaviour).

        Used only when the lane cache carries no per-lane vehicle IDs (e.g.
        synthetic test fixtures or external callers). The global throughput
        increment is split equally across agents — high variance and diluted
        credit; prefer the local path. Normalised by
        ``config.reward.throughput_norm_divisor``.

        Returns
        -------
        float in [0, 1]
        """
        if num_tls <= 0:
            logger.debug("[%s] throughput_reward: num_tls=0 → 0.0", tls_id)
            return 0.0

        norm   = throughput_delta / max(self.config.reward.throughput_norm_divisor, 1.0)
        result = float(np.clip(norm, 0.0, 1.0)) / float(num_tls)
        logger.debug("[%s] throughput_reward=%.4f (delta=%.1f)", tls_id, result, throughput_delta)
        return result

    def _compute_switch_penalty(
        self,
        tls_id: str,
        switch_counts: Mapping[str, float],
    ) -> float:
        """Phase-switching cost for this agent.

        Returns the raw switch count (typically 0 or 1 per step); the scale is
        controlled by the corresponding weight in ``RewardConfig.weights``.

        Returns
        -------
        float >= 0
        """
        result = max(self._safe_float(switch_counts.get(tls_id, 0.0)), 0.0)
        logger.debug("[%s] switch_penalty=%.4f", tls_id, result)
        return result

    def _compute_low_speed_penalty(
        self,
        tls_id: str,
        approach_keys: Sequence[str],
        approach_metrics: Mapping[str, Mapping[str, float]],
    ) -> float:
        """Penalise mean approach speed below the configured threshold.

        Each approach's ``avg_speed_norm`` is the vehicle-count-weighted mean
        across its lanes (via ``aggregate_approach_metrics``), and the term
        takes the unweighted mean across controlled approaches. In vision mode
        this is estimated from ByteTrack pixel displacement and is noisier than
        SUMO — consider reducing or zeroing its weight for vision deployment.

        Returns
        -------
        float in [0, config.reward.low_speed_threshold]
        """
        if not approach_keys:
            logger.debug("[%s] low_speed_penalty: no approaches → 0.0", tls_id)
            return 0.0

        speed_vals = [
            self._safe_metric(approach_metrics.get(key, {}), "avg_speed_norm")
            for key in approach_keys
        ]
        mean_speed = float(np.mean(speed_vals))
        threshold  = float(self.config.reward.low_speed_threshold)
        result     = max(threshold - mean_speed, 0.0)
        logger.debug("[%s] low_speed_penalty=%.4f (mean_spd=%.4f)", tls_id, result, mean_speed)
        return result

    def _compute_queue_occupancy_penalty(
        self,
        tls_id: str,
        approach_keys: Sequence[str],
        approach_metrics: Mapping[str, Mapping[str, float]],
    ) -> float:
        """Mean halted-queue occupancy across controlled approaches (linear).

        Computes ``mean_a(q_a)`` where ``q_a`` is each approach's aggregated
        ``effective_queue_norm`` (halted-vehicle length fraction, unweighted
        mean over the approach's lanes). This is a linear halted-queue
        occupancy penalty — NOT accumulated vehicle delay — complementing the
        quadratic anti-starvation ``queue_penalty`` (``mean_a(q_a^2)``). It is
        fully camera-computable with zero privileged simulator reads.

        (Renamed from ``waiting_penalty`` / ``waiting_time`` in revision 1.3.0;
        weight lookup keeps ``waiting_time`` as a backward-compatible alias.)

        Returns
        -------
        float in [0, 1]
        """
        if not approach_keys:
            logger.debug("[%s] queue_occupancy_penalty: no approaches → 0.0", tls_id)
            return 0.0

        q_vals = [
            self._safe_metric(approach_metrics.get(key, {}), "effective_queue_norm")
            for key in approach_keys
        ]
        result = float(np.clip(np.mean(q_vals), 0.0, 1.0))
        logger.debug("[%s] queue_occupancy_penalty=%.4f", tls_id, result)
        return result

    # ------------------------------------------------------------------
    # Privileged-reward components (A5: exact SUMO quantities per lane)
    # ------------------------------------------------------------------

    def _compute_privileged_queue_penalty(
        self,
        tls_id: str,
        lane_ids: Sequence[str],
        lane_cache: LaneCache,
    ) -> float:
        """Per-lane exact halted-vehicle count penalty (``privileged_halt_count_norm``)."""
        if not lane_ids:
            return 0.0
        q_vals = [
            self._safe_metric(lane_cache.get(lid, {}), "privileged_halt_count_norm")
            for lid in lane_ids
        ]
        if self.config.reward.queue_mean_of_squares:
            value = float(np.mean(np.square(q_vals)))
        else:
            value = float(np.square(np.mean(q_vals)))
        return float(np.clip(value, 0.0, 1.0))

    def _compute_privileged_pressure_penalty(
        self,
        tls_id: str,
        green_lanes: Optional[Sequence[str]],
        red_lanes: Optional[Sequence[str]],
        lane_cache: LaneCache,
    ) -> float:
        """Per-lane signed pressure from exact halted-vehicle counts."""
        if not green_lanes or not red_lanes:
            return 0.0
        sum_green = sum(
            self._safe_metric(lane_cache.get(lid, {}), "privileged_halt_count_norm")
            for lid in green_lanes
        )
        sum_red = sum(
            self._safe_metric(lane_cache.get(lid, {}), "privileged_halt_count_norm")
            for lid in red_lanes
        )
        n_lanes = max(len(green_lanes) + len(red_lanes), 1)
        raw = (sum_red - sum_green) / n_lanes
        if not self.config.reward.pressure_signed:
            raw = max(raw, 0.0)
        return float(np.clip(raw, -1.0, 1.0))

    def _compute_privileged_waiting_penalty(
        self,
        tls_id: str,
        lane_ids: Sequence[str],
        lane_cache: LaneCache,
    ) -> float:
        """Mean exact accumulated waiting time across controlled lanes."""
        if not lane_ids:
            return 0.0
        w_vals = [
            self._safe_metric(lane_cache.get(lid, {}), "privileged_waiting_norm")
            for lid in lane_ids
        ]
        return float(np.clip(np.mean(w_vals), 0.0, 1.0))

    def _resolve_green_red_lanes(
        self,
        tls_ids: Sequence[str],
        action_info: ActionInfo,
        lane_ids_by_tls: Mapping[str, Tuple[str, ...]],
    ) -> Dict[str, Tuple[Optional[Tuple[str, ...]], Optional[Tuple[str, ...]]]]:
        """Resolve per-lane green and red partitions for privileged pressure."""
        green_info = action_info.get("green_lanes_by_tls")
        if not isinstance(green_info, Mapping):
            return {tls_id: (None, None) for tls_id in tls_ids}

        result: Dict[str, Tuple[Optional[Tuple[str, ...]], Optional[Tuple[str, ...]]]] = {}
        for tls_id in tls_ids:
            raw = green_info.get(tls_id)
            if isinstance(raw, (list, tuple)):
                green_set = {str(x) for x in raw}
                all_lanes = lane_ids_by_tls.get(tls_id, ())
                green = tuple(lid for lid in all_lanes if lid in green_set)
                red = tuple(lid for lid in all_lanes if lid not in green_set)
                result[tls_id] = (green, red)
            else:
                result[tls_id] = (None, None)
        return result

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def _resolve_lane_ids_by_tls(
        self,
        tls_ids: Sequence[str],
        action_info: ActionInfo,
    ) -> Dict[str, Tuple[str, ...]]:
        """Resolve controlled lane IDs for each TLS from action_info."""
        mapping: Any = (
            action_info.get("lane_ids_by_tls")
            or action_info.get("controlled_lanes_dict")
        )
        if isinstance(mapping, Mapping):
            result: Dict[str, Tuple[str, ...]] = {}
            for tls_id in tls_ids:
                raw = mapping.get(tls_id)
                if isinstance(raw, (list, tuple)):
                    result[tls_id] = tuple(str(x) for x in raw)
                elif raw is not None:
                    result[tls_id] = (str(raw),)
                else:
                    result[tls_id] = ()
            return result

        fallback = action_info.get("lane_ids")
        if isinstance(fallback, (list, tuple)):
            shared = tuple(str(x) for x in fallback)
            return {tls_id: shared for tls_id in tls_ids}

        return {tls_id: () for tls_id in tls_ids}

    def _resolve_green_red_approaches(
        self,
        tls_ids: Sequence[str],
        action_info: ActionInfo,
        approaches_by_tls: Mapping[str, Sequence[Tuple[str, List[str]]]],
        *,
        external_state: bool = False,
    ) -> Dict[str, Tuple[Optional[Tuple[str, ...]], Optional[Tuple[str, ...]]]]:
        """Resolve green approaches and red approaches (complement) per TLS.

        Maps ``green_lanes_by_tls`` (populated from ``lane_groups`` group A/B)
        to approach keys using the same edge-grouping rule as observations.
        If ``green_lanes_by_tls`` is absent from ``action_info``, entries are
        ``(None, None)`` and ``pressure_penalty`` returns 0.0.
        """
        green_info = action_info.get("green_lanes_by_tls")
        if not isinstance(green_info, Mapping):
            return {tls_id: (None, None) for tls_id in tls_ids}

        result: Dict[str, Tuple[Optional[Tuple[str, ...]], Optional[Tuple[str, ...]]]] = {}
        for tls_id in tls_ids:
            raw = green_info.get(tls_id)
            if isinstance(raw, (list, tuple)):
                green_keys = {
                    key
                    for key, _ in group_lanes_by_approach(
                        tuple(str(x) for x in raw), external_state=external_state
                    )
                }
                all_keys = [key for key, _ in approaches_by_tls.get(tls_id, ())]
                green = tuple(key for key in all_keys if key in green_keys)
                red = tuple(key for key in all_keys if key not in green_keys)
                result[tls_id] = (green, red)
            else:
                result[tls_id] = (None, None)

        return result

    def _resolve_throughput_delta(self, action_info: ActionInfo) -> float:
        """Compute per-step throughput increment.

        Precedence:
        1. ``delta_passed`` — pre-computed; used as-is.
        2. ``current_passed`` − ``prev_passed`` — caller-managed tracking.
        3. ``current_passed`` − ``_prev_passed_total`` — internal tracking.

        The internal ``_prev_passed_total`` is updated **only** when the caller
        does not supply ``prev_passed``.  This prevents state corruption when
        calls alternate between internal and external tracking.
        """
        if "delta_passed" in action_info:
            return max(self._safe_float(action_info["delta_passed"]), 0.0)

        current = action_info.get("current_passed") or action_info.get("passed")
        if current is None:
            return 0.0

        current_f          = max(self._safe_float(current), 0.0)
        caller_owns_prev   = "prev_passed" in action_info
        prev_f             = max(
            self._safe_float(
                action_info.get("prev_passed", self._prev_passed_total)
            ),
            0.0,
        )
        delta = max(current_f - prev_f, 0.0)

        if not caller_owns_prev:
            self._prev_passed_total = current_f

        return delta

    def _resolve_switch_counts(
        self,
        tls_ids: Sequence[str],
        action_info: ActionInfo,
    ) -> Dict[str, float]:
        """Resolve per-TLS phase-switch counts from action_info."""
        switches_by_tls = action_info.get("switches_by_tls")
        if isinstance(switches_by_tls, Mapping):
            return {
                tls_id: max(self._safe_float(switches_by_tls.get(tls_id, 0.0)), 0.0)
                for tls_id in tls_ids
            }

        num_switches = action_info.get("num_switches")
        if num_switches is not None and tls_ids:
            shared = max(self._safe_float(num_switches), 0.0) / float(len(tls_ids))
            return {tls_id: shared for tls_id in tls_ids}

        switched_tls = action_info.get("switched_tls")
        if isinstance(switched_tls, Mapping):
            return {
                tls_id: 1.0 if switched_tls.get(tls_id, False) else 0.0
                for tls_id in tls_ids
            }

        return {tls_id: 0.0 for tls_id in tls_ids}

    # ------------------------------------------------------------------
    # Micro-utilities
    # ------------------------------------------------------------------

    def _weight(self, weights: Mapping[str, float], key: str) -> float:
        try:
            return float(weights.get(key, 0.0))
        except Exception:
            return 0.0

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(default if value is None else value)
        except Exception:
            return float(default)

    def _safe_metric(
        self, metrics: LaneMetrics, key: str, default: float = 0.0
    ) -> float:
        try:
            return self._safe_float(metrics.get(key, default), default=default)
        except Exception:
            return float(default)


__all__ = ["RewardCalculator", "LaneCache", "LaneMetrics", "ActionInfo", "RewardDict"]
