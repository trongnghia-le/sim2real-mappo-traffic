# Reward Function — Design Reference

**File:** `src/traffic_env/components/rewards.py`  
**Config:** `src/traffic_env/config.py` → `RewardConfig`, `DEFAULT_REWARD_WEIGHTS`  
**Revision:** **1.3.0** (approach-level reward alignment — see §6 for the change log)

> **LEGACY WARNING:** All paper results require `env_cfg.version >= 1.3.0`
> (guarded by `PolicyLoader` and `eval_compare.py`). All 1.2.0-era checkpoints
> (per-lane reward aggregation, per-lane ID diffs, un-enforced env max-green)
> and legacy checkpoint `20260418_215140` (pre-1.1.0, bug R1 inert pressure)
> are invalid for paper results; do not pool results across schema versions.

---

## 1. Reward Equation

Every reward term operates on **approach aggregates** ($a \in A$, where each controlled approach groups all incoming lanes of one parent edge via `group_lanes_by_approach` and aggregates them via `aggregate_approach_metrics` — the exact same rule used by `ObservationBuilder`: unweighted mean across lanes for queue/occupancy, vehicle-count-weighted mean for speed):

$$
R(t) = w_q \cdot \text{queue}(t)
     + w_{pr} \cdot \text{pressure}(t)
     + w_t \cdot \text{throughput}(t)
     + w_s \cdot \text{switch}(t)
     + w_{sp} \cdot \text{low\_speed}(t)
     + w_w \cdot \text{queue\_occupancy}(t)
$$

**Final output:**

$$
r_{\text{agent}} = \text{clip}(R(t) \times \text{scale},\ \text{clip\_low},\ \text{clip\_high})
$$

All component functions return values in `[0, 1]` except `pressure` which is **signed** in `[−1, 1]` and `switch_penalty` which is unbounded above. Signs are applied via weights. **Penalty weights are negative; reward weights are positive** — for the signed pressure term this means wrong-side green is penalised AND correct-side green is rewarded.

---

## 2. Component Reference

### 2.1 `queue_penalty` — Nonlinear Congestion *(1.3.0: approach-level mean of squares)*

$$
\text{queue}(t) = \text{clip}\!\left(\frac{1}{|A|}\sum_{a \in A} q_a^{\,2},\; 0,\; 1\right)
\quad\text{where}\quad
q_a = \frac{1}{|L_a|}\sum_{l \in L_a} \text{effective\_queue\_norm}(l)
$$

| Property | Value |
|---|---|
| Range | `[0, 1]` |
| Default weight | `−1.0` |
| SUMO source | Approach-averaged `effective_queue_norm` (`Σ halted_len / lane_len`, `v ≤ 0.1 m/s`) |
| Camera counterpart | Halted-vehicle length fraction in approach ROI polygon |
| Ablation flag | `queue_mean_of_squares=False` computes $(\frac{1}{|A|}\sum_{a \in A} q_a)^2$ |

**Rationale:** Squaring provides near-zero gradient in free-flow conditions and strong gradient during sustained congestion (PRESSLIGHT's saturation-aware design). By Jensen's inequality `mean_a(q_a²) ≥ (mean_a q_a)²` with equality only when all approaches are equal, so one saturated approach among empty ones cannot hide behind a low average (anti-starvation guard).

---

### 2.2 `pressure_penalty` — Phase-Aware Approach Queue Imbalance *(PRESSLIGHT / CoLight; 1.3.0: approach-level SIGNED)*

$$
\text{pressure}(t) = \text{clip}\!\left(\frac{\sum_{a \in A_{\text{red}}} q_a - \sum_{a \in A_{\text{green}}} q_a}{|A|},\; -1,\; 1\right)
$$

| Property | Value |
|---|---|
| Range | **`[−1, 1]`** (signed; `pressure_signed=False` clips below at `0`) |
| Default weight | `−0.5` |
| Green/red membership | TLS group A / group B approaches from `lane_groups` (wired via `green_lanes_by_tls`) |
| Returns | `0.0` (neutral) if phase-group info is absent |
| Camera counterpart | Same signed group A/B difference over approach ROI `q_a` |

**Rationale:** Core PRESSLIGHT signal (Wei et al., KDD 2019) at approach granularity. Positive value = red approaches carry more queue than green approaches (wrong phase held; penalised by `−0.5`). Negative value = green serves the more congested approaches (rewarded by `−0.5`, restoring gradient on correct decisions).

**Bug R1 (FIXED 2026-06-12):** `MappoTrafficEnv.step` previously never passed `green_lanes_by_tls`, leaving pressure silently `0.0` in all earlier runs (including `20260418_215140`). It is now populated every step from `lane_groups` (see §7).

---

### 2.3 `throughput_reward` — LOCAL Approach-Exit Count *(1.3.0)*

**Primary path (1.3.0):** per-agent count of vehicles that exited any controlled approach edge since the previous decision step. For each approach $a$, the active vehicle set is the **union** of vehicle IDs across all lanes $l \in L_a$ of that approach edge:

$$
\text{ids}_t(a) = \bigcup_{l \in L_a} \text{ids}_t(l),
\qquad
\text{throughput}(t) = \text{clip}\!\left(\frac{1}{D}\sum_{a \in A} \bigl|\,\text{ids}_{t-1}(a) \setminus \text{ids}_t(a)\,\bigr|,\; 0,\; 1\right)
$$

| Property | Value |
|---|---|
| Range | `[0, 1]` |
| Default weight | `+1.0` |
| `D` (divisor) | `config.reward.throughput_norm_divisor` = 20 |
| SUMO source | Per-approach union of `lane_cache[lane_id]["vehicle_ids"]`, differenced across steps |
| Camera counterpart | Stop-line-crossing count in `src/vision/state_extractor.py` (`LaneROI.stop_line`, bbox bottom-centre crossing in exit direction, `min_track_frames=3`, counted at most once per track ID; tracks disappearing inside the ROI are NOT exits) |

Because all lanes of an approach share a single set, a vehicle changing lanes within the same approach edge yields **0 exits** (fixing the 1.2.0 per-lane double-counting bug in sub-lane mixed traffic). Each agent is credited only for traffic it served — no cross-agent splitting. Note that SUMO set-difference exits and camera stop-line crossings are **not identical mechanisms**: on the camera side, a track lost inside the ROI due to occlusion or detector drop must not be counted as an exit, so `StateExtractor` uses an explicit stop-line segment (`LaneROI.stop_line`, exposed per decision step via `StateExtractor.get_step_exits()` and `build_state(..., return_info=True)["approach_exits"]`) rather than raw ID disappearance.

**Fallback path (pre-1.2.0 behaviour, untouched):** when the lane cache carries no `vehicle_ids` (synthetic test fixtures, external callers), the global `simulation.getArrived()` delta is split equally across agents:

$$
\text{throughput}_{\text{fallback}}(t) = \frac{\text{clip}(\Delta V / D,\; 0,\; 1)}{N_{\text{agents}}}
\quad\text{where}\quad
\Delta V = V_{\text{current}} - V_{\text{prev}}
$$

The switch is automatic per step (`has_approach_ids` in `calculate_rewards`).

---

### 2.4 `switch_penalty` — Phase-Switching Cost

$$
\text{switch}(t) = \mathbb{1}[\text{phase changed at step } t]
$$

| Property | Value |
|---|---|
| Range | `[0, ∞)` — typically `{0, 1}` |
| Default weight | `−0.1` |
| Source | `action_info["switches_by_tls"]` |

Prevents rapid phase cycling. No camera dependency.

---

### 2.5 `low_speed_penalty` — Mean Approach Speed Below Threshold *(1.3.0)*

$$
\text{low\_speed}(t) = \max(\tau_s - \overline{v},\; 0),
\quad
\overline{v} = \frac{1}{|A|}\sum_{a \in A} v_a,
\quad
v_a = \frac{\sum_{l \in L_a} n_l\,\text{avg\_speed\_norm}(l)}{\sum_{l \in L_a} n_l}
$$

(falling back to unweighted lane mean on empty approaches where $\sum_{l \in L_a} n_l = 0$, identical to `aggregate_approach_metrics`).

| Property | Value |
|---|---|
| Range | `[0, τ_s]` |
| Default weight | `−0.2` |
| `τ_s` | `config.reward.low_speed_threshold` = 0.20 |
| Camera counterpart | Mean ByteTrack displacement speed across tracked vehicles in each approach ROI |

> **Vision deployment note:** Speed estimated from ByteTrack pixel displacement is significantly noisier than SUMO speed. Consider setting `"low_speed_penalty": 0.0` in `RewardConfig.weights` when evaluating reward diagnostics on vision streams.

---

### 2.6 `queue_occupancy` (formerly `waiting_penalty`) — Mean Halted-Queue Occupancy *(1.3.0)*

$$
\text{queue\_occupancy}(t) = \text{clip}\!\left(\frac{1}{|A|}\sum_{a \in A} q_a,\; 0,\; 1\right)
\quad\text{where}\quad
q_a = \frac{1}{|L_a|}\sum_{l \in L_a} \text{effective\_queue\_norm}(l)
$$

| Property | Value |
|---|---|
| Range | `[0, 1]` |
| Default weight | `−0.3` (weight key `"queue_occupancy"`; backward-compatible fallback to legacy `"waiting_time"`) |
| SUMO source | Mean approach `effective_queue_norm` (halted-vehicle length fraction, `v ≤ 0.1 m/s`) |
| Camera counterpart | Mean halted-vehicle length fraction across approach ROIs (same camera quantity as `queue_penalty`) |

**Clarification (1.3.0):** This term is the **linear mean halted-queue occupancy** across approaches, **not** accumulated vehicle delay (`getWaitingTime` is never read by the default reward). It provides a linear congestion penalty complementing the quadratic anti-starvation `queue_penalty` (`mean_a(q_a²)`), so every reward term has a camera-side computation at approach granularity in both the camera-proxy and privileged-observation reference arms. The weight key was renamed from `"waiting_time"` to `"queue_occupancy"` in 1.3.0, while `RewardCalculator` falls back to `"waiting_time"` when loading pre-1.3.0 weight dicts.

### 2.7 Privileged-Reward Ablation (`--reward-preset privileged`, 1.3.0)

To isolate the camera-computable reward design from the observation restriction, `train_ppo.py --reward-preset privileged` sets `RewardConfig.privileged_reward=True` while keeping the **26-dim camera-proxy observation unchanged**. In this ablation arm only, `RewardCalculator` substitutes exact per-lane simulator quantities:

- `queue_penalty`: per-lane mean of squared `privileged_halt_count_norm` (exact halted counts),
- `pressure_penalty`: per-lane signed `privileged_halt_count_norm` imbalance across green/red lanes,
- `throughput_reward`: global `simulation.getArrived()` delta split across agents,
- `queue_occupancy` slot: per-lane mean `privileged_waiting_norm` (exact accumulated SUMO waiting time).

`switch_penalty` and `low_speed_penalty` keep their standard definitions and weights.

---

## 3. Value Ranges with Default Weights

| Component | Raw range | Weight | Contribution |
|---|---|---|---|
| `queue_penalty` | `[0, 1]` | `−1.0` | `[−1.0, 0]` |
| `pressure_penalty` | `[−1, 1]` | `−0.5` | `[−0.5, +0.5]` |
| `throughput_reward` | `[0, 1]` | `+1.0` | `[0, +1.0]` |
| `switch_penalty` | `[0, 1]` | `−0.1` | `[−0.1, 0]` |
| `low_speed_penalty` | `[0, 0.2]` | `−0.2` | `[−0.04, 0]` |
| `queue_occupancy` | `[0, 1]` | `−0.3` | `[−0.3, 0]` |
| **Total (raw)** | | | **`[−1.94, +1.50]`** |
| **After clip** | | | **`[−2.0, +1.5]`** |

---

## 4. `action_info` Protocol

```python
action_info = {
    # Required
    "lane_ids_by_tls": {
        "J0": ["lane_id_0", "lane_id_1", ...],
        "J2": ["lane_id_4", "lane_id_5", ...],
    },

    # Recommended — enables pressure_penalty (mapped to approach groups via lane_groups)
    "green_lanes_by_tls": {
        "J0": ["lane_id_0", "lane_id_1"],   # currently green group
        "J2": ["lane_id_4", "lane_id_5"],
    },

    # Throughput FALLBACK (used only when lane_cache lacks "vehicle_ids");
    # one of these three:
    "delta_passed": 5,                # preferred: pre-computed delta
    "current_passed": 120,            # OR: current cumulative + prev below
    "prev_passed": 115,
    # (if neither, internal _prev_passed_total is used)

    # Switch info (one of these)
    "switches_by_tls": {"J0": True, "J2": False},  # preferred
    "num_switches": 1,                              # OR: distributed equally
    "switched_tls": {"J0": True, "J2": False},      # OR: boolean map

    # Optional
    "step_length_seconds": 5,  # kept for compatibility; no longer used
}
```

**Local approach throughput (1.3.0) does not use `action_info`** — it unions
`lane_cache[lane_id]["vehicle_ids"]` over all lanes belonging to each approach
edge and diffs the per-approach ID sets across consecutive `calculate_rewards`
calls. `RewardCalculator.reset()` clears `_prev_approach_vehicle_ids` at episode
boundaries.

---

## 5. Sim-to-Real Alignment (Approach Granularity)

The reward is **only computed during training in SUMO**, never at deployment. Every reward term has a camera-side computation at approach granularity ($a \in A$: incoming edge in SUMO ↔ approach ROI in camera) so the training objective depends exclusively on quantities a roadside camera can observe in motorcycle-dominant mixed traffic:

- **`queue_penalty`**: SUMO `mean_a(q_a²)` where `q_a` is the unweighted mean of `effective_queue_norm` (`Σ halted_len / lane_len`, `v ≤ 0.1 m/s`) over lanes of approach `a` `<->` camera `mean_a(q_a²)` where `q_a` is the halted-vehicle length fraction (`Σ class_len_px / roi_len_px`, `v < 0.1 m/s`) in approach ROI `a`.
- **`pressure_penalty`**: SUMO `(Σ_{a∈A_red} q_a − Σ_{a∈A_green} q_a) / |A|` partitioned by group A/B approaches (`lane_groups`) `<->` camera signed difference `(Σ_{a∈A_red} q_a − Σ_{a∈A_green} q_a) / |A|` over group A/B approach ROIs using camera `q_a`.
- **`throughput_reward`**: SUMO `clip((1/D) Σ_a |ids_{t-1}(a) \ ids_t(a)|, 0, 1)` on per-approach vehicle-ID unions across all lanes of edge `a` (intra-approach lane changes = 0 exits) `<->` camera `clip((1/D) Σ_a crossings_t(a), 0, 1)` from `StateExtractor.get_step_exits()` (`src/vision/state_extractor.py`: stop-line crossing on bbox bottom-centre, `min_track_frames=3`, counted at most once per track ID; tracks disappearing inside the ROI are NOT exits).
- **`switch_penalty`**: SUMO `1[phase switched at step t]` from signal-controller action state `<->` camera/deployment identical signal-controller action logic (zero vision dependency).
- **`low_speed_penalty`**: SUMO `max(τ_s − mean_a(v_a), 0)` where `v_a` is the vehicle-count-weighted mean of `avg_speed_norm` across lanes of approach `a` `<->` camera `max(τ_s − mean_a(v_a), 0)` where `v_a` is the mean ByteTrack displacement speed (`Δpos / Δt / speed_cap`) across tracked vehicles in approach ROI `a`.
- **`queue_occupancy`**: SUMO `mean_a(q_a)` unweighted mean of approach halted-queue occupancy `q_a` across approaches `<->` camera `mean_a(q_a)` mean halted-vehicle length fraction across approach ROIs (linear counterpart to `queue_penalty`).

**Recommended weights for vision-aligned reward diagnostics:**

```python
from src.traffic_env.config import RewardConfig

vision_weights = RewardConfig(weights={
    "queue":            -1.0,
    "pressure":         -0.5,
    "throughput":        0.5,   # reduced: stop-line tracker noise under occlusion
    "switch_penalty":   -0.1,
    "low_speed_penalty": 0.0,   # disabled: ByteTrack speed estimate too noisy
    "queue_occupancy":  -0.3,
})
```

---

## 6. Change Log

### Revision 1.3.0 (approach-level reward alignment)

| Change | Old behaviour (1.2.0) | Fixed behaviour (1.3.0) |
|---|---|---|
| A1 — Approach-level reward | Iterated raw controlled lanes (`lane_ids`, `green_lane_ids`, `red_lane_ids`) — inconsistent with approach-aggregated observations | Every term aggregates controlled lanes into approaches via `group_lanes_by_approach` + `aggregate_approach_metrics` (exact same rule as `ObservationBuilder`: unweighted mean for queue, vehicle-count-weighted mean for speed) |
| A2 — Approach-exit throughput | Per-lane ID set diffs counted intra-approach lane changes as exits; docstrings claimed SUMO and camera mechanisms were "identical" | Per-approach union ID sets (`Σ_a \|prev_a \ cur_a\|`), so lane changes inside an edge yield 0 exits; camera counterpart documented as stop-line-crossing count on tracked IDs (track disappearance inside ROI ≠ exit); global `getArrived` fallback untouched |
| A3 — `queue_occupancy` naming | Term documented as "accumulated delay" / keyed `"waiting_time"` despite always using `effective_queue_norm` | Documented as mean halted-queue occupancy across approaches; weight key renamed to `"queue_occupancy"` with backward-compatible fallback to legacy `"waiting_time"` |
| A5 — Privileged-reward ablation | Only observation-restriction arm existed (`--obs-mode privileged`) | Added `--reward-preset privileged` (`RewardConfig.privileged_reward=True`): keeps 26-dim camera obs with exact per-lane simulator queue/pressure/waiting and global arrival throughput |
| V1 — Camera stop-line exits | `src/vision/state_extractor.py` had no exit counting | Added `LaneROI.stop_line` + directed bottom-centre crossing counter (`min_track_frames=3`, single count per ID, exposed via `get_step_exits()` and `build_state(..., return_info=True)`) |

### Revision 1.2.0 (2026-06-12, W5 campaign)

| Bug | Old behaviour | Fixed behaviour |
|---|---|---|
| R3 | Pressure one-sided `max(red−green, 0)` — zero gradient whenever the allocation was correct | **Signed** `(Σq_red − Σq_green)/n_total ∈ [−1, 1]` — penalty for wrong-side green AND reward for correct-side green |
| R4 | Queue penalty `mean(q)²` — single-lane starvation averaged away | `mean(q²)` — mean of squares (Jensen, anti-starvation) |
| R5 | Global `getArrived()` delta split equally across agents — variance + diluted credit | **Local** per-agent lane-exit count from vehicle-ID set differences; global path kept as automatic fallback |

### Revision 1.1.0 (2026-06-12, audit v3)

| Bug | Old behaviour | Fixed behaviour |
|---|---|---|
| R1 | `green_lanes_by_tls` never passed by `MappoTrafficEnv.step` → pressure term silently `0.0` in ALL training runs to date | Wired in `multi_agent.py` from manual lane groups (see §7) |

### Earlier revision

| Issue | Old behaviour | Fixed behaviour |
|---|---|---|
| `waiting_penalty` formula | Multiplied by `step_length/max_green` → max ≤ 0.083 | Direct `effective_queue_norm` proxy; range `[0, 1]` |
| `_prev_passed_total` side-effect | Always mutated even when caller supplied `prev_passed` | Only mutated when caller does **not** supply `prev_passed` |
| Double-counting queue | `queue_penalty` + `high_queue_penalty` both used `effective_queue_norm` | `high_queue_penalty` removed; replaced by `pressure_penalty` |
| Linear queue penalty | `0.85 × queue + 0.15 × occupancy` (magic numbers) | `mean_queue ** 2` (nonlinear, no magic numbers) |
| Pressure signal | Absent | `pressure_penalty` (PRESSLIGHT formulation) added |
| `tls_id` unused in private methods | Accepted but silently ignored | Used for `logger.debug` output in every method |
| Dead code | `_weight_with_default` never called | Removed |
| Reward clip | `[−2.0, +2.0]` — never fired (actual range `[−1.24, +0.50]`) | `[−2.0, +1.5]` — both bounds now active |

---

## 7. `pressure_penalty` Wiring in `multi_agent.py` (DONE — bug R1)

The pressure component returns `0.0` until the environment populates
`green_lanes_by_tls` in `action_info`. **This is wired since 2026-06-12** —
`MappoTrafficEnv.step` maps the applied green phase back to its manual lane
group every step (`src/traffic_env/envs/multi_agent.py`, reward section):

```python
# multi_agent.py step() — actual implementation
lane_groups = self.config.get_lane_groups()
green_lanes_by_tls: Dict[str, List[str]] = {}
for tls_id in self.tls_ids:
    groups    = lane_groups.get(tls_id)
    phase_map = self._green_phase_map.get(tls_id, [])
    target    = target_phases.get(tls_id)
    if groups and target in phase_map:
        group_idx = phase_map.index(target)
        green_lanes_by_tls[tls_id] = list(groups[0] if group_idx == 0 else groups[1])

action_info["green_lanes_by_tls"] = green_lanes_by_tls
```

`RewardCalculator` groups `green_lanes_by_tls` and `lane_ids_by_tls` by parent
edge so pressure operates on the TLS's group A/B approaches.

---

## 8. Configuration Reference

```python
# src/traffic_env/config.py

DEFAULT_REWARD_WEIGHTS = {
    "queue":            -1.0,
    "pressure":         -0.5,
    "throughput":        1.0,
    "switch_penalty":   -0.1,
    "low_speed_penalty":-0.2,
    "queue_occupancy":  -0.3,   # legacy alias "waiting_time" still accepted
}

@dataclass
class RewardConfig:
    weights:                  dict  = DEFAULT_REWARD_WEIGHTS
    reward_scale:             float = 1.0
    reward_clip_low:          float = -2.0
    reward_clip_high:         float = 1.5
    throughput_norm_divisor:  float = 20.0   # vehicles per step at saturation
    low_speed_threshold:      float = 0.20   # normalised speed below which penalty applies
    jam_speed_threshold:      float = 0.50   # informational; not used in reward
    jam_queue_threshold:      float = 0.50   # informational; not used in reward
    pressure_signed:          bool  = True
    queue_mean_of_squares:    bool  = True
    privileged_reward:        bool  = False   # --reward-preset privileged ablation (26-dim camera obs, exact SUMO reward)
```

---

## 9. References

- Wei et al. (2019). *PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network.* KDD 2019.
- Wei et al. (2019). *CoLight: Learning Network-level Cooperation for Traffic Signal Control.* CIKM 2019.
- Chen et al. (2020). *Toward A Thousand Lights: Decentralized Deep Reinforcement Learning for Large-Scale Traffic Signal Control.* AAAI 2020.
