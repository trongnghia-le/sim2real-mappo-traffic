"""Shared tripinfo CLI flags and parsing for baselines, evaluation and pilot runs."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Canonical extra SUMO tripinfo flags used by EVERY evaluation path (both
# MappoTrafficEnv runs and SUMO-native actuated runs) so all controllers write
# identical XML schemas and unfinished vehicles are explicitly accounted for.
TRIPINFO_EXTRA_ARGS: Tuple[str, ...] = ("--tripinfo-output.write-unfinished",)


def tripinfo_sumo_args(tripinfo_path: str | Path) -> List[str]:
    """Return the shared SUMO CLI args for writing tripinfo XML."""
    return ["--tripinfo-output", str(tripinfo_path), *TRIPINFO_EXTRA_ARGS]


def _is_completed_trip(trip: ET.Element) -> bool:
    """Return True iff a SUMO 1.24 <tripinfo> entry represents a completed trip.

    SUMO 1.24 semantics with ``--tripinfo-output.write-unfinished``:
      * Completed trips have ``arrival >= 0`` (arrival timestamp in seconds)
        and ``vaporized=""`` (empty).
      * Unfinished in-flight trips at episode end are written with
        ``arrival="-1.00"`` (and ``vaporized="end"`` or ``vaporized=""``).
      * Prematurely removed vehicles (teleport/collision/vaporizer/end) carry
        a non-empty ``vaporized`` attribute.
    """
    arrival_attr = trip.get("arrival")
    arrival = float(arrival_attr) if arrival_attr is not None else 0.0
    vaporized = (trip.get("vaporized") or "").strip()
    return arrival >= 0.0 and vaporized == ""


def parse_tripinfo_means(path: str | Path) -> Dict[str, float]:
    """Parse a SUMO --tripinfo-output file into network-level metrics.

    Primary metrics are computed strictly over **completed** trips so truncated
    in-flight vehicles cannot artificially lower travel/waiting means:
        n_trips                        count of completed trips
        mean_travel_time_s             mean 'duration' of completed trips
        mean_waiting_time_s            mean 'waitingTime' of completed trips
        mean_time_loss_s               mean 'timeLoss' of completed trips
        p95_waiting_time_s             95th percentile 'waitingTime' (completed)
        total_co2_mg                   sum of <emissions CO2_abs> (completed; NaN if absent)
        total_fuel_mg                  sum of <emissions fuel_abs> (completed; NaN if absent)

    Additional unfinished-aware columns prevent controllers that trap vehicles
    in the network from looking artificially better:
        n_unfinished                   count of unfinished / vaporized trips
        mean_time_loss_incl_unfinished mean 'timeLoss' across all (completed + unfinished) trips
        mean_travel_time_incl_unfinished mean 'duration' across all (completed + unfinished) trips
    """
    path = Path(path)
    comp_durations: List[float] = []
    comp_waits: List[float] = []
    comp_losses: List[float] = []
    comp_co2: List[float] = []
    comp_fuel: List[float] = []

    all_durations: List[float] = []
    all_losses: List[float] = []
    n_unfinished = 0

    for trip in ET.parse(path).getroot().iter("tripinfo"):
        dur = float(trip.get("duration", 0.0))
        wait = float(trip.get("waitingTime", 0.0))
        loss = float(trip.get("timeLoss", 0.0))
        all_durations.append(dur)
        all_losses.append(loss)

        if _is_completed_trip(trip):
            comp_durations.append(dur)
            comp_waits.append(wait)
            comp_losses.append(loss)
            em = trip.find("emissions")
            if em is not None:
                comp_co2.append(float(em.get("CO2_abs", 0.0)))
                comp_fuel.append(float(em.get("fuel_abs", 0.0)))
        else:
            n_unfinished += 1

    mean_loss_all = float(np.mean(all_losses)) if all_losses else float("nan")
    mean_dur_all = float(np.mean(all_durations)) if all_durations else float("nan")

    if not comp_durations:
        return {
            "n_trips": 0,
            "mean_travel_time_s": float("nan"),
            "mean_waiting_time_s": float("nan"),
            "mean_time_loss_s": float("nan"),
            "p95_waiting_time_s": float("nan"),
            "total_co2_mg": float("nan"),
            "total_fuel_mg": float("nan"),
            "n_unfinished": n_unfinished,
            "mean_time_loss_incl_unfinished": mean_loss_all,
            "mean_travel_time_incl_unfinished": mean_dur_all,
        }

    return {
        "n_trips": len(comp_durations),
        "mean_travel_time_s": float(np.mean(comp_durations)),
        "mean_waiting_time_s": float(np.mean(comp_waits)),
        "mean_time_loss_s": float(np.mean(comp_losses)),
        "p95_waiting_time_s": float(np.percentile(comp_waits, 95)),
        # emissions are post-hoc only (never in the reward); NaN when the SUMO
        # emissions device was not active for this run.
        "total_co2_mg": float(np.sum(comp_co2)) if comp_co2 else float("nan"),
        "total_fuel_mg": float(np.sum(comp_fuel)) if comp_fuel else float("nan"),
        "n_unfinished": n_unfinished,
        "mean_time_loss_incl_unfinished": mean_loss_all,
        "mean_travel_time_incl_unfinished": mean_dur_all,
    }


__all__ = [
    "TRIPINFO_EXTRA_ARGS",
    "tripinfo_sumo_args",
    "parse_tripinfo_means",
]
