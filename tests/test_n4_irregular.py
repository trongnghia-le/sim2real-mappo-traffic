"""
tests/test_n4_irregular.py
==========================
Unit tests for the third evaluation topology ``n4_irregular`` (3x6 irregular
grid, 18 signalized junctions, non-uniform spacing). No live SUMO required.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiment.runners import eval_compare
from scripts import check_obs_match, generate_demand, generate_networks

ROOT = Path(__file__).resolve().parents[1]
N3_DIR = ROOT / "sumo_configs" / "networks" / "n3_grid"
N4_DIR = ROOT / "sumo_configs" / "networks" / "n4_irregular"


@pytest.mark.unit
class TestN4IrregularTopology:
    def test_required_files_exist_and_vtypes_match_n3(self):
        for fname in (
            "intersections.net.xml",
            "vtypes.add.xml",
            "lane_groups.json",
            "sumo_config.sumocfg",
            "sumo_config_eval.sumocfg",
            "demand_train.trips.xml",
            "demand_train.rou.xml",
            "demand_eval.trips.xml",
            "demand_eval.rou.xml",
        ):
            assert (N4_DIR / fname).exists(), f"missing {fname} in n4_irregular"

        assert (N4_DIR / "vtypes.add.xml").read_bytes() == (
            N3_DIR / "vtypes.add.xml"
        ).read_bytes()

    def test_net_geometry_attributes_and_tls_mirror_n3(self):
        root = ET.parse(N4_DIR / "intersections.net.xml").getroot()

        tls_juncs = {
            j.get("id"): (float(j.get("x")), float(j.get("y")))
            for j in root.iter("junction")
            if j.get("type") == "traffic_light"
        }
        dead_juncs = [
            j for j in root.iter("junction") if j.get("type") == "dead_end"
        ]
        assert len(tls_juncs) == 18
        assert len(dead_juncs) == 18

        xs = sorted({x for x, _ in tls_juncs.values()})
        ys = sorted({y for _, y in tls_juncs.values()})
        assert len(xs) == 6
        assert len(ys) == 3
        col_gaps = [round(xs[i + 1] - xs[i], 2) for i in range(len(xs) - 1)]
        row_gaps = [round(ys[i + 1] - ys[i], 2) for i in range(len(ys) - 1)]
        assert col_gaps == [200.0, 350.0, 150.0, 400.0, 250.0]
        assert row_gaps == [180.0, 300.0]

        edges = [e for e in root.iter("edge") if e.get("function") != "internal"]
        assert len(edges) == 90  # 54 internal + 36 fringe directed edges
        for edge in edges:
            lanes = list(edge.iter("lane"))
            assert len(lanes) == 2
            assert edge.get("priority") == "-1"
            for lane in lanes:
                assert float(lane.get("speed")) == pytest.approx(13.89)

        # Mirror n3_grid TLS phase layout (2 green phases A/B + 3s yellows)
        expected_phases = [
            ("30", "GGGgrrrrGGGgrrrr"),
            ("3", "yyyyrrrryyyyrrrr"),
            ("30", "rrrrGGGgrrrrGGGg"),
            ("3", "rrrryyyyrrrryyyy"),
        ]
        logics = list(root.iter("tlLogic"))
        assert len(logics) == 18
        for logic in logics:
            assert logic.get("type") == "static"
            phases = [(p.get("duration"), p.get("state")) for p in logic.iter("phase")]
            assert phases == expected_phases

        # 4-way intersections with no turnarounds -> 16 controlled connections each
        conn_counts: dict[str, int] = {}
        for conn in root.iter("connection"):
            tl = conn.get("tl")
            if tl:
                conn_counts[tl] = conn_counts.get(tl, 0) + 1
        assert len(conn_counts) == 18
        assert set(conn_counts.values()) == {16}

    def test_lane_groups_derived_and_pass_topology_check(self):
        net_path = N4_DIR / "intersections.net.xml"
        derived = generate_networks.derive_lane_groups(net_path)
        assert len(derived) == 18

        data = json.loads((N4_DIR / "lane_groups.json").read_text(encoding="utf-8"))
        assert data["network"] == "n4_irregular"
        assert len(data["tls_ids"]) == 18
        loaded = {
            t: (list(pair[0]), list(pair[1]))
            for t, pair in data["lane_groups"].items()
        }
        assert loaded == derived
        for tls_id, (ga, gb) in loaded.items():
            assert len(ga) == 4 and len(gb) == 4, f"{tls_id}: expected 4+4 lanes"

        failures = check_obs_match._validate_lane_groups_on_net(
            net_path, loaded, label="networks/n4_irregular"
        )
        assert failures == 0
        assert check_obs_match.EXPECTED_NETWORK_TLS["n4_irregular"] == 18

    def test_eval_compare_registration_and_scenario_resolution(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["eval_compare.py", "--network", "n4_irregular"]
        )
        args = eval_compare.parse_args()
        assert args.network == "n4_irregular"

        scenario = eval_compare.resolve_scenario(
            SimpleNamespace(network="n4_irregular", split="eval", sumo_cfg=None, lane_groups=None)
        )
        assert len(scenario.tls_ids) == 18
        assert scenario.sumo_cfg == N4_DIR / "sumo_config_eval.sumocfg"
        assert scenario.net_path == N4_DIR / "intersections.net.xml"

    def test_demand_entries_exits_and_n3_trip_determinism(self, tmp_path):
        entries, exits = generate_demand.find_entries_exits(
            N4_DIR / "intersections.net.xml"
        )
        assert len(entries) == 18
        assert len(exits) == 18

        # Verify generate_demand produces byte-identical trips for n3_grid
        n3_entries, n3_exits = generate_demand.find_entries_exits(
            N3_DIR / "intersections.net.xml"
        )
        knots = generate_demand.trapezoid_profile(low=0.40, peak=1.67, medium=0.83)
        scale = len(n3_entries) / generate_demand.N1_REFERENCE_ENTRIES
        rng = np.random.default_rng(42)
        departs = generate_demand.sample_departures(
            knots, scale, generate_demand.EPISODE_SECONDS, rng
        )
        tmp_trips = tmp_path / "demand_train.trips.xml"
        generate_demand.write_trips(
            tmp_trips, departs, n3_entries, n3_exits, rng, id_prefix="train_"
        )
        assert tmp_trips.read_bytes() == (
            N3_DIR / "demand_train.trips.xml"
        ).read_bytes()

    def test_build_irregular_grid_xml_helper(self, tmp_path):
        nodes_p = tmp_path / "nodes.nod.xml"
        edges_p = tmp_path / "edges.edg.xml"
        generate_networks.build_irregular_grid_xml(
            generate_networks.N4_COL_GAPS_M,
            generate_networks.N4_ROW_GAPS_M,
            generate_networks.N4_ATTACH_LENGTH_M,
            nodes_p,
            edges_p,
        )
        nodes = list(ET.parse(nodes_p).getroot().iter("node"))
        edges = list(ET.parse(edges_p).getroot().iter("edge"))
        assert len(nodes) == 36
        assert len(edges) == 90
