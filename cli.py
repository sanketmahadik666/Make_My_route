"""
Simple CLI for EV route feasibility and recommendation.
Usage:
  python cli.py feasibility --start 52.52 13.405 --end 52.53 13.41 --capacity 50000 --efficiency 200
  python cli.py recommend --start 52.52 13.405 --end 52.53 13.41
  python cli.py map --start 52.52 13.405 --end 52.53 13.41 --out map.html
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_network_config, get_charging_config
from network.loader import load_road_network, get_graph_with_elevation
from charging.stations import load_charging_stations, snap_stations_to_graph
from energy.energy_model import get_default_energy_model
from energy.soh_soc import usable_energy_wh, is_route_feasible
from routing.energy_routing import add_energy_cost_to_graph, path_energy_cost
from routing.charging_logic import select_charging_stops
from recommendation.ev_ranker import rank_evs_for_route, EVSpec
from visualization.map_render import route_to_html_file
import osmnx as ox


def _get_graph_and_stations():
    cfg = get_network_config()
    G = load_road_network(cfg)
    G = get_graph_with_elevation(G, cfg)
    cc = get_charging_config()
    stations = load_charging_stations(cc.stations_path, cc.api_base_url)
    stations = snap_stations_to_graph(G, stations)
    return G, stations


def cmd_feasibility(args):
    G, stations = _get_graph_and_stations()
    model = get_default_energy_model()
    G = add_energy_cost_to_graph(G, args.efficiency, model)
    start_node = int(ox.distance.nearest_nodes(G, [args.start_lon], [args.start_lat])[0])
    end_node = int(ox.distance.nearest_nodes(G, [args.end_lon], [args.end_lat])[0])
    usable = usable_energy_wh(args.capacity, args.soc, args.soh)
    path, charges, total = select_charging_stops(G, start_node, end_node, stations, usable, args.efficiency, model)
    feasible = total < 1e9 and (total <= usable or len(charges) > 0)
    out = {
        "feasible": feasible,
        "total_energy_wh": total,
        "usable_energy_wh": usable,
        "charging_stops": len(charges),
        "path_nodes": len(path),
    }
    print(json.dumps(out, indent=2))
    return 0 if feasible else 1


def cmd_recommend(args):
    G, stations = _get_graph_and_stations()
    ev_specs = [
        EVSpec(id=f"ev_{i}", battery_capacity_wh=e["capacity"], efficiency_wh_per_km=e["efficiency"])
        for i, e in enumerate(args.evs)
    ]
    start_node = int(ox.distance.nearest_nodes(G, [args.start_lon], [args.start_lat])[0])
    end_node = int(ox.distance.nearest_nodes(G, [args.end_lon], [args.end_lat])[0])
    results = rank_evs_for_route(G, start_node, end_node, ev_specs, stations)
    out = [
        {"rank": i + 1, "ev_id": r.ev_id, "feasible": r.feasible, "total_energy_wh": r.total_energy_wh, "charging_stops": len(r.charging_stops)}
        for i, r in enumerate(results)
    ]
    print(json.dumps(out, indent=2))
    return 0


def cmd_map(args):
    G, stations = _get_graph_and_stations()
    model = get_default_energy_model()
    G = add_energy_cost_to_graph(G, args.efficiency, model)
    start_node = int(ox.distance.nearest_nodes(G, [args.start_lon], [args.start_lat])[0])
    end_node = int(ox.distance.nearest_nodes(G, [args.end_lon], [args.end_lat])[0])
    path, _, _ = select_charging_stops(G, start_node, end_node, stations, 1e9, args.efficiency, model)
    path_out = Path(args.out)
    route_to_html_file(G, path, path_out, stations_snapped=stations)
    print(f"Map saved to {path_out}")
    return 0


def main():
    p = argparse.ArgumentParser(description="EV Route Feasibility CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    # feasibility
    f = sub.add_parser("feasibility")
    f.add_argument("--start", type=float, nargs=2, metavar=("LAT", "LON"), required=True, dest="start_lat_lon")
    f.add_argument("--end", type=float, nargs=2, metavar=("LAT", "LON"), required=True, dest="end_lat_lon")
    f.add_argument("--capacity", type=float, default=50000, help="Battery capacity Wh")
    f.add_argument("--efficiency", type=float, default=200, help="Wh/km")
    f.add_argument("--soc", type=float, default=1.0)
    f.add_argument("--soh", type=float, default=1.0)
    f.set_defaults(run=cmd_feasibility)
    # recommend
    r = sub.add_parser("recommend")
    r.add_argument("--start", type=float, nargs=2, metavar=("LAT", "LON"), required=True, dest="start_lat_lon")
    r.add_argument("--end", type=float, nargs=2, metavar=("LAT", "LON"), required=True, dest="end_lat_lon")
    r.add_argument("--evs", type=json.loads, default='[{"capacity": 50000, "efficiency": 180}, {"capacity": 70000, "efficiency": 220}]', help='JSON list of {"capacity": Wh, "efficiency": Wh/km}')
    r.set_defaults(run=cmd_recommend)
    # map
    m = sub.add_parser("map")
    m.add_argument("--start", type=float, nargs=2, metavar=("LAT", "LON"), required=True, dest="start_lat_lon")
    m.add_argument("--end", type=float, nargs=2, metavar=("LAT", "LON"), required=True, dest="end_lat_lon")
    m.add_argument("--out", default="ev_route_map.html")
    m.add_argument("--efficiency", type=float, default=200)
    m.set_defaults(run=cmd_map)

    args = p.parse_args()
    args.start_lat, args.start_lon = args.start_lat_lon[0], args.start_lat_lon[1]
    args.end_lat, args.end_lon = args.end_lat_lon[0], args.end_lat_lon[1]
    if args.cmd == "recommend" and isinstance(args.evs, str):
        args.evs = json.loads(args.evs)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
