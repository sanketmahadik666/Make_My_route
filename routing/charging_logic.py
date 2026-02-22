"""
Charging logic: when route energy exceeds usable energy, find reachable charging stations
and select best charger by distance and power. Uses NetworkX for path costs only.
"""
from typing import Any, List, Optional, Tuple

import networkx as nx

from energy.energy_model import EnergyModel, get_default_energy_model
from routing.energy_routing import energy_shortest_path, path_energy_cost


def reachable_chargers(
    G: nx.MultiDiGraph,
    from_node: int,
    stations_snapped: List[dict],
    remaining_wh: float,
    efficiency_wh_per_km: float,
    model: Optional[EnergyModel] = None,
) -> List[Tuple[dict, float]]:
    """
    From a node, list charging stations reachable within remaining_wh.
    Returns list of (station, energy_wh_to_reach).
    """
    if model is None:
        model = get_default_energy_model()
    weight = "energy_cost"
    reachable = []
    for st in stations_snapped:
        node_id = st.get("node_id")
        if node_id is None or node_id == from_node:
            continue
        try:
            _, cost = energy_shortest_path(G, from_node, node_id, weight=weight)
        except Exception:
            continue
        if cost <= remaining_wh and cost < float("inf"):
            reachable.append((st, cost))
    return reachable


def select_charging_stops(
    G: nx.MultiDiGraph,
    start_node: int,
    end_node: int,
    stations_snapped: List[dict],
    usable_energy_wh: float,
    efficiency_wh_per_km: float,
    model: Optional[EnergyModel] = None,
) -> Tuple[List[int], List[dict], float]:
    """
    If direct route fits in usable_energy_wh, return (path, [], total_energy).
    Otherwise find reachable chargers, pick best (by power and then by distance),
    then plan from charger to end (recursive). Returns (path_with_stops, charging_stations_used, total_energy).
    """
    if model is None:
        model = get_default_energy_model()
    path, direct_energy = energy_shortest_path(G, start_node, end_node, weight="energy_cost")
    if not path:
        return [], [], float("inf")
    if direct_energy <= usable_energy_wh:
        return path, [], direct_energy

    # Need charging: find best reachable charger from start
    reachable = reachable_chargers(
        G, start_node, stations_snapped, usable_energy_wh, efficiency_wh_per_km, model
    )
    if not reachable:
        return path, [], direct_energy  # return direct path but caller can treat as infeasible

    # Best: prefer higher power, then lower energy to reach
    def score(item: Tuple[dict, float]) -> Tuple[float, float]:
        st, energy_to = item
        power = st.get("power_kw", 0) or 0
        return (-power, energy_to)

    reachable.sort(key=score)
    best_station, energy_to_charger = reachable[0]
    charger_node = best_station["node_id"]

    # From charger to end (assume full charge at stop: usable_energy_wh again for next segment)
    sub_path, sub_charges, sub_energy = select_charging_stops(
        G, charger_node, end_node, stations_snapped, usable_energy_wh, efficiency_wh_per_km, model
    )
    if not sub_path:
        return path, [best_station], direct_energy

    # Build full path: start -> charger -> ... -> end
    path_to_charger, _, _ = energy_shortest_path(G, start_node, charger_node, weight="energy_cost")
    full_path = path_to_charger[:-1] + sub_path
    total_energy = path_energy_cost(G, full_path, "energy_cost")
    charges = [best_station] + sub_charges
    return full_path, charges, total_energy
