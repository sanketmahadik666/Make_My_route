"""
Energy-aware routing: assign energy_cost to graph edges and use NetworkX shortest path.
Do NOT modify routing internals; use only nx.dijkstra_path / nx.astar_path with weight.
"""
from typing import Any, List, Optional

import networkx as nx

from energy.energy_model import EnergyModel, get_default_energy_model


def add_energy_cost_to_graph(
    G: nx.MultiDiGraph,
    efficiency_wh_per_km: float,
    model: Optional[EnergyModel] = None,
) -> nx.MultiDiGraph:
    """
    Add energy_cost (Wh) to each edge using the energy model.
    Reads length (m) and grade from edges; uses 0 for grade if missing.
    """
    if model is None:
        model = get_default_energy_model()

    for u, v, k, data in G.edges(keys=True, data=True):
        length_m = data.get("length", 0.0) or 0.0
        grade = data.get("grade", 0.0) or 0.0
        energy_wh = model.predict_wh(length_m, grade, efficiency_wh_per_km)
        G.edges[u, v, k]["energy_cost"] = max(0.0, energy_wh)
    return G


def energy_shortest_path(
    G: nx.MultiDiGraph,
    source: int,
    target: int,
    weight: str = "energy_cost",
) -> tuple[List[int], float]:
    """
    Shortest path by energy cost using NetworkX Dijkstra. Does not modify routing internals.
    Returns (node list, total energy cost in Wh).
    """
    try:
        path = nx.dijkstra_path(G, source, target, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [], float("inf")

    total = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        # MultiDiGraph: take min energy among parallel edges
        min_cost = min(
            G.edges[u, v, k].get(weight, 0.0)
            for k in G[u][v]
        )
        total += min_cost
    return path, total


def path_energy_cost(G: nx.MultiDiGraph, path: List[int], weight: str = "energy_cost") -> float:
    """Compute total energy cost along a node path."""
    total = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        total += min(G.edges[u, v, k].get(weight, 0.0) for k in G[u][v])
    return total
