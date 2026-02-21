"""
EV recommendation: given a route and list of EVs, evaluate feasibility for each
and rank by least energy usage and fewest charging stops.
"""
from dataclasses import dataclass
from typing import Any, List, Optional

import networkx as nx

from energy.energy_model import EnergyModel, get_default_energy_model
from energy.soh_soc import is_route_feasible, usable_energy_wh
from routing.charging_logic import select_charging_stops
from routing.energy_routing import add_energy_cost_to_graph, energy_shortest_path


@dataclass
class EVSpec:
    """EV specification for ranking."""
    id: str
    battery_capacity_wh: float
    efficiency_wh_per_km: float
    soc: float = 1.0
    soh: float = 1.0


@dataclass
class EVRouteResult:
    """Result of evaluating one EV on a route."""
    ev_id: str
    feasible: bool
    total_energy_wh: float
    charging_stops: List[dict]
    path: List[int]
    usable_energy_wh: float
    rank_score: float  # lower is better


def rank_evs_for_route(
    G: nx.MultiDiGraph,
    start_node: int,
    end_node: int,
    ev_specs: List[EVSpec],
    stations_snapped: List[dict],
    model: Optional[EnergyModel] = None,
) -> List[EVRouteResult]:
    """
    Evaluate each EV on the route and rank by:
    1. Feasibility (infeasible last)
    2. Fewer charging stops better
    3. Less total energy better
    """
    if model is None:
        model = get_default_energy_model()

    results: List[EVRouteResult] = []
    for ev in ev_specs:
        G_energized = add_energy_cost_to_graph(G.copy(), ev.efficiency_wh_per_km, model)
        usable = usable_energy_wh(ev.battery_capacity_wh, ev.soc, ev.soh)
        path, charges, total_energy = select_charging_stops(
            G_energized, start_node, end_node, stations_snapped, usable, ev.efficiency_wh_per_km, model
        )
        feasible = total_energy < float("inf") and (
            total_energy <= usable or len(charges) > 0
        )
        # Rank: feasible first, then by (num_stops, total_energy)
        n_stops = len(charges)
        rank_score = (0 if feasible else 1e9) + n_stops * 1e6 + total_energy
        results.append(EVRouteResult(
            ev_id=ev.id,
            feasible=feasible,
            total_energy_wh=total_energy,
            charging_stops=charges,
            path=path,
            usable_energy_wh=usable,
            rank_score=rank_score,
        ))

    results.sort(key=lambda r: r.rank_score)
    return results
