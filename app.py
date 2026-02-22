"""
EV Route Feasibility and Recommendation API.
Exposes: route feasibility check, EV recommendation for a route, optional map export.
"""
import html
import os
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from config import get_network_config, get_charging_config
from network.loader import load_road_network, get_graph_with_elevation
from charging.stations import load_charging_stations, snap_stations_to_graph
from energy.energy_model import get_default_energy_model
from energy.soh_soc import is_route_feasible, usable_energy_wh
from routing.energy_routing import add_energy_cost_to_graph, energy_shortest_path, path_energy_cost
from routing.charging_logic import select_charging_stops
from recommendation.ev_ranker import rank_evs_for_route, EVSpec
from visualization.map_render import route_to_folium_map, route_to_html_file, route_to_html_string
import osmnx as ox

_STATIC_DIR = Path(__file__).resolve().parent / "static"

# Load graph and stations once at startup (lazy or on first request)
_graph: Optional[Any] = None
_stations_snapped: Optional[List[dict]] = None


def get_graph():
    global _graph
    if _graph is None:
        cfg = get_network_config()
        _graph = load_road_network(cfg)
        _graph = get_graph_with_elevation(_graph, cfg)
    return _graph


def get_stations_snapped():
    global _stations_snapped
    if _stations_snapped is None:
        G = get_graph()
        cc = get_charging_config()
        stations = load_charging_stations(cc.stations_path, cc.api_base_url)
        _stations_snapped = snap_stations_to_graph(G, stations)
    return _stations_snapped


app = FastAPI(
    title="EV Route Feasibility & Recommendation",
    description="MVP: route feasibility check and EV ranking for a route using OSMnx + NetworkX.",
)


@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the test UI."""
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return index.read_text(encoding="utf-8")
    return "<html><body><h1>EV Route API</h1><p><a href='/docs'>Docs</a> | <a href='/health'>Health</a></p></body></html>"


@app.get("/ui", response_class=HTMLResponse)
def ui():
    """Serve the test UI (same as /)."""
    return root()


# --- Request/Response models ---


class FeasibilityRequest(BaseModel):
    start_lat: float = Field(..., description="Start latitude")
    start_lon: float = Field(..., description="Start longitude")
    end_lat: float = Field(..., description="End latitude")
    end_lon: float = Field(..., description="End longitude")
    battery_capacity_wh: float = Field(..., gt=0, description="Battery capacity in Wh")
    efficiency_wh_per_km: float = Field(..., gt=0, description="Vehicle efficiency Wh/km")
    soc: float = Field(1.0, ge=0, le=1, description="State of charge 0-1")
    soh: float = Field(1.0, ge=0, le=1, description="State of health 0-1")
    margin_ratio: float = Field(1.0, gt=0, description="Required reserve margin")


class FeasibilityResponse(BaseModel):
    feasible: bool
    total_energy_wh: float
    usable_energy_wh: float
    path_found: bool
    charging_stops: List[dict] = []
    message: str = ""


class EVSpecRequest(BaseModel):
    id: str
    battery_capacity_wh: float
    efficiency_wh_per_km: float
    soc: float = 1.0
    soh: float = 1.0


class RecommendationRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    ev_specs: List[EVSpecRequest]


class EVResultResponse(BaseModel):
    ev_id: str
    feasible: bool
    total_energy_wh: float
    charging_stops: List[dict]
    usable_energy_wh: float
    rank: int


class RecommendationResponse(BaseModel):
    rankings: List[EVResultResponse]
    path_node_count: int


# --- Endpoints ---


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/route/feasibility", response_model=FeasibilityResponse)
def check_feasibility(req: FeasibilityRequest) -> FeasibilityResponse:
    """Check if a route is feasible for the given EV and battery state (SOC/SOH)."""
    G = get_graph().copy()
    stations = get_stations_snapped()
    model = get_default_energy_model()
    G = add_energy_cost_to_graph(G, req.efficiency_wh_per_km, model)

    start_node = int(ox.distance.nearest_nodes(G, [req.start_lon], [req.start_lat])[0])
    end_node = int(ox.distance.nearest_nodes(G, [req.end_lon], [req.end_lat])[0])
    usable = usable_energy_wh(req.battery_capacity_wh, req.soc, req.soh)
    path, charges, total_energy = select_charging_stops(
        G, start_node, end_node, stations, usable, req.efficiency_wh_per_km, model
    )
    path_found = len(path) > 0 and total_energy < float("inf")
    feasible, _ = is_route_feasible(
        total_energy, req.battery_capacity_wh, req.soc, req.soh, req.margin_ratio
    )
    # With charging, route can be feasible even if direct energy > usable
    if path_found and len(charges) > 0:
        feasible = True

    return FeasibilityResponse(
        feasible=feasible,
        total_energy_wh=total_energy,
        usable_energy_wh=usable,
        path_found=path_found,
        charging_stops=charges,
        message="Route feasible with charging stops" if charges else ("Feasible" if feasible else "Infeasible"),
    )


@app.post("/route/recommend", response_model=RecommendationResponse)
def recommend_evs(req: RecommendationRequest) -> RecommendationResponse:
    """Rank EVs for the given route by least energy and fewest charging stops."""
    G = get_graph()
    stations = get_stations_snapped()
    start_node = int(ox.distance.nearest_nodes(G, [req.start_lon], [req.start_lat])[0])
    end_node = int(ox.distance.nearest_nodes(G, [req.end_lon], [req.end_lat])[0])
    ev_specs = [
        EVSpec(
            id=e.id,
            battery_capacity_wh=e.battery_capacity_wh,
            efficiency_wh_per_km=e.efficiency_wh_per_km,
            soc=e.soc,
            soh=e.soh,
        )
        for e in req.ev_specs
    ]
    results = rank_evs_for_route(G, start_node, end_node, ev_specs, stations)
    path_node_count = len(results[0].path) if results else 0
    return RecommendationResponse(
        rankings=[
            EVResultResponse(
                ev_id=r.ev_id,
                feasible=r.feasible,
                total_energy_wh=r.total_energy_wh,
                charging_stops=r.charging_stops,
                usable_energy_wh=r.usable_energy_wh,
                rank=i + 1,
            )
            for i, r in enumerate(results)
        ],
        path_node_count=path_node_count,
    )


@app.get("/map", response_class=HTMLResponse)
def get_map(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    efficiency_wh_per_km: float = 200.0,
):
    """Generate Folium map HTML for the route and charging stations (for UI iframe)."""
    try:
        G = get_graph().copy()
        stations = get_stations_snapped()
        model = get_default_energy_model()
        G = add_energy_cost_to_graph(G, efficiency_wh_per_km, model)
        start_node = int(ox.distance.nearest_nodes(G, [start_lon], [start_lat])[0])
        end_node = int(ox.distance.nearest_nodes(G, [end_lon], [end_lat])[0])
        path, _, _ = select_charging_stops(G, start_node, end_node, stations, 1e9, efficiency_wh_per_km, model)
        html = route_to_html_string(G, path, stations_snapped=stations)
        return HTMLResponse(html)
    except Exception as e:
        msg = html.escape(f"{e!r}")
        return HTMLResponse(
            f"<html><body style='font-family:sans-serif;padding:2rem;'><h2>Map error</h2><pre>{msg}</pre></body></html>"
        )


@app.post("/map/export")
def export_map(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    efficiency_wh_per_km: float = 200.0,
):
    """Generate Folium map HTML for the route and charging stations. Returns path to file."""
    G = get_graph().copy()
    stations = get_stations_snapped()
    model = get_default_energy_model()
    G = add_energy_cost_to_graph(G, efficiency_wh_per_km, model)
    start_node = int(ox.distance.nearest_nodes(G, [start_lon], [start_lat])[0])
    end_node = int(ox.distance.nearest_nodes(G, [end_lon], [end_lat])[0])
    path, _, _ = select_charging_stops(G, start_node, end_node, stations, 1e9, efficiency_wh_per_km, model)

    out_dir = Path(os.environ.get("EV_MAP_OUTPUT_DIR", "."))
    out_path = out_dir / "ev_route_map.html"
    route_to_html_file(G, path, out_path, stations_snapped=stations)
    return FileResponse(out_path, media_type="text/html", filename="ev_route_map.html")