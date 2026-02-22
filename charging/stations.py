"""
Load charging stations from placeholder dataset or API and snap them to nearest graph nodes.
Uses OSMnx nearest_nodes for snapping.
"""
import json
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
import osmnx as ox


def load_charging_stations(
    stations_path: Optional[str] = None,
    api_base_url: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Load charging stations from a JSON/CSV file or return a placeholder list.
    Each station must have: lat, lon, and optionally power_kw, name, id.
    """
    if stations_path and Path(stations_path).exists():
        path = Path(stations_path)
        if path.suffix.lower() == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return _normalize_stations(data)
            if isinstance(data, dict) and "stations" in data:
                return _normalize_stations(data["stations"])
            return _normalize_stations(data.get("features", []))
        if path.suffix.lower() == ".csv":
            import pandas as pd
            df = pd.read_csv(path)
            return _normalize_stations(df.to_dict("records"))

    # Placeholder dataset: stations with lat, lon, power_kw (no external API call by default)
    return _placeholder_stations()


def _normalize_stations(rows: list) -> list[dict[str, Any]]:
    out = []
    for i, r in enumerate(rows):
        if isinstance(r, dict):
            lat = r.get("lat") or r.get("latitude") or r.get("y")
            lon = r.get("lon") or r.get("longitude") or r.get("x")
            if lat is None or lon is None:
                continue
            out.append({
                "id": r.get("id", f"station_{i}"),
                "lat": float(lat),
                "lon": float(lon),
                "power_kw": float(r.get("power_kw", r.get("power", 50))),
                "name": r.get("name", ""),
            })
    return out


def _placeholder_stations() -> list[dict[str, Any]]:
    """Built-in placeholder stations (e.g. around a default region). Not hardcoded to one city."""
    # Generic offsets from a center so any bbox/place can overlay; real data should replace this
    base_lat, base_lon = 52.52, 13.405
    return [
        {"id": "ph_1", "lat": base_lat + 0.01, "lon": base_lon + 0.01, "power_kw": 150, "name": "Placeholder 1"},
        {"id": "ph_2", "lat": base_lat - 0.008, "lon": base_lon + 0.015, "power_kw": 50, "name": "Placeholder 2"},
        {"id": "ph_3", "lat": base_lat + 0.005, "lon": base_lon - 0.01, "power_kw": 350, "name": "Placeholder 3"},
    ]


def snap_stations_to_graph(
    G: nx.MultiDiGraph,
    stations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Snap each charging station to the nearest graph node using OSMnx.
    Adds 'node_id' to each station record; preserves original lat/lon.
    """
    if not stations:
        return []

    lats = np.array([s["lat"] for s in stations])
    lons = np.array([s["lon"] for s in stations])
    # OSMnx: graph may be unprojected (lat/lon) or projected; nearest_nodes accepts (G, X, Y)
    # In OSMnx unprojected, x=lon, y=lat
    node_ids = ox.distance.nearest_nodes(G, lons, lats)

    if isinstance(node_ids, np.ndarray):
        node_ids = node_ids.tolist()
    else:
        node_ids = [node_ids]

    out = []
    for s, nid in zip(stations, node_ids):
        out.append({**s, "node_id": int(nid)})
    return out
