"""
Antigravity AI — Charger Client (F-007 partial)
OpenStreetMap OSMnx integration for EV charging stations (No API key required).
Replaces OpenChargeMap since it requires an API key.

Charging station data provided by OpenStreetMap
Licensed under Open Data Commons Open Database License (ODbL)
https://www.openstreetmap.org/copyright
"""

import os
import json
import time
import hashlib
import pandas as pd
import osmnx as ox
import re
from config import STATION_CACHE_DIR, STATION_CACHE_TTL, OCM_MIN_POWER_KW


os.makedirs(STATION_CACHE_DIR, exist_ok=True)


def _cache_path(lat: float, lon: float, radius_km: float) -> str:
    """Generate cache filename for station lookup."""
    key = f"{round(lat, 3)}:{round(lon, 3)}:{radius_km}"
    safe = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(STATION_CACHE_DIR, f"{safe}.json")


def _cache_read(path: str) -> list | None:
    if not os.path.exists(path):
        return None
    age = time.time() - os.path.getmtime(path)
    if age > STATION_CACHE_TTL:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _cache_write(path: str, data: list):
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except IOError as e:
        print(f"[ChargerClient] Cache write failed: {e}")


def fetch_charging_stations(
    lat: float,
    lon: float,
    radius_km: float = 10.0,
    min_power_kw: float = 0.0,
) -> list[dict]:
    """
    Fetch charging stations from OpenStreetMap via OSMnx.
    Returns normalized list simulating the OCM structure.
    """
    if min_power_kw <= 0:
        min_power_kw = OCM_MIN_POWER_KW

    cache_path = _cache_path(lat, lon, radius_km)
    cached = _cache_read(cache_path)
    if cached is not None:
        print(f"[ChargerClient] Cache hit: {len(cached)} stations from OSM")
        return cached

    print(f"[ChargerClient] Fetching stations from OSMnx: ({lat:.4f}, {lon:.4f}) r={radius_km}km")
    
    tags = {"amenity": "charging_station"}
    try:
        gdf = ox.features_from_point((lat, lon), tags, dist=radius_km * 1000)
        if gdf.empty:
            print("[ChargerClient] No stations found in this area.")
            return []
    except Exception as e:
        print(f"[ChargerClient] OSMnx query failed: {e}")
        return []

    # Get centroids for polygons/multipolygons
    # convert to projected CRS for centroid calculation to suppress warnings, then back to EPSG:4326?
    # OSMnx usually handles this, but .centroid on EPSG:4326 throws a warning.
    # We will just suppress it or use it as is.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        gdf["lat"] = gdf.geometry.centroid.y
        gdf["lon"] = gdf.geometry.centroid.x

    stations = []
    
    for idx, row in gdf.iterrows():
        station_id = str(idx)
        
        # Power extraction
        power = 0.0
        power_str = None
        for col in ["socket:type2:output", "socket:ccs:output", "socket:chademo:output", "max_power", "capacity:output"]:
            if col in row and pd.notna(row[col]):
                power_str = str(row[col])
                break
                
        if power_str:
            import re
            match = re.search(r"(\d+(\.\d+)?)", power_str)
            if match:
                power = float(match.group(1))

        # Only filter by power if we ACTUALLY have power metadata, 
        # otherwise assume it's acceptable so we don't drop 90% of OSM stations.
        if power > 0 and power < min_power_kw:
            continue

        fast_charge = power >= 40.0

        connector_types = []
        if "socket:type2" in row and pd.notna(row["socket:type2"]): connector_types.append("Type 2")
        if "socket:type2_combo" in row and pd.notna(row["socket:type2_combo"]): connector_types.append("CCS")
        if "socket:ccs" in row and pd.notna(row["socket:ccs"]): connector_types.append("CCS")
        if "socket:chademo" in row and pd.notna(row["socket:chademo"]): connector_types.append("CHAdeMO")
        
        # Fallback if no specific socket tags found
        if not connector_types:
            connector_types.append("Standard / CCS")
            fast_charge = True  # Assume true for route planning if unknown

        operator = row["operator"] if "operator" in row and pd.notna(row["operator"]) else "Public Charger"
        name = row["name"] if "name" in row and pd.notna(row["name"]) else "Charging Station"
        
        capacity = 1
        if "capacity" in row and pd.notna(row["capacity"]):
            try:
                capacity = int(str(row["capacity"]).split(";")[0])
            except ValueError:
                capacity = 1

        stations.append({
            "ocm_id": station_id,
            "uuid": station_id,
            "connection_id": 0,
            "name": name,
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "power_kw": power,
            "connector_type": ", ".join(connector_types) if connector_types else "Unknown",
            "connector_type_id": 0,
            "is_fast_charge": fast_charge,
            "is_operational": True, 
            "current_type": "DC" if fast_charge else ("AC" if power < 40 else "Unknown"),
            "quantity": capacity,
            "operator": operator,
            "usage_type": "Public",
            "number_of_points": capacity,
            "distance_km": None 
        })

    print(f"[ChargerClient] Found {len(stations)} valid stations in OSM (min {min_power_kw}kW)")
    _cache_write(cache_path, stations)
    return stations


def fetch_stations_along_corridor(
    route_nodes: list[int],
    G,
    radius_km: float = 8.0,
) -> list[dict]:
    """
    Fetch charging stations near the route corridor.
    Queries at origin, midpoint, and destination.
    Maps each station to its nearest graph node.
    """
    if not route_nodes or len(route_nodes) < 2:
        return []

    sample_indices = [0, len(route_nodes) // 2, len(route_nodes) - 1]
    all_stations = []
    seen_ids = set()

    for idx in sample_indices:
        node = route_nodes[idx]
        lat = float(G.nodes[node]["y"])
        lon = float(G.nodes[node]["x"])
        stations = fetch_charging_stations(lat, lon, radius_km)

        for s in stations:
            if s["ocm_id"] not in seen_ids:
                seen_ids.add(s["ocm_id"])
                all_stations.append(s)

    if not all_stations:
        return []

    all_stations = _map_to_graph_nodes(all_stations, G)
    print(f"[ChargerClient] Corridor total: {len(all_stations)} unique stations")
    return all_stations


def _map_to_graph_nodes(stations: list[dict], G) -> list[dict]:
    """Map each station to its nearest OSMnx graph node."""
    valid = [s for s in stations if s.get("lat") is not None and s.get("lon") is not None]
    if not valid:
        return []

    lons = [s["lon"] for s in valid]
    lats = [s["lat"] for s in valid]

    try:
        nearest = ox.nearest_nodes(G, X=lons, Y=lats)
        if not hasattr(nearest, "__iter__"):
            nearest = [nearest]

        for station, node_id in zip(valid, nearest):
            station["graph_node"] = int(node_id)
            station["graph_node_lat"] = float(G.nodes[node_id]["y"])
            station["graph_node_lon"] = float(G.nodes[node_id]["x"])
    except Exception as e:
        print(f"[ChargerClient] Node mapping failed: {e}")
        return []

    return valid
