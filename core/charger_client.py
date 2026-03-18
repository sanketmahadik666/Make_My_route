"""
Antigravity AI — Charger Client (F-007 partial)
OpenChargeMap API integration with 24h disk caching and graph node mapping.
"""

import os
import json
import time
import hashlib
import requests
import osmnx as ox
from config import OCM_API_KEY, STATION_CACHE_DIR, STATION_CACHE_TTL, OCM_MIN_POWER_KW


os.makedirs(STATION_CACHE_DIR, exist_ok=True)

OCM_BASE = "https://api.openchargemap.io/v3/poi/"


def _cache_path(lat: float, lon: float, radius_km: float) -> str:
    """Generate cache filename for station lookup."""
    key = f"{round(lat, 3)}:{round(lon, 3)}:{radius_km}"
    safe = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(STATION_CACHE_DIR, f"{safe}.json")


def _cache_read(path: str) -> list | None:
    """Read from disk cache if fresh (within TTL)."""
    if not os.path.exists(path):
        return None
    age = time.time() - os.path.getmtime(path)
    if age > STATION_CACHE_TTL:
        return None
    with open(path) as f:
        return json.load(f)


def _cache_read_stale(path: str) -> list | None:
    """Read from disk cache even if expired — fallback."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _cache_write(path: str, data: list):
    """Write station data to disk cache."""
    with open(path, "w") as f:
        json.dump(data, f)


def fetch_charging_stations(
    lat: float,
    lon: float,
    radius_km: float = 10.0,
    min_power_kw: float = None,
) -> list[dict]:
    """
    Fetch charging stations from OpenChargeMap API.
    Uses 24h disk cache to avoid repeated API calls.

    Returns list of normalized station dicts with:
    - ocm_id, name, lat, lon, power_kw, connector_type, is_fast_charge, status
    """
    if min_power_kw is None:
        min_power_kw = OCM_MIN_POWER_KW

    # Check cache first
    cache_path = _cache_path(lat, lon, radius_km)
    cached = _cache_read(cache_path)
    if cached is not None:
        print(f"[ChargerClient] Cache hit: {len(cached)} stations")
        return cached

    # API call
    if not OCM_API_KEY:
        print("[ChargerClient] No OCM API key — returning empty stations (degraded mode)")
        return []

    print(f"[ChargerClient] Fetching stations from OCM API: ({lat:.3f}, {lon:.3f}) r={radius_km}km")
    try:
        params = {
            "output": "json",
            "key": OCM_API_KEY,
            "latitude": lat,
            "longitude": lon,
            "distance": radius_km,
            "distanceunit": "KM",
            "maxresults": 100,
        }
        response = requests.get(OCM_BASE, params=params, timeout=10)
        response.raise_for_status()
        raw_stations = response.json()
    except Exception as e:
        print(f"[ChargerClient] OCM API failed: {e}")
        # Try stale cache
        stale = _cache_read_stale(cache_path)
        if stale:
            print(f"[ChargerClient] Serving stale cache: {len(stale)} stations")
            return stale
        return []

    # Normalize
    stations = _normalize_stations(raw_stations, min_power_kw)
    print(f"[ChargerClient] Found {len(stations)} stations (filtered from {len(raw_stations)})")

    # Cache result
    _cache_write(cache_path, stations)

    return stations


def _normalize_stations(raw: list, min_power_kw: float) -> list[dict]:
    """Normalize OCM response into clean station dicts."""
    result = []
    for s in raw:
        addr = s.get("AddressInfo", {})
        lat = addr.get("Latitude")
        lon = addr.get("Longitude")

        # Filter out stations with null coordinates
        if lat is None or lon is None:
            continue

        connections = s.get("Connections", [])
        for c in connections:
            power_kw = c.get("PowerKW") or 0
            if power_kw < min_power_kw:
                continue

            result.append({
                "ocm_id": s.get("ID"),
                "name": addr.get("Title", "Unknown Station"),
                "lat": float(lat),
                "lon": float(lon),
                "power_kw": float(power_kw),
                "connector_type": (c.get("ConnectionType") or {}).get("Title", "Unknown"),
                "is_fast_charge": (c.get("Level") or {}).get("IsFastChargeCapable", False),
                "status": (s.get("StatusType") or {}).get("Title", "Unknown"),
            })

    return result


def fetch_stations_along_corridor(
    route_nodes: list[int],
    G,
    radius_km: float = 8.0,
) -> list[dict]:
    """
    Fetch charging stations near the route corridor midpoint.
    Maps each station to its nearest graph node.
    """
    if not route_nodes:
        return []

    # Use route midpoint
    mid_idx = len(route_nodes) // 2
    mid_node = route_nodes[mid_idx]
    mid_lat = float(G.nodes[mid_node]["y"])
    mid_lon = float(G.nodes[mid_node]["x"])

    stations = fetch_charging_stations(mid_lat, mid_lon, radius_km)

    if not stations:
        return []

    # Map stations to nearest graph nodes
    stations = _map_to_graph_nodes(stations, G)

    return stations


def _map_to_graph_nodes(stations: list[dict], G) -> list[dict]:
    """Map each station to its nearest OSMnx graph node."""
    # Filter valid coordinates
    valid = [s for s in stations if s.get("lat") and s.get("lon")]
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
