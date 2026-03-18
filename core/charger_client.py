"""
Antigravity AI — Charger Client (F-007 partial)
OpenChargeMap API integration with 24h disk caching and graph node mapping.

Charging station data provided by OpenChargeMap (https://openchargemap.org)
Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0)
https://creativecommons.org/licenses/by/4.0/
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

# ── OCM Reference IDs ──
# StatusTypeID 50 = "Operational"
# LevelID 3 = "Level 3: High (Over 40kW)" → IsFastChargeCapable = true
# ConnectionTypeID 25 = "Type 2 (Socket Only)"
# ConnectionTypeID 33 = "CCS (Type 2)"
# ConnectionTypeID 2  = "CHAdeMO"
# CurrentTypeID 30 = "DC"

# Map OCM ConnectionTypeID → human-readable connector name
CONNECTOR_MAP = {
    1:  "Type 1 (J1772)",
    2:  "CHAdeMO",
    25: "Type 2 (Socket)",
    27: "Tesla Supercharger",
    30: "Tesla (Model S/X)",
    32: "CCS (Type 1)",
    33: "CCS (Type 2)",
    36: "Type 2 (Tethered)",
    1036: "Type 2 (IEC 62196)",
}


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
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _cache_read_stale(path: str) -> list | None:
    """Read from disk cache even if expired — fallback."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _cache_write(path: str, data: list):
    """Write station data to disk cache."""
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
    Fetch charging stations from OpenChargeMap API.
    Uses 24h disk cache to avoid repeated API calls.

    OCM API docs: https://openchargemap.org/site/develop/api

    Returns list of normalized station dicts with:
    - ocm_id, uuid, name, lat, lon, power_kw, connector_type, connector_type_id,
      is_fast_charge, is_operational, operator_name, usage_type, number_of_points,
      amps, voltage, current_type, quantity, distance_km
    """
    if min_power_kw <= 0:
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

    print(f"[ChargerClient] Fetching stations from OCM API: ({lat:.4f}, {lon:.4f}) r={radius_km}km")
    try:
        params = {
            "key": OCM_API_KEY,
            "output": "json",
            "latitude": lat,
            "longitude": lon,
            "distance": radius_km,
            "distanceunit": "KM",
            "maxresults": 100,
            "compact": True,          # Remove nested reference objects (returns IDs only)
            "verbose": False,         # Smaller payload, nulls removed
            "statustypeid": "50",     # Only operational stations
            "camelcase": True,        # Consistent property names
        }
        response = requests.get(OCM_BASE, params=params, timeout=10)
        response.raise_for_status()
        raw_stations = response.json()
    except requests.exceptions.Timeout:
        print("[ChargerClient] OCM API timed out")
        stale = _cache_read_stale(cache_path)
        if stale:
            print(f"[ChargerClient] Serving stale cache: {len(stale)} stations")
            return stale
        return []
    except requests.exceptions.RequestException as e:
        print(f"[ChargerClient] OCM API request failed: {e}")
        stale = _cache_read_stale(cache_path)
        if stale:
            print(f"[ChargerClient] Serving stale cache: {len(stale)} stations")
            return stale
        return []
    except (ValueError, json.JSONDecodeError) as e:
        print(f"[ChargerClient] OCM API response parse failed: {e}")
        return []

    # Normalize
    stations = _normalize_stations(raw_stations, min_power_kw)
    print(f"[ChargerClient] Found {len(stations)} connections across "
          f"{len(raw_stations)} POIs (min {min_power_kw}kW filter)")

    # Cache result
    _cache_write(cache_path, stations)

    return stations


def _normalize_stations(raw: list, min_power_kw: float) -> list[dict]:
    """
    Normalize OCM API response into clean station dicts.

    OCM POI structure (key fields):
    ├── ID, UUID
    ├── AddressInfo.Latitude/Longitude/Title/Town/Postcode
    ├── OperatorInfo.Title
    ├── UsageType.Title
    ├── StatusType.IsOperational / Title
    ├── NumberOfPoints
    ├── Connections[] (one per connector at a site)
    │   ├── ConnectionTypeID → CONNECTOR_MAP lookup
    │   ├── ConnectionType.Title / FormalName
    │   ├── Level.IsFastChargeCapable (bool)
    │   ├── LevelID (1=Low, 2=Medium, 3=High≥40kW)
    │   ├── PowerKW (peak available kW, may be null)
    │   ├── Amps, Voltage (may be null)
    │   ├── CurrentType.Description ("DC", "AC (Single-Phase)", "AC (Three-Phase)")
    │   ├── Quantity (number of equipment items with this spec)
    │   └── StatusTypeID / StatusType.IsOperational
    └── DataProvider.License
    """
    result = []
    seen_connections = set()  # Deduplicate by (ocm_id, connection_id)

    for poi in raw:
        addr = poi.get("addressInfo") or poi.get("AddressInfo") or {}
        lat = addr.get("latitude") or addr.get("Latitude")
        lon = addr.get("longitude") or addr.get("Longitude")

        # Skip stations with null coordinates
        if lat is None or lon is None:
            continue

        # POI-level fields
        ocm_id = poi.get("id") or poi.get("ID")
        uuid = poi.get("uuid") or poi.get("UUID")
        site_name = (addr.get("title") or addr.get("Title")
                     or addr.get("addressLine1") or addr.get("AddressLine1")
                     or "Unknown Station")
        town = addr.get("town") or addr.get("Town") or ""
        distance = addr.get("distance") or addr.get("Distance")
        number_of_points = poi.get("numberOfPoints") or poi.get("NumberOfPoints") or 0

        # Operator info (may be compact/null)
        operator_info = poi.get("operatorInfo") or poi.get("OperatorInfo") or {}
        operator_name = (operator_info.get("title") or operator_info.get("Title")
                         or operator_info.get("description") or "Unknown")

        # Usage type
        usage_type_obj = poi.get("usageType") or poi.get("UsageType") or {}
        usage_type = (usage_type_obj.get("title") or usage_type_obj.get("Title")
                      or usage_type_obj.get("description") or "Unknown")

        # Site-level operational status
        status_type_obj = poi.get("statusType") or poi.get("StatusType") or {}
        is_operational = status_type_obj.get("isOperational",
                         status_type_obj.get("IsOperational", True))

        # Process each connection at this POI
        connections = poi.get("connections") or poi.get("Connections") or []
        for conn in connections:
            conn_id = conn.get("id") or conn.get("ID") or 0
            dedup_key = (ocm_id, conn_id)
            if dedup_key in seen_connections:
                continue
            seen_connections.add(dedup_key)

            # Power (may be null — estimate from Amps × Voltage if missing)
            power_kw = conn.get("powerKW") or conn.get("PowerKW")
            amps = conn.get("amps") or conn.get("Amps")
            voltage = conn.get("voltage") or conn.get("Voltage")

            if power_kw is None and amps and voltage:
                power_kw = (amps * voltage) / 1000.0

            if power_kw is None:
                power_kw = 0.0
            else:
                power_kw = float(power_kw)

            # Filter by minimum power
            if power_kw < min_power_kw:
                continue

            # Connection type
            conn_type_id = conn.get("connectionTypeID") or conn.get("ConnectionTypeID")
            conn_type_obj = conn.get("connectionType") or conn.get("ConnectionType") or {}
            connector_type = (
                conn_type_obj.get("title")
                or conn_type_obj.get("Title")
                or conn_type_obj.get("description")
                or CONNECTOR_MAP.get(conn_type_id, "Unknown")
            )
            formal_name = conn_type_obj.get("formalName") or conn_type_obj.get("FormalName") or ""

            # Fast charge detection
            # Primary: Level.IsFastChargeCapable
            # Fallback: LevelID == 3 means ≥40kW
            # Fallback: power_kw >= 40
            level_obj = conn.get("level") or conn.get("Level") or {}
            is_fast_charge = level_obj.get("isFastChargeCapable",
                             level_obj.get("IsFastChargeCapable"))
            if is_fast_charge is None:
                level_id = conn.get("levelID") or conn.get("LevelID") or 0
                is_fast_charge = (level_id == 3) or (power_kw >= 40.0)
            else:
                is_fast_charge = bool(is_fast_charge)

            # Current type (AC/DC)
            current_type_obj = conn.get("currentType") or conn.get("CurrentType") or {}
            current_type = (current_type_obj.get("description")
                           or current_type_obj.get("Description")
                           or current_type_obj.get("title")
                           or current_type_obj.get("Title")
                           or "Unknown")

            # Connection-level status
            conn_status_obj = conn.get("statusType") or conn.get("StatusType") or status_type_obj
            conn_operational = conn_status_obj.get("isOperational",
                              conn_status_obj.get("IsOperational", is_operational))

            # Quantity of equipment items with this spec
            quantity = conn.get("quantity") or conn.get("Quantity") or 1

            result.append({
                "ocm_id": ocm_id,
                "uuid": uuid,
                "connection_id": conn_id,
                "name": f"{site_name}" + (f", {town}" if town else ""),
                "lat": float(lat),
                "lon": float(lon),
                "power_kw": round(power_kw, 1),
                "connector_type": connector_type,
                "connector_type_id": conn_type_id,
                "formal_name": formal_name,
                "is_fast_charge": is_fast_charge,
                "is_operational": bool(conn_operational),
                "current_type": current_type,
                "amps": float(amps) if amps else None,
                "voltage": float(voltage) if voltage else None,
                "quantity": quantity,
                "operator": operator_name,
                "usage_type": usage_type,
                "number_of_points": number_of_points,
                "distance_km": round(float(distance), 2) if distance else None,
            })

    return result


def fetch_stations_along_corridor(
    route_nodes: list[int],
    G,
    radius_km: float = 8.0,
) -> list[dict]:
    """
    Fetch charging stations near the route corridor.
    Queries at origin, midpoint, and destination to cover the full corridor.
    Maps each station to its nearest graph node.
    Deduplicates by OCM ID + connection ID.
    """
    if not route_nodes or len(route_nodes) < 2:
        return []

    # Sample 3 points along route: start, mid, end
    sample_indices = [0, len(route_nodes) // 2, len(route_nodes) - 1]
    all_stations = []
    seen_keys = set()

    for idx in sample_indices:
        node = route_nodes[idx]
        lat = float(G.nodes[node]["y"])
        lon = float(G.nodes[node]["x"])
        stations = fetch_charging_stations(lat, lon, radius_km)

        for s in stations:
            key = (s.get("ocm_id"), s.get("connection_id"))
            if key not in seen_keys:
                seen_keys.add(key)
                all_stations.append(s)

    if not all_stations:
        return []

    # Map stations to nearest graph nodes
    all_stations = _map_to_graph_nodes(all_stations, G)

    print(f"[ChargerClient] Corridor total: {len(all_stations)} unique connections")
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
