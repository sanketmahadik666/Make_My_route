"""
Antigravity AI — Charger Client
Integrates API Ninjas EVCharger API with aggressive 0.5-degree grid permanent caching.
Features OSMnx graceful fallback if API Ninjas fails.
"""

import os
import json
import time
import hashlib
import math
import requests
import pandas as pd
import osmnx as ox

from config import STATION_CACHE_DIR, STATION_CACHE_TTL, OCM_MIN_POWER_KW, API_NINJAS_KEY

os.makedirs(STATION_CACHE_DIR, exist_ok=True)


def _haversine(lat1, lon1, lat2, lon2):
    """Compute distance between two points on Earth in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * (2 * math.asin(math.sqrt(a)))


def _fetch_api_ninjas_grid(center_lat: float, center_lon: float) -> list | None:
    """Fetches a 50km radius around the grid center from API Ninjas and normalizes stations."""
    if not API_NINJAS_KEY:
        print("[ChargerClient] API_NINJAS_KEY not configured. Skipping.")
        return None
        
    url = f"https://api.api-ninjas.com/v1/evcharger?lat={center_lat}&lon={center_lon}&distance=50&limit=30"
    headers = {"X-Api-Key": API_NINJAS_KEY}
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            print(f"[ChargerClient] API Ninjas error: HTTP {resp.status_code} - {resp.text}")
            return None
            
        data = resp.json()
        stations = []
        
        for s in data:
            if not s.get('is_active', True):
                continue
                
            connectors = []
            max_power = 0.0
            is_fast = False
            
            for conn in s.get('connections', []):
                ctype = conn.get('type_name', 'Unknown')
                lvl = conn.get('level', 2)
                
                if lvl == 1:
                    power = 3.7
                elif lvl == 2:
                    power = 7.4
                elif lvl == 3:
                    power = 50.0
                    is_fast = True
                else:
                    power = 7.4
                
                if power > max_power:
                    max_power = power
                connectors.append(ctype)
                
            if max_power == 0.0:
                max_power = 7.4
                
            station = {
                "ocm_id": f"ninja_{s.get('latitude')}_{s.get('longitude')}",
                "uuid": f"ninja_{s.get('latitude')}_{s.get('longitude')}",
                "name": s.get('name', 'API Ninjas Charger'),
                "lat": float(s.get('latitude', 0)),
                "lon": float(s.get('longitude', 0)),
                "power_kw": float(max_power),
                "connector_type": ", ".join(list(set(connectors))) if connectors else "Standard",
                "is_fast_charge": is_fast,
                "is_operational": s.get('is_active', True),
                "operator": "Public",
                "usage_type": "Public",
                "number_of_points": sum(c.get('num_connectors', 1) for c in s.get('connections', [])),
            }
            stations.append(station)
            
        return stations
    except Exception as e:
        print(f"[ChargerClient] API Ninjas exception: {e}")
        return None


def _fetch_ocm_grid(center_lat: float, center_lon: float) -> list | None:
    """Fetches a 50km radius around the grid from OpenChargeMap (OCM)."""
    from config import OCM_API_KEY
    if not OCM_API_KEY:
        print("[ChargerClient] OCM_API_KEY not configured. Skipping OCM.")
        return None
        
    url = "https://api.openchargemap.io/v3/poi/"
    params = {
        "output": "json",
        "key": OCM_API_KEY,
        "latitude": center_lat,
        "longitude": center_lon,
        "distance": 50,
        "distanceunit": "KM",
        "maxresults": 100,
        "statustype": 50 # Operational
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"[ChargerClient] OCM error: HTTP {resp.status_code}")
            return None
            
        data = resp.json()
        stations = []
        
        for s in data:
            addr = s.get("AddressInfo", {})
            conns = s.get("Connections", [])
            
            max_power = 0.0
            is_fast = False
            connector_types = []
            
            for c in conns:
                power = c.get("PowerKW") or 0.0
                if power > max_power:
                    max_power = float(power)
                lvl = c.get("Level", {})
                if lvl.get("IsFastChargeCapable", False):
                    is_fast = True
                ctype = c.get("ConnectionType", {})
                if ctype.get("Title"):
                    connector_types.append(ctype["Title"])
                    
            if max_power == 0.0:
                max_power = 7.4
                
            stations.append({
                "ocm_id": f"ocm_{s.get('ID')}",
                "uuid": f"ocm_{s.get('ID')}",
                "name": addr.get("Title", "OCM Charger"),
                "lat": float(addr.get("Latitude", 0)),
                "lon": float(addr.get("Longitude", 0)),
                "power_kw": max_power,
                "connector_type": ", ".join(list(set(connector_types))) if connector_types else "Standard",
                "is_fast_charge": is_fast,
                "is_operational": True,
                "operator": s.get("OperatorInfo", {}).get("Title", "Public"),
                "usage_type": "Public",
                "number_of_points": s.get("NumberOfPoints", 1),
            })
            
        return stations
    except Exception as e:
        print(f"[ChargerClient] OCM exception: {e}")
        return None

def fetch_charging_stations(
    lat: float,
    lon: float,
    radius_km: float = 10.0,
    min_power_kw: float = 0.0,
) -> list[dict]:
    """
    Fetch charging stations using API Ninjas with permanent 0.5-degree grid caching.
    Falls back to OSMnx if API Ninjas fails or returns no stations within radius.
    """
    if min_power_kw <= 0:
        min_power_kw = OCM_MIN_POWER_KW

    # 1. Compute 0.5-degree grid center for permanent caching
    center_lat = round(lat * 2) / 2
    center_lon = round(lon * 2) / 2
    bucket_path = os.path.join(STATION_CACHE_DIR, f"ninja_grid_{center_lat}_{center_lon}.json")
    
    bucket_stations = None
    if os.path.exists(bucket_path):
        try:
            with open(bucket_path, "r") as f:
                bucket_stations = json.load(f)
            print(f"[ChargerClient] API Ninjas permanent cache hit for grid ({center_lat}, {center_lon})")
        except Exception as e:
            print(f"[ChargerClient] Error reading bucket: {e}")
            pass
            
    if bucket_stations is None or len(bucket_stations) == 0:
        print(f"[ChargerClient] Unexplored/Empty grid ({center_lat}, {center_lon}). Calling API APIs...")
        bucket_stations = _fetch_api_ninjas_grid(center_lat, center_lon)
        
        # If API Ninjas returns empty, try OCM
        if not bucket_stations:
            print(f"[ChargerClient] API Ninjas empty for grid. Falling back to OpenChargeMap...")
            bucket_stations = _fetch_ocm_grid(center_lat, center_lon)
            
        if bucket_stations is not None:
            # Save permanently to accumulate an offline database
            with open(bucket_path, "w") as f:
                json.dump(bucket_stations, f)
                
    # Filter the bucket stations by actual distance and power
    if bucket_stations is not None:
        valid_stations = []
        for s in bucket_stations:
            dist = _haversine(lat, lon, s['lat'], s['lon'])
            if dist <= radius_km and s['power_kw'] >= min_power_kw:
                valid_stations.append(s)
                
        if valid_stations:
            print(f"[ChargerClient] Returning {len(valid_stations)} stations from API Ninjas data")
            return valid_stations

    # 2. OSMnx Fallback
    print(f"[ChargerClient] API Ninjas empty for area. Falling back to OSMnx for ({lat:.4f}, {lon:.4f})")
    
    # Simple OSMnx cache for fallback
    key = f"osmnx_{round(lat, 3)}:{round(lon, 3)}:{radius_km}"
    safe = hashlib.md5(key.encode()).hexdigest()
    osm_cache_path = os.path.join(STATION_CACHE_DIR, f"{safe}.json")
    
    if os.path.exists(osm_cache_path) and (time.time() - os.path.getmtime(osm_cache_path)) < STATION_CACHE_TTL:
        try:
            with open(osm_cache_path) as f:
                return json.load(f)
        except Exception:
            pass

    tags = {"amenity": "charging_station"}
    try:
        gdf = ox.features_from_point((lat, lon), tags, dist=radius_km * 1000)
        if gdf.empty:
            return []
    except Exception as e:
        print(f"[ChargerClient] OSMnx query failed: {e}")
        return []

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        gdf["lat"] = gdf.geometry.centroid.y
        gdf["lon"] = gdf.geometry.centroid.x

    stations = []
    for idx, row in gdf.iterrows():
        station_id = str(idx)
        power = 0.0
        
        for col in ["socket:type2:output", "socket:ccs:output", "socket:chademo:output", "max_power", "capacity:output"]:
            if col in row and pd.notna(row[col]):
                power_str = str(row[col])
                import re
                match = re.search(r"(\d+(\.\d+)?)", power_str)
                if match:
                    power = float(match.group(1))
                break

        if power > 0 and power < min_power_kw:
            continue

        fast_charge = power >= 40.0
        connector_types = []
        if "socket:type2" in row and pd.notna(row["socket:type2"]): connector_types.append("Type 2")
        if "socket:type2_combo" in row and pd.notna(row["socket:type2_combo"]): connector_types.append("CCS")
        if "socket:ccs" in row and pd.notna(row["socket:ccs"]): connector_types.append("CCS")
        if "socket:chademo" in row and pd.notna(row["socket:chademo"]): connector_types.append("CHAdeMO")
        
        if not connector_types:
            connector_types.append("Standard / CCS")
            fast_charge = True

        capacity = 1
        if "capacity" in row and pd.notna(row["capacity"]):
            try:
                capacity = int(str(row["capacity"]).split(";")[0])
            except ValueError:
                capacity = 1

        stations.append({
            "ocm_id": f"osm_{station_id}",
            "uuid": f"osm_{station_id}",
            "name": row["name"] if "name" in row and pd.notna(row["name"]) else "OSM Charging Station",
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "power_kw": power if power > 0 else 7.4,
            "connector_type": ", ".join(connector_types),
            "is_fast_charge": fast_charge,
            "is_operational": True,
            "operator": row["operator"] if "operator" in row and pd.notna(row["operator"]) else "Public Charger",
            "usage_type": "Public",
            "number_of_points": capacity,
        })

    print(f"[ChargerClient] Found {len(stations)} stations via OSMnx")
    try:
        with open(osm_cache_path, "w") as f:
            json.dump(stations, f)
    except Exception:
        pass
        
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
