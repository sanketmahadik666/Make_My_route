"""
Application configuration. Region and options are loaded from environment
or defaults to avoid hardcoding region-specific values.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class NetworkConfig:
    """Road network loading configuration."""
    # Place query for OSMnx (e.g. "Berlin, Germany") or None to use bbox
    place_query: Optional[str] = None
    # Bounding box (north, south, east, west) used if place_query is None
    bbox: Optional[tuple[float, float, float, float]] = None
    # Fallback: (lat, lon) and radius in meters for graph_from_point
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    dist_meters: float = 5000
    network_type: str = "drive"
    # Elevation: set ELEVATION_RASTER_PATH for local raster, or use API
    elevation_raster_path: Optional[str] = None
    elevation_api_key: Optional[str] = None


@dataclass
class ChargingConfig:
    """Charging stations data source."""
    # Path to JSON/CSV placeholder file, or None to use built-in stub
    stations_path: Optional[str] = None
    # Optional API base URL for future extension
    api_base_url: Optional[str] = None


def get_network_config() -> NetworkConfig:
    return NetworkConfig(
        place_query=os.environ.get("EV_PLACE_QUERY"),
        bbox=_parse_bbox(os.environ.get("EV_BBOX")),
        center_lat=_float_or_none(os.environ.get("EV_CENTER_LAT")),
        center_lon=_float_or_none(os.environ.get("EV_CENTER_LON")),
        dist_meters=float(os.environ.get("EV_DIST_METERS", "5000")),
        network_type=os.environ.get("EV_NETWORK_TYPE", "drive"),
        elevation_raster_path=os.environ.get("ELEVATION_RASTER_PATH"),
        elevation_api_key=os.environ.get("ELEVATION_API_KEY"),
    )


def get_charging_config() -> ChargingConfig:
    return ChargingConfig(
        stations_path=os.environ.get("EV_CHARGING_STATIONS_PATH"),
        api_base_url=os.environ.get("EV_CHARGING_API_URL"),
    )


def _parse_bbox(s: Optional[str]) -> Optional[tuple[float, float, float, float]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        return None
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    except ValueError:
        return None


def _float_or_none(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None
