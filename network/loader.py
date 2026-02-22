"""
Load road network using OSMnx. Graph stores distance (length) and elevation (grade) when available.
Uses only prebuilt OSMnx/NetworkX; no custom routing algorithms here.
"""
from typing import Optional

import networkx as nx
import osmnx as ox

from config import NetworkConfig, get_network_config


def load_road_network(config: Optional[NetworkConfig] = None) -> nx.MultiDiGraph:
    """
    Load a road network for the configured region.
    Uses graph_from_place, graph_from_bbox, or graph_from_point depending on config.
    """
    cfg = config or get_network_config()

    if cfg.place_query:
        G = ox.graph_from_place(cfg.place_query, network_type=cfg.network_type, simplify=True)
    elif cfg.bbox:
        # bbox: (north, south, east, west)
        G = ox.graph_from_bbox(
            bbox=(cfg.bbox[0], cfg.bbox[1], cfg.bbox[2], cfg.bbox[3]),
            network_type=cfg.network_type,
            simplify=True,
        )
    elif cfg.center_lat is not None and cfg.center_lon is not None:
        G = ox.graph_from_point(
            (cfg.center_lat, cfg.center_lon),
            dist=cfg.dist_meters,
            dist_type="bbox",
            network_type=cfg.network_type,
            simplify=True,
        )
    else:
        # Default: use a well-known data-rich region (configurable via env)
        G = ox.graph_from_point((52.52, 13.405), dist=5000, dist_type="bbox", network_type="drive", simplify=True)

    # Ensure length is on edges (OSMnx usually adds it; add_edge_lengths if missing)
    if not any(G.edges(data=True)):
        return G
    first_edge = next(iter(G.edges(data=True)))
    if "length" not in first_edge[2]:
        G = ox.add_edge_lengths(G)

    return G


def get_graph_with_elevation(G: nx.MultiDiGraph, config: Optional[NetworkConfig] = None) -> nx.MultiDiGraph:
    """
    Add node elevations and edge grades to the graph when a data source is configured.
    If no raster or API key is set, edges get grade=0 so energy model can still run.
    """
    cfg = config or get_network_config()

    # Add elevations from raster if path is set
    if cfg.elevation_raster_path:
        try:
            G = ox.elevation.add_node_elevations_raster(G, cfg.elevation_raster_path)
            G = ox.elevation.add_edge_grades(G, add_absolute=True)
        except Exception:
            _set_default_grade(G)
        return G

    # Optional: Google Elevation API (requires key)
    if cfg.elevation_api_key:
        try:
            G = ox.elevation.add_node_elevations_google(G, api_key=cfg.elevation_api_key)
            G = ox.elevation.add_edge_grades(G, add_absolute=True)
        except Exception:
            _set_default_grade(G)
        return G

    # No elevation source: set grade to 0 so energy model has a slope value
    _set_default_grade(G)
    return G


def _set_default_grade(G: nx.MultiDiGraph) -> None:
    """Set grade and grade_abs to 0 for all edges when elevation is unavailable."""
    for u, v, k, data in G.edges(keys=True, data=True):
        G.edges[u, v, k]["grade"] = 0.0
        G.edges[u, v, k]["grade_abs"] = 0.0
