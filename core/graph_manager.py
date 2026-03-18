"""
Antigravity AI — Graph Manager (F-001, F-002)
Downloads, caches, and enriches OSMnx road network graphs.
"""

import os
import osmnx as ox
import networkx as nx
from config import GRAPH_CACHE_DIR, GOOGLE_ELEVATION_API_KEY


os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)


def _cache_path(region: str) -> str:
    """Generate safe filename for region cache."""
    safe = region.lower().replace(",", "").replace(" ", "_").strip("_")
    return os.path.join(GRAPH_CACHE_DIR, f"{safe}.graphml")


def get_graph(region: str) -> nx.MultiDiGraph:
    """
    Load road graph for a region.
    1. Check GraphML cache → load from file (~2s)
    2. Cache miss → download from OSM Overpass API (~60–90s)
    3. Add speed + travel time attributes
    4. Save to cache for next time
    """
    path = _cache_path(region)

    if os.path.exists(path):
        print(f"[GraphManager] Loading cached graph: {path}")
        G = ox.load_graphml(path)
        print(f"[GraphManager] Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    print(f"[GraphManager] Downloading graph for '{region}' from OSM...")
    G = ox.graph_from_place(region, network_type="drive")

    # Add speed and travel time attributes
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # Save to cache
    ox.save_graphml(G, filepath=path)
    print(f"[GraphManager] Graph cached: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


def add_elevation_to_graph(G: nx.MultiDiGraph, api_key: str = None) -> nx.MultiDiGraph:
    """
    Add elevation data to graph nodes and compute edge grades.
    Falls back to grade=0.0 if no API key is provided.

    Grade formula: grade = (elevation_end - elevation_start) / edge_length
    Positive = uphill (more energy), Negative = downhill (regen potential)
    """
    key = api_key or GOOGLE_ELEVATION_API_KEY

    if not key:
        print("[GraphManager] No elevation API key — setting grade=0.0 on all edges (degraded mode)")
        for u, v, k, data in G.edges(data=True, keys=True):
            G[u][v][k]["grade"] = 0.0
        return G

    try:
        print("[GraphManager] Fetching elevation data from Google API...")
        G = ox.elevation.add_node_elevations_google(G, api_key=key)
        G = ox.elevation.add_edge_grades(G)
        print("[GraphManager] Elevation + grade enrichment complete")
    except Exception as e:
        print(f"[GraphManager] Elevation API failed: {e} — falling back to grade=0.0")
        for u, v, k, data in G.edges(data=True, keys=True):
            if "grade" not in data:
                G[u][v][k]["grade"] = 0.0

    return G


def get_graph_stats(G: nx.MultiDiGraph) -> dict:
    """Return summary stats for the health endpoint."""
    total_edges = G.number_of_edges()
    enriched = sum(1 for _, _, d in G.edges(data=True) if "energy_kwh" in d)

    return {
        "nodes": G.number_of_nodes(),
        "edges": total_edges,
        "enriched_edges": enriched,
        "coverage_pct": round(enriched / total_edges * 100, 1) if total_edges else 0,
        "has_elevation": any("grade" in d and d["grade"] != 0.0
                            for _, _, d in G.edges(data=True)),
    }
