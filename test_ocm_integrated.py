import asyncio
import os
import sys

# Ensure correct path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.graph_manager import get_graph
from orchestrator.ocm_client import retrieve_stations_for_route
from config import REGION

async def main():
    print(f"Loading graph for {REGION}...")
    try:
        G = get_graph(REGION)
    except Exception as e:
        print(f"Failed to load graph: {e}")
        return

    # Mock a route through Nashik center
    # Find a node near center of Nashik
    lat, lon = 20.0063, 73.7900
    import osmnx as ox
    try:
        center_node = ox.nearest_nodes(G, X=lon, Y=lat)
    except Exception as e:
        print(f"Failed to find nearest node: {e}")
        return

    route_nodes = [center_node, center_node] # Dummy 2-node route

    ev_profile = {
        "max_charge_rate_kw": 50.0,
        "connector_types": ["CCS2", "Type2"]
    }

    print("Fetching stations for route...")
    try:
        stations = await retrieve_stations_for_route(
            route_nodes=route_nodes,
            G=G,
            ev_profile=ev_profile,
            radius_km=15.0,
            country_code="IN"
        )
        print(f"\nSuccessfully retrieved {len(stations)} stations!")
        for i, s in enumerate(stations[:5]):
            print(f"[{i+1}] {s.name} ({s.best_connection.effective_power_kw}kW) - Score: {s.ranking_score}")
    except Exception as e:
        print(f"Error during retrieval: {e}")

if __name__ == "__main__":
    asyncio.run(main())
