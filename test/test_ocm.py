import asyncio
from orchestrator.ocm_client import retrieve_stations_for_route

class MockNode:
    def __init__(self, y, x):
        self.y = y
        self.x = x
    def __getitem__(self, key):
        return getattr(self, key)
        
class MockGraph:
    def __init__(self):
        self.nodes = {1: MockNode(20.0063, 73.79)}

async def test_ocm_retrieval():
    print("1. Mocking Graph to bypass OSMnx download...")
    G = MockGraph()
    start_node = 1

    ev_profile = {
        "connector_types": ["CCS2", "Type2", "CHAdeMO", "Type1_J1772"],
        "max_charge_rate_kw": 150.0
    }
    
    print("\n2. Executing `retrieve_stations_for_route`...")
    stations = await retrieve_stations_for_route(
        route_nodes=[start_node, start_node, start_node],
        G=G, # Passing mock 
        ev_profile=ev_profile,
        radius_km=15.0,
        country_code="IN"
    )
    
    print(f"\n3. Results:")
    print(f"Total stations dynamically parsed, normalized, and ranked: {len(stations)}\n")
    
    for idx, s in enumerate(stations[:5]):
        conn = s.best_connection
        ctype = conn.connector_type if conn else "Unknown"
        cpow = conn.effective_power_kw if conn else 0.0
        print(f"[{idx+1}] {s.name}")
        print(f"    Operator : {s.operator_name}")
        print(f"    Score    : {s.ranking_score}")
        print(f"    Power    : {cpow} kW ({ctype})")
        print(f"    Graph ID : {s.graph_node}")

if __name__ == "__main__":
    asyncio.run(test_ocm_retrieval())
