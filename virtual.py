import folium
from folium.plugins import AntPath
import networkx as nx
import math

BASE_LAT, BASE_LON = 17.0500, 74.4200

LOCATIONS = {
    "engg_gate": {"name": "Engineering Gate", "coords": (BASE_LAT, BASE_LON)},
    "canteen": {"name": "Swad Aswad Canteen", "coords": (BASE_LAT + 0.00025, BASE_LON + 0.00040)},
    "public_school": {"name": "Public School", "coords": (BASE_LAT + 0.00040, BASE_LON - 0.00035)},
    "admin_building": {"name": "Administrative Building", "coords": (BASE_LAT + 0.00055, BASE_LON + 0.00010)},
    "library": {"name": "Library", "coords": (BASE_LAT + 0.00065, BASE_LON + 0.00045)},
    "mech_dept": {"name": "Mechanical Department", "coords": (BASE_LAT + 0.00060, BASE_LON - 0.00015)},
    "electrical_dept": {"name": "Electrical Department", "coords": (BASE_LAT + 0.00075, BASE_LON - 0.00020)},
    "iot_dept": {"name": "IoT Department", "coords": (BASE_LAT + 0.00075, BASE_LON + 0.00005)},
    "cse_block": {"name": "CSE Block", "coords": (BASE_LAT + 0.00085, BASE_LON + 0.00015)},
    "aero_block": {"name": "Aeronautical Department", "coords": (BASE_LAT + 0.00095, BASE_LON + 0.00015)},
    "food_block": {"name": "Food Department", "coords": (BASE_LAT + 0.00105, BASE_LON + 0.00015)},
    "aids_block": {"name": "AIDS Department", "coords": (BASE_LAT + 0.00115, BASE_LON + 0.00015)},
    "coe_block": {"name": "COE Department", "coords": (BASE_LAT + 0.00125, BASE_LON + 0.00015)},
    "civil_block": {"name": "Civil Department", "coords": (BASE_LAT + 0.0010, BASE_LON - 0.00025)},
    "bba_bca": {"name": "BBA / BCA Department", "coords": (BASE_LAT + 0.0012, BASE_LON - 0.00030)},
    "basic_science": {"name": "Basic Science Department", "coords": (BASE_LAT + 0.0014, BASE_LON - 0.00035)}
}

CONNECTIONS = [
    ("engg_gate", "canteen"),
    ("engg_gate", "public_school"),
    ("canteen", "admin_building"),
    ("admin_building", "library"),
    ("admin_building", "mech_dept"),
    ("mech_dept", "electrical_dept"),
    ("electrical_dept", "iot_dept"),
    ("iot_dept", "cse_block"),
    ("cse_block", "aero_block"),
    ("aero_block", "food_block"),
    ("food_block", "aids_block"),
    ("aids_block", "coe_block"),
    ("admin_building", "civil_block"),
    ("civil_block", "bba_bca"),
    ("bba_bca", "basic_science")
]

def distance(a, b):
    lat1, lon1 = LOCATIONS[a]["coords"]
    lat2, lon2 = LOCATIONS[b]["coords"]
    R = 6371000
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    return R * (dphi ** 2 + dlambda ** 2) ** 0.5

G = nx.Graph()
for a, b in CONNECTIONS:
    G.add_edge(a, b, weight=distance(a, b))

def generate_map(start=None, end=None):
    m = folium.Map(location=[BASE_LAT, BASE_LON], zoom_start=18)

    # Background gray paths
    for a, b in CONNECTIONS:
        coords = [LOCATIONS[a]["coords"], LOCATIONS[b]["coords"]]
        folium.PolyLine(coords, color="lightgray", weight=4, opacity=0.5).add_to(m)

    # All locations as gray
    for key, loc in LOCATIONS.items():
        folium.CircleMarker(
            location=loc["coords"],
            radius=5,
            color="gray",
            fill=True,
            fill_color="lightgray",
            popup=loc["name"]
        ).add_to(m)

    # Highlight selected route
    if start and end:
        try:
            path = nx.shortest_path(G, start, end, weight="weight")
            route_coords = [LOCATIONS[node]["coords"] for node in path]
            AntPath(route_coords, color="blue", weight=7, opacity=0.9, delay=600).add_to(m)

            folium.Marker(LOCATIONS[start]["coords"], popup="Start", icon=folium.Icon(color="green")).add_to(m)
            folium.Marker(LOCATIONS[end]["coords"], popup="Destination", icon=folium.Icon(color="red")).add_to(m)
        except nx.NetworkXNoPath:
            pass

    return m.get_root().render()
