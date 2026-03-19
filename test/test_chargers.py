import osmnx as ox
from core.charger_client import fetch_charging_stations

# Use Nashik center: 20.0063, 73.79
print("Fetching with min_power=0.0")
stations = fetch_charging_stations(20.0063, 73.79, 10.0, 0.0)
print(f"Total stations found (power filter 0): {len(stations)}")
for s in stations[:2]:
    print(s)

print("Fetching with typical min_power=7.0")
stations_7 = fetch_charging_stations(20.0063, 73.79, 10.0, 7.0)
print(f"Total stations found (power filter 7): {len(stations_7)}")

# Let's inspect raw OSMnx tags
print("Raw OSMNX tags for charging_station:")
gdf = ox.features_from_point((20.0063, 73.79), tags={"amenity": "charging_station"}, dist=10000)
print(f"Features: {len(gdf)}")
if not gdf.empty:
    for col in gdf.columns:
        if "socket" in col or col in ["power", "max_power", "capacity"]:
            print(f"- {col}")
