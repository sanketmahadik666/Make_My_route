import os
from config import API_NINJAS_KEY
from core.charger_client import _fetch_api_ninjas_grid

print(f"Key loaded: {bool(API_NINJAS_KEY)}")
print("Fetching from grid...")
grid = _fetch_api_ninjas_grid(37.5, -122.0)
if grid is None:
    print("No response from API")
else:
    print(f"Success! Found {len(grid)} stations.")
    for s in grid[:2]:
        print(s)
