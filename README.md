# EV Route Feasibility & Recommendation MVP

Minimal production-quality MVP for **Electric Vehicle route feasibility and recommendation**, using only prebuilt routing (OSMnx + NetworkX). Battery constraints (SOC/SOH) are applied outside the routing engine.

## Pipeline Overview

1. **Road network** – OSMnx loads a drive network for a configurable region; edges get `length` (m) and optionally `grade` (elevation) when a raster or API is provided.
2. **Charging stations** – Loaded from a placeholder dataset or JSON/CSV file and snapped to the nearest graph nodes via OSMnx.
3. **Energy model** – Simple ML model in `energy/energy_model.py`: inputs are distance, slope, and vehicle efficiency; output is energy in Wh. Implemented with scikit-learn (no custom algorithms).
4. **Battery constraints** – `energy/soh_soc.py`: `usable_energy = battery_capacity * SOC * SOH`; feasibility checks use this outside the router.
5. **Energy-aware routing** – Edges get an `energy_cost` (Wh) from the energy model; NetworkX Dijkstra (or A*) uses `energy_cost` as weight. No modification of routing internals.
6. **Charging logic** – If route energy exceeds usable energy, `routing/charging_logic.py` finds reachable chargers and selects the best by distance and power.
7. **EV recommendation** – Given a route and a list of EVs, `recommendation/ev_ranker.py` evaluates each EV and ranks by feasibility, then by fewest charging stops and least energy.
8. **API** – FastAPI in `app.py`: route feasibility check and EV recommendation.
9. **Visualization** – Folium: route, charging stations, start/end in an HTML map.

## Project Structure

```
Make_My_route/
├── config.py              # Region and options (env, no hardcoded region)
├── network/
│   ├── loader.py          # OSMnx graph load + elevation (length, grade)
├── charging/
│   ├── stations.py        # Load stations, snap to graph (OSMnx nearest_nodes)
├── energy/
│   ├── energy_model.py    # ML energy prediction (Wh); isolated
│   ├── soh_soc.py         # usable_energy, feasibility checks
├── routing/
│   ├── energy_routing.py  # Add energy_cost, NetworkX shortest path
│   ├── charging_logic.py  # Reachable chargers, select stops
├── recommendation/
│   ├── ev_ranker.py       # Rank EVs for a route
├── visualization/
│   ├── map_render.py      # Folium route + stations + start/end
├── app.py                 # FastAPI: feasibility, recommend, map export
├── cli.py                 # CLI for feasibility, recommend, map
├── requirements.txt
└── README.md
```

## How to Run the MVP

### 1. Install dependencies

```bash
cd Make_My_route
pip install -r requirements.txt
```

### 2. Configure region (optional)

Set one of the following so the graph is not hardcoded:

- **Place query:** `EV_PLACE_QUERY="Berlin, Germany"`
- **Bounding box:** `EV_BBOX="52.6,52.4,13.6,13.2"` (north,south,east,west)
- **Point + radius:** `EV_CENTER_LAT=52.52` `EV_CENTER_LON=13.405` `EV_DIST_METERS=5000`

If none are set, the app uses a default point (Berlin center, 5 km) so the MVP runs without config. The first run may take a minute while OSMnx downloads the road network from OpenStreetMap.

### 3. Run the API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- **Route feasibility:** `POST /route/feasibility` with JSON body (start_lat/lon, end_lat/lon, battery_capacity_wh, efficiency_wh_per_km, soc, soh).
- **EV recommendation:** `POST /route/recommend` with start/end and a list of EV specs (id, battery_capacity_wh, efficiency_wh_per_km, soc, soh).
- **Map export:** `POST /map/export?start_lat=...&start_lon=...&end_lat=...&end_lon=...` to generate `ev_route_map.html`.

### 4. Run the CLI

```bash
# Feasibility
python cli.py feasibility --start 52.52 13.405 --end 52.53 13.41 --capacity 50000 --efficiency 200

# Rank EVs (default EV list or --evs '[{"capacity":50000,"efficiency":180},...]')
python cli.py recommend --start 52.52 13.405 --end 52.53 13.41

# Export Folium map
python cli.py map --start 52.52 13.405 --end 52.53 13.41 --out ev_route_map.html
```

## How to Extend

- **SOH models:** Keep using `energy/soh_soc.py` for `usable_energy`; plug in a separate SOH estimator and pass its output as `soh` (e.g. from a trained model or API).
- **Real-time data:** Replace placeholder charging in `charging/stations.py` with an API client; keep the same interface (list of dicts with lat, lon, power_kw). Optionally add live traffic by adjusting edge weights before calling the same routing.
- **Scaling:** Load graph once per worker; use a smaller bbox/place or simplify the graph for large regions. Replace in-memory graph with a persistent store and same NetworkX interface if needed.
- **Richer energy model:** Replace or retrain the model in `energy/energy_model.py` (e.g. more features, different sklearn model or another library) without changing callers that use `predict_energy_wh` / `EnergyModel.predict_wh`.
- **Other regions:** No code change; set `EV_PLACE_QUERY`, `EV_BBOX`, or center + `EV_DIST_METERS` for the desired region.

## Constraints Respected

- **Only prebuilt routing:** OSMnx for graph and snapping; NetworkX for shortest path (Dijkstra/A*) with `energy_cost` as weight. No custom routing algorithms.
- **Battery constraints outside router:** SOC/SOH and `usable_energy` are applied in `energy/soh_soc.py` and in charging logic; the routing engine only sees edge weights.
- **Modular layout:** Network, charging, energy, routing, recommendation, and visualization are separate; business logic is not mixed across modules.
