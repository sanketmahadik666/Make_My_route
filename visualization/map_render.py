"""
Use Folium to display route, charging stations, and start/end points.
"""
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional

import folium
import networkx as nx


def route_to_folium_map(
    G: nx.MultiDiGraph,
    path: List[int],
    stations_snapped: Optional[List[dict]] = None,
    start_node: Optional[int] = None,
    end_node: Optional[int] = None,
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    zoom_start: int = 13,
) -> folium.Map:
    """
    Build a Folium map with route polyline, charging stations, and start/end markers.
    """
    if not path and not stations_snapped:
        center_lat = center_lat or 52.52
        center_lon = center_lon or 13.405
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
        return m

    # Get center from first node or stations
    if path and path[0] in G:
        n = G.nodes[path[0]]
        y, x = n.get("y"), n.get("x")
        if y is not None and x is not None:
            center_lat = center_lat or y
            center_lon = center_lon or x
    if center_lat is None:
        center_lat = 52.52
    if center_lon is None:
        center_lon = 13.405

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    # Route polyline (nodes have x=lon, y=lat in OSMnx)
    if path:
        coords = []
        for node_id in path:
            if node_id in G.nodes:
                n = G.nodes[node_id]
                lat, lon = n.get("y"), n.get("x")
                if lat is not None and lon is not None:
                    coords.append([lat, lon])
        if len(coords) >= 2:
            folium.PolyLine(coords, color="blue", weight=5, opacity=0.7).add_to(m)
        if coords:
            folium.Marker(coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
            folium.Marker(coords[-1], popup="End", icon=folium.Icon(color="red")).add_to(m)

    # Charging stations
    if stations_snapped:
        for st in stations_snapped:
            lat = st.get("lat")
            lon = st.get("lon")
            if lat is not None and lon is not None:
                folium.Marker(
                    [lat, lon],
                    popup=f"Charger {st.get('name', st.get('id', ''))} ({st.get('power_kw', '')} kW)",
                    icon=folium.Icon(color="orange", icon="flash"),
                ).add_to(m)

    return m


def route_to_html_file(
    G: nx.MultiDiGraph,
    path: List[int],
    filepath: str | Path,
    stations_snapped: Optional[List[dict]] = None,
    **kwargs: Any,
) -> Path:
    """Render map to HTML file."""
    m = route_to_folium_map(G, path, stations_snapped=stations_snapped, **kwargs)
    path_out = Path(filepath)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(path_out))
    return path_out


class _UncloseableBytesIO(BytesIO):
    """BytesIO that does not close on close(), so getvalue() works after Folium save()."""

    def close(self) -> None:
        pass


def route_to_html_string(
    G: nx.MultiDiGraph,
    path: List[int],
    stations_snapped: Optional[List[dict]] = None,
    **kwargs: Any,
) -> str:
    """Render map to HTML string for in-memory serving."""
    m = route_to_folium_map(G, path, stations_snapped=stations_snapped, **kwargs)
    # Folium save() writes bytes and closes the stream; use uncloseable buffer
    buf = _UncloseableBytesIO()
    m.save(buf)
    return buf.getvalue().decode("utf-8")
