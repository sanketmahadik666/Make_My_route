"""
Battery constraints: SOC and SOH determine usable energy.
usable_energy = battery_capacity_wh * SOC * SOH
Feasibility checks applied outside the routing engine.
"""
from typing import List, Tuple


def usable_energy_wh(
    battery_capacity_wh: float,
    soc: float,
    soh: float,
) -> float:
    """
    Compute usable battery energy in Wh.
    SOC and SOH in [0, 1] (e.g. 0.8 = 80%).
    """
    soc = max(0.0, min(1.0, soc))
    soh = max(0.0, min(1.0, soh))
    return battery_capacity_wh * soc * soh


def is_route_feasible(
    total_energy_wh: float,
    battery_capacity_wh: float,
    soc: float,
    soh: float,
    margin_ratio: float = 1.0,
) -> Tuple[bool, float]:
    """
    Check if a route is feasible given battery constraints.
    margin_ratio: require usable_energy >= total_energy_wh * margin_ratio (e.g. 1.1 for 10% reserve).
    Returns (feasible, usable_energy_wh).
    """
    usable = usable_energy_wh(battery_capacity_wh, soc, soh)
    required = total_energy_wh * margin_ratio
    return (usable >= required, usable)
