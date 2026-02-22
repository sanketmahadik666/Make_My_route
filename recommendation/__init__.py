"""EV recommendation: rank EVs by feasibility, energy usage, and charging stops."""
from .ev_ranker import rank_evs_for_route, EVSpec

__all__ = ["rank_evs_for_route", "EVSpec"]
