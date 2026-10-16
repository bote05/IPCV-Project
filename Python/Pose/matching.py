"""One-to-one matching with a distance limit and optional unmatched rows."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_pairs(cost: np.ndarray, max_distance: float) -> list[tuple[int, int]]:
    """Maximize the number of valid pairs, then minimize their total distance."""
    if not cost.size:
        return []
    valid = np.isfinite(cost) & (cost <= max_distance)
    # One extra match must outweigh any reduction in the total distance.
    unmatched_cost = (min(cost.shape) + 1) * (cost[valid].max(initial=0.0) + 1.0)
    unmatched_options = np.full((len(cost), len(cost)), unmatched_cost)
    costs = np.concatenate([np.where(valid, cost, np.inf), unmatched_options], axis=1)
    rows, cols = linear_sum_assignment(costs)
    return [(int(row), int(col)) for row, col in zip(rows, cols) if col < cost.shape[1]]
