"""
backend/rl_allocator.py
RL-inspired budget allocator using random search with diminishing-returns revenue model.

Reward function:
    reward = expected_revenue - ad_spend

Expected revenue per channel:
    expected_revenue = weight * log(1 + spend)

where weight is derived from the channel's attributed revenue ratio.

The optimizer runs N random allocation strategies (Dirichlet samples),
evaluates each one using the reward function, and returns the best.
"""
import numpy as np
from typing import Any


def _compute_weights(attribution_data: list[dict]) -> tuple[list[str], np.ndarray]:
    """Derive revenue weights from attribution data."""
    channels = [d["channel"] for d in attribution_data]
    revenues  = np.array([d["attributed_revenue"] for d in attribution_data], dtype=float)

    total = revenues.sum()
    if total == 0:
        weights = np.ones(len(channels)) / len(channels)
    else:
        weights = revenues / total

    return channels, weights


def _expected_reward(spend_vec: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute total reward for an allocation.
    reward = Σ [ weight_i * log(1 + spend_i) ] - total_spend
    The log models diminishing returns: doubling spend < doubling revenue.
    """
    expected_revenue = float(np.sum(weights * np.log1p(spend_vec)))
    total_spend      = float(spend_vec.sum())
    return expected_revenue - total_spend


def optimize_budget_allocation(
    total_budget: float,
    attribution_data: list[dict],
    n_iterations: int = 20_000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run random-search RL budget optimization.

    Parameters
    ----------
    total_budget     : total marketing budget to allocate
    attribution_data : list of {channel, attributed_revenue, conversions}
    n_iterations     : number of random strategies to evaluate
    seed             : reproducibility seed

    Returns
    -------
    {
        "channels"            : [...],
        "allocation_pcts"     : [...],   # fractions, sum to 1
        "recommended_budgets" : [...],   # dollar amounts
        "expected_roi_index"  : float,   # relative reward score
    }
    """
    if not attribution_data:
        raise ValueError("attribution_data is empty")

    rng = np.random.default_rng(seed)
    channels, weights = _compute_weights(attribution_data)
    n = len(channels)

    best_reward  = -np.inf
    best_alloc   = np.ones(n) / n  # equal split as baseline

    # Dirichlet random search — naturally sums to 1
    # Use a range of concentration parameters to explore from even to skewed splits
    for concentration in [0.5, 1.0, 2.0, 5.0]:
        alpha = weights * concentration * n + 0.1   # weight-biased Dirichlet
        samples = rng.dirichlet(alpha, size=n_iterations // 4)
        spend_matrix = samples * total_budget         # (n_iter, n_channels)

        rewards = np.array([
            _expected_reward(spend_vec, weights)
            for spend_vec in spend_matrix
        ])

        best_idx = int(np.argmax(rewards))
        if rewards[best_idx] > best_reward:
            best_reward = rewards[best_idx]
            best_alloc  = samples[best_idx]

    recommended = best_alloc * total_budget

    return {
        "channels":            channels,
        "allocation_pcts":     best_alloc.tolist(),
        "recommended_budgets": recommended.tolist(),
        "expected_roi_index":  round(best_reward, 4),
    }
