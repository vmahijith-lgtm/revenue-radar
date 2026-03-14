"""
backend/rl_allocator.py
Proper RL Budget Allocator — Incremental Thompson Sampling Bandit.

ALGORITHM: Greedy Incremental Allocation with Thompson Sampling
════════════════════════════════════════════════════════════════
Distributes the total budget dollar-by-dollar using a Beta-Bernoulli
bandit. At each step the agent:

  1. SAMPLES  θᵢ ~ Beta(αᵢ, βᵢ) per channel       ← exploration
  2. COMPUTES marginal return = θᵢ × wᵢ / (1 + spendᵢ)
             (diminishing returns: each extra dollar is worth less)
  3. ALLOCATES the next increment to argmax(marginal return)  ← action
  4. UPDATES  the winner's posterior with observed ROAS       ← learning

WHY THIS WORKS
  With a log utility model the KKT optimality condition is:
      wᵢ / (1 + spendᵢ) = λ  ∀ i   (equalize marginal returns)
  → spendᵢ ∝ wᵢ   (proportional to attribution weight)

  The incremental bandit converges to this optimum because:
  - High-weight channels have higher marginal returns → win more rounds
  - As their spend grows, marginal return falls → lower-weight channels
    catch up at the right proportion
  - Thompson Sampling adds exploration so uncertain channels get
    occasional budget to refine their posterior

REWARD / UPDATE
  After allocating increment Δ to channel i:
    ROAS_i    = wᵢ × log(1 + spendᵢ) / spendᵢ
    normalised = ROAS_i / max_possible_ROAS (clipped to [0,1])
    αᵢ += normalised   (Bayesian success count)
    βᵢ += 1 − normalised
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────
# Bandit arm
# ─────────────────────────────────────────────────────────────
@dataclass
class BanditArm:
    channel:     str
    weight:      float   # revenue weight from attribution data
    alpha:       float   # Beta prior: successes
    beta:        float   # Beta prior: failures

    def sample(self, rng: np.random.Generator) -> float:
        """Thompson sample: draw θ ~ Beta(α, β)."""
        return float(rng.beta(max(self.alpha, 1e-6), max(self.beta, 1e-6)))

    def marginal_return(self, theta: float, current_spend: float) -> float:
        """Expected marginal revenue of next dollar (diminishing returns)."""
        return theta * self.weight / (1.0 + current_spend)

    def update(self, roas: float, max_roas: float):
        """Update Beta posterior with normalised ROAS observation."""
        if max_roas > 0:
            normalised = max(0.0, min(roas / max_roas, 1.0))
        else:
            normalised = 0.5
        self.alpha += normalised
        self.beta  += (1.0 - normalised)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _compute_weights(attribution_data: list[dict]) -> list[float]:
    revenues = np.array([max(d["attributed_revenue"], 0.0) for d in attribution_data])
    total    = revenues.sum()
    if total == 0:
        return [1.0 / len(attribution_data)] * len(attribution_data)
    return (revenues / total).tolist()


def _roas(weight: float, total_spend: float) -> float:
    """Return on Ad Spend: revenue per dollar spent."""
    if total_spend <= 0:
        return 0.0
    return weight * np.log1p(total_spend) / total_spend


# ─────────────────────────────────────────────────────────────
# Main optimizer
# ─────────────────────────────────────────────────────────────
def optimize_budget_allocation(
    total_budget: float,
    attribution_data: list[dict],
    n_steps: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Incremental Thompson Sampling budget allocator.

    Allocates `total_budget` across channels in `n_steps` equal increments.
    At each step the channel with the highest Thompson-sampled marginal
    return wins the increment.  Posteriors are updated with the winner's
    ROAS, so the agent learns which channels are truly high-value.

    Parameters
    ----------
    total_budget     : total dollars to allocate
    attribution_data : [{channel, attributed_revenue, conversions}, ...]
    n_steps          : number of allocation increments (resolution)
    seed             : random seed for reproducibility

    Returns
    -------
    dict with channels, allocation_pcts, recommended_budgets,
         expected_roi_index, episodes_run, algorithm
    """
    if not attribution_data:
        raise ValueError("attribution_data is empty")

    rng       = np.random.default_rng(seed)
    weights   = _compute_weights(attribution_data)
    increment = total_budget / n_steps

    # Initialise arms — prior biased toward attribution weight
    # α = 1 + w*50 means Paid Search (w≈0.4) starts at α=21 vs
    # Unknown (w≈0.09) at α=5.5 — a meaningful head-start
    arms = [
        BanditArm(
            channel=d["channel"],
            weight=w,
            alpha=1.0 + w * 50,
            beta=1.0,
        )
        for d, w in zip(attribution_data, weights)
    ]

    spend = np.zeros(len(arms))           # cumulative spend per channel
    history: list[float] = []            # net reward per step

    for step in range(n_steps):
        # 1. Thompson sample
        thetas = np.array([arm.sample(rng) for arm in arms])

        # 2. Marginal returns at current spend levels
        marginals = np.array([
            arm.marginal_return(theta, spend[i])
            for i, (arm, theta) in enumerate(zip(arms, thetas))
        ])

        # 3. Allocate increment to winner
        winner = int(np.argmax(marginals))
        spend[winner] += increment

        # 4. Compute ROAS for winner and update its posterior
        winner_roas = _roas(arms[winner].weight, spend[winner])
        max_roas    = max(_roas(arm.weight, s + 1) for arm, s in zip(arms, spend))
        arms[winner].update(winner_roas, max_roas)

        # Track net reward (revenue - spend)
        net_reward = sum(
            arm.weight * np.log1p(s) - s
            for arm, s in zip(arms, spend)
        )
        history.append(net_reward)

    # ── Compute meaningful ROI estimate ──────────────────────
    # Scale the log-utility output to dollar revenue using total historical
    # attributed revenue as the reference scale.
    total_attributed_revenue = sum(d["attributed_revenue"] for d in attribution_data)

    # Estimated revenue with the recommended spending (log diminishing returns, scaled)
    # Normalise by log(1 + historical avg spend) to anchor to observed revenue
    historical_avg_spend = total_budget / len(arms)   # proxy for typical spend per channel
    rev_scale = total_attributed_revenue / max(
        sum(arm.weight * np.log1p(historical_avg_spend) for arm in arms), 1e-9
    )

    estimated_revenue = sum(
        arm.weight * np.log1p(s) * rev_scale
        for arm, s in zip(arms, spend)
    )
    roi_pct = (estimated_revenue - total_budget) / total_budget * 100

    # Final allocation percentages
    total_spent = spend.sum()
    alloc_pcts  = (spend / total_spent).tolist() if total_spent > 0 else [1/len(arms)] * len(arms)

    return {
        "channels":            [arm.channel for arm in arms],
        "allocation_pcts":     alloc_pcts,
        "recommended_budgets": spend.tolist(),
        "expected_roi_index":  round(roi_pct, 1),        # estimated ROI %
        "estimated_revenue":   round(estimated_revenue, 2),
        "episodes_run":        n_steps,
        "algorithm":           "Incremental Thompson Sampling (Beta-Bernoulli bandit)",
    }
