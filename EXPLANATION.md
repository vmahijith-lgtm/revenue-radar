# Attribution Engine — Complete Technical & Business Explanation

---

## Table of Contents
1. [What This Project Does](#1-what-this-project-does)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Attribution Models — Math & Logic](#4-attribution-models--math--logic)
5. [RL Budget Allocator — Math & Logic](#5-rl-budget-allocator--math--logic)
6. [Understanding Every Result & Term](#6-understanding-every-result--term)
7. [How to Use Each Result](#7-how-to-use-each-result)
8. [Industry Value & Novelty](#8-industry-value--novelty)

---

## 1. What This Project Does

The Attribution Engine answers two of the most important questions in marketing:

> **"Which channels actually drove my revenue?"**
> **"How should I split my budget to maximise return?"**

Traditional analytics tools can tell you *clicks* and *impressions*. This system goes further — it tells you **how much revenue each channel deserves credit for**, using five different statistical models, and then uses a **Reinforcement Learning agent** to recommend exactly how much to spend on each channel next time.

---

## 2. End-to-End System Architecture

```
User uploads CSV (clickstream data)
          │
          ▼
    Streamlit Dashboard
    • Validates schema
    • Ingests into DuckDB (raw_clicks table)
          │
          ▼
    dbt Pipeline (runs automatically)
    ┌─────────────────────────────────────────┐
    │  seed: channel_spend.csv               │
    │  model 1: heuristic_attribution.sql    │
    │  model 2: markov_attribution.py        │
    │  model 3: final_attribution.sql        │
    │  model 4: roi_attribution.sql          │
    └─────────────────────────────────────────┘
          │
          ▼
    DuckDB (dev.duckdb)
    Attribution result tables
          │
          ├──► Streamlit Dashboard
          │    (visualise attribution, ROI, compare models)
          │
          └──► FastAPI (/optimize-budget)
               • loads attribution data
               • runs Thompson Sampling RL
               └──► Budget Allocation results
                    └──► Streamlit (Budget Allocation page)
```

### Components

| Component | Technology | Role |
|---|---|---|
| Data store | DuckDB | Embedded analytical database — no server needed |
| Pipeline | dbt-core + dbt-duckdb | SQL and Python model execution, lineage tracking |
| Dashboard | Streamlit | Interactive web UI |
| Budget API | FastAPI + uvicorn | REST API for the RL optimizer |
| RL Engine | NumPy (custom) | Thompson Sampling bandit agent |

---

## 3. Data Pipeline

### Input: Clickstream Data

Each row in your CSV represents one **marketing touchpoint** — a moment when a user interacted with a channel before (or without) converting.

| Column | Meaning |
|---|---|
| `event_id` | Unique identifier for this touch event |
| `user_id` | Identifies the user (links touches into a journey) |
| `timestamp` | When the touch happened |
| `channel` | Marketing channel (e.g. "Paid Search", "Email") |
| `conversion` | 1 = this user converted (bought/signed up), 0 = did not |
| `conversion_value` | Revenue value of the conversion ($) |
| `cost` | Ad spend attributed to this touch event |
| `user_engagement` | (optional) engagement score 0–1 |
| `touch_number` | Position in the user's journey (1 = first) |

### What dbt does

dbt (data build tool) is a transformation layer that turns raw tables into analytics-ready models, with:
- **Lineage tracking** — every model knows its dependencies
- **Incremental builds** — only re-runs what changed
- **SQL + Python models** — heuristics in SQL, Markov in Python

---

## 4. Attribution Models — Math & Logic

### The Core Problem

A user's purchase journey may look like this:
```
Day 1: Google Ad → Day 3: Email → Day 5: Direct visit → 💰 Purchase $200
```

Which channel gets credit for the $200? This is the **attribution problem**.

---

### Model 1: First Touch Attribution

**Idea:** The channel that first introduced the user to your brand gets 100% of the credit.

**Math:**
```
credit(channel) = conversion_value   if touch_number = 1 (for that user)
                = 0                  otherwise
```

**When to use:** Measuring brand discovery, top-of-funnel effectiveness.

**Limitation:** Ignores every touchpoint that nurtured the user toward conversion.

---

### Model 2: Last Touch Attribution

**Idea:** The channel that was the final push before conversion gets 100% of the credit.

**Math:**
```
credit(channel) = conversion_value   if this touch is the last before conversion
                = 0                  otherwise
```

**When to use:** Measuring closing/conversion-focused channels.

**Limitation:** Ignores awareness and consideration phases. Overvalues retargeting.

---

### Model 3: U-Shaped Attribution (Position-Based)

**Idea:** Split credit between first and last touches (most important), with remaining credit spread across the middle.

**Default weights:**
```
First touch:  40%
Last touch:   40%
Middle touches: 20% split equally
```

**Math (for a journey with N touches):**
```
credit_first  = 0.40 × conversion_value
credit_last   = 0.40 × conversion_value
credit_middle = 0.20 × conversion_value / (N - 2)   [for each middle touch]
```

**When to use:** Balancing awareness and conversion credit. The most commonly used model in industry.

**Novelty here:** The weights are user-configurable via sliders in the dashboard — you can tune 40/40/20 to match your business reality.

---

### Model 4: Time Decay Attribution

**Idea:** Channels closer to the conversion get exponentially more credit. Older touches decay in importance.

**Math:**
```
decay_weight(t) = exp(−λ × Δt)

where:
  Δt = days between touch and conversion
  λ  = decay rate (controls how fast credit fades)

credit_i = (decay_weight_i / Σ decay_weight_j) × conversion_value
```

**When to use:** Short sales cycles where recent influence matters most (e.g. flash sales, SaaS trials).

**Limitation:** Undervalues long-term brand building.

---

### Model 5: Markov Chain Attribution

**Idea:** Treat the user journey as a probabilistic graph. Each channel is a state. Measure how much conversion probability drops when a channel is removed — that drop is its credit.

**Math:**

**Step 1 — Build the transition matrix.**
From your data, compute the probability of moving from channel A to channel B:
```
P(A → B) = count(journeys where B follows A) / count(journeys through A)
```

Special states:
- `Start` — beginning of every journey
- `Conversion` — successful purchase
- `Null` — journey ended without conversion

**Step 2 — Compute overall conversion rate.**
Using the transition matrix, calculate the probability that a journey starting at `Start` reaches `Conversion`.

**Step 3 — Removal Effect.**
For each channel C, temporarily remove it from the graph (set all transitions through C to go to `Null`), and recompute the conversion rate:
```
removal_effect(C) = (P_full − P_without_C) / P_full
```

**Step 4 — Attribute revenue.**
```
credit(C) = removal_effect(C) / Σ removal_effect(i) × total_conversion_value
```

**Why it is superior:** It is the only model that captures true **incremental value** — if removing Email drops conversions by 30%, Email deserves 30% of the credit, regardless of where it sits in the journey.

**Industry use:** Used by Facebook's Attribution product, Google's Data-Driven Attribution, and sophisticated in-house analytics teams.

---

### Model 6: Model Comparison (Final Attribution)

The `final_attribution` table combines all five models in one view:

| channel | val_first_touch | val_last_touch | val_u_shaped | val_time_decay | val_markov |
|---|---|---|---|---|---|
| Paid Search | $X | $X | $X | $X | $X |
| Email | $X | $X | $X | $X | $X |

**Use for:** Identifying where models disagree. If Markov gives a channel much less credit than Last Touch, that channel is likely a retargeting channel that appears at the end of journeys but doesn't independently drive them.

---

### Model 7: ROI Attribution

**Math:**
```
ROI(channel, model) = (attributed_revenue(channel, model) − spend(channel))
                      / spend(channel) × 100%
```

**Example:** Paid Search attributed $68,000 via Markov with $15,000 spend → ROI = +353%.

**A zero-spend channel** shows N/A for ROI since we cannot compute return on $0 spend.

---

## 5. RL Budget Allocator — Math & Logic

### Problem Statement

Given a total budget B and attribution data, find the spend per channel `(s₁, s₂, ..., sₙ)` such that:
```
maximise: Σ wᵢ × log(1 + sᵢ)
subject to: Σ sᵢ = B,   sᵢ ≥ 0
```

where `wᵢ` is channel i's revenue weight from attribution.

### Why log(1 + spend)?

This is the **diminishing returns** model — universally validated in advertising:

```
spend:    $0     → $1,000  → $2,000  → $10,000
revenue:  $0     → $120K   → $160K   → $230K   (not linear)
```

Doubling spend does not double revenue. The log function captures this mathematically.

### The Optimal Allocation (Analytical)

Setting the gradient of the objective equal via Lagrange multipliers (KKT conditions):
```
∂/∂sᵢ [wᵢ × log(1 + sᵢ)] = λ   ∀ i

→ wᵢ / (1 + sᵢ) = λ

→ sᵢ* = wᵢ/λ − 1

→ sᵢ* ≈ wᵢ × (B + n) − 1   (approximately proportional to wᵢ)
```

**The optimal spend is proportional to channel attribution weight**, adjusted for diminishing returns.

### Thompson Sampling RL Agent

Rather than computing the analytical solution directly, the system uses a Reinforcement Learning agent that **discovers** this optimum through experience:

**Per episode (1000 episodes total):**
```
1. SAMPLE:   θᵢ ~ Beta(αᵢ, βᵢ)       ← agent's belief about channel value
2. COMPUTE:  marginalᵢ = θᵢ × wᵢ / (1 + spendᵢ)   ← expected next-dollar return
3. ALLOCATE: spend[argmax(marginal)] += increment    ← give next dollar to best channel
4. UPDATE:   ROAS_winner = wᵢ × log(1+spend) / spend
             αᵢ += normalised_ROAS                  ← Bayesian learning
             βᵢ += 1 − normalised_ROAS
```

**Why Thompson Sampling?**
- **Exploration**: Channels with uncertain posteriors (wide Beta) occasionally get explored — prevents locking in on early lucky channels
- **Exploitation**: Channels with consistently high ROAS dominate over time
- **Convergence**: Mathematically proven to yield regret O(√T·log T) — the best achievable for this class of problems
- **Used in production** at Google Ads, Meta Ads, Netflix recommendation

### Beta Distribution — Your Agent's Belief

Each channel's `Beta(α, β)` posterior represents the agent's confidence:
```
α = accumulated successes (high ROAS episodes)
β = accumulated failures  (low ROAS episodes)
mean(Beta) = α / (α + β)
```

A channel with `Beta(50, 5)` is confidently good. A channel with `Beta(3, 4)` is uncertain.

---

## 6. Understanding Every Result & Term

### Attribution Dashboard

| Term | Meaning |
|---|---|
| **Users** | Unique users in your data who had at least one touch |
| **Conversions** | Total number of purchases/sign-ups |
| **Revenue** | Sum of all conversion values ($) |
| **Ad Spend** | Sum of all cost values across all touches |
| **Blended ROI** | (Revenue − Spend) / Spend × 100% — overall return |
| **Attributed Revenue** | How much revenue each model assigns to each channel |
| **val_first_touch** | Revenue attributed under First Touch model |
| **val_last_touch** | Revenue attributed under Last Touch model |
| **val_u_shaped** | Revenue attributed under U-Shaped (40/40/20) model |
| **val_time_decay** | Revenue attributed under Time Decay model |
| **val_markov** | Revenue attributed under Markov Chain model |
| **removal_effect** | (Markov only) Drop in conversion rate when a channel is removed |

### Budget Allocation Page

| Term | Meaning |
|---|---|
| **Total Budget** | The dollar amount you input to allocate |
| **Channels** | Number of marketing channels detected from your data |
| **Estimated Revenue** | Projected revenue from this allocation using the log-utility model scaled to historical revenue |
| **Estimated ROI** | (Estimated Revenue − Budget) / Budget × 100% |
| **Budget %** | Fraction of total budget recommended for this channel |
| **Recommended Budget** | Exact dollar amount to spend on this channel |

### ROI Comparison Chart

Each bar shows: `(attributed_revenue − channel_spend) / channel_spend × 100%`

- A **positive ROI** bar = that channel generates more revenue than it costs
- A **negative ROI** bar = that channel costs more than it generates (under this model)
- Different bar heights across models for the same channel = **model disagreement** → investigate

---

## 7. How to Use Each Result

### Scenario A: Evaluating Channel Performance

1. Open **Model Comparison** tab — see all five models side by side
2. Channels where **all models agree** (similar bar heights) → high-confidence performers
3. Channels where **models disagree sharply** → investigate (e.g. last-touch inflated by retargeting)
4. Use **Markov** as the most data-driven ground truth for large datasets

### Scenario B: Budget Planning

1. Open **ROI Comparison** tab — identify channels with consistently positive ROI across all models
2. Open **Budget Allocation** page — input next quarter's budget
3. Read recommended spend per channel directly from the allocation table
4. Use **Export CSV** to share with your media buyer or ad platform

### Scenario C: Detecting Over/Under-Investment

Compare **First Touch vs Last Touch** ROI:
- If Last Touch ROI >> First Touch ROI → you are likely **over-investing in retargeting** and under-crediting awareness
- If First Touch ROI >> Last Touch ROI → your top-of-funnel is converting well; consider scaling

### Scenario D: Choosing an Attribution Model

| Your business | Recommended model |
|---|---|
| Long sales cycles (B2B, SaaS) | U-Shaped — values first contact and closing |
| Short purchase cycles (e-commerce) | Time Decay — recent touches drive decisions |
| Large datasets, trust the data | Markov Chain — no assumptions required |
| Simple measurement baseline | Last Touch — universally understood |
| Benchmarking only | Model Comparison — run all, compare |

---

## 8. Industry Value & Novelty

### The $500 Billion Problem

Digital advertising spend globally exceeded $600 billion in 2024. Industry studies estimate **~40% is misallocated** due to poor attribution — brands overspend on retargeting and underspend on awareness because they use Last Touch by default.

This project addresses that directly.

### What Makes This Novel

#### 1. Open-Source, Local, Private
Enterprise attribution tools (Rockerbox, Northbeam, Triple Whale) cost $2,000–$20,000/month and send your customer data to third-party servers. This system runs entirely **on your machine**, with no data leaving, using only open-source technology.

#### 2. Five Models in One Pipeline
Most teams pick one attribution model and stick with it. This system runs all five simultaneously on the same data, enabling **model triangulation** — understanding where models agree gives higher confidence; where they disagree reveals channel complexity.

#### 3. True Incremental Attribution (Markov)
Markov Chain removal effects are the closest open approximation to **true incrementality** (the gold standard in ad measurement). Normally this requires expensive holdout experiments. The Markov model provides a statistical proxy from your existing data.

#### 4. RL-Powered Budget Optimization
The Thompson Sampling agent goes beyond descriptive analytics into **prescriptive analytics** — not just "what happened" but "what should you do next". The Bayesian posterior naturally incorporates uncertainty: channels with less data get explored more until the agent is confident.

#### 5. Dynamic Channel Support
Unlike fixed dashboards, this system detects whatever channels exist in your data. A retailer using TikTok, Influencer, and SMS will get a correctly configured system without any code changes.

#### 6. dbt as the Analytics Layer
Using dbt introduces **software engineering best practices** to analytics:
- Version-controlled SQL and Python models
- Reproducible builds
- Data lineage (understand what feeds into what)
- Easy extension: add a new attribution model by adding one SQL file

### Comparison to Commercial Alternatives

| Feature | This Project | Northbeam | Triple Whale | Google Analytics |
|---|---|---|---|---|
| Markov Attribution | ✅ | ✅ | Limited | ❌ |
| RL Budget Optimizer | ✅ | ❌ | ❌ | ❌ |
| Data stays local | ✅ | ❌ | ❌ | ❌ |
| Open source | ✅ | ❌ | ❌ | ❌ |
| Cost | Free | $2K+/mo | $300+/mo | Free (limited) |
| Custom models | ✅ | Limited | Limited | ❌ |
| Any data source | ✅ | Limited | Shopify/Meta | GA only |

### Real-World Impact

A brand spending $100,000/month on ads with 5 channels:
- If Last Touch overvalues Retargeting by 30%, they are wasting ~$30,000/month
- Correcting attribution using Markov and reallocating via the RL engine recovers $30,000 in wasted spend, or alternatively increases revenue by ~$90,000–$150,000 at typical e-commerce ROAS of 3–5×
- Over 12 months: **$360,000–$1.8M in recovered value**

---

## Summary

This project combines:

| Domain | Technology | Purpose |
|---|---|---|
| Data Engineering | DuckDB + dbt | Fast, reproducible analytics pipeline |
| Statistics | 5 attribution models | Revenue attribution from clickstream data |
| Reinforcement Learning | Thompson Sampling | Optimal budget allocation under uncertainty |
| Software Engineering | FastAPI + Streamlit | Production-ready, user-friendly interface |

It transforms raw clickstream data into actionable marketing investment recommendations — a capability previously available only to enterprise teams with large budgets and data science departments.
