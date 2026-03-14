"""
backend/main.py
FastAPI backend for the AI Budget Allocation feature.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.attribution_loader import load_attribution_data
from backend.rl_allocator import optimize_budget_allocation

app = FastAPI(
    title="Attribution Engine – Budget Allocation API",
    description="RL-powered marketing budget optimizer backed by attribution data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────
class BudgetRequest(BaseModel):
    total_budget: float = Field(..., gt=0, description="Total marketing budget in dollars")


class ChannelAllocation(BaseModel):
    channel:             str
    budget_percent:      float   # 0–1
    recommended_budget:  float   # dollars


class BudgetResponse(BaseModel):
    total_budget:       float
    expected_roi_index: float
    data_source:        str
    allocations:        list[ChannelAllocation]


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/attribution-data")
def get_attribution_data():
    """Return the raw attribution data used by the allocator."""
    data = load_attribution_data()
    return {"data": data, "count": len(data)}


@app.post("/optimize-budget", response_model=BudgetResponse)
def optimize_budget(req: BudgetRequest):
    """
    Run RL budget optimization and return recommended allocations.

    Input:  { "total_budget": 10000 }
    Output: allocations per channel with budget % and dollar amount.
    """
    attribution_data = load_attribution_data()

    if not attribution_data:
        raise HTTPException(status_code=503, detail="No attribution data available. Run the attribution pipeline first.")

    try:
        result = optimize_budget_allocation(
            total_budget=req.total_budget,
            attribution_data=attribution_data,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

    from backend.attribution_loader import _DB_PATH
    source = "DuckDB (live attribution)" if _DB_PATH.exists() else "Sample data (fallback)"

    allocations = [
        ChannelAllocation(
            channel=ch,
            budget_percent=round(pct, 4),
            recommended_budget=round(bud, 2),
        )
        for ch, pct, bud in zip(
            result["channels"],
            result["allocation_pcts"],
            result["recommended_budgets"],
        )
    ]

    return BudgetResponse(
        total_budget=req.total_budget,
        expected_roi_index=result["expected_roi_index"],
        data_source=source,
        allocations=allocations,
    )
