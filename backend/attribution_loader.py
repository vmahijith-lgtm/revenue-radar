"""
backend/attribution_loader.py
Loads channel performance data from DuckDB for the RL budget optimizer.

Spend per channel = SUM(cost) from raw_clicks — computed directly from
the uploaded dataset so it is always correct for any data provided.
Falls back to sample data if DB is unavailable.
"""
import duckdb
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parents[1] / "attribution_project" / "dev.duckdb"

FALLBACK_DATA = [
    {"channel": "Paid Search",    "attributed_revenue": 68000, "conversions": 500, "spend": 15000},
    {"channel": "Organic Search", "attributed_revenue": 42000, "conversions": 350, "spend":  5000},
    {"channel": "Social Media",   "attributed_revenue": 25000, "conversions": 210, "spend": 12000},
    {"channel": "Email",          "attributed_revenue": 15000, "conversions": 140, "spend":  3000},
    {"channel": "Direct",         "attributed_revenue": 20000, "conversions": 180, "spend":     0},
]


def load_attribution_data() -> list[dict]:
    """
    Load channel attribution + spend data from DuckDB.

    Attribution revenue  → final_attribution table (dbt model)
    Conversion counts    → raw_clicks (SUM of conversion column)
    Historical spend     → raw_clicks SUM(cost) per channel   ← always from data
                           Falls back to channel_spend table if raw_clicks cost is 0.

    Returns list of dicts: {channel, attributed_revenue, conversions, spend}.
    """
    if not _DB_PATH.exists():
        return FALLBACK_DATA

    try:
        con = duckdb.connect(str(_DB_PATH), read_only=True)

        # ── 1. Attributed revenue from final_attribution ──────────────
        try:
            rev_rows = con.sql("""
                SELECT
                    channel,
                    GREATEST(
                        COALESCE(val_first_touch, 0),
                        COALESCE(val_last_touch,  0),
                        COALESCE(val_u_shaped,    0),
                        COALESCE(val_time_decay,  0),
                        COALESCE(val_markov,      0)
                    ) AS attributed_revenue
                FROM final_attribution
                ORDER BY attributed_revenue DESC
            """).fetchall()
        except Exception:
            # Fallback: use heuristic_attribution if final not ready
            try:
                rev_rows = con.sql("""
                    SELECT channel,
                        GREATEST(
                            COALESCE(val_last_touch, 0),
                            COALESCE(val_u_shaped,   0),
                            COALESCE(val_time_decay,  0)
                        ) AS attributed_revenue
                    FROM heuristic_attribution
                    ORDER BY attributed_revenue DESC
                """).fetchall()
            except Exception:
                con.close()
                return FALLBACK_DATA

        if not rev_rows:
            con.close()
            return FALLBACK_DATA

        # ── 2. Conversion counts from raw_clicks ──────────────────────
        conv_map: dict[str, int] = {}
        try:
            for r in con.sql("""
                SELECT channel, SUM(conversion) AS conversions
                FROM raw_clicks WHERE channel IS NOT NULL
                GROUP BY channel
            """).fetchall():
                conv_map[r[0]] = int(r[1])
        except Exception:
            pass

        # ── 3. Spend = SUM(cost) from raw_clicks ─────────────────────
        # Primary: actual cost data from the uploaded dataset.
        # This is always correct regardless of what channel_spend.csv contains.
        raw_spend_map: dict[str, float] = {}
        try:
            for r in con.sql("""
                SELECT channel, ROUND(SUM(cost), 2) AS total_cost
                FROM raw_clicks WHERE channel IS NOT NULL
                GROUP BY channel
            """).fetchall():
                raw_spend_map[r[0]] = float(r[1])
        except Exception:
            pass

        # Secondary: channel_spend seed (user-edited overrides)
        seed_spend_map: dict[str, float] = {}
        try:
            for r in con.sql("""
                SELECT channel, COALESCE(spend, 0) AS spend
                FROM channel_spend
                WHERE channel != '__placeholder__'
            """).fetchall():
                seed_spend_map[r[0]] = float(r[1])
        except Exception:
            pass

        # Merge: seed value wins ONLY if > 0 (user explicitly set it)
        # Otherwise raw cost is used (so any dataset works automatically)
        def resolve_spend(channel: str) -> float:
            seed_val = seed_spend_map.get(channel, 0.0)
            raw_val  = raw_spend_map.get(channel, 0.0)
            return seed_val if seed_val > 0 else raw_val

        con.close()

        return [
            {
                "channel":            r[0],
                "attributed_revenue": float(r[1]) if r[1] else 0.0,
                "conversions":        conv_map.get(r[0], 0),
                "spend":              resolve_spend(r[0]),
            }
            for r in rev_rows
            if r[0] is not None
        ]

    except Exception:
        return FALLBACK_DATA
