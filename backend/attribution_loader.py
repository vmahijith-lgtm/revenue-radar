"""
backend/attribution_loader.py
Loads channel performance data from the existing DuckDB attribution tables.
Falls back to sample data if DB is unavailable.
"""
import duckdb
from pathlib import Path

# Path to the existing attribution DuckDB
_DB_PATH = Path(__file__).resolve().parents[1] / "attribution_project" / "dev.duckdb"

# Fallback sample data (used when DB is not available)
FALLBACK_DATA = [
    {"channel": "Paid Search",    "attributed_revenue": 68000, "conversions": 500},
    {"channel": "Organic Search", "attributed_revenue": 42000, "conversions": 350},
    {"channel": "Social Media",   "attributed_revenue": 25000, "conversions": 210},
    {"channel": "Email",          "attributed_revenue": 15000, "conversions": 140},
    {"channel": "Direct",         "attributed_revenue": 20000, "conversions": 180},
]


def load_attribution_data() -> list[dict]:
    """
    Load channel attribution data from DuckDB (final_attribution table).
    Returns list of dicts with: channel, attributed_revenue, conversions.
    """
    if not _DB_PATH.exists():
        return FALLBACK_DATA

    try:
        con = duckdb.connect(str(_DB_PATH), read_only=True)

        # Try final_attribution first (has all model values)
        try:
            rows = con.sql("""
                SELECT
                    channel,
                    GREATEST(val_first_touch, val_last_touch, val_u_shaped,
                             val_time_decay, COALESCE(val_markov, 0)) AS attributed_revenue
                FROM final_attribution
                ORDER BY attributed_revenue DESC
            """).fetchall()
            cols = ["channel", "attributed_revenue"]
        except Exception:
            # Fallback: use heuristic_attribution
            rows = con.sql("""
                SELECT channel, val_last_touch AS attributed_revenue
                FROM heuristic_attribution
                ORDER BY attributed_revenue DESC
            """).fetchall()
            cols = ["channel", "attributed_revenue"]

        # Get conversion counts from raw_clicks
        conv_rows = con.sql("""
            SELECT channel, SUM(conversion) AS conversions
            FROM raw_clicks
            WHERE conversion = 1
            GROUP BY channel
            ORDER BY channel
        """).fetchall()
        conv_map = {r[0]: int(r[1]) for r in conv_rows}

        con.close()

        if not rows:
            return FALLBACK_DATA

        return [
            {
                "channel": r[0],
                "attributed_revenue": float(r[1]) if r[1] else 0.0,
                "conversions": conv_map.get(r[0], 0),
            }
            for r in rows
            if r[0] is not None
        ]

    except Exception:
        return FALLBACK_DATA
