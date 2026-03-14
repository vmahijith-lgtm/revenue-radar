"""
backend/attribution_loader.py
Loads channel performance data from the existing DuckDB attribution tables.
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
    Returns list of dicts: {channel, attributed_revenue, conversions, spend}.
    spend = 0 for channels with no entry in channel_spend.
    """
    if not _DB_PATH.exists():
        return FALLBACK_DATA

    try:
        con = duckdb.connect(str(_DB_PATH), read_only=True)

        # ── Revenue per channel from final_attribution ────────────────────
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
            rev_rows = con.sql("""
                SELECT channel, COALESCE(val_last_touch, 0) AS attributed_revenue
                FROM heuristic_attribution
                ORDER BY attributed_revenue DESC
            """).fetchall()

        if not rev_rows:
            con.close()
            return FALLBACK_DATA

        # ── Conversion counts from raw_clicks ─────────────────────────────
        conv_map = {}
        try:
            for r in con.sql("""
                SELECT channel, SUM(conversion) AS conversions
                FROM raw_clicks
                WHERE channel IS NOT NULL
                GROUP BY channel
            """).fetchall():
                conv_map[r[0]] = int(r[1])
        except Exception:
            pass

        # ── Historical spend per channel from channel_spend seed ──────────
        spend_map = {}
        try:
            for r in con.sql(
                "SELECT channel, COALESCE(spend, 0) AS spend FROM channel_spend"
            ).fetchall():
                spend_map[r[0]] = float(r[1])
        except Exception:
            pass

        con.close()

        return [
            {
                "channel":            r[0],
                "attributed_revenue": float(r[1]) if r[1] else 0.0,
                "conversions":        conv_map.get(r[0], 0),
                "spend":              spend_map.get(r[0], 0.0),
            }
            for r in rev_rows
            if r[0] is not None
        ]

    except Exception:
        return FALLBACK_DATA
