"""
utils.py – Shared pipeline utilities
"""
import csv
import duckdb
from pathlib import Path

PROJECT_ROOT     = Path(__file__).resolve().parent
DB_PATH          = PROJECT_ROOT / "attribution_project" / "dev.duckdb"
CHANNEL_SPEND_CSV = PROJECT_ROOT / "attribution_project" / "seeds" / "channel_spend.csv"

DEFAULT_SPEND = 10_000  # default $ spend per channel if not specified


def get_channels_from_db(db_path: Path = DB_PATH) -> list[str]:
    """Return the unique channels currently in raw_clicks."""
    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute(
        "SELECT DISTINCT channel FROM raw_clicks WHERE channel IS NOT NULL ORDER BY channel"
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


def get_channels_from_df(df) -> list[str]:
    """Return sorted unique channels from an in-memory DataFrame."""
    return sorted(df["channel"].dropna().unique().tolist())


def write_channel_spend_csv(channels: list[str], spend_map: dict = None, path: Path = CHANNEL_SPEND_CSV):
    """
    Write channel_spend.csv with one row per channel.

    Parameters
    ----------
    channels  : list of channel names
    spend_map : optional dict {channel: spend_value}; defaults to DEFAULT_SPEND for missing channels
    path      : destination CSV path
    """
    spend_map = spend_map or {}
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["channel", "spend"])
        for ch in sorted(channels):
            writer.writerow([ch, spend_map.get(ch, DEFAULT_SPEND)])


def sync_channel_spend_from_db(db_path: Path = DB_PATH, spend_map: dict = None):
    """
    Read channels from the DB and (re)write channel_spend.csv.
    Safe to call any time after raw_clicks is populated.
    """
    channels = get_channels_from_db(db_path)
    if not channels:
        return
    write_channel_spend_csv(channels, spend_map=spend_map)
