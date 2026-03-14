"""
run_pipeline.py – dbt orchestrator.

Auto-syncs channel_spend from raw_clicks costs before running dbt.
Works for ANY dataset — no manual edits to channel_spend.csv needed.

Usage:
  1. Upload your CSV via the dashboard, OR run:  python sample1.py
  2. Then run:  python run_pipeline.py
"""
import subprocess
from pathlib import Path
import sys

PROJECT_ROOT    = Path(__file__).resolve().parent
DBT_PROJECT_DIR = PROJECT_ROOT / "attribution_project"
PROFILES_DIR    = PROJECT_ROOT / "profiles"

sys.path.insert(0, str(PROJECT_ROOT))


def run_cmd(cmd, cwd=None):
    print(f"\n▶  {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def main():
    # ── Step 1: Compute spend from actual raw_clicks cost data ───────
    # This ensures channel_spend.csv always reflects the uploaded dataset,
    # regardless of what channel names or cost values are present.
    print("💰 Syncing spend from raw_clicks cost column…")
    try:
        from utils import sync_spend_from_raw_clicks
        spend_map = sync_spend_from_raw_clicks()
        if spend_map:
            print(f"   ✔ {len(spend_map)} channels  "
                  f"(total spend: ${sum(spend_map.values()):,.0f})")
            for ch, sp in sorted(spend_map.items()):
                print(f"      {ch}: ${sp:,.2f}")
        else:
            print("   ⚠  No spend data (raw_clicks may be empty or cost column is zero).")
    except Exception as e:
        print(f"   ⚠  Could not sync spend: {e} — using existing channel_spend.csv")

    # ── Step 2: Seed spend into DuckDB ───────────────────────────────
    run_cmd(f'dbt seed --profiles-dir "{PROFILES_DIR}"', cwd=DBT_PROJECT_DIR)

    # ── Step 3: Run attribution models ───────────────────────────────
    run_cmd(f'dbt run  --profiles-dir "{PROFILES_DIR}"', cwd=DBT_PROJECT_DIR)

    print("\n✅ Attribution pipeline complete.")
    print("👉 Launch the dashboard:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()