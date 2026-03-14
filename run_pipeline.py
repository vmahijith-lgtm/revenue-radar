import os
import subprocess
from pathlib import Path
from utils import sync_channel_spend_from_db

PROJECT_ROOT    = Path(__file__).resolve().parent
DBT_PROJECT_DIR = PROJECT_ROOT / "attribution_project"
PROFILES_DIR    = PROJECT_ROOT / "profiles"


def run_cmd(cmd, cwd=None):
    print(f"\n🟢 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def main():
    # 1. Generate synthetic click data → DuckDB
    run_cmd("python sample1.py", cwd=PROJECT_ROOT)

    # 2. Auto-generate channel_spend.csv from actual channels in DB
    print("\n🟢 Syncing channel_spend.csv from raw_clicks channels…")
    sync_channel_spend_from_db()
    print("   ✔ channel_spend.csv updated")

    # 3. Seed channel spend table
    run_cmd(
        f'dbt seed --profiles-dir "{PROFILES_DIR}"',
        cwd=DBT_PROJECT_DIR,
    )

    # 4. Run all dbt models
    run_cmd(
        f'dbt run  --profiles-dir "{PROFILES_DIR}"',
        cwd=DBT_PROJECT_DIR,
    )

    print("\n✅ Pipeline complete: data generated → seeded → dbt models updated.")
    print("👉 Launch the dashboard with:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()