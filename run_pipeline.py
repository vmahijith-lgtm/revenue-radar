"""
run_pipeline.py – dbt orchestrator

Runs dbt seed + dbt run against the DuckDB database.
Data must already be loaded into raw_clicks before running this.

To load data:
  - Upload a CSV through the dashboard (recommended), OR
  - Run `python sample1.py` to generate synthetic data first
"""
import subprocess
from pathlib import Path

PROJECT_ROOT    = Path(__file__).resolve().parent
DBT_PROJECT_DIR = PROJECT_ROOT / "attribution_project"
PROFILES_DIR    = PROJECT_ROOT / "profiles"


def run_cmd(cmd, cwd=None):
    print(f"\n🟢 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def main():
    # Seed channel spend reference data
    run_cmd(
        f'dbt seed --profiles-dir "{PROFILES_DIR}"',
        cwd=DBT_PROJECT_DIR,
    )

    # Run attribution models
    run_cmd(
        f'dbt run  --profiles-dir "{PROFILES_DIR}"',
        cwd=DBT_PROJECT_DIR,
    )

    print("\n✅ Attribution models updated.")
    print("👉 Launch the dashboard with:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()