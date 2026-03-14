import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DBT_PROJECT_DIR = PROJECT_ROOT / "attribution_project"
PROFILES_DIR = PROJECT_ROOT / "profiles"


def run_cmd(cmd, cwd=None):
    print(f"\n🟢 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def main():
    # 1. Regenerate raw_clicks data into DuckDB
    run_cmd("python sample1.py", cwd=PROJECT_ROOT)

    # 2. Seed static reference data (channel_spend.csv → DuckDB)
    run_cmd(
        f'dbt seed --profiles-dir "{PROFILES_DIR}"',
        cwd=DBT_PROJECT_DIR,
    )

    # 3. Run dbt models (heuristics + markov + final + ROI)
    run_cmd(
        f'dbt run --profiles-dir "{PROFILES_DIR}"',
        cwd=DBT_PROJECT_DIR,
    )

    print("\n✅ Pipeline complete: data generated → seeded → dbt models updated.")
    print("👉 Launch the dashboard with:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()