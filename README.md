# 🛡️ Attribution Engine

> Multi-touch marketing attribution pipeline built on **DuckDB + dbt + Streamlit**.

Generates or accepts your own clickstream data, applies five attribution models, computes per-channel ROI, and visualises results in an interactive dashboard.

---

## Architecture

```
sample1.py  (or upload via dashboard)
      ↓
  DuckDB  raw_clicks
      ↓
  dbt seed  (channel_spend.csv)
      ↓
  dbt models → heuristic · markov · final · roi
      ↓
  dashboard.py  (Streamlit)
```

---

## Quickstart

```bash
# 1. Create virtualenv & install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run full pipeline
python run_pipeline.py

# 3. Launch dashboard
streamlit run dashboard.py
```

---

## Attribution Models

| Model | Description |
|---|---|
| **First Touch** | 100% credit to first channel |
| **Last Touch** | 100% credit to final channel |
| **U-Shaped** | 40/40/20 — first, last, middle |
| **Time Decay** | Exponential credit toward conversion |
| **Markov Chain** | Data-driven removal effect |
| **ROI Comparison** | (Revenue − Spend) / Spend × 100 per model |

---

## Project Structure

```
attribution_engine/
├── dashboard.py              # Streamlit app
├── run_pipeline.py           # Pipeline orchestrator
├── sample1.py                # Synthetic data generator
├── requirements.txt
├── .streamlit/config.toml    # Theme & server config
├── profiles/profiles.yml     # dbt DuckDB connection
└── attribution_project/      # dbt project
    ├── dbt_project.yml
    ├── seeds/channel_spend.csv
    └── models/
        ├── sources.yml
        ├── heuristic_attribution.sql
        ├── markov_attribution.py
        ├── final_attribution.sql
        └── roi_attribution.sql
```

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (make sure `*.duckdb` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set **Main file path**: `dashboard.py`
4. Add a **Secrets** entry if needed (none required for default config)
5. Click **Deploy**

> **Note**: The app will show a friendly "no data yet" screen until the pipeline is run. For cloud deployments, upload your CSV directly through the sidebar uploader.

---

## dbt Commands

```bash
cd attribution_project

dbt debug  --profiles-dir ../profiles   # Test connection
dbt seed   --profiles-dir ../profiles   # Load channel spend
dbt run    --profiles-dir ../profiles   # Run all models
dbt test   --profiles-dir ../profiles   # Run schema tests
```

---

## Configuration

**U-Shaped weights** — adjust in the dashboard sidebar (no restart needed).

**dbt vars** — edit `attribution_project/dbt_project.yml`:
```yaml
vars:
  u_shape:
    first: 0.4
    last: 0.4
    middle: 0.2
```

**DuckDB path override**:
```bash
export DBT_DUCKDB_PATH=/path/to/custom.duckdb
```

---

## License

MIT
