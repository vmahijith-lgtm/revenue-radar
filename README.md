# 🛡️ Attribution Engine

A production-ready, **multi-touch marketing attribution** pipeline built on **DuckDB + dbt + Streamlit**.

It generates synthetic clickstream data, applies five attribution models (First Touch, Last Touch, U-Shaped, Time Decay, Markov Chain), computes per-channel ROI, and visualises results in an interactive dashboard.

---

## 📐 Architecture

```
sample1.py          →   DuckDB (raw_clicks)
                              ↓
                    dbt seed (channel_spend)
                              ↓
              dbt models (heuristic → markov → final → ROI)
                              ↓
                    dashboard.py (Streamlit UI)
```

---

## 🚀 Quickstart

### 1. Create & activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline (one command)
```bash
python run_pipeline.py
```
This will:
- Generate ~100 k synthetic click events → `attribution_project/dev.duckdb`
- Seed channel spend data (`seeds/channel_spend.csv`)
- Run all four dbt models

### 4. Launch the dashboard
```bash
streamlit run dashboard.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📂 Project Structure

```
attribution_engine/
├── dashboard.py              # Streamlit dashboard
├── run_pipeline.py           # Full pipeline orchestrator
├── sample1.py                # Synthetic data generator
├── see.py                    # Quick DB inspection script
├── requirements.txt
├── profiles/
│   └── profiles.yml          # dbt DuckDB connection profile
└── attribution_project/      # dbt project
    ├── dbt_project.yml
    ├── seeds/
    │   └── channel_spend.csv # Channel marketing spend
    ├── models/
    │   ├── sources.yml
    │   ├── heuristic_attribution.sql   # First/Last/U-Shaped/Time-Decay
    │   ├── markov_attribution.py       # Markov Chain (dbt Python model)
    │   ├── final_attribution.sql       # Combined comparison table
    │   └── roi_attribution.sql         # ROI% per channel per model
    └── tests/
```

---

## 📊 Attribution Models

| Model | Description |
|-------|-------------|
| **First Touch** | 100% credit to the first channel a user touched |
| **Last Touch** | 100% credit to the final channel before conversion |
| **U-Shaped** | 40% first, 40% last, 20% distributed across middle touches |
| **Time Decay** | Exponentially more credit to touches closer to conversion |
| **Markov Chain** | Data-driven removal effect — credit ∝ how much conversions drop when a channel is removed |
| **ROI Comparison** | Cross-model ROI% = (Attributed Revenue − Spend) / Spend × 100 |

---

## ⚙️ Configuration

### dbt Profile
The project uses a local `profiles/profiles.yml` (already included). You can override the DuckDB path via:
```bash
export DBT_DUCKDB_PATH=/path/to/your.duckdb
```

### U-Shaped weights
Edit `attribution_project/dbt_project.yml` → `vars.u_shape` to adjust the 40/40/20 split:
```yaml
vars:
  u_shape:
    first: 0.4
    last: 0.4
    middle: 0.2
```

---

## 🧪 Running dbt commands manually

```bash
cd attribution_project

# Check connection
dbt debug --profiles-dir ../profiles

# Load seed data only
dbt seed --profiles-dir ../profiles

# Run all models
dbt run --profiles-dir ../profiles

# Run tests
dbt test --profiles-dir ../profiles
```

---

## 📝 License

MIT
