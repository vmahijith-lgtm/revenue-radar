# 🎯 Attribution Engine

> **Multi-touch marketing attribution + AI budget allocation** — powered by dbt, DuckDB, Thompson Sampling RL, FastAPI, and Streamlit.

Upload your clickstream data, run 5 attribution models simultaneously, and use a reinforcement learning optimizer to allocate your marketing budget across channels.

---

## ✨ Features

| Feature | Description |
|---|---|
| **5 Attribution Models** | First Touch, Last Touch, U-Shaped (40/40/20), Time Decay, Markov Chain |
| **ROI Comparison** | Side-by-side ROI% per channel across all models |
| **AI Budget Allocation** | Thompson Sampling RL agent allocates budget using ROAS-based returns |
| **Live Data** | Upload any CSV clickstream → full pipeline runs in seconds |
| **Any Channel Names** | Works with any channel names and cost structures |

---

## 🏗️ Architecture

```
Your CSV Data
     ↓
Streamlit Dashboard (dashboard.py)
     ↓
DuckDB ← dbt models (heuristic_attribution, markov_attribution,
     │               final_attribution, roi_attribution)
     ↓
FastAPI  (/optimize-budget)
     ↓
Thompson Sampling RL Allocator (rl_allocator.py)
     ↓
Budget Allocation Results → Streamlit UI
```

---

## 📁 Project Structure

```
attribution_engine/
├── dashboard.py                  # Main Streamlit app (upload, results, spend editor)
├── run_pipeline.py               # CLI pipeline runner (dbt seed + dbt run)
├── sample1.py                    # Synthetic data generator
├── utils.py                      # Shared DB helpers
├── requirements.txt
├── start.sh                      # One-command startup script
│
├── backend/
│   ├── main.py                   # FastAPI app (/health, /attribution-data, /optimize-budget)
│   ├── rl_allocator.py           # Thompson Sampling budget optimizer
│   └── attribution_loader.py     # Loads channel data from DuckDB
│
├── pages/
│   └── budget_allocation.py     # Streamlit Budget Allocation page
│
├── attribution_project/          # dbt project
│   ├── models/
│   │   ├── heuristic_attribution.sql   # First Touch, Last Touch, U-Shaped, Time Decay
│   │   ├── markov_attribution.py       # Markov Chain (removal effect)
│   │   ├── final_attribution.sql       # Combined model comparison
│   │   └── roi_attribution.sql         # ROI% per channel per model
│   └── seeds/
│       └── channel_spend.csv           # Auto-generated from uploaded data
│
└── profiles/
    └── profiles.yml              # dbt DuckDB connection config
```

---

## 🚀 Quick Start

### 1. Clone & set up

```bash
git clone https://github.com/YOUR_USERNAME/attribution_engine.git
cd attribution_engine

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Generate sample data

```bash
python sample1.py
# Creates generated_clicks.csv — upload this via the dashboard
```

### 3. Start everything

```bash
chmod +x start.sh
./start.sh
```

Or manually in two terminals:

```bash
# Terminal 1 — FastAPI backend
uvicorn backend.main:app --reload

# Terminal 2 — Streamlit dashboard
streamlit run dashboard.py
```

### 4. Open the dashboard

- Attribution Dashboard: http://localhost:8501
- Budget Allocation: http://localhost:8501 → sidebar → "budget allocation"
- API docs: http://localhost:8000/docs

---

## 📊 Input Data Format

Upload a CSV with these columns:

| Column | Type | Description |
|---|---|---|
| `event_id` | string | Unique identifier for each touch event |
| `user_id` | string | User identifier (links journey touches) |
| `timestamp` | datetime | When the touch happened |
| `channel` | string | Marketing channel name (any name works) |
| `conversion` | int (0/1) | Whether this touch resulted in a conversion |
| `conversion_value` | float | Revenue from the conversion (0 if no conversion) |
| `cost` | float | Ad spend for this touch (0 for organic channels) |
| `user_engagement` | float | Engagement score 0–1 (optional, defaults to 1.0) |
| `touch_number` | int | Position in the journey (optional, auto-computed) |

📥 Download the template from the dashboard sidebar.

---

## 🧠 Attribution Models — How They Work

### First Touch
100% of conversion value goes to the **first channel** the user ever touched.
```
User journey: Paid Search → Email → Direct ($200)
First Touch credit: Paid Search = $200
```

### Last Touch
100% goes to the channel that was **last touched before conversion**.
```
Last Touch credit: Direct = $200
```

### U-Shaped (Position-Based)
40% to first touch, 40% to last touch, 20% split equally across middle touches.
```
Paid Search = $80 (40%)  |  Email = $40 (20%)  |  Direct = $80 (40%)
```

### Time Decay
Exponential decay — recent touches get more credit. Weight = 0.5^(reverse_position − 1).
```
Direct (most recent): highest weight
Paid Search (oldest): lowest weight
```

### Markov Chain
Computes **removal effect** for each channel — how many conversions would be lost if that channel was removed. Attribution weight ∝ removal effect.

---

## 💰 AI Budget Allocation

Uses **Thompson Sampling** (Beta-Bernoulli Bandit), a Bayesian reinforcement learning algorithm:

1. Each channel is a bandit arm with a Beta(α, β) posterior
2. Over 1,000 steps, the agent samples θ from each arm's distribution
3. Budget is allocated to the arm with the highest marginal return: `θ × (weight / (1 + current_spend))`
4. Posterior is updated based on ROAS reward
5. Final allocation = posterior means, scaled to the total budget

**Estimated Revenue** = `ROAS_per_channel × recommended_spend`  
Where `ROAS = historical_attributed_revenue / historical_spend`

---

## 🌐 Deployment

### Option A — Local (recommended for development)
```bash
./start.sh
```

### Option B — Docker
```bash
docker build -t attribution-engine .
docker run -p 8000:8000 -p 8501:8501 attribution-engine
```

### Option C — Railway / Render / Heroku
1. Push to GitHub
2. Connect your repo to Railway/Render
3. Set start command: `./start.sh`
4. Set env vars if needed (see `.env.example`)

### Option D — Cloud VM (EC2, GCP, Azure)
```bash
git clone YOUR_REPO
cd attribution_engine
pip install -r requirements.txt
nohup ./start.sh &
```

---

## 🔧 Configuration

| File | Purpose |
|---|---|
| `attribution_project/dbt_project.yml` | dbt settings, U-Shaped weights (default 40/40/20) |
| `attribution_project/seeds/channel_spend.csv` | Auto-generated on each upload — do not edit manually |
| `profiles/profiles.yml` | DuckDB connection path |
| `.streamlit/config.toml` | Streamlit theme / server settings |

To change the U-Shaped weights, edit `dbt_project.yml`:
```yaml
vars:
  u_shape:
    first: 0.4   # 40% to first touch
    last:  0.4   # 40% to last touch
    middle: 0.2  # 20% split across middle touches
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| dbt-core | ≥1.9 | SQL transformation pipeline |
| dbt-duckdb | ≥1.9 | DuckDB adapter for dbt |
| duckdb | ≥1.2 | Embedded analytical database |
| streamlit | ≥1.40 | Web dashboard |
| fastapi | ≥0.110 | Budget allocation API |
| uvicorn | ≥0.29 | ASGI server |
| pandas | ≥2.2 | Data manipulation |
| numpy | ≥1.26 | RL algorithm numerics |
| plotly | ≥5.20 | Interactive charts |
| faker | ≥25.0 | Sample data generation |

---

## 🏭 Industry Value

This project replicates capabilities of commercial tools like **Northbeam**, **Triple Whale**, and **Rockerbox** — at zero cost:

- Multi-touch attribution removes the "last-click bias" in standard analytics
- Markov Chain attribution is industry standard for data-driven channel valuation
- RL budget allocation maximises forward-looking ROAS, not historical averages

---

## 📄 License

MIT License — free to use, modify, and deploy.
