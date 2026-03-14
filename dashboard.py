import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import subprocess
from pathlib import Path
from utils import (
    get_channels_from_df,
    write_channel_spend_csv,
    DEFAULT_SPEND,
)

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Attribution Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(139,92,246,0.06));
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(124,58,237,0.18);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase; color: #a78bfa;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.6rem; font-weight: 700; color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid rgba(124,58,237,0.2);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #c4b5fd; }

.page-title {
    font-size: 1.8rem; font-weight: 800;
    background: linear-gradient(90deg, #c4b5fd, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}
.page-subtitle { font-size: 0.85rem; color: #64748b; margin-bottom: 1.5rem; }

/* Upload landing page */
.upload-hero {
    text-align: center;
    padding: 4rem 2rem;
    border: 2px dashed rgba(124,58,237,0.35);
    border-radius: 20px;
    margin: 2rem 0;
    background: rgba(124,58,237,0.04);
}
.upload-hero h2 { color: #c4b5fd; font-size: 1.5rem; margin-bottom: 0.5rem; }
.upload-hero p  { color: #64748b; font-size: 0.9rem; }

[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(124,58,237,0.4);
    border-radius: 10px;
    padding: 0.5rem;
}
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    border: none !important; font-weight: 600;
}
hr { border-color: rgba(124,58,237,0.15); margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent
DB_PATH         = PROJECT_ROOT / "attribution_project" / "dev.duckdb"
DBT_PROJECT_DIR = PROJECT_ROOT / "attribution_project"
PROFILES_DIR    = PROJECT_ROOT / "profiles"

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
REQUIRED_COLS = {"event_id", "user_id", "timestamp", "channel", "conversion", "conversion_value"}

SAMPLE_CSV = (
    "event_id,user_id,timestamp,channel,conversion,conversion_value,cost,user_engagement,touch_number\n"
    "evt-001,user-A,2024-01-01 10:00:00,Paid Search,0,0.0,2.5,0.8,1\n"
    "evt-002,user-A,2024-01-02 11:00:00,Email,0,0.0,0.3,0.8,2\n"
    "evt-003,user-A,2024-01-03 12:00:00,Direct,1,150.0,0.0,0.8,3\n"
    "evt-004,user-B,2024-01-01 09:00:00,Social Media,0,0.0,1.8,0.6,1\n"
    "evt-005,user-B,2024-01-02 14:00:00,Organic Search,1,95.0,0.5,0.6,2\n"
)

MODEL_LABELS = {
    "val_first_touch": "First Touch", "val_last_touch": "Last Touch",
    "val_u_shaped": "U-Shaped",       "val_time_decay": "Time Decay",
    "val_markov": "Markov",           "roi_first_touch": "First Touch",
    "roi_last_touch": "Last Touch",   "roi_u_shaped": "U-Shaped",
    "roi_time_decay": "Time Decay",   "roi_markov": "Markov",
}

MODELS = {
    "First Touch":      {"table": "heuristic_attribution", "col": "val_first_touch",  "group": "Single Model"},
    "Last Touch":       {"table": "heuristic_attribution", "col": "val_last_touch",   "group": "Single Model"},
    "U-Shaped":         {"table": "heuristic_attribution", "col": "val_u_shaped",     "group": "Single Model"},
    "Time Decay":       {"table": "heuristic_attribution", "col": "val_time_decay",   "group": "Single Model"},
    "Markov Chain":     {"table": "markov_attribution",    "col": "attributed_value", "group": "Single Model"},
    "Model Comparison": {"table": "final_attribution",     "col": None,               "group": "Compare"},
    "ROI Comparison":   {"table": "roi_attribution",       "col": None,               "group": "Compare"},
}

MODEL_DESC = {
    "First Touch":      "100% credit to the first touchpoint.",
    "Last Touch":       "100% credit to the last touchpoint before conversion.",
    "U-Shaped":         "40% first · 40% last · 20% distributed across middle touches.",
    "Time Decay":       "Exponentially more credit to touches closer to conversion.",
    "Markov Chain":     "Data-driven: credit based on how much conversions drop when a channel is removed.",
    "Model Comparison": "All attribution models side-by-side for every channel.",
    "ROI Comparison":   "Return on spend % per channel, across all attribution models.",
}

# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
if "results_ready" not in st.session_state: st.session_state["results_ready"] = False
if "cache_ts"      not in st.session_state: st.session_state["cache_ts"]      = 0

ts = st.session_state["cache_ts"]

# ─────────────────────────────────────────────────────────────
# DB & cache helpers
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_con():
    if not DB_PATH.exists():
        return None
    try:
        return duckdb.connect(str(DB_PATH), read_only=True)
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_kpis(_ts):
    con = get_con()
    if con is None: return None
    try:
        return con.sql("""
            SELECT COUNT(DISTINCT user_id) AS users,
                   SUM(conversion)         AS conversions,
                   SUM(conversion_value)   AS revenue,
                   SUM(cost)               AS spend
            FROM raw_clicks
        """).df().iloc[0]
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_current_channels(_ts):
    con = get_con()
    if con is None: return []
    try:
        return [r[0] for r in con.sql(
            "SELECT DISTINCT channel FROM raw_clicks WHERE channel IS NOT NULL ORDER BY channel"
        ).fetchall()]
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_current_spend(_ts):
    con = get_con()
    if con is None: return {}
    try:
        return {r[0]: r[1] for r in con.sql("SELECT channel, spend FROM channel_spend").fetchall()}
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def get_available_models_cached(_ts):
    con = get_con()
    if con is None: return []
    available = []
    for table, names in {
        "heuristic_attribution": ["First Touch", "Last Touch", "U-Shaped", "Time Decay"],
        "final_attribution":     ["Model Comparison"],
        "roi_attribution":       ["ROI Comparison"],
    }.items():
        try:
            con.sql(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
            available.extend(names)
        except Exception:
            pass
    try:
        row = con.sql("SELECT status FROM markov_attribution LIMIT 1").fetchone()
        if row and row[0] == "success":
            available.insert(4, "Markov Chain")
    except Exception:
        pass
    return available


@st.cache_data(ttl=300, show_spinner=False)
def fetch_model_data(_ts, model_name):
    con = get_con()
    if con is None: return None, None
    try:
        if model_name == "Model Comparison":
            df = con.sql("SELECT channel,val_first_touch,val_last_touch,val_u_shaped,val_time_decay,val_markov FROM final_attribution").df()
            plot_df = df.melt("channel", var_name="model", value_name="revenue")
            plot_df["model"] = plot_df["model"].map(MODEL_LABELS)
        elif model_name == "ROI Comparison":
            df = con.sql("SELECT channel,roi_first_touch,roi_last_touch,roi_u_shaped,roi_time_decay,roi_markov FROM roi_attribution").df()
            plot_df = df.melt("channel", var_name="model", value_name="roi_pct")
            plot_df["model"] = plot_df["model"].map(MODEL_LABELS)
        else:
            cfg = MODELS[model_name]
            df = con.sql(f"SELECT channel, {cfg['col']} AS value FROM {cfg['table']} ORDER BY 2 DESC").df()
            plot_df = df.assign(model=model_name)
        return df, plot_df
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────
# Pipeline helpers
# ─────────────────────────────────────────────────────────────
def validate_csv(df):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        return False, f"Missing columns: **{', '.join(sorted(missing))}**"
    if len(df) == 0:
        return False, "File has no data rows."
    return True, "OK"


def ingest_to_duckdb(df: pd.DataFrame) -> int:
    df = df.copy()
    if "cost"            not in df.columns: df["cost"]            = 0.0
    if "user_engagement" not in df.columns: df["user_engagement"] = 1.0
    if "touch_number"    not in df.columns:
        df = df.sort_values(["user_id", "timestamp"])
        df["touch_number"] = df.groupby("user_id").cumcount() + 1
    df["conversion"]       = df["conversion"].astype(int)
    df["conversion_value"] = df["conversion_value"].astype(float)
    df["cost"]             = df["cost"].astype(float)
    df["user_engagement"]  = df["user_engagement"].astype(float)
    df["touch_number"]     = df["touch_number"].astype(int)
    df["timestamp"]        = pd.to_datetime(df["timestamp"])
    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE raw_clicks AS SELECT * FROM df")
    n = con.execute("SELECT COUNT(*) FROM raw_clicks").fetchone()[0]
    con.close()
    return n


def run_dbt_pipeline():
    cmds = [
        f'dbt seed --profiles-dir "{PROFILES_DIR}"',
        f'dbt run  --profiles-dir "{PROFILES_DIR}"',
    ]
    log_parts = []
    for cmd in cmds:
        r = subprocess.run(cmd, shell=True, cwd=DBT_PROJECT_DIR, capture_output=True, text=True)
        log_parts.append(f"$ {cmd}\n{r.stdout}{r.stderr}")
        if r.returncode != 0:
            return False, "\n".join(log_parts)
    return True, "\n".join(log_parts)


def bust_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state["cache_ts"] += 1


# ─────────────────────────────────────────────────────────────
# Upload form — reused in both sidebar and main page
# ─────────────────────────────────────────────────────────────
def render_upload_form(key_prefix: str):
    """Render the file uploader + spend inputs + run button."""
    st.download_button(
        "📄 Download sample CSV template",
        SAMPLE_CSV,
        file_name="sample_clicks.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"{key_prefix}_dl",
    )

    uploaded_file = st.file_uploader(
        "Upload your clickstream CSV",
        type=["csv"],
        label_visibility="visible",
        key=f"{key_prefix}_uploader",
    )

    if not uploaded_file:
        return

    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Cannot read file: {e}")
        return

    ok, msg = validate_csv(df_raw)
    if not ok:
        st.error(f"❌ {msg}")
        return

    channels = get_channels_from_df(df_raw)
    st.success(f"✅ **{len(df_raw):,} rows** · {len(channels)} channels detected")

    with st.expander("Preview (first 5 rows)"):
        st.dataframe(df_raw.head(5), use_container_width=True)

    st.markdown("**💰 Set channel spend (used to calculate ROI)**")
    existing_spend = fetch_current_spend(ts)
    spend_map = {}
    for ch in channels:
        spend_map[ch] = st.number_input(
            ch, min_value=0,
            value=int(existing_spend.get(ch, DEFAULT_SPEND)),
            step=500, key=f"{key_prefix}_spend_{ch}",
        )

    if st.button("🚀 Run Attribution", type="primary", use_container_width=True, key=f"{key_prefix}_run"):
        with st.status("Running pipeline…", expanded=True) as status:
            st.write("📝 Loading data into DuckDB…")
            try:
                n = ingest_to_duckdb(df_raw)
                st.write(f"   ✔ {n:,} rows loaded")
            except Exception as e:
                status.update(label="Ingestion failed", state="error")
                st.error(str(e))
                return

            st.write("💰 Saving channel spend…")
            write_channel_spend_csv(channels, spend_map=spend_map)

            st.write("⚙️ Running dbt attribution models…")
            ok_dbt, log = run_dbt_pipeline()

            if ok_dbt:
                status.update(label="✅ Attribution complete!", state="complete")
                st.session_state["results_ready"] = True
                bust_caches()
                st.rerun()
            else:
                status.update(label="dbt failed", state="error")
                with st.expander("Error log"):
                    st.code(log, language="bash")


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Attribution Engine")
    st.markdown("<hr>", unsafe_allow_html=True)

    if st.session_state["results_ready"]:
        # Show spend editor + option to re-upload
        upload_tab, spend_tab = st.tabs(["📥 Re-upload", "💰 Spend"])

        with upload_tab:
            render_upload_form("sidebar")

        with spend_tab:
            existing_channels = fetch_current_channels(ts)
            existing_spend    = fetch_current_spend(ts)
            if existing_channels:
                st.caption("Adjust spend and re-run dbt to update ROI.")
                updated_spend = {}
                for ch in existing_channels:
                    updated_spend[ch] = st.number_input(
                        ch, min_value=0,
                        value=int(existing_spend.get(ch, DEFAULT_SPEND)),
                        step=500, key=f"spend_ex_{ch}",
                    )
                if st.button("💾 Save & Rerun", use_container_width=True, type="primary"):
                    with st.spinner("Updating…"):
                        write_channel_spend_csv(existing_channels, spend_map=updated_spend)
                        ok, log = run_dbt_pipeline()
                        if ok:
                            bust_caches()
                            st.rerun()
                        else:
                            st.error("dbt failed")
                            with st.expander("Error log"):
                                st.code(log, language="bash")
    else:
        st.caption("Upload your data and run attribution to see results.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("dbt only: `python run_pipeline.py`  \nSynthetic data: `python sample1.py`")


# ─────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">🛡️ Attribution Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Multi-touch marketing attribution · powered by dbt + DuckDB</p>', unsafe_allow_html=True)

# ── Gate: show upload landing if no data uploaded yet ─────────
if not st.session_state["results_ready"]:
    st.markdown("""
    <div class="upload-hero">
        <h2>📂 Upload your clickstream data to get started</h2>
        <p>Attribution results will appear here once you upload a CSV and run the pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    render_upload_form("main")
    st.stop()

# ── Results view ─────────────────────────────────────────────
kpis = fetch_kpis(ts)
if kpis is not None:
    spend   = float(kpis["spend"])
    revenue = float(kpis["revenue"])
    roi     = ((revenue - spend) / spend * 100) if spend > 0 else 0
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Users",       f"{int(kpis['users']):,}")
    k2.metric("Conversions", f"{int(kpis['conversions']):,}")
    k3.metric("Revenue",     f"${revenue:,.0f}")
    k4.metric("Ad Spend",    f"${spend:,.0f}")
    k5.metric("Blended ROI", f"{roi:.1f}%")

st.markdown("<hr>", unsafe_allow_html=True)

valid_models = get_available_models_cached(ts)
if not valid_models:
    st.warning("No attribution models built yet.")
    st.stop()

tabs = st.tabs(valid_models)

for model_name, tab in zip(valid_models, tabs):
    with tab:
        st.caption(MODEL_DESC.get(model_name, ""))

        with st.spinner("Loading…"):
            df, plot_df = fetch_model_data(ts, model_name)

        if df is None:
            st.error(f"No data for **{model_name}**.")
            continue

        chart_col, table_col = st.columns([3, 2], gap="large")

        with chart_col:
            if model_name == "Model Comparison":
                fig = px.bar(plot_df, x="channel", y="revenue", color="model", barmode="group",
                             color_discrete_sequence=px.colors.qualitative.Vivid,
                             labels={"revenue": "Attributed Revenue ($)", "channel": ""})
            elif model_name == "ROI Comparison":
                fig = px.bar(plot_df, x="channel", y="roi_pct", color="model", barmode="group",
                             color_discrete_sequence=px.colors.qualitative.Vivid,
                             labels={"roi_pct": "ROI (%)", "channel": ""})
                fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
            else:
                fig = px.bar(plot_df, x="channel", y="value", color="channel",
                             color_discrete_sequence=px.colors.qualitative.Vivid,
                             labels={"value": "Attributed Revenue ($)", "channel": ""})

            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", size=12),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=11)),
                yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
                margin=dict(l=0, r=0, t=10, b=0), bargap=0.2,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with table_col:
            st.dataframe(
                df.style.format({c: "{:,.2f}" for c in df.select_dtypes("number").columns}),
                use_container_width=True, height=360,
            )
            st.download_button(
                "⬇️ Export CSV",
                df.to_csv(index=False).encode(),
                file_name=f"{model_name.replace(' ','_').lower()}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"dl_{model_name}",
            )
