import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import subprocess
from pathlib import Path
from utils import (
    get_channels_from_df,
    write_channel_spend_csv,
    sync_channel_spend_from_db,
    DEFAULT_SPEND,
    CHANNEL_SPEND_CSV,
)

# ─────────────────────────────────────────────────────────────
# Page config (must be first)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Attribution Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent
DB_PATH         = PROJECT_ROOT / "attribution_project" / "dev.duckdb"
DBT_PROJECT_DIR = PROJECT_ROOT / "attribution_project"
PROFILES_DIR    = PROJECT_ROOT / "profiles"

# ─────────────────────────────────────────────────────────────
# Schema
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
    "First Touch":      {"table": "heuristic_attribution", "col": "val_first_touch"},
    "Last Touch":       {"table": "heuristic_attribution", "col": "val_last_touch"},
    "U-Shaped":         {"table": "heuristic_attribution", "col": "val_u_shaped"},
    "Time Decay":       {"table": "heuristic_attribution", "col": "val_time_decay"},
    "Markov Chain":     {"table": "markov_attribution",    "col": "attributed_value"},
    "Model Comparison": {"table": "final_attribution",     "col": None},
    "ROI Comparison":   {"table": "roi_attribution",       "col": None},
}

MODEL_DESC = {
    "First Touch":      "100% credit to the first channel.",
    "Last Touch":       "100% credit to the last channel.",
    "U-Shaped":         "40/40/20 split: first, last, middle.",
    "Time Decay":       "Exponential credit toward conversion.",
    "Markov Chain":     "Data-driven removal-effect attribution.",
    "Model Comparison": "Side-by-side revenue across all models.",
    "ROI Comparison":   "ROI% per channel per model.",
}

# ─────────────────────────────────────────────────────────────
# DB helpers
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
    if con is None:
        return None
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
    """Return channels currently in the DB (for spend editor)."""
    con = get_con()
    if con is None:
        return []
    try:
        rows = con.sql(
            "SELECT DISTINCT channel FROM raw_clicks WHERE channel IS NOT NULL ORDER BY channel"
        ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_current_spend(_ts):
    """Return current channel spend from DuckDB seed table."""
    con = get_con()
    if con is None:
        return {}
    try:
        rows = con.sql("SELECT channel, spend FROM channel_spend").fetchall()
        return {r[0]: r[1] for r in rows}
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def get_available_models_cached(_ts):
    con = get_con()
    if con is None:
        return []
    available = []
    checks = {
        "heuristic_attribution": ["First Touch", "Last Touch", "U-Shaped", "Time Decay"],
        "final_attribution":     ["Model Comparison"],
        "roi_attribution":       ["ROI Comparison"],
    }
    for table, names in checks.items():
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
    if con is None:
        return None, None
    try:
        if model_name == "Model Comparison":
            df = con.sql(
                "SELECT channel,val_first_touch,val_last_touch,val_u_shaped,val_time_decay,val_markov "
                "FROM final_attribution"
            ).df()
            plot_df = df.melt("channel", var_name="model", value_name="revenue")
            plot_df["model"] = plot_df["model"].map(MODEL_LABELS)
        elif model_name == "ROI Comparison":
            df = con.sql(
                "SELECT channel,roi_first_touch,roi_last_touch,roi_u_shaped,roi_time_decay,roi_markov "
                "FROM roi_attribution"
            ).df()
            plot_df = df.melt("channel", var_name="model", value_name="roi_pct")
            plot_df["model"] = plot_df["model"].map(MODEL_LABELS)
        else:
            cfg = MODELS[model_name]
            df = con.sql(
                f"SELECT channel, {cfg['col']} AS value FROM {cfg['table']} ORDER BY 2 DESC"
            ).df()
            plot_df = df.assign(model=model_name)
        return df, plot_df
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────
# Upload helpers
# ─────────────────────────────────────────────────────────────
def validate_csv(df: pd.DataFrame):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        return False, f"Missing columns: **{', '.join(sorted(missing))}**"
    if len(df) == 0:
        return False, "File contains no data rows."
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
# Session state
# ─────────────────────────────────────────────────────────────
if "cache_ts"         not in st.session_state: st.session_state["cache_ts"]         = 0
if "upload_df"        not in st.session_state: st.session_state["upload_df"]        = None
if "upload_channels"  not in st.session_state: st.session_state["upload_channels"]  = []
if "spend_confirmed"  not in st.session_state: st.session_state["spend_confirmed"]  = False

ts = st.session_state["cache_ts"]

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Attribution Engine")
    st.markdown("---")

    # ── Upload ───────────────────────────────────────────────
    st.markdown("### 📥 Upload Click Data")
    st.caption("Required columns: `event_id`, `user_id`, `timestamp`, `channel`, `conversion`, `conversion_value`")

    st.download_button(
        "📄 Download sample template",
        SAMPLE_CSV,
        file_name="sample_clicks.csv",
        mime="text/csv",
        use_container_width=True,
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Cannot read file: {e}")
            df_raw = None

        if df_raw is not None:
            ok, msg = validate_csv(df_raw)
            if not ok:
                st.error(f"❌ {msg}")
            else:
                channels = get_channels_from_df(df_raw)
                st.success(f"✅ {len(df_raw):,} rows · **{len(channels)} channels** detected")

                with st.expander("Preview data"):
                    st.dataframe(df_raw.head(5), use_container_width=True)

                # ── Per-channel spend config ─────────────────
                st.markdown("#### 💰 Set Channel Spend")
                st.caption(
                    "Enter how much was spent on each channel. "
                    "This is used to calculate ROI. Leave at 0 if unknown."
                )

                # Prefill from existing spend data if available
                existing_spend = fetch_current_spend(ts)
                spend_map = {}
                for ch in channels:
                    default_val = int(existing_spend.get(ch, DEFAULT_SPEND))
                    spend_map[ch] = st.number_input(
                        ch,
                        min_value=0,
                        value=default_val,
                        step=500,
                        key=f"spend_{ch}",
                    )

                if st.button("🚀 Run Attribution", type="primary", use_container_width=True):
                    with st.status("Running pipeline…", expanded=True) as status:
                        st.write("📝 Writing data to DuckDB…")
                        try:
                            n = ingest_to_duckdb(df_raw)
                            st.write(f"   ✔ {n:,} rows loaded into `raw_clicks`")
                        except Exception as e:
                            status.update(label="Ingestion failed", state="error")
                            st.error(str(e))
                            st.stop()

                        st.write("💰 Saving channel spend config…")
                        write_channel_spend_csv(channels, spend_map=spend_map)
                        st.write(f"   ✔ {len(channels)} channels written to `channel_spend.csv`")

                        st.write("⚙️ Running dbt models…")
                        ok, log = run_dbt_pipeline()

                        if ok:
                            status.update(label="✅ Done!", state="complete")
                            bust_caches()
                            st.rerun()
                        else:
                            status.update(label="dbt failed", state="error")
                            with st.expander("Error log"):
                                st.code(log, language="bash")

    # ── Channel Spend Editor (when data already in DB) ───────
    elif DB_PATH.exists():
        st.markdown("### 💰 Channel Spend")
        st.caption("Edit spend values and click **Save & Rerun** to update ROI.")
        existing_channels = fetch_current_channels(ts)
        existing_spend    = fetch_current_spend(ts)

        if existing_channels:
            updated_spend = {}
            for ch in existing_channels:
                updated_spend[ch] = st.number_input(
                    ch,
                    min_value=0,
                    value=int(existing_spend.get(ch, DEFAULT_SPEND)),
                    step=500,
                    key=f"existing_spend_{ch}",
                )

            if st.button("💾 Save & Rerun dbt", use_container_width=True):
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

    st.markdown("---")
    st.caption("💡 Run `python run_pipeline.py` to regenerate synthetic data.")


# ─────────────────────────────────────────────────────────────
# MAIN – guard: no DB yet
# ─────────────────────────────────────────────────────────────
if not DB_PATH.exists():
    st.title("🛡️ Attribution Engine")
    st.info("👈 **No data yet.** Upload a CSV in the sidebar, or run `python run_pipeline.py` to generate synthetic data.")
    st.stop()

con = get_con()
if con is None:
    st.error("Cannot connect to the database.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────────────────────
st.markdown("## 🛡️ Attribution Dashboard")

kpis = fetch_kpis(ts)
if kpis is not None:
    spend   = float(kpis["spend"])
    revenue = float(kpis["revenue"])
    roi     = ((revenue - spend) / spend * 100) if spend > 0 else 0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👤 Users",       f"{int(kpis['users']):,}")
    c2.metric("✅ Conversions", f"{int(kpis['conversions']):,}")
    c3.metric("💰 Revenue",     f"${revenue:,.0f}")
    c4.metric("📢 Spend",       f"${spend:,.0f}")
    c5.metric("📈 Blended ROI", f"{roi:.1f}%")
    st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Model selector
# ─────────────────────────────────────────────────────────────
valid_models = get_available_models_cached(ts)

if not valid_models:
    st.warning("No attribution models found. Run the pipeline first.")
    st.stop()

col_sel, col_info = st.columns([3, 1])
with col_sel:
    selected = st.selectbox("Attribution Model", valid_models, label_visibility="collapsed")
with col_info:
    st.caption(MODEL_DESC.get(selected, ""))

# ─────────────────────────────────────────────────────────────
# Fetch & render
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading…"):
    df, plot_df = fetch_model_data(ts, selected)

if df is None:
    st.error(f"Could not load data for **{selected}**.")
    st.stop()

chart_col, table_col = st.columns([3, 2])

with chart_col:
    st.subheader(selected)
    if selected == "Model Comparison":
        fig = px.bar(plot_df, x="channel", y="revenue",  color="model", barmode="group",
                     labels={"revenue": "Attributed Revenue ($)", "channel": "Channel"})
    elif selected == "ROI Comparison":
        fig = px.bar(plot_df, x="channel", y="roi_pct",  color="model", barmode="group",
                     labels={"roi_pct": "ROI (%)", "channel": "Channel"})
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
    else:
        fig = px.bar(plot_df, x="channel", y="value", color="channel",
                     labels={"value": "Attributed Revenue ($)", "channel": "Channel"})

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

with table_col:
    st.subheader("Data Table")
    st.dataframe(df, use_container_width=True, height=380)
    st.download_button(
        "⬇️ Export as CSV",
        df.to_csv(index=False).encode(),
        file_name=f"{selected.replace(' ','_').lower()}.csv",
        mime="text/csv",
        use_container_width=True,
    )
