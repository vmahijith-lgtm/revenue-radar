import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import subprocess
import io
from pathlib import Path

st.set_page_config(page_title="Attribution Engine", layout="wide", page_icon="🛡️")

# --- PATHS ---
PROJECT_ROOT    = Path(__file__).resolve().parent
DB_PATH         = PROJECT_ROOT / "attribution_project" / "dev.duckdb"
DBT_PROJECT_DIR = PROJECT_ROOT / "attribution_project"
PROFILES_DIR    = PROJECT_ROOT / "profiles"

# ─────────────────────────────────────────────────────────────
# Expected schema for uploaded CSVs
# ─────────────────────────────────────────────────────────────
REQUIRED_COLS = {
    "event_id", "user_id", "timestamp", "channel",
    "conversion", "conversion_value",
}
OPTIONAL_COLS = {"cost", "user_engagement", "touch_number"}
ALL_COLS = REQUIRED_COLS | OPTIONAL_COLS


# ─────────────────────────────────────────────────────────────
# Sample template for download
# ─────────────────────────────────────────────────────────────
SAMPLE_CSV = """event_id,user_id,timestamp,channel,conversion,conversion_value,cost,user_engagement,touch_number
evt-001,user-A,2024-01-01 10:00:00,Paid Search,0,0.0,2.5,0.8,1
evt-002,user-A,2024-01-02 11:00:00,Email,0,0.0,0.3,0.8,2
evt-003,user-A,2024-01-03 12:00:00,Direct,1,150.0,0.0,0.8,3
evt-004,user-B,2024-01-01 09:00:00,Social Media,0,0.0,1.8,0.6,1
evt-005,user-B,2024-01-02 14:00:00,Organic Search,1,95.0,0.5,0.6,2
"""


# ─────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_con():
    try:
        return duckdb.connect(str(DB_PATH), read_only=True)
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


def get_available_models(con):
    models = {
        "First Touch":      {"available": False, "table": "heuristic_attribution", "col": "val_first_touch"},
        "Last Touch":       {"available": False, "table": "heuristic_attribution", "col": "val_last_touch"},
        "U-Shaped":         {"available": False, "table": "heuristic_attribution", "col": "val_u_shaped"},
        "Time Decay":       {"available": False, "table": "heuristic_attribution", "col": "val_time_decay"},
        "Markov Chain":     {"available": False, "table": "markov_attribution",    "col": "attributed_value"},
        "Model Comparison": {"available": False, "table": "final_attribution",     "col": None},
        "ROI Comparison":   {"available": False, "table": "roi_attribution",       "col": None},
    }

    checks = {
        "heuristic_attribution": ["First Touch", "Last Touch", "U-Shaped", "Time Decay"],
        "final_attribution":     ["Model Comparison"],
        "roi_attribution":       ["ROI Comparison"],
    }

    for table, names in checks.items():
        try:
            con.sql(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
            for n in names:
                models[n]["available"] = True
        except Exception:
            pass

    try:
        row = con.sql("SELECT status, error_msg FROM markov_attribution LIMIT 1").fetchone()
        if row and row[0] == "success":
            models["Markov Chain"]["available"] = True
        else:
            st.session_state["markov_error"] = row[1] if row else "Unknown Error"
    except Exception:
        pass

    return models


# ─────────────────────────────────────────────────────────────
# Upload helpers
# ─────────────────────────────────────────────────────────────
def validate_csv(df: pd.DataFrame):
    """Return (ok: bool, message: str)."""
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        return False, f"Missing required columns: {', '.join(sorted(missing))}"
    if len(df) == 0:
        return False, "Uploaded file contains no data rows."
    # Check channel values
    return True, "OK"


def ingest_to_duckdb(df: pd.DataFrame):
    """Write uploaded DataFrame to DuckDB as raw_clicks (write mode)."""
    # Fill optional columns with defaults if absent
    df = df.copy()
    if "cost" not in df.columns:
        df["cost"] = 0.0
    if "user_engagement" not in df.columns:
        df["user_engagement"] = 1.0
    if "touch_number" not in df.columns:
        # Compute from timestamp order per user
        df["touch_number"] = (
            df.sort_values(["user_id", "timestamp"])
              .groupby("user_id")
              .cumcount() + 1
        )

    # Ensure correct dtypes
    df["conversion"]       = df["conversion"].astype(int)
    df["conversion_value"] = df["conversion_value"].astype(float)
    df["cost"]             = df["cost"].astype(float)
    df["user_engagement"]  = df["user_engagement"].astype(float)
    df["touch_number"]     = df["touch_number"].astype(int)
    df["timestamp"]        = pd.to_datetime(df["timestamp"])

    con = duckdb.connect(str(DB_PATH))  # write mode
    con.execute("CREATE OR REPLACE TABLE raw_clicks AS SELECT * FROM df")
    row_count = con.execute("SELECT COUNT(*) FROM raw_clicks").fetchone()[0]
    con.close()
    return row_count


def run_dbt_pipeline():
    """Run dbt seed + dbt run and return (success, log)."""
    full_log = []
    for cmd in [
        f'dbt seed --profiles-dir "{PROFILES_DIR}"',
        f'dbt run  --profiles-dir "{PROFILES_DIR}"',
    ]:
        result = subprocess.run(
            cmd, shell=True, cwd=DBT_PROJECT_DIR,
            capture_output=True, text=True,
        )
        full_log.append(f"$ {cmd}\n{result.stdout}{result.stderr}")
        if result.returncode != 0:
            return False, "\n".join(full_log)
    return True, "\n".join(full_log)


# ─────────────────────────────────────────────────────────────
# SIDEBAR – Upload Section
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Data Controls")

    st.download_button(
        label="📄 Download sample CSV template",
        data=SAMPLE_CSV,
        file_name="sample_clicks.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("📥 Upload Your Click Data")
    st.caption(
        "Upload a CSV with your own clickstream data. "
        "Required columns: `event_id`, `user_id`, `timestamp`, `channel`, "
        "`conversion`, `conversion_value`."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=["csv"], label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            df_upload = None

        if df_upload is not None:
            ok, msg = validate_csv(df_upload)
            if not ok:
                st.error(f"❌ Validation failed: {msg}")
            else:
                st.success(f"✅ {len(df_upload):,} rows detected — looks good!")

                # Preview
                with st.expander("Preview (first 5 rows)"):
                    st.dataframe(df_upload.head(5), use_container_width=True)

                if st.button("🚀 Ingest & Run Attribution", use_container_width=True, type="primary"):
                    with st.status("Processing your data…", expanded=True) as status:
                        # Step 1: Write to DuckDB
                        st.write("📝 Writing data to DuckDB…")
                        try:
                            row_count = ingest_to_duckdb(df_upload)
                            st.write(f"   ✔ {row_count:,} rows loaded into `raw_clicks`")
                        except Exception as e:
                            st.error(f"DuckDB write failed: {e}")
                            status.update(label="Failed", state="error")
                            st.stop()

                        # Step 2: Run dbt
                        st.write("⚙️ Running dbt models…")
                        success, dbt_log = run_dbt_pipeline()

                        if success:
                            status.update(label="✅ Pipeline complete!", state="complete")
                            # Bust the cached connection so dashboard refreshes
                            st.cache_resource.clear()
                            st.rerun()
                        else:
                            status.update(label="dbt failed", state="error")
                            with st.expander("dbt error log"):
                                st.code(dbt_log)

    st.markdown("---")
    st.caption("💡 Or regenerate synthetic data by running `python run_pipeline.py` in the terminal.")


# ─────────────────────────────────────────────────────────────
# MAIN – Attribution Dashboard
# ─────────────────────────────────────────────────────────────
con = get_con()

if con is None:
    st.stop()

st.title("🛡️ Anti-Fragile Attribution Dashboard")

model_config  = get_available_models(con)
valid_options = [name for name, cfg in model_config.items() if cfg["available"]]

if not valid_options:
    st.error("🚨 No attribution models found. Run the pipeline or upload data first.")
    st.stop()

if "Markov Chain" not in valid_options:
    st.warning(
        f"⚠️ Markov Chain model is offline. "
        f"(Error: {st.session_state.get('markov_error', 'Not built')})"
    )

selected_model_name = st.selectbox("Select Attribution Model", valid_options)

cfg = model_config[selected_model_name]

# ── Fetch data ──────────────────────────────────────────────
if selected_model_name == "Model Comparison":
    df = con.sql("""
        SELECT channel, val_first_touch, val_last_touch,
               val_u_shaped, val_time_decay, val_markov
        FROM final_attribution
    """).df()
    plot_df = df.melt(
        id_vars="channel",
        value_vars=["val_first_touch","val_last_touch","val_u_shaped","val_time_decay","val_markov"],
        var_name="model", value_name="revenue",
    )

elif selected_model_name == "ROI Comparison":
    df = con.sql("""
        SELECT channel, roi_first_touch, roi_last_touch,
               roi_u_shaped, roi_time_decay, roi_markov
        FROM roi_attribution
    """).df()
    plot_df = df.melt(
        id_vars="channel",
        value_vars=["roi_first_touch","roi_last_touch","roi_u_shaped","roi_time_decay","roi_markov"],
        var_name="model", value_name="roi_pct",
    )

else:
    df = con.sql(f"""
        SELECT channel, {cfg['col']} AS revenue
        FROM {cfg['table']}
        ORDER BY 2 DESC
    """).df()
    plot_df = df.copy()
    plot_df["model"] = selected_model_name

# ── Quick stats ─────────────────────────────────────────────
try:
    stats = con.sql("""
        SELECT
            COUNT(DISTINCT user_id)      AS users,
            SUM(conversion)              AS conversions,
            SUM(conversion_value)        AS revenue,
            SUM(cost)                    AS spend
        FROM raw_clicks
    """).df().iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👤 Unique Users",    f"{int(stats.users):,}")
    c2.metric("✅ Conversions",     f"{int(stats.conversions):,}")
    c3.metric("💰 Total Revenue",   f"${stats.revenue:,.0f}")
    c4.metric("📊 Total Spend",     f"${stats.spend:,.0f}")
except Exception:
    pass

st.markdown("---")

# ── Chart & table ────────────────────────────────────────────
st.subheader(f"Results: {selected_model_name}")
col1, col2 = st.columns([2, 1])

with col1:
    if selected_model_name == "Model Comparison":
        fig = px.bar(plot_df, x="channel", y="revenue", color="model", barmode="group")
    elif selected_model_name == "ROI Comparison":
        fig = px.bar(plot_df, x="channel", y="roi_pct", color="model", barmode="group")
        fig.update_layout(yaxis_title="ROI (%)")
    else:
        fig = px.bar(plot_df, x="channel", y="revenue", color="channel")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.dataframe(df, use_container_width=True)
