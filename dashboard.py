import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Resilient Attribution", layout="wide")

# --- CONFIG: path to the SAME DB dbt uses ---
PROJECT_ROOT = Path(__file__).resolve().parent          # attribution_engine/
DB_PATH = PROJECT_ROOT / "attribution_project" / "dev.duckdb"


# --- DEFENSIVE FUNCTION 1: Safe Connection ---
@st.cache_resource
def get_con():
    try:
        # Read-only so dbt can still write to it
        return duckdb.connect(str(DB_PATH), read_only=True)
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


# --- DEFENSIVE FUNCTION 2: Model Introspection ---
def get_available_models(con):
    models = {
        "First Touch": {
            "available": False,
            "table": "heuristic_attribution",
            "col": "val_first_touch",
        },
        "Last Touch": {
            "available": False,
            "table": "heuristic_attribution",
            "col": "val_last_touch",
        },
        "U-Shaped": {
            "available": False,
            "table": "heuristic_attribution",
            "col": "val_u_shaped",
        },
        "Time Decay": {
            "available": False,
            "table": "heuristic_attribution",
            "col": "val_time_decay",
        },
        "Markov Chain": {
            "available": False,
            "table": "markov_attribution",
            "col": "attributed_value",
        },
        "Model Comparison": {
            "available": False,
            "table": "final_attribution",
            "col": None,
        },
        "ROI Comparison": {
            "available": False,
            "table": "roi_attribution",
            "col": None,
        },
    }

    # Check heuristics exist
    try:
        con.sql("SELECT 1 FROM heuristic_attribution LIMIT 1").fetchone()
        models["First Touch"]["available"] = True
        models["Last Touch"]["available"] = True
        models["U-Shaped"]["available"] = True
        models["Time Decay"]["available"] = True
    except Exception:
        pass

    # Check Markov table
    try:
        status_check = con.sql(
            "SELECT status, error_msg FROM markov_attribution LIMIT 1"
        ).fetchone()
        if status_check and status_check[0] == "success":
            models["Markov Chain"]["available"] = True
        else:
            st.session_state["markov_error"] = (
                status_check[1] if status_check else "Unknown Error"
            )
    except Exception:
        pass

    # Check final_attribution
    try:
        con.sql("SELECT 1 FROM final_attribution LIMIT 1").fetchone()
        models["Model Comparison"]["available"] = True
    except Exception:
        pass

    # Check ROI table
    try:
        con.sql("SELECT 1 FROM roi_attribution LIMIT 1").fetchone()
        models["ROI Comparison"]["available"] = True
    except Exception:
        pass

    return models


# --- APP LOGIC ---
con = get_con()

if con is None:
    st.stop()

st.title("🛡️ Anti-Fragile Attribution Dashboard")

model_config = get_available_models(con)
valid_options = [name for name, cfg in model_config.items() if cfg["available"]]

if not valid_options:
    st.error("🚨 Critical System Failure: No attribution models available. Run dbt.")
    st.stop()

# Model selector
selected_model_name = st.selectbox("Select Model", valid_options)

# Warn if Markov missing
if "Markov Chain" not in valid_options:
    st.warning(
        f"⚠️ Markov Chain model is offline. "
        f"(Error: {st.session_state.get('markov_error', 'Not built')})"
    )

# Fetch data
cfg = model_config[selected_model_name]

if selected_model_name == "Model Comparison":
    query = """
        SELECT
            channel,
            val_first_touch,
            val_last_touch,
            val_u_shaped,
            val_time_decay,
            val_markov
        FROM final_attribution
    """
    df = con.sql(query).df()

    plot_df = df.melt(
        id_vars="channel",
        value_vars=[
            "val_first_touch",
            "val_last_touch",
            "val_u_shaped",
            "val_time_decay",
            "val_markov",
        ],
        var_name="model",
        value_name="revenue",
    )

elif selected_model_name == "ROI Comparison":
    query = """
        SELECT
            channel,
            roi_first_touch,
            roi_last_touch,
            roi_u_shaped,
            roi_time_decay,
            roi_markov
        FROM roi_attribution
    """
    df = con.sql(query).df()

    plot_df = df.melt(
        id_vars="channel",
        value_vars=[
            "roi_first_touch",
            "roi_last_touch",
            "roi_u_shaped",
            "roi_time_decay",
            "roi_markov",
        ],
        var_name="model",
        value_name="roi_pct",
    )

else:
    query = f"""
        SELECT channel, {cfg['col']} AS revenue
        FROM {cfg['table']}
        ORDER BY 2 DESC
    """
    df = con.sql(query).df()
    plot_df = df.copy()
    plot_df["model"] = selected_model_name

# Visualization
st.subheader(f"Results: {selected_model_name}")
col1, col2 = st.columns([2, 1])

with col1:
    if selected_model_name == "Model Comparison":
        fig = px.bar(
            plot_df,
            x="channel",
            y="revenue",
            color="model",
            barmode="group",
        )
    elif selected_model_name == "ROI Comparison":
        fig = px.bar(
            plot_df,
            x="channel",
            y="roi_pct",
            color="model",
            barmode="group",
        )
        fig.update_layout(yaxis_title="ROI (%)")
    else:
        fig = px.bar(
            plot_df,
            x="channel",
            y="revenue",
            color="channel",
        )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.dataframe(df)
