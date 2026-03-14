"""
pages/budget_allocation.py
AI Budget Allocation – Streamlit multipage page.
Detected automatically by Streamlit when running:  streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

API_URL = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────
# Custom CSS (green accent to distinguish from attribution purple)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.06));
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(16,185,129,0.15);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase; color: #6ee7b7;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.6rem; font-weight: 700; color: #f1f5f9;
}
.page-title {
    font-size: 1.8rem; font-weight: 800;
    background: linear-gradient(90deg, #6ee7b7, #3b82f6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.page-subtitle { font-size: 0.85rem; color: #64748b; margin-bottom: 1.5rem; }
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #059669, #047857) !important;
    border: none !important; font-weight: 600;
}
hr { border-color: rgba(16,185,129,0.15); margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">💰 AI Budget Allocation</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-subtitle">Allocate your marketing budget across channels using '
    'reinforcement learning and live attribution data.</p>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# API health check
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=10, show_spinner=False)
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


if not check_api():
    st.error(
        "⚠️ **FastAPI server is not running.**\n\n"
        "Open a terminal and run:\n"
        "```bash\n"
        "source venv/bin/activate\n"
        "uvicorn backend.main:app --reload\n"
        "```"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────
# Load attribution data preview
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def load_preview():
    try:
        r = requests.get(f"{API_URL}/attribution-data", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


preview = load_preview()

# ─────────────────────────────────────────────────────────────
# Layout: controls (left) · data preview (right)
# ─────────────────────────────────────────────────────────────
input_col, preview_col = st.columns([2, 3], gap="large")

with input_col:
    st.markdown("### ⚙️ Configure")

    total_budget = st.number_input(
        "Total Marketing Budget ($)",
        min_value=100.0,
        max_value=100_000_000.0,
        value=10_000.0,
        step=500.0,
        format="%.2f",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**How it works**")
    st.markdown("""
- Loads live channel performance from your attribution pipeline
- Fits a **diminishing-returns revenue model** per channel  
  `revenue = weight × log(1 + spend)`
- Runs **20,000 Dirichlet random strategies**
- Returns the allocation with the highest **net reward**  
  `reward = expected revenue − ad spend`
    """)

    run_btn = st.button("🚀 Optimize Budget", type="primary", use_container_width=True)

with preview_col:
    st.markdown("### 📊 Attribution Data (input to optimizer)")
    if preview and preview.get("data"):
        df_prev = pd.DataFrame(preview["data"])
        df_prev["attributed_revenue"] = df_prev["attributed_revenue"].map("${:,.0f}".format)
        df_prev["conversions"]        = df_prev["conversions"].map("{:,}".format)
        df_prev.columns               = ["Channel", "Attributed Revenue", "Conversions"]
        st.dataframe(df_prev, use_container_width=True, hide_index=True)
        st.caption(f"Source: **{preview.get('data_source', 'DuckDB')}** · {preview.get('count', len(preview['data']))} channels")
    else:
        st.info("No attribution data available. Run `python run_pipeline.py` after uploading data.")


# ─────────────────────────────────────────────────────────────
# Run optimization on button click
# ─────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running RL optimizer — evaluating 20,000 strategies…"):
        try:
            resp = requests.post(
                f"{API_URL}/optimize-budget",
                json={"total_budget": total_budget},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Is `uvicorn backend.main:app --reload` running?")
            st.stop()
        except requests.exceptions.HTTPError as e:
            try:
                detail = e.response.json().get("detail", str(e))
            except Exception:
                detail = e.response.text or str(e)
            st.error(f"API error ({e.response.status_code}): {detail}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    allocations = data["allocations"]
    df = pd.DataFrame(allocations)
    df["pct_label"] = (df["budget_percent"] * 100).round(1).astype(str) + "%"

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## 📈 Optimization Results")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Budget",       f"${data['total_budget']:,.2f}")
    k2.metric("Channels",           str(len(allocations)))
    k3.metric("ROI Index",          str(data["expected_roi_index"]))
    k4.metric("Data",               "Live" if "DuckDB" in data.get("data_source", "") else "Sample")

    st.caption(f"📦 Data source: **{data.get('data_source', '')}**")
    st.markdown("<hr>", unsafe_allow_html=True)

    chart_col, table_col = st.columns([3, 2], gap="large")

    with chart_col:
        # Bar chart
        fig_bar = px.bar(
            df, x="channel", y="recommended_budget", color="channel",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            labels={"recommended_budget": "Budget ($)", "channel": ""},
            title="Recommended Budget per Channel",
            text="pct_label",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", size=12), showlegend=False,
            yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        # Donut chart
        fig_pie = px.pie(
            df, names="channel", values="recommended_budget", hole=0.55,
            color_discrete_sequence=px.colors.qualitative.Vivid,
            title="Budget Share",
        )
        fig_pie.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", size=12),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    with table_col:
        st.markdown("**Allocation Table**")
        display = df[["channel", "pct_label", "recommended_budget"]].copy()
        display.columns = ["Channel", "Budget %", "Recommended ($)"]
        display["Recommended ($)"] = display["Recommended ($)"].map("${:,.2f}".format)
        st.dataframe(display, use_container_width=True, hide_index=True, height=320)

        st.download_button(
            "⬇️ Export CSV",
            df[["channel", "budget_percent", "recommended_budget"]].to_csv(index=False).encode(),
            file_name="budget_allocation.csv",
            mime="text/csv",
            use_container_width=True,
        )
