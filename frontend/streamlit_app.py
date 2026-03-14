"""
frontend/streamlit_app.py
Main entry point for the Attribution Engine Streamlit app.
Provides navigation to Attribution Dashboard and AI Budget Allocation.
"""
import streamlit as st

st.set_page_config(
    page_title="Attribution Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; }
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.hero {
    text-align: center;
    padding: 3rem 2rem 2rem;
}
.hero h1 {
    font-size: 2.5rem; font-weight: 900;
    background: linear-gradient(90deg, #c4b5fd, #818cf8, #6ee7b7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.hero p { color: #64748b; font-size: 1rem; }
.nav-card {
    background: rgba(26,26,46,0.7);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
}
.nav-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(124,58,237,0.2);
}
.nav-card h2 { font-size: 1.3rem; margin: 1rem 0 0.5rem; }
.nav-card p  { color: #64748b; font-size: 0.85rem; }
hr { border-color: rgba(124,58,237,0.15); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🛡️ Attribution Engine</h1>
    <p>Multi-touch marketing attribution · AI-powered budget optimization · powered by dbt + DuckDB</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="nav-card">
        <div style="font-size:3rem">🛡️</div>
        <h2>Revenue Attribution</h2>
        <p>Upload clickstream data and run multi-touch attribution models:
        First Touch, Last Touch, U-Shaped, Time Decay, and Markov Chain.</p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("dashboard.py", label="→ Open Attribution Dashboard", use_container_width=True)

with col2:
    st.markdown("""
    <div class="nav-card">
        <div style="font-size:3rem">💰</div>
        <h2>AI Budget Allocation</h2>
        <p>Enter your total marketing budget and let the RL optimizer recommend
        the optimal spend per channel based on your attribution data.</p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("frontend/pages/budget_allocation.py", label="→ Open Budget Allocation", use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#475569; font-size:0.8rem">
    Start the API first: <code>uvicorn backend.main:app --reload</code>
    &nbsp;·&nbsp;
    Then the UI: <code>streamlit run dashboard.py</code>
</div>
""", unsafe_allow_html=True)
