"""
RLIC Dashboard - Home Page

Interactive analysis portal for economic indicators and asset returns.

Run with: streamlit run src/dashboard/Home.py
"""

import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.navigation import ANALYSES, render_sidebar

st.set_page_config(page_title="RLIC Dashboard", page_icon="🏠", layout="wide")

# Sidebar: analysis selector
render_sidebar(current_page="Home")

# Main content
st.title("🏠 RLIC Dashboard")

st.markdown("""
Interactive analysis portal for economic indicators and asset returns.

**Select an analysis** in the sidebar or click a card below to explore.
""")

# Analysis cards in 2x2 grid
col1, col2 = st.columns(2)

cards = [
    ('investment_clock', col1, "11 Sectors • 4 Phases • Monthly Data"),
    ('spy_retailirsa', col1, "RETAILIRSA • SPY • Lead-Lag Analysis"),
    ('xlp_retailirsa', col1, "RETAILIRSA • XLP • Lead-Lag Analysis"),
    ('spy_indpro', col2, "INDPRO • SPY • Regime Analysis"),
    ('xlre_orders_inv', col2, "Orders/Inv Ratio • XLRE • Backtest"),
    ('xly_retailirsa', col2, "RETAILIRSA • XLY • Lead-Lag Analysis"),
]

for analysis_id, col, caption in cards:
    info = ANALYSES[analysis_id]
    is_selected = st.session_state.selected_analysis == analysis_id

    with col:
        with st.container(border=True):
            if is_selected:
                st.markdown(f"### ✓ {info['icon']} {info['name']}")
            else:
                st.markdown(f"### {info['icon']} {info['name']}")

            st.markdown(info['description'])
            st.caption(caption)

            btn_label = "Currently Selected" if is_selected else "Select & Explore →"
            if st.button(btn_label, key=f"btn_{analysis_id}",
                        use_container_width=True, disabled=is_selected):
                st.session_state.selected_analysis = analysis_id
                st.switch_page("pages/2_📊_Overview.py")

# Summary stats
st.markdown("---")
st.subheader("📊 Available Data")

stat_cols = st.columns(4)
stat_cols[0].metric("Analyses", "6")
stat_cols[1].metric("Indicators", "12+")
stat_cols[2].metric("Data Range", "1990-2024")
stat_cols[3].metric("Update Freq", "Monthly")
