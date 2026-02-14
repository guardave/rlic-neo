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

from src.dashboard.navigation import render_sidebar
from src.dashboard.config_db import get_all_analyses

st.set_page_config(page_title="RLIC Dashboard", page_icon="🏠", layout="wide")

# Sidebar: analysis selector
render_sidebar(current_page="Home")

# Main content
st.title(":material/home: RLIC Dashboard")

st.markdown("""
Interactive analysis portal for economic indicators and asset returns.

**Select an analysis** in the sidebar or click a card below to explore.
""")

# Load analyses from database
all_analyses = get_all_analyses()

# Analysis cards in 2-column grid
col1, col2 = st.columns(2)

for a in all_analyses:
    col = col1 if a.get('home_column', 1) == 1 else col2
    is_selected = st.session_state.selected_analysis == a['id']

    with col:
        with st.container(border=True):
            if is_selected:
                st.markdown(f"### :material/{a['icon']}: {a['name']} :material/check_circle:")
            else:
                st.markdown(f"### :material/{a['icon']}: {a['name']}")

            st.markdown(a['description'])
            st.caption(a.get('caption', ''))

            btn_label = "Currently Selected" if is_selected else "Select & Explore"
            if st.button(btn_label, key=f"btn_{a['id']}",
                        use_container_width=True, disabled=is_selected):
                st.session_state.selected_analysis = a['id']
                st.switch_page("pages/2_📊_Overview.py")

# Summary stats
st.markdown("---")
st.subheader(":material/bar_chart: Available Data")

stat_cols = st.columns(4)
stat_cols[0].metric("Analyses", str(len(all_analyses)))
stat_cols[1].metric("Indicators", "12+")
stat_cols[2].metric("Data Range", "1990-2024")
stat_cols[3].metric("Update Freq", "Monthly")
