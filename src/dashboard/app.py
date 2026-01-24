"""
RLIC Dashboard - Main Entry Point

Interactive dashboard portal for analyzing economic indicators
and their relationship to asset returns.

Run with: streamlit run src/dashboard/app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.navigation import render_sidebar

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="RLIC Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
        # RLIC Enhancement Dashboard

        Interactive analysis portal for the Royal London Investment Clock project.

        **Features:**
        - Economic indicator analysis
        - Lead-lag relationship detection
        - Regime-based performance analysis
        - Backtesting with signal lag

        Built with Streamlit and Plotly.
        """
    }
)


def main():
    """Main app entry point - Home page."""

    # Sidebar (no home button on home page)
    render_sidebar(show_home=False)

    # Main content
    st.title("🏠 RLIC Dashboard")

    st.markdown("""
    Interactive analysis portal for economic indicators and asset returns.

    **Select an analysis in the sidebar** or click a card below to explore.
    """)

    # Analysis cards in 2x2 grid
    col1, col2 = st.columns(2)

    with col1:
        # Investment Clock
        with st.container(border=True):
            st.subheader("📈 Investment Clock Sectors")
            st.markdown("Sector performance across Investment Clock phases using Orders/Inventories and PPI.")
            st.caption("11 Sectors • 4 Phases • Monthly Data")
            if st.button("Select & Explore →", key="btn_ic", use_container_width=True):
                st.session_state.selected_analysis = 'investment_clock'
                st.switch_page("pages/2_📊_Overview.py")

        # SPY vs Retail
        with st.container(border=True):
            st.subheader("🏪 SPY vs Retail Inv/Sales")
            st.markdown("Retail inventory-to-sales ratio and S&P 500 returns relationship.")
            st.caption("RETAILIRSA • SPY • Lead-Lag Analysis")
            if st.button("Select & Explore →", key="btn_retail", use_container_width=True):
                st.session_state.selected_analysis = 'spy_retailirsa'
                st.switch_page("pages/2_📊_Overview.py")

    with col2:
        # SPY vs INDPRO
        with st.container(border=True):
            st.subheader("🏭 SPY vs Industrial Production")
            st.markdown("Industrial production and S&P 500 returns relationship.")
            st.caption("INDPRO • SPY • Regime Analysis")
            if st.button("Select & Explore →", key="btn_indpro", use_container_width=True):
                st.session_state.selected_analysis = 'spy_indpro'
                st.switch_page("pages/2_📊_Overview.py")

        # XLRE vs Orders/Inv
        with st.container(border=True):
            st.subheader("🏠 XLRE vs Orders/Inventories")
            st.markdown("Real estate sector vs manufacturing orders-to-inventories ratio.")
            st.caption("Orders/Inv Ratio • XLRE • Backtest")
            if st.button("Select & Explore →", key="btn_xlre", use_container_width=True):
                st.session_state.selected_analysis = 'xlre_orders_inv'
                st.switch_page("pages/2_📊_Overview.py")

    # Quick stats
    st.markdown("---")
    st.subheader("📊 Quick Stats")

    stat_cols = st.columns(4)
    stat_cols[0].metric("Analyses", "4")
    stat_cols[1].metric("Indicators", "12+")
    stat_cols[2].metric("Data Range", "1990-2024")
    stat_cols[3].metric("Update Freq", "Monthly")


if __name__ == "__main__":
    main()
