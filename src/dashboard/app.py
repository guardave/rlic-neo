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

# Analysis definitions
ANALYSES = {
    'investment_clock': {
        'name': 'Investment Clock Sectors',
        'icon': '📈',
        'short': 'IC Sectors',
        'description': 'Sector performance across Investment Clock phases'
    },
    'spy_retailirsa': {
        'name': 'SPY vs Retail Inv/Sales',
        'icon': '🏪',
        'short': 'SPY-Retail',
        'description': 'Retail inventory-to-sales ratio vs S&P 500'
    },
    'spy_indpro': {
        'name': 'SPY vs Industrial Production',
        'icon': '🏭',
        'short': 'SPY-INDPRO',
        'description': 'Industrial production vs S&P 500'
    },
    'xlre_orders_inv': {
        'name': 'XLRE vs Orders/Inventories',
        'icon': '🏠',
        'short': 'XLRE-O/I',
        'description': 'Real estate vs manufacturing orders ratio'
    }
}

# Session state initialization
if 'selected_analysis' not in st.session_state:
    st.session_state.selected_analysis = 'spy_retailirsa'


def render_analysis_selector():
    """Render the global analysis selector at the top of every page."""
    # Top bar with analysis selector
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        selected = st.selectbox(
            "📊 Focus Analysis",
            options=list(ANALYSES.keys()),
            format_func=lambda x: f"{ANALYSES[x]['icon']} {ANALYSES[x]['name']}",
            index=list(ANALYSES.keys()).index(st.session_state.selected_analysis),
            key='global_analysis_selector',
            label_visibility="collapsed"
        )

        if selected != st.session_state.selected_analysis:
            st.session_state.selected_analysis = selected
            st.rerun()

    st.markdown("---")
    return st.session_state.selected_analysis


def render_sidebar_nav():
    """Render the sidebar with current analysis info."""
    analysis_id = st.session_state.selected_analysis
    analysis = ANALYSES[analysis_id]

    with st.sidebar:
        # Current analysis header
        st.markdown(f"## {analysis['icon']} {analysis['short']}")
        st.caption(analysis['description'])
        st.markdown("---")
        st.caption("RLIC Enhancement Project v0.1")


def main():
    """Main app entry point - Home page."""

    # Global analysis selector at top
    render_analysis_selector()

    # Sidebar navigation
    render_sidebar_nav()

    # Main content
    st.title("🏠 RLIC Dashboard")

    st.markdown("""
    Interactive analysis portal for economic indicators and asset returns.

    **Select an analysis above** or click a card below to explore.
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
