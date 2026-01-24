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

# Session state initialization
if 'selected_analysis' not in st.session_state:
    st.session_state.selected_analysis = None

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}


def main():
    """Main app entry point - redirects to Catalog page."""

    # Sidebar with navigation info
    with st.sidebar:
        st.title("📊 RLIC Dashboard")
        st.markdown("---")
        st.markdown("""
        **Navigation:**
        - 🏠 **Catalog**: Analysis overview
        - 📊 **Overview**: Key metrics
        - 📈 **Correlation**: Relationship analysis
        - 🔄 **Lead-Lag**: Timing analysis
        - 🎯 **Regimes**: Phase analysis
        - 💰 **Backtests**: Strategy testing
        - 🔮 **Forecasts**: Predictions
        """)

        st.markdown("---")

        # Analysis selector
        st.subheader("Select Analysis")
        analyses = {
            'investment_clock': 'Investment Clock Sectors',
            'spy_retailirsa': 'SPY vs Retail Inv/Sales',
            'spy_indpro': 'SPY vs Industrial Production',
            'xlre_orders_inv': 'XLRE vs Orders/Inventories'
        }

        selected = st.selectbox(
            "Analysis:",
            options=list(analyses.keys()),
            format_func=lambda x: analyses[x],
            key='analysis_selector'
        )
        st.session_state.selected_analysis = selected

        st.markdown("---")
        st.caption("RLIC Enhancement Project v0.1")

    # Main content - Welcome page
    st.title("🏠 Welcome to RLIC Dashboard")

    st.markdown("""
    This dashboard provides interactive analysis of economic indicators
    and their relationship to asset returns, based on the Investment Clock framework.

    ### Available Analyses

    Select an analysis from the sidebar or click on a card below to explore.
    """)

    # Analysis cards
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("📈 Investment Clock Sectors")
            st.markdown("""
            Analyze sector performance across Investment Clock phases
            using Orders/Inventories and PPI as indicators.
            """)
            st.caption("11 Sectors • 4 Phases • Monthly Data")
            if st.button("Explore →", key="btn_ic"):
                st.session_state.selected_analysis = 'investment_clock'
                st.switch_page("pages/2_📊_Overview.py")

        with st.container(border=True):
            st.subheader("🏪 SPY vs Retail Inv/Sales")
            st.markdown("""
            Analyze the relationship between retail inventory-to-sales
            ratio and S&P 500 returns.
            """)
            st.caption("RETAILIRSA • SPY • Lead-Lag Analysis")
            if st.button("Explore →", key="btn_retail"):
                st.session_state.selected_analysis = 'spy_retailirsa'
                st.switch_page("pages/2_📊_Overview.py")

    with col2:
        with st.container(border=True):
            st.subheader("🏭 SPY vs Industrial Production")
            st.markdown("""
            Analyze the relationship between industrial production
            and S&P 500 returns.
            """)
            st.caption("INDPRO • SPY • Regime Analysis")
            if st.button("Explore →", key="btn_indpro"):
                st.session_state.selected_analysis = 'spy_indpro'
                st.switch_page("pages/2_📊_Overview.py")

        with st.container(border=True):
            st.subheader("🏠 XLRE vs Orders/Inventories")
            st.markdown("""
            Analyze real estate sector performance relative to
            manufacturing orders-to-inventories ratio.
            """)
            st.caption("Orders/Inv Ratio • XLRE • Backtest")
            if st.button("Explore →", key="btn_xlre"):
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
