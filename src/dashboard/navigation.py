"""
Shared navigation components for RLIC Dashboard.

All pages should import and use these functions for consistent navigation.
"""

import streamlit as st

# Analysis definitions - single source of truth
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


def init_session_state():
    """Initialize session state with defaults."""
    if 'selected_analysis' not in st.session_state:
        st.session_state.selected_analysis = 'spy_retailirsa'


def render_analysis_selector():
    """
    Render the global analysis selector at the top of the page.
    Should be called at the start of every page.

    Returns:
        str: The selected analysis ID
    """
    init_session_state()

    # Top bar with centered analysis selector
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
    """
    Render the sidebar with current analysis info and navigation links.
    Should be called after render_analysis_selector().
    """
    init_session_state()

    analysis_id = st.session_state.selected_analysis
    analysis = ANALYSES[analysis_id]

    with st.sidebar:
        # Current analysis header
        st.markdown(f"## {analysis['icon']} {analysis['short']}")
        st.caption(analysis['description'])
        st.markdown("---")

        # Navigation links
        st.markdown("**Sections**")
        st.page_link("app.py", label="🏠 Home")
        st.page_link("pages/1_🏠_Catalog.py", label="📋 Catalog")
        st.page_link("pages/2_📊_Overview.py", label="📊 Overview")
        st.page_link("pages/3_📖_Qualitative.py", label="📖 Qualitative")
        st.page_link("pages/4_📈_Correlation.py", label="📈 Correlation")
        st.page_link("pages/5_🔄_Lead_Lag.py", label="🔄 Lead-Lag")
        st.page_link("pages/6_🎯_Regimes.py", label="🎯 Regimes")
        st.page_link("pages/7_💰_Backtests.py", label="💰 Backtests")
        st.page_link("pages/8_🔮_Forecasts.py", label="🔮 Forecasts")

        st.markdown("---")
        st.caption("RLIC Enhancement Project v0.1")


def get_current_analysis():
    """Get the current analysis info dict."""
    init_session_state()
    return ANALYSES[st.session_state.selected_analysis]


def get_analysis_title():
    """Get a formatted title for the current analysis."""
    analysis = get_current_analysis()
    return f"{analysis['icon']} {analysis['name']}"
