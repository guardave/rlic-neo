"""
Shared navigation components for RLIC Dashboard.

Layout:
- Left pane (sidebar): Home button, Analysis dropdown, Section links
- Right pane: Breadcrumb, then page content

Uses CSS to hide Streamlit's default navigation for full control.
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

# Section pages for navigation
SECTIONS = [
    {'page': 'pages/1_🏠_Catalog.py', 'label': '📋 Catalog', 'name': 'Catalog'},
    {'page': 'pages/2_📊_Overview.py', 'label': '📊 Overview', 'name': 'Overview'},
    {'page': 'pages/3_📖_Qualitative.py', 'label': '📖 Qualitative', 'name': 'Qualitative'},
    {'page': 'pages/4_📈_Correlation.py', 'label': '📈 Correlation', 'name': 'Correlation'},
    {'page': 'pages/5_🔄_Lead_Lag.py', 'label': '🔄 Lead-Lag', 'name': 'Lead-Lag'},
    {'page': 'pages/6_🎯_Regimes.py', 'label': '🎯 Regimes', 'name': 'Regimes'},
    {'page': 'pages/7_💰_Backtests.py', 'label': '💰 Backtests', 'name': 'Backtests'},
    {'page': 'pages/8_🔮_Forecasts.py', 'label': '🔮 Forecasts', 'name': 'Forecasts'},
]


def init_session_state():
    """Initialize session state with defaults."""
    if 'selected_analysis' not in st.session_state:
        st.session_state.selected_analysis = 'spy_retailirsa'


def hide_default_nav():
    """Hide Streamlit's default sidebar navigation."""
    st.markdown("""
    <style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar(show_home: bool = True, current_page: str = None):
    """
    Render the full sidebar with custom navigation.

    Args:
        show_home: Whether to show the home button (False on home page)
        current_page: Current page name for highlighting

    Returns:
        str: The selected analysis ID
    """
    init_session_state()
    hide_default_nav()

    with st.sidebar:
        # Home button (not shown on home page)
        if show_home:
            if st.button("🏠 Home", use_container_width=True):
                st.switch_page("app.py")

        # Focus analysis selector
        selected = st.selectbox(
            "Focus Analysis",
            options=list(ANALYSES.keys()),
            format_func=lambda x: f"{ANALYSES[x]['icon']} {ANALYSES[x]['name']}",
            index=list(ANALYSES.keys()).index(st.session_state.selected_analysis),
            key='global_analysis_selector',
            label_visibility="collapsed"
        )

        if selected != st.session_state.selected_analysis:
            st.session_state.selected_analysis = selected
            st.rerun()

        # Analysis description
        analysis = ANALYSES[st.session_state.selected_analysis]
        st.caption(analysis['description'])

        st.markdown("---")

        # Section navigation links
        for section in SECTIONS:
            is_current = current_page == section['name']
            if is_current:
                st.markdown(f"**→ {section['label']}**")
            else:
                st.page_link(section['page'], label=section['label'])

        st.markdown("---")
        st.caption("RLIC Enhancement Project v0.1")

    return st.session_state.selected_analysis


def render_breadcrumb(current_page: str):
    """
    Render breadcrumb at top of content area.

    Args:
        current_page: Name of current page (e.g., "Overview", "Correlation")
    """
    init_session_state()
    analysis = ANALYSES[st.session_state.selected_analysis]
    st.markdown(f"**{analysis['icon']} {analysis['short']}** / {current_page}")
    st.markdown("---")


def get_current_analysis():
    """Get the current analysis info dict."""
    init_session_state()
    return ANALYSES[st.session_state.selected_analysis]


def get_analysis_title():
    """Get a formatted title for the current analysis."""
    analysis = get_current_analysis()
    return f"{analysis['icon']} {analysis['name']}"


# Backwards compatibility aliases
def render_top_bar(current_page: str = None):
    """Deprecated: Use render_sidebar() + render_breadcrumb() instead."""
    analysis_id = render_sidebar(current_page=current_page)
    if current_page:
        render_breadcrumb(current_page)
    return analysis_id


def render_sidebar_nav():
    """Deprecated: Use render_sidebar() instead."""
    render_sidebar()


def render_analysis_selector():
    """Deprecated: Use render_sidebar() instead."""
    return render_sidebar()
