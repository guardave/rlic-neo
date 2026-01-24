"""
Shared navigation components for RLIC Dashboard.

Layout:
- Top pane: Home button | Focus analysis dropdown | Breadcrumb
- Left pane (sidebar): Focus analysis title (Streamlit handles section links)
- Right pane: Page content
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


def render_top_bar(current_page: str = None):
    """
    Render the top navigation bar.

    Args:
        current_page: Name of current page for breadcrumb (e.g., "Overview", "Correlation")

    Returns:
        str: The selected analysis ID
    """
    init_session_state()

    # Top bar: Home | Analysis Selector | Breadcrumb
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("🏠 Home", use_container_width=True):
            st.switch_page("app.py")

    with col2:
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

    with col3:
        if current_page:
            analysis = ANALYSES[st.session_state.selected_analysis]
            st.markdown(f"**{analysis['short']}** / {current_page}")

    st.markdown("---")
    return st.session_state.selected_analysis


def render_sidebar():
    """
    Render the sidebar with focus analysis title.
    Streamlit's built-in navigation handles section links automatically.
    """
    init_session_state()

    analysis = ANALYSES[st.session_state.selected_analysis]

    with st.sidebar:
        # Focus analysis title
        st.markdown(f"## {analysis['icon']} {analysis['name']}")
        st.caption(analysis['description'])
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


# Backwards compatibility aliases
def render_analysis_selector():
    """Deprecated: Use render_top_bar() instead."""
    return render_top_bar()


def render_sidebar_nav():
    """Deprecated: Use render_sidebar() instead."""
    render_sidebar()
