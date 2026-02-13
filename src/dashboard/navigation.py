"""
Shared navigation components for RLIC Dashboard.

Layout:
- Left pane (sidebar): Streamlit's default nav at top, then Home icon, Analysis dropdown
- Right pane: Breadcrumb, then page content
"""

import streamlit as st

# Analysis definitions - single source of truth
# Format: "Target vs Indicator" (what we predict vs what predicts it)
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
        'description': 'S&P 500 vs retail inventory-to-sales ratio'
    },
    'spy_indpro': {
        'name': 'SPY vs Industrial Production',
        'icon': '🏭',
        'short': 'SPY-INDPRO',
        'description': 'S&P 500 vs industrial production index'
    },
    'xlre_orders_inv': {
        'name': 'XLRE vs Orders/Inventories',
        'icon': '🏠',
        'short': 'XLRE-O/I',
        'description': 'Real estate sector vs orders-to-inventories ratio'
    },
    'xlp_retailirsa': {
        'name': 'XLP vs Retail Inv/Sales',
        'icon': '🛒',
        'short': 'XLP-Retail',
        'description': 'Consumer Staples sector vs retail inventory-to-sales ratio'
    },
    'xly_retailirsa': {
        'name': 'XLY vs Retail Inv/Sales',
        'icon': '🛍️',
        'short': 'XLY-Retail',
        'description': 'Consumer Discretionary sector vs retail inventory-to-sales ratio'
    },
    'xlre_newhomesales': {
        'name': 'XLRE vs New Home Sales',
        'icon': '🏡',
        'short': 'XLRE-NHS',
        'description': 'Real estate sector vs new home sales (lag +8 significant)'
    },
    'xli_ism_mfg': {
        'name': 'XLI vs ISM Manufacturing PMI',
        'icon': '🏭',
        'short': 'XLI-MFG',
        'description': 'Industrials sector vs ISM Manufacturing PMI (confirmatory)'
    },
    'xli_ism_svc': {
        'name': 'XLI vs ISM Services PMI',
        'icon': '🏢',
        'short': 'XLI-SVC',
        'description': 'Industrials sector vs ISM Services PMI (confirmatory)'
    }
}


def init_session_state():
    """Initialize session state with defaults."""
    if 'selected_analysis' not in st.session_state:
        st.session_state.selected_analysis = 'spy_retailirsa'


def render_sidebar(current_page: str = None):
    """
    Render the sidebar with navigation controls.
    Streamlit's default page navigation appears at the top automatically.

    Args:
        current_page: Current page name (for future use)

    Returns:
        str: The selected analysis ID
    """
    init_session_state()

    with st.sidebar:
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
