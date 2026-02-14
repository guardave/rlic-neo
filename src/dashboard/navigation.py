"""
Shared navigation components for RLIC Dashboard.

Layout:
- Left pane (sidebar): Streamlit's default nav at top, then Home icon, Analysis dropdown
- Right pane: Breadcrumb, then page content

Data source: SQLite config database (config_db.py)
"""

import streamlit as st
from src.dashboard.config_db import get_all_analyses, get_analysis_config


def _get_analyses_dict():
    """Build analyses dict from SQLite database."""
    analyses = get_all_analyses()
    return {
        a['id']: {
            'name': a['name'],
            'icon': a['icon'],
            'short': a['short_name'],
            'description': a['description'],
        }
        for a in analyses
    }


# Lazy-loaded analyses dict (populated on first access)
_analyses_cache = None


def _get_cached_analyses():
    """Get cached analyses dict, loading from DB on first call."""
    global _analyses_cache
    if _analyses_cache is None:
        _analyses_cache = _get_analyses_dict()
    return _analyses_cache




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
    analyses = _get_cached_analyses()

    with st.sidebar:
        # Focus analysis selector with Material Icons
        selected = st.selectbox(
            "Focus Analysis",
            options=list(analyses.keys()),
            format_func=lambda x: analyses[x]['name'],
            index=list(analyses.keys()).index(st.session_state.selected_analysis),
            key='global_analysis_selector',
            label_visibility="collapsed"
        )

        if selected != st.session_state.selected_analysis:
            st.session_state.selected_analysis = selected
            st.rerun()

        # Analysis description
        analysis = analyses[st.session_state.selected_analysis]
        st.caption(analysis['description'])

        st.markdown("---")
        st.caption("RLIC Enhancement Project v0.2")

    return st.session_state.selected_analysis


def render_breadcrumb(current_page: str):
    """
    Render breadcrumb at top of content area.

    Args:
        current_page: Name of current page (e.g., "Overview", "Correlation")
    """
    init_session_state()
    analyses = _get_cached_analyses()
    analysis = analyses[st.session_state.selected_analysis]
    st.markdown(f":material/{analysis['icon']}: **{analysis['short']}** / {current_page}")
    st.markdown("---")


def get_current_analysis():
    """Get the current analysis info dict."""
    init_session_state()
    analyses = _get_cached_analyses()
    return analyses[st.session_state.selected_analysis]


def get_analysis_title():
    """Get a formatted title for the current analysis."""
    analysis = get_current_analysis()
    return f":material/{analysis['icon']}: {analysis['name']}"
