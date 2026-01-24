"""
Catalog Page - Portal landing with analysis cards.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.components import render_regime_badge, format_pct
from src.dashboard.data_loader import load_analysis_data, get_cache_info
from src.dashboard.analysis_engine import get_current_regime, run_full_analysis

st.set_page_config(page_title="Catalog | RLIC", page_icon="🏠", layout="wide")

st.title("🏠 Analysis Catalog")
st.markdown("Select an analysis to explore in detail.")

# Analysis metadata
ANALYSES = {
    'investment_clock': {
        'title': 'Investment Clock Sectors',
        'description': 'Sector performance across IC phases using Orders/Inv + PPI',
        'indicator': 'Orders/Inv Ratio + PPI',
        'target': '11 Sectors',
        'icon': '📈'
    },
    'spy_retailirsa': {
        'title': 'SPY vs Retail Inv/Sales',
        'description': 'Retail inventory-to-sales ratio leading indicator for S&P 500',
        'indicator': 'RETAILIRSA',
        'target': 'SPY',
        'icon': '🏪'
    },
    'spy_indpro': {
        'title': 'SPY vs Industrial Production',
        'description': 'Industrial production index relationship with S&P 500',
        'indicator': 'INDPRO',
        'target': 'SPY',
        'icon': '🏭'
    },
    'xlre_orders_inv': {
        'title': 'XLRE vs Orders/Inventories',
        'description': 'Real estate sector vs manufacturing health indicator',
        'indicator': 'Orders/Inv Ratio',
        'target': 'XLRE',
        'icon': '🏠'
    }
}

# Search/filter
col1, col2 = st.columns([3, 1])
with col1:
    search = st.text_input("🔍 Search analyses", placeholder="Type to filter...")
with col2:
    sort_by = st.selectbox("Sort by", ["Name", "Recent", "Indicator"])

# Filter analyses
filtered = {k: v for k, v in ANALYSES.items()
            if search.lower() in v['title'].lower()
            or search.lower() in v['description'].lower()
            or search.lower() in v['indicator'].lower()}

if not filtered:
    st.warning("No analyses match your search.")
else:
    # Display as 2x2 grid
    cols = st.columns(2)
    for i, (analysis_id, meta) in enumerate(filtered.items()):
        with cols[i % 2]:
            with st.container(border=True):
                st.subheader(f"{meta['icon']} {meta['title']}")
                st.markdown(meta['description'])

                # Try to load and show current regime
                try:
                    # This is a placeholder - would load real data
                    st.markdown(f"**Indicator:** {meta['indicator']}")
                    st.markdown(f"**Target:** {meta['target']}")
                except Exception:
                    pass

                st.caption("Click to explore →")
                if st.button("View Analysis", key=f"btn_{analysis_id}"):
                    st.session_state.selected_analysis = analysis_id
                    st.switch_page("pages/2_📊_Overview.py")

# Cache info in expander
with st.expander("📦 Cache Status"):
    try:
        cache_info = get_cache_info()
        if not cache_info.empty:
            st.dataframe(cache_info, use_container_width=True)
        else:
            st.info("No cached data found.")
    except Exception as e:
        st.error(f"Error loading cache info: {e}")
