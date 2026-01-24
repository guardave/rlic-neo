"""
Correlation Page - Relationship analysis between indicator and target.
"""

import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.navigation import (
    render_analysis_selector, render_sidebar_nav, get_analysis_title
)
from src.dashboard.components import (
    plot_heatmap, plot_scatter, plot_rolling_correlation, format_number
)
from src.dashboard.data_loader import load_analysis_data
from src.dashboard.analysis_engine import (
    create_derivatives, correlation_analysis, correlation_with_pvalues,
    rolling_correlation
)

st.set_page_config(page_title="Correlation | RLIC", page_icon="📈", layout="wide")

# Global analysis selector at top
analysis_id = render_analysis_selector()

# Sidebar navigation
render_sidebar_nav()

# Page title
st.title(f"📈 Correlation: {get_analysis_title()}")

# Settings in sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("Settings")
    rolling_window = st.slider("Rolling Window (months)", 12, 60, 36)

try:
    with st.spinner("Loading data..."):
        data = load_analysis_data(analysis_id)

    if data.empty:
        st.error("No data available.")
        st.stop()

    # Identify columns based on analysis type
    if analysis_id == 'investment_clock':
        # For investment clock, look for specific indicator and sector returns
        indicator_cols = [c for c in data.columns if 'orders_inv' in c.lower() or 'ppi' in c.lower()]
        return_cols = [c for c in data.columns if c.endswith('_return')]
        if not indicator_cols:
            indicator_cols = ['orders_inv_ratio'] if 'orders_inv_ratio' in data.columns else [data.columns[0]]
    elif analysis_id == 'spy_retailirsa':
        indicator_cols = [c for c in data.columns if 'retail' in c.lower() and not c.endswith('_return')]
        return_cols = [c for c in data.columns if c.endswith('_return')]
        if not indicator_cols:
            indicator_cols = [c for c in data.columns if c not in ['SPY', 'regime'] and not c.endswith('_return')]
        if not return_cols and 'SPY' in data.columns:
            data['SPY_return'] = data['SPY'].pct_change()
            return_cols = ['SPY_return']
    elif analysis_id == 'spy_indpro':
        indicator_cols = [c for c in data.columns if 'indpro' in c.lower() or 'industrial' in c.lower()]
        return_cols = [c for c in data.columns if c.endswith('_return')]
        if not indicator_cols:
            indicator_cols = [c for c in data.columns if c not in ['SPY', 'regime'] and not c.endswith('_return')]
        if not return_cols and 'SPY' in data.columns:
            data['SPY_return'] = data['SPY'].pct_change()
            return_cols = ['SPY_return']
    elif analysis_id == 'xlre_orders_inv':
        indicator_cols = [c for c in data.columns if ('order' in c.lower() or 'oi' in c.lower()) and not c.endswith('_return')]
        return_cols = [c for c in data.columns if c.endswith('_return')]
        if not indicator_cols:
            indicator_cols = [c for c in data.columns if c not in ['XLRE', 'regime'] and not c.endswith('_return')]
        if not return_cols and 'XLRE' in data.columns:
            data['XLRE_return'] = data['XLRE'].pct_change()
            return_cols = ['XLRE_return']
    else:
        indicator_cols = [c for c in data.columns if not c.endswith('_return') and c != 'regime']
        return_cols = [c for c in data.columns if c.endswith('_return')]

    # Fallback if still empty
    if not indicator_cols:
        indicator_cols = [c for c in data.columns if not c.endswith('_return') and c != 'regime']
    if not return_cols:
        # Try to find a price column and compute returns
        price_cols = [c for c in data.columns if c in ['SPY', 'XLRE', 'QQQ', 'IWM']]
        if price_cols:
            price_col = price_cols[0]
            data[f'{price_col}_return'] = data[price_col].pct_change()
            return_cols = [f'{price_col}_return']

    if not indicator_cols or not return_cols:
        st.error(f"Could not identify indicator or return columns. Columns: {data.columns.tolist()}")
        st.stop()

    indicator_col = indicator_cols[0]
    return_col = return_cols[0]

    # Create derivatives
    indicator_name = indicator_col.replace('_', ' ').title()
    target_name = return_col.replace('_return', '').upper()

    # Add derivative columns if not present
    for suffix, periods in [('MoM', 1), ('QoQ', 3), ('YoY', 12)]:
        col_name = f"{indicator_col}_{suffix}"
        if col_name not in data.columns:
            data[col_name] = data[indicator_col].pct_change(periods)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    st.markdown("Correlation between indicator derivatives and target returns.")

    x_cols = [indicator_col] + [f"{indicator_col}_{s}" for s in ['MoM', 'QoQ', 'YoY']
                                if f"{indicator_col}_{s}" in data.columns]
    y_cols = [return_col]

    corr_matrix = correlation_analysis(data, x_cols, y_cols)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig_heatmap = plot_heatmap(corr_matrix, title="Correlation Heatmap")
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col2:
        st.markdown("**Interpretation:**")
        for idx in corr_matrix.index:
            val = corr_matrix.loc[idx, return_col]
            strength = "Strong" if abs(val) > 0.5 else "Moderate" if abs(val) > 0.3 else "Weak"
            direction = "positive" if val > 0 else "negative"
            st.markdown(f"- **{idx}**: {format_number(val, 3)} ({strength} {direction})")

    # Scatter plot
    st.subheader("Scatter Plot")
    col1, col2 = st.columns(2)

    with col1:
        x_choice = st.selectbox("X-axis", x_cols, index=0)
    with col2:
        color_by = st.selectbox("Color by", ["None", "regime"] +
                               [c for c in data.columns if 'regime' not in c.lower()][:3])

    color_col = None if color_by == "None" else color_by if color_by in data.columns else None

    fig_scatter = plot_scatter(
        data, x_choice, return_col,
        color_col=color_col,
        title=f"{x_choice} vs {return_col}"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Correlation stats
    corr_stats = correlation_with_pvalues(data[x_choice], data[return_col])
    stat_cols = st.columns(3)
    stat_cols[0].metric("Correlation", format_number(corr_stats['correlation'], 3))
    stat_cols[1].metric("P-value", format_number(corr_stats['pvalue'], 4))
    stat_cols[2].metric("Observations", corr_stats['n_obs'])

    # Rolling correlation
    st.subheader("Rolling Correlation")
    st.markdown(f"Rolling {rolling_window}-month correlation over time.")

    rolling_corr = rolling_correlation(
        data[indicator_col], data[return_col], window=rolling_window
    )

    fig_rolling = plot_rolling_correlation(
        rolling_corr,
        title=f"Rolling {rolling_window}-Month Correlation"
    )
    st.plotly_chart(fig_rolling, use_container_width=True)

    # Stats on rolling correlation
    if not rolling_corr.empty:
        rcorr_cols = st.columns(4)
        rcorr_cols[0].metric("Mean", format_number(rolling_corr.mean(), 3))
        rcorr_cols[1].metric("Std Dev", format_number(rolling_corr.std(), 3))
        rcorr_cols[2].metric("Min", format_number(rolling_corr.min(), 3))
        rcorr_cols[3].metric("Max", format_number(rolling_corr.max(), 3))

except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)
