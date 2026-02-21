"""
Correlation Page - Relationship analysis between indicator and target.
"""

import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.navigation import render_sidebar, render_breadcrumb, get_analysis_title
from src.dashboard.components import (
    plot_heatmap, plot_scatter, plot_rolling_correlation, format_number
)
from src.dashboard.data_loader import load_analysis_data
from src.dashboard.config_db import resolve_columns
from src.dashboard.analysis_engine import (
    create_derivatives, correlation_analysis, correlation_with_pvalues,
    rolling_correlation
)
from src.dashboard.interpretation import render_annotation

st.set_page_config(page_title="Correlation | RLIC", page_icon="📈", layout="wide")

# Sidebar: Home, analysis selector, section links
analysis_id = render_sidebar(current_page="Correlation")

# Content: Breadcrumb, then page
render_breadcrumb("Correlation")
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

    # Resolve columns from config database
    resolved = resolve_columns(analysis_id, data)
    data = resolved['data']
    indicator_cols = resolved['indicator_cols']
    return_cols = resolved['return_cols']
    indicator_col = resolved['indicator_col']
    return_col = resolved['return_col']

    if not indicator_col or not return_col:
        st.error(f"Could not identify indicator or return columns. Columns: {data.columns.tolist()}")
        st.stop()

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
        st.plotly_chart(fig_heatmap, width='stretch')

    with col2:
        st.markdown("**Interpretation:**")
        for idx in corr_matrix.index:
            val = corr_matrix.loc[idx, return_col]
            strength = "Strong" if abs(val) > 0.5 else "Moderate" if abs(val) > 0.3 else "Weak"
            direction = "positive" if val > 0 else "negative"
            st.markdown(f"- **{idx}**: {format_number(val, 3)} ({strength} {direction})")

    # Annotation: level correlation
    render_annotation(analysis_id, 'correlation.level',
                     indicator_name=indicator_col, target_name=return_col)

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
    st.plotly_chart(fig_scatter, width='stretch')

    # Correlation stats
    corr_stats = correlation_with_pvalues(data[x_choice], data[return_col])
    stat_cols = st.columns(3)
    stat_cols[0].metric("Correlation", format_number(corr_stats['correlation'], 3))
    stat_cols[1].metric("P-value", format_number(corr_stats['pvalue'], 4))
    stat_cols[2].metric("Observations", corr_stats['n_obs'])

    # Annotation: change correlation
    render_annotation(analysis_id, 'correlation.change',
                     indicator_name=indicator_col, target_name=return_col)

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
    st.plotly_chart(fig_rolling, width='stretch')

    # Stats on rolling correlation
    if not rolling_corr.empty:
        rcorr_cols = st.columns(4)
        rcorr_cols[0].metric("Mean", format_number(rolling_corr.mean(), 3))
        rcorr_cols[1].metric("Std Dev", format_number(rolling_corr.std(), 3))
        rcorr_cols[2].metric("Min", format_number(rolling_corr.min(), 3))
        rcorr_cols[3].metric("Max", format_number(rolling_corr.max(), 3))

    # Annotation: rolling correlation
    render_annotation(analysis_id, 'correlation.rolling',
                     indicator_name=indicator_col, target_name=return_col)

except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)
