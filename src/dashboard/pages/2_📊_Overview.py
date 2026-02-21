"""
Overview Page - Key metrics and main charts for selected analysis.
"""

import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.navigation import render_sidebar, render_breadcrumb, get_analysis_title
from src.dashboard.components import (
    plot_timeseries, plot_dual_axis, render_kpi_row,
    render_regime_badge, format_pct, format_number, plot_regime_timeline
)
from src.dashboard.data_loader import load_analysis_data
from src.dashboard.config_db import resolve_columns
from src.dashboard.analysis_engine import (
    create_derivatives, get_current_regime, define_regimes_direction,
    regime_performance
)
from src.dashboard.interpretation import render_annotation
from src.dashboard.components import plot_regime_boxplot

st.set_page_config(page_title="Overview | RLIC", page_icon="📊", layout="wide")

# Sidebar: Home, analysis selector, section links
analysis_id = render_sidebar(current_page="Overview")

# Content: Breadcrumb, then page
render_breadcrumb("Overview")
st.title(f"📊 Overview: {get_analysis_title()}")

# Load data
try:
    with st.spinner("Loading data..."):
        data = load_analysis_data(analysis_id)

    if data.empty:
        st.error("No data available for this analysis.")
        st.stop()

    # Resolve columns from config database
    resolved = resolve_columns(analysis_id, data)
    data = resolved['data']
    indicator_cols = resolved['indicator_cols']
    return_cols = resolved['return_cols']
    indicator_col = resolved['indicator_col']
    return_col = resolved['return_col']

    if not indicator_col or not return_col:
        st.warning("Data structure not as expected.")
        st.write(f"Columns: {data.columns.tolist()}")
        st.dataframe(data.head())
        st.stop()

    # Summary annotation (auto-generated or override)
    render_annotation(analysis_id, 'overview.summary',
                     indicator_name=indicator_col, target_name=return_col)

    # Derive stem name (strip _Level suffix if present) for derivative column naming
    def _stem(col_name):
        return col_name[:-6] if col_name.endswith('_Level') else col_name

    # Ensure all derivatives exist for indicator (Level, MoM, QoQ, YoY)
    ind_stem = _stem(indicator_col)
    derivs = create_derivatives(data[indicator_col], ind_stem)
    for c in derivs.columns:
        if c not in data.columns:
            data[c] = derivs[c]

    # Define regime if not present
    if 'regime' not in data.columns:
        data['regime'] = define_regimes_direction(data, indicator_col)

    # Current regime
    current_regime = get_current_regime(data['regime'])

    # KPIs
    st.subheader("Key Metrics")
    latest_indicator = data[indicator_col].dropna().iloc[-1]
    latest_return = data[return_col].dropna().iloc[-1]
    avg_return = data[return_col].mean()

    # Calculate regime performance
    regime_perf = regime_performance(data, 'regime', return_col)

    metrics = [
        {'label': 'Current Regime', 'value': current_regime},
        {'label': f'{indicator_col} (Latest)', 'value': format_number(latest_indicator, 3)},
        {'label': f'{return_col.replace("_return", "")} Return (Latest)',
         'value': format_pct(latest_return)},
        {'label': 'Avg Monthly Return', 'value': format_pct(avg_return)}
    ]
    render_kpi_row(metrics, columns=4)

    # Main chart - dual axis with derivative selectors
    st.subheader("Time Series")
    price_col = resolved.get('price_col')
    chart_y2_base = price_col or return_col
    tgt_stem = _stem(chart_y2_base)

    # Ensure all derivatives exist for target (Level, MoM, QoQ, YoY)
    derivs = create_derivatives(data[chart_y2_base], tgt_stem)
    for c in derivs.columns:
        if c not in data.columns:
            data[c] = derivs[c]

    # Derivative selector dropdowns
    DERIV_OPTIONS = ["Level", "MoM", "QoQ", "YoY"]
    sel1, sel2 = st.columns(2)
    with sel1:
        y1_deriv = st.selectbox("Indicator", DERIV_OPTIONS, index=0)
    with sel2:
        y2_deriv = st.selectbox("Target", DERIV_OPTIONS, index=0)

    # Build column names using stem (e.g., CassShip_YoY, SPY_Level)
    y1_col = f"{ind_stem}_{y1_deriv}"
    y2_col = f"{tgt_stem}_{y2_deriv}"

    # Fallback to base column if derivative not found
    if y1_col not in data.columns:
        y1_col = indicator_col
    if y2_col not in data.columns:
        y2_col = chart_y2_base

    fig = plot_dual_axis(
        data,
        y1_col=y1_col,
        y2_col=y2_col,
        title=f"{ind_stem} ({y1_deriv}) vs {tgt_stem} ({y2_deriv})",
        y1_name=f"{ind_stem} {y1_deriv}",
        y2_name=f"{tgt_stem} {y2_deriv}"
    )
    st.plotly_chart(fig, width='stretch')

    # Regime timeline
    st.subheader("Regime History")
    fig_regime = plot_regime_timeline(data, 'regime', title="Regime Over Time")
    st.plotly_chart(fig_regime, width='stretch')

    # Regime performance summary
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Regime Performance")
        if not regime_perf.empty:
            st.dataframe(
                regime_perf.style.format({
                    'mean_return': '{:.4f}',
                    'std_return': '{:.4f}',
                    'sharpe_ratio': '{:.2f}',
                    'pct_positive': '{:.1%}'
                }),
                width='stretch'
            )

    with col2:
        st.subheader("Returns by Regime")
        if 'regime' in data.columns:
            fig_box = plot_regime_boxplot(
                data, 'regime', return_col,
                title="Return Distribution by Regime"
            )
            st.plotly_chart(fig_box, width='stretch')

    # Regime interpretation
    render_annotation(analysis_id, 'regime.performance',
                     indicator_name=indicator_col, target_name=return_col)

    # Data preview
    with st.expander("📋 Raw Data Preview"):
        st.dataframe(data.tail(20), width='stretch')

except Exception as e:
    st.error(f"Error loading analysis: {e}")
    st.exception(e)
