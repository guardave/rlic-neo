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

    # Create derivatives if not present
    if f"{indicator_col}_MoM" not in data.columns:
        derivs = create_derivatives(data[indicator_col], indicator_col)
        data = data.join(derivs[[f"{indicator_col}_MoM"]])

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

    # Main chart - dual axis
    st.subheader("Time Series")
    fig = plot_dual_axis(
        data,
        y1_col=indicator_col,
        y2_col=return_col,
        title=f"{indicator_col} vs {return_col}",
        y1_name=indicator_col,
        y2_name="Return"
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
        st.subheader("Regime Distribution")
        regime_counts = data['regime'].value_counts()
        st.bar_chart(regime_counts)

    # Data preview
    with st.expander("📋 Raw Data Preview"):
        st.dataframe(data.tail(20), width='stretch')

except Exception as e:
    st.error(f"Error loading analysis: {e}")
    st.exception(e)
