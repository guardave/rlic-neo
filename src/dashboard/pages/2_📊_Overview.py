"""
Overview Page - Key metrics and main charts for selected analysis.
"""

import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.navigation import render_top_bar, render_sidebar, get_analysis_title
from src.dashboard.components import (
    plot_timeseries, plot_dual_axis, render_kpi_row,
    render_regime_badge, format_pct, format_number, plot_regime_timeline
)
from src.dashboard.data_loader import load_analysis_data
from src.dashboard.analysis_engine import (
    create_derivatives, get_current_regime, define_regimes_direction,
    regime_performance
)

st.set_page_config(page_title="Overview | RLIC", page_icon="📊", layout="wide")

# Top bar: Home | Analysis Selector | Breadcrumb
analysis_id = render_top_bar("Overview")

# Sidebar with focus analysis title
render_sidebar()

# Page title
st.title(f"📊 Overview: {get_analysis_title()}")

# Load data
try:
    with st.spinner("Loading data..."):
        data = load_analysis_data(analysis_id)

    if data.empty:
        st.error("No data available for this analysis.")
        st.stop()

    # Identify columns based on analysis type
    if analysis_id == 'investment_clock':
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
        price_cols = [c for c in data.columns if c in ['SPY', 'XLRE', 'QQQ', 'IWM']]
        if price_cols:
            price_col = price_cols[0]
            data[f'{price_col}_return'] = data[price_col].pct_change()
            return_cols = [f'{price_col}_return']

    if not indicator_cols or not return_cols:
        st.warning("Data structure not as expected.")
        st.write(f"Columns: {data.columns.tolist()}")
        st.dataframe(data.head())
        st.stop()

    indicator_col = indicator_cols[0]
    return_col = return_cols[0]

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
    st.plotly_chart(fig, use_container_width=True)

    # Regime timeline
    st.subheader("Regime History")
    fig_regime = plot_regime_timeline(data, 'regime', title="Regime Over Time")
    st.plotly_chart(fig_regime, use_container_width=True)

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
                use_container_width=True
            )

    with col2:
        st.subheader("Regime Distribution")
        regime_counts = data['regime'].value_counts()
        st.bar_chart(regime_counts)

    # Data preview
    with st.expander("📋 Raw Data Preview"):
        st.dataframe(data.tail(20), use_container_width=True)

except Exception as e:
    st.error(f"Error loading analysis: {e}")
    st.exception(e)
