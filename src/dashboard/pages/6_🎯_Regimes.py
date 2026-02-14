"""
Regimes Page - Phase analysis and conditional performance.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.navigation import render_sidebar, render_breadcrumb, get_analysis_title
from src.dashboard.components import (
    plot_regime_timeline, plot_regime_boxplot, plot_regime_performance_bars,
    render_regime_badge, format_pct, format_number
)
from src.dashboard.data_loader import load_analysis_data
from src.dashboard.config_db import resolve_columns
from src.dashboard.analysis_engine import (
    define_regimes_direction, define_regimes_level,
    regime_performance, regime_transition_matrix, get_current_regime
)

st.set_page_config(page_title="Regimes | RLIC", page_icon="🎯", layout="wide")

# Sidebar: Home, analysis selector, section links
analysis_id = render_sidebar(current_page="Regimes")

# Content: Breadcrumb, then page
render_breadcrumb("Regimes")
st.title(f"🎯 Regimes: {get_analysis_title()}")

# Settings in sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("Regime Definition")
    regime_method = st.radio(
        "Method",
        ["Direction (Rising/Falling)", "Level (High/Low)"],
        index=0
    )
    if regime_method == "Direction (Rising/Falling)":
        short_window = st.slider("Short MA Window", 1, 6, 3)
        long_window = st.slider("Long MA Window", 3, 12, 6)
    else:
        threshold = st.selectbox("Threshold", ["median", "mean"])

try:
    with st.spinner("Loading data..."):
        data = load_analysis_data(analysis_id)

    if data.empty:
        st.error("No data available.")
        st.stop()

    # Resolve columns from config database (trading context uses lagged indicators)
    resolved = resolve_columns(analysis_id, data, context='trading')
    data = resolved['data']
    indicator_cols = resolved['indicator_cols']
    return_cols = resolved['return_cols']
    indicator_col = resolved['indicator_col']
    return_col = resolved['return_col']

    if not indicator_col or not return_col:
        st.error(f"Could not identify columns. Available: {data.columns.tolist()}")
        st.stop()

    # Dynamic lag exploration from config
    lag_config = resolved.get('lag_config', {})
    base_col = lag_config.get('base_col')
    if base_col and base_col in data.columns:
        with st.sidebar:
            st.markdown("---")
            st.subheader("Lag Exploration")
            selected_lag = st.slider(
                "Indicator Lag (months)",
                min_value=lag_config.get('min', -12),
                max_value=lag_config.get('max', 12),
                value=lag_config.get('default', 0),
                help="Positive = indicator leads target by N months"
            )
            if selected_lag != 0:
                lagged_col = f"{base_col}_lag{selected_lag}"
                data[lagged_col] = data[base_col].shift(selected_lag)
                indicator_col = lagged_col
                st.caption(f"Using {base_col} shifted by {selected_lag} months")
            else:
                st.caption(f"Using {indicator_col} (no lag)")

    # Define regime based on selected method
    if regime_method == "Direction (Rising/Falling)":
        data['regime'] = define_regimes_direction(
            data, indicator_col, short_window, long_window
        )
    else:
        data['regime'] = define_regimes_level(data, indicator_col, threshold)

    # Current regime
    current_regime = get_current_regime(data['regime'])

    # Current regime display
    st.subheader("Current Regime")
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        render_regime_badge(current_regime, size='large')
    with col2:
        latest_value = data[indicator_col].dropna().iloc[-1]
        st.metric(f"Latest {indicator_col}", format_number(latest_value, 3))
    with col3:
        regime_duration = (data['regime'] == current_regime).iloc[::-1].cumsum().iloc[-1]
        st.metric("Regime Duration", f"{regime_duration} months")

    # Regime timeline
    st.subheader("Regime Timeline")
    fig_timeline = plot_regime_timeline(data, 'regime')
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Performance by regime
    st.subheader("Performance by Regime")

    regime_perf = regime_performance(data, 'regime', return_col)

    if not regime_perf.empty:
        col1, col2 = st.columns(2)

        with col1:
            fig_returns = plot_regime_performance_bars(
                regime_perf, metric='mean_return',
                title="Mean Monthly Return by Regime"
            )
            st.plotly_chart(fig_returns, use_container_width=True)

        with col2:
            fig_sharpe = plot_regime_performance_bars(
                regime_perf, metric='sharpe_ratio',
                title="Sharpe Ratio by Regime"
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)

        # Return distribution
        st.subheader("Return Distribution by Regime")
        fig_box = plot_regime_boxplot(
            data, 'regime', return_col,
            title="Monthly Returns by Regime"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Performance table
        st.subheader("Detailed Performance Metrics")
        perf_display = regime_perf.copy()
        perf_display.columns = [c.replace('_', ' ').title() for c in perf_display.columns]

        st.dataframe(
            perf_display.style.format({
                'Mean Return': '{:.4f}',
                'Std Return': '{:.4f}',
                'Sharpe Ratio': '{:.2f}',
                'Pct Positive': '{:.1%}',
                'Min Return': '{:.4f}',
                'Max Return': '{:.4f}'
            }),
            use_container_width=True
        )

    # Transition matrix
    st.subheader("Regime Transition Probabilities")
    st.markdown("Probability of transitioning from one regime to another.")

    try:
        trans_matrix = regime_transition_matrix(data['regime'])

        if trans_matrix is not None and not trans_matrix.empty and trans_matrix.shape[0] > 0:
            # Convert to numeric and handle any non-numeric values
            trans_display = trans_matrix.copy()
            for col in trans_display.columns:
                trans_display[col] = pd.to_numeric(trans_display[col], errors='coerce').fillna(0)

            st.dataframe(
                trans_display.style.format('{:.1%}').background_gradient(cmap='Blues'),
                use_container_width=True
            )
        else:
            st.info("Not enough data to compute transition probabilities.")
    except Exception as tm_error:
        st.warning(f"Could not compute transition matrix: {tm_error}")

        # Average regime duration
        if 'regime' in data.columns:
            regime_counts = data['regime'].value_counts()
            st.subheader("Regime Distribution")
            col1, col2 = st.columns(2)

            with col1:
                st.bar_chart(regime_counts)

            with col2:
                total = regime_counts.sum()
                for regime, count in regime_counts.items():
                    pct = count / total * 100
                    st.markdown(f"**{regime}**: {count} months ({pct:.1f}%)")

except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)
