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
    elif analysis_id == 'xlp_retailirsa':
        indicator_cols = [c for c in data.columns if 'retail' in c.lower() and not c.endswith('_return')]
        return_cols = [c for c in data.columns if c.endswith('_return')]
        if not indicator_cols:
            indicator_cols = [c for c in data.columns if c not in ['XLP', 'regime'] and not c.endswith('_return')]
        if not return_cols and 'XLP' in data.columns:
            data['XLP_return'] = data['XLP'].pct_change()
            return_cols = ['XLP_return']
    elif analysis_id == 'xly_retailirsa':
        indicator_cols = [c for c in data.columns if 'retail' in c.lower() and not c.endswith('_return')]
        return_cols = [c for c in data.columns if c.endswith('_return')]
        if not indicator_cols:
            indicator_cols = [c for c in data.columns if c not in ['XLY', 'regime'] and not c.endswith('_return')]
        if not return_cols and 'XLY' in data.columns:
            data['XLY_return'] = data['XLY'].pct_change()
            return_cols = ['XLY_return']
    elif analysis_id == 'xlre_newhomesales':
        # Use lagged indicator for regime definition (lag +8)
        indicator_cols = ['NewHomeSales_YoY_Lagged'] if 'NewHomeSales_YoY_Lagged' in data.columns else ['NewHomeSales_YoY']
        return_cols = ['XLRE_Returns'] if 'XLRE_Returns' in data.columns else []
        # Use pre-computed regime if available
        if 'Regime' in data.columns and 'regime' not in data.columns:
            data['regime'] = data['Regime']
    elif analysis_id == 'xli_ism_mfg':
        indicator_cols = ['ISM_Mfg_PMI_Level_Lagged'] if 'ISM_Mfg_PMI_Level_Lagged' in data.columns else ['ISM_Mfg_PMI_Level']
        return_cols = ['XLI_Returns'] if 'XLI_Returns' in data.columns else []
        if 'Regime' in data.columns and 'regime' not in data.columns:
            data['regime'] = data['Regime']
    else:
        indicator_cols = [c for c in data.columns if not c.endswith('_return') and c != 'regime']
        return_cols = [c for c in data.columns if c.endswith('_return')]

    # Fallback if still empty
    if not indicator_cols:
        indicator_cols = [c for c in data.columns if not c.endswith('_return') and c != 'regime']
    if not return_cols:
        price_cols = [c for c in data.columns if c in ['SPY', 'XLRE', 'XLP', 'XLY', 'QQQ', 'IWM']]
        if price_cols:
            price_col = price_cols[0]
            data[f'{price_col}_return'] = data[price_col].pct_change()
            return_cols = [f'{price_col}_return']

    if not indicator_cols or not return_cols:
        st.error(f"Could not identify columns. Available: {data.columns.tolist()}")
        st.stop()

    indicator_col = indicator_cols[0]
    return_col = return_cols[0]

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
