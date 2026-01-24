"""
Backtests Page - Strategy testing and performance metrics.
"""

import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.navigation import render_sidebar, render_breadcrumb, get_analysis_title
from src.dashboard.components import (
    plot_equity_curve, plot_drawdown, render_kpi_row,
    format_pct, format_number
)
from src.dashboard.data_loader import load_analysis_data
from src.dashboard.analysis_engine import (
    define_regimes_direction, simple_backtest, regime_conditional_backtest,
    backtest_metrics
)

st.set_page_config(page_title="Backtests | RLIC", page_icon="💰", layout="wide")

# Sidebar: Home button, analysis selector
analysis_id = render_sidebar()

# Content: Breadcrumb, then page
render_breadcrumb("Backtests")
st.title(f"💰 Backtests: {get_analysis_title()}")

# Settings in sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("Backtest Settings")
    signal_lag = st.slider("Signal Lag (months)", 0, 3, 1,
                          help="Lag between signal and trade execution to avoid look-ahead bias")
    strategy_type = st.radio(
        "Strategy Type",
        ["Regime-Based", "Signal-Based"],
        index=0
    )

    if strategy_type == "Regime-Based":
        st.markdown("**Long in regime:**")
        long_rising = st.checkbox("Rising", value=True)
        long_falling = st.checkbox("Falling", value=False)

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
        st.error(f"Could not identify columns. Available: {data.columns.tolist()}")
        st.stop()

    indicator_col = indicator_cols[0]
    return_col = return_cols[0]

    # Define regime
    data['regime'] = define_regimes_direction(data, indicator_col)

    # Run backtest
    if strategy_type == "Regime-Based":
        long_regimes = []
        if long_rising:
            long_regimes.append('Rising')
        if long_falling:
            long_regimes.append('Falling')

        if not long_regimes:
            st.warning("Select at least one regime to go long.")
            st.stop()

        results = regime_conditional_backtest(
            data, 'regime', return_col,
            long_regimes=long_regimes,
            lag=signal_lag
        )
        strategy_desc = f"Long in {', '.join(long_regimes)} regime(s)"

    else:
        # Signal-based: long when MoM is positive
        mom_col = f"{indicator_col}_MoM"
        if mom_col not in data.columns:
            data[mom_col] = data[indicator_col].pct_change(1)

        results = simple_backtest(data, mom_col, return_col, lag=signal_lag)
        strategy_desc = f"Long when {mom_col} > 0"

    # Calculate metrics
    strategy_metrics = backtest_metrics(results['strategy_return'])
    benchmark_metrics = backtest_metrics(results['benchmark_return'])

    # Display strategy description
    st.info(f"**Strategy**: {strategy_desc} | **Signal Lag**: {signal_lag} month(s)")

    # KPI comparison
    st.subheader("Performance Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Strategy")
        metrics = [
            {'label': 'Total Return', 'value': format_pct(strategy_metrics.get('total_return', 0))},
            {'label': 'Ann. Return', 'value': format_pct(strategy_metrics.get('annualized_return', 0))},
            {'label': 'Sharpe Ratio', 'value': format_number(strategy_metrics.get('sharpe_ratio', 0), 2)},
            {'label': 'Max Drawdown', 'value': format_pct(strategy_metrics.get('max_drawdown', 0))}
        ]
        render_kpi_row(metrics, columns=4)

    with col2:
        st.markdown("### 📊 Benchmark (Buy & Hold)")
        metrics = [
            {'label': 'Total Return', 'value': format_pct(benchmark_metrics.get('total_return', 0))},
            {'label': 'Ann. Return', 'value': format_pct(benchmark_metrics.get('annualized_return', 0))},
            {'label': 'Sharpe Ratio', 'value': format_number(benchmark_metrics.get('sharpe_ratio', 0), 2)},
            {'label': 'Max Drawdown', 'value': format_pct(benchmark_metrics.get('max_drawdown', 0))}
        ]
        render_kpi_row(metrics, columns=4)

    # Equity curve
    st.subheader("Equity Curve")
    fig_equity = plot_equity_curve(
        results,
        strategy_col='strategy_cumulative',
        benchmark_col='benchmark_cumulative',
        title="Cumulative Returns: Strategy vs Benchmark"
    )
    st.plotly_chart(fig_equity, use_container_width=True)

    # Drawdown
    st.subheader("Drawdown")
    col1, col2 = st.columns(2)

    with col1:
        fig_dd_strat = plot_drawdown(results, 'strategy_return', title="Strategy Drawdown")
        st.plotly_chart(fig_dd_strat, use_container_width=True)

    with col2:
        fig_dd_bench = plot_drawdown(results, 'benchmark_return', title="Benchmark Drawdown")
        st.plotly_chart(fig_dd_bench, use_container_width=True)

    # Position analysis
    st.subheader("Position Analysis")
    col1, col2, col3 = st.columns(3)

    if 'position' in results.columns:
        position_counts = results['position'].value_counts()
        total_periods = len(results['position'].dropna())

        long_pct = position_counts.get(1, 0) / total_periods * 100 if total_periods > 0 else 0
        flat_pct = position_counts.get(0, 0) / total_periods * 100 if total_periods > 0 else 0

        col1.metric("Time in Market", f"{long_pct:.1f}%")
        col2.metric("Time Out", f"{flat_pct:.1f}%")
        col3.metric("Win Rate", format_pct(strategy_metrics.get('win_rate', 0)))

    # Detailed metrics table
    with st.expander("📋 Detailed Metrics"):
        import pandas as pd
        comparison = pd.DataFrame({
            'Metric': ['Total Return', 'Annualized Return', 'Annualized Volatility',
                      'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Periods'],
            'Strategy': [
                format_pct(strategy_metrics.get('total_return', 0)),
                format_pct(strategy_metrics.get('annualized_return', 0)),
                format_pct(strategy_metrics.get('annualized_volatility', 0)),
                format_number(strategy_metrics.get('sharpe_ratio', 0), 2),
                format_pct(strategy_metrics.get('max_drawdown', 0)),
                format_pct(strategy_metrics.get('win_rate', 0)),
                strategy_metrics.get('n_periods', 0)
            ],
            'Benchmark': [
                format_pct(benchmark_metrics.get('total_return', 0)),
                format_pct(benchmark_metrics.get('annualized_return', 0)),
                format_pct(benchmark_metrics.get('annualized_volatility', 0)),
                format_number(benchmark_metrics.get('sharpe_ratio', 0), 2),
                format_pct(benchmark_metrics.get('max_drawdown', 0)),
                format_pct(benchmark_metrics.get('win_rate', 0)),
                benchmark_metrics.get('n_periods', 0)
            ]
        })
        st.dataframe(comparison, use_container_width=True)

    # Raw results
    with st.expander("📋 Raw Backtest Data"):
        st.dataframe(results.tail(24), use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)
