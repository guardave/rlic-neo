"""
Forecasts Page - Predictions and forward-looking analysis.
"""

import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.navigation import render_sidebar, render_breadcrumb, get_analysis_title
from src.dashboard.components import (
    plot_timeseries, render_regime_badge, format_pct, format_number
)
from src.dashboard.data_loader import load_analysis_data
from src.dashboard.analysis_engine import (
    define_regimes_direction, get_current_regime, regime_performance,
    find_optimal_lag, leadlag_analysis
)

st.set_page_config(page_title="Forecasts | RLIC", page_icon="🔮", layout="wide")

# Sidebar: Home, analysis selector, section links
analysis_id = render_sidebar(current_page="Forecasts")

# Content: Breadcrumb, then page
render_breadcrumb("Forecasts")

# Page title
st.title(f"🔮 Forecasts: {get_analysis_title()}")

st.info("⚠️ **Disclaimer**: Forecasts are based on historical patterns and should not be considered financial advice.")

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

    # Current state
    st.subheader("Current State")

    col1, col2, col3 = st.columns(3)

    current_regime = get_current_regime(data['regime'])
    with col1:
        st.markdown("**Current Regime**")
        render_regime_badge(current_regime, size='large')

    with col2:
        latest_indicator = data[indicator_col].dropna().iloc[-1]
        prev_indicator = data[indicator_col].dropna().iloc[-2]
        change = (latest_indicator - prev_indicator) / prev_indicator
        st.metric(
            f"{indicator_col} (Latest)",
            format_number(latest_indicator, 3),
            delta=format_pct(change)
        )

    with col3:
        latest_date = data.index[-1]
        st.metric("Data Through", latest_date.strftime("%Y-%m"))

    # Historical regime performance
    st.subheader("Expected Performance Based on Current Regime")

    regime_perf = regime_performance(data, 'regime', return_col)

    if not regime_perf.empty:
        current_perf = regime_perf[regime_perf['regime'] == current_regime]

        if not current_perf.empty:
            perf = current_perf.iloc[0]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Expected Monthly Return", format_pct(perf['mean_return']))
            col2.metric("Historical Volatility", format_pct(perf['std_return']))
            col3.metric("Win Rate", format_pct(perf['pct_positive']))
            col4.metric("Sample Size", f"{int(perf['n_periods'])} months")

            st.markdown(f"""
            **Interpretation**: Based on historical data, when the indicator is in a **{current_regime}** regime,
            the average monthly return has been **{format_pct(perf['mean_return'])}** with
            **{format_pct(perf['pct_positive'])}** of months showing positive returns.
            """)
        else:
            st.warning(f"No historical data for {current_regime} regime.")

    # Lead-lag based forecast
    st.subheader("Lead-Lag Signal")

    mom_col = f"{indicator_col}_MoM"
    if mom_col not in data.columns:
        data[mom_col] = data[indicator_col].pct_change(1)

    leadlag_results = leadlag_analysis(data, mom_col, return_col, max_lag=12)
    optimal = find_optimal_lag(leadlag_results)

    if optimal['optimal_lag'] > 0:
        st.markdown(f"""
        The indicator leads the target by **{optimal['optimal_lag']} month(s)**
        with a correlation of **{format_number(optimal['correlation'], 3)}**.
        """)

        # Recent indicator trend
        recent_mom = data[mom_col].dropna().iloc[-1]
        signal = "POSITIVE" if recent_mom > 0 else "NEGATIVE"
        signal_color = "green" if recent_mom > 0 else "red"

        st.markdown(f"""
        **Latest Signal**: The indicator MoM is <span style="color:{signal_color};font-weight:bold">{signal}</span>
        ({format_pct(recent_mom)}), suggesting that in {optimal['optimal_lag']} month(s),
        the target may move in the same direction.
        """, unsafe_allow_html=True)

    else:
        st.info("No significant leading relationship detected.")

    # Scenario analysis
    st.subheader("Scenario Analysis")
    st.markdown("What if the regime changes?")

    scenario_cols = st.columns(len(regime_perf))
    for i, (_, row) in enumerate(regime_perf.iterrows()):
        with scenario_cols[i]:
            regime = row['regime']
            is_current = regime == current_regime
            st.markdown(f"### {'→ ' if is_current else ''}{regime}")
            if is_current:
                st.caption("(Current)")
            st.metric("Avg Return", format_pct(row['mean_return']))
            st.metric("Sharpe", format_number(row['sharpe_ratio'], 2))

    # Recent trend
    st.subheader("Recent Trend")

    fig = plot_timeseries(
        data.iloc[-36:],  # Last 3 years
        y_cols=[indicator_col],
        regime_col='regime',
        title=f"Recent {indicator_col} with Regime Background"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Outlook summary
    st.subheader("Outlook Summary")

    with st.container(border=True):
        col1, col2 = st.columns([1, 3])

        with col1:
            render_regime_badge(current_regime, size='large')

        with col2:
            if current_regime == 'Rising':
                st.markdown("""
                📈 **Outlook**: The indicator is in a rising trend.
                Historically, this has been associated with positive returns.
                Consider maintaining or increasing exposure.
                """)
            else:
                st.markdown("""
                📉 **Outlook**: The indicator is in a falling trend.
                Historically, this has been associated with lower or negative returns.
                Consider reducing exposure or staying defensive.
                """)

    st.caption("Last updated: " + data.index[-1].strftime("%Y-%m-%d"))

except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)
