"""
Lead-Lag Page - Timing analysis and Granger causality.
"""

import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.navigation import render_sidebar, render_breadcrumb, get_analysis_title
from src.dashboard.components import plot_leadlag_bars, format_number
from src.dashboard.data_loader import load_analysis_data
from src.dashboard.config_db import resolve_columns
from src.dashboard.analysis_engine import (
    create_derivatives, leadlag_analysis, find_optimal_lag,
    granger_causality_test, is_stationary
)

st.set_page_config(page_title="Lead-Lag | RLIC", page_icon="🔄", layout="wide")

# Sidebar: Home, analysis selector, section links
analysis_id = render_sidebar(current_page="Lead-Lag")

# Content: Breadcrumb, then page
render_breadcrumb("Lead-Lag")
st.title(f"🔄 Lead-Lag: {get_analysis_title()}")

# Settings in sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("Settings")
    # Extended range for analyses like XLRE/NewHomeSales that need 0-24 months
    max_lag = st.slider("Max Lag (months)", 6, 24, 24 if analysis_id == 'xlre_newhomesales' else 12)
    granger_max_lag = st.slider("Granger Max Lag", 2, 12, 6)

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
        st.error(f"Could not identify columns. Available: {data.columns.tolist()}")
        st.stop()

    # Create MoM if not present
    mom_col = f"{indicator_col}_MoM"
    if mom_col not in data.columns:
        data[mom_col] = data[indicator_col].pct_change(1)

    # Lead-Lag Analysis
    st.subheader("Cross-Correlation at Different Lags")
    st.markdown("""
    - **Positive lag**: Indicator leads target (predictive)
    - **Negative lag**: Target leads indicator
    - **Red bars**: Statistically significant (p < 0.05)
    """)

    leadlag_results = leadlag_analysis(data, mom_col, return_col, max_lag=max_lag)

    if not leadlag_results.empty:
        fig = plot_leadlag_bars(leadlag_results, title="Lead-Lag Cross-Correlation")
        st.plotly_chart(fig, width='stretch')

        # Optimal lag
        optimal = find_optimal_lag(leadlag_results)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Optimal Lag", f"{optimal['optimal_lag']} months")
        col2.metric("Correlation", format_number(optimal['correlation'], 3))
        col3.metric("P-value", format_number(optimal.get('pvalue', 0), 4))
        col4.metric("Observations", optimal.get('n_obs', 'N/A'))

        # Interpretation
        if optimal['optimal_lag'] > 0:
            st.success(f"✅ **{indicator_col}** leads **{return_col}** by {optimal['optimal_lag']} month(s). "
                      "This indicator may have predictive power.")
        elif optimal['optimal_lag'] < 0:
            st.info(f"ℹ️ **{return_col}** leads **{indicator_col}** by {abs(optimal['optimal_lag'])} month(s). "
                   "The target moves before the indicator.")
        else:
            st.warning("⚠️ No clear lead-lag relationship detected at lag 0.")

    # Granger Causality
    st.subheader("Granger Causality Test")
    st.markdown("""
    Tests whether past values of the indicator help predict future target values
    beyond what past target values alone can predict.
    """)

    # Check stationarity
    col1, col2 = st.columns(2)
    with col1:
        ind_stationary = is_stationary(data[mom_col])
        if ind_stationary:
            st.success(f"✅ {mom_col} is stationary (ADF test p < 0.05)")
        else:
            st.warning(f"⚠️ {mom_col} may not be stationary")

    with col2:
        ret_stationary = is_stationary(data[return_col])
        if ret_stationary:
            st.success(f"✅ {return_col} is stationary")
        else:
            st.warning(f"⚠️ {return_col} may not be stationary")

    granger_results = granger_causality_test(
        data, mom_col, return_col, max_lag=granger_max_lag
    )

    if not granger_results.empty:
        st.markdown(f"**H₀**: {mom_col} does NOT Granger-cause {return_col}")
        st.markdown(f"**H₁**: {mom_col} DOES Granger-cause {return_col}")

        # Display results
        granger_display = granger_results.copy()
        granger_display['significant'] = granger_display['pvalue'] < 0.05

        st.dataframe(
            granger_display.style.format({
                'f_statistic': '{:.2f}',
                'pvalue': '{:.4f}'
            }).map(
                lambda x: 'background-color: #d4edda' if x else '',
                subset=['significant']
            ),
            width='stretch'
        )

        # Summary
        sig_lags = granger_display[granger_display['significant']]['lag'].tolist()
        if sig_lags:
            st.success(f"✅ Granger causality significant at lag(s): {sig_lags}")
        else:
            st.info("ℹ️ No significant Granger causality detected at tested lags.")

    else:
        st.warning("Could not perform Granger causality test. Check data availability.")

    # Lead-lag results table
    with st.expander("📋 Full Lead-Lag Results"):
        if not leadlag_results.empty:
            st.dataframe(
                leadlag_results.style.format({
                    'correlation': '{:.4f}',
                    'pvalue': '{:.4f}'
                }),
                width='stretch'
            )

except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)
