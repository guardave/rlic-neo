"""
Lead-Lag Page - Timing analysis and Granger causality.
"""

import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.navigation import render_sidebar, render_breadcrumb, get_analysis_title
from src.dashboard.components import plot_leadlag_bars, plot_scatter, format_number
from src.dashboard.data_loader import load_analysis_data
from src.dashboard.config_db import resolve_columns
from src.dashboard.analysis_engine import (
    create_derivatives, leadlag_analysis, find_optimal_lag,
    granger_causality_test, granger_bidirectional,
    identify_deepdive_lags, lag_scatter_data, is_stationary,
    correlation_with_pvalues
)
from src.dashboard.interpretation import render_annotation

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

        # Annotation: cross-correlation interpretation
        render_annotation(analysis_id, 'leadlag.crosscorr',
                         indicator_name=indicator_col, target_name=return_col)

    # Bi-directional Granger Causality
    st.subheader("Granger Causality Test (Bi-directional)")
    st.markdown("""
    Tests whether past values of one series help predict the other,
    tested in **both directions** to classify the relationship.
    """)

    # Stationarity checks
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

    # Bi-directional Granger
    bg = granger_bidirectional(data, mom_col, return_col, max_lag=granger_max_lag)

    col_fwd, col_rev = st.columns(2)

    with col_fwd:
        st.markdown(f"**{mom_col} → {return_col}**")
        render_annotation(analysis_id, 'leadlag.granger_fwd',
                         indicator_name=mom_col, target_name=return_col)
        if not bg['forward'].empty:
            gf_display = bg['forward'].copy()
            gf_display['significant'] = gf_display['pvalue'] < 0.05
            st.dataframe(
                gf_display.style.format({'f_statistic': '{:.2f}', 'pvalue': '{:.4f}'})
                .map(lambda x: 'background-color: #d4edda' if x else '', subset=['significant']),
                width='stretch'
            )

    with col_rev:
        st.markdown(f"**{return_col} → {mom_col}**")
        render_annotation(analysis_id, 'leadlag.granger_rev',
                         indicator_name=mom_col, target_name=return_col)
        if not bg['reverse'].empty:
            gr_display = bg['reverse'].copy()
            gr_display['significant'] = gr_display['pvalue'] < 0.05
            st.dataframe(
                gr_display.style.format({'f_statistic': '{:.2f}', 'pvalue': '{:.4f}'})
                .map(lambda x: 'background-color: #d4edda' if x else '', subset=['significant']),
                width='stretch'
            )

    # Granger verdict
    render_annotation(analysis_id, 'leadlag.granger_verdict',
                     indicator_name=indicator_col, target_name=return_col)

    # Deep-Dive: Lag-Specific Scatter Verification
    st.subheader("Lag Deep-Dive Verification")
    st.markdown("Scatter plots at key lags to verify cross-correlation and Granger findings.")

    if not leadlag_results.empty:
        deepdive_lags = identify_deepdive_lags(
            leadlag_results, bg['forward'], bg['reverse'], top_n=3
        )

        if deepdive_lags:
            cols = st.columns(min(len(deepdive_lags), 3))
            for i, dd in enumerate(deepdive_lags[:3]):
                with cols[i]:
                    lag_val = dd['lag']
                    scatter_df = lag_scatter_data(data, mom_col, return_col, lag_val)
                    if not scatter_df.empty:
                        corr = correlation_with_pvalues(scatter_df['x_lagged'], scatter_df['y'])
                        r_val = corr['correlation']
                        p_val = corr['pvalue']

                        lag_label = f"Lag {lag_val}" if lag_val != 0 else "Lag 0 (contemporaneous)"
                        fig = plot_scatter(
                            scatter_df, 'x_lagged', 'y',
                            title=f"{lag_label}\nr={r_val:.4f}, p={p_val:.4f}"
                        )
                        fig.update_layout(
                            xaxis_title=f"{mom_col} (t{lag_val:+d})" if lag_val != 0 else mom_col,
                            yaxis_title=return_col,
                            height=350
                        )
                        st.plotly_chart(fig, width='stretch')
                        st.caption(f"Source: {dd['source']} | n={len(scatter_df)}")
        else:
            st.info("No significant lags identified for deep-dive analysis.")

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
