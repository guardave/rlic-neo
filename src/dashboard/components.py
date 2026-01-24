"""
Reusable Streamlit Components for RLIC Dashboard.

All chart components use Plotly for interactivity (zoom, pan, hover).
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union

# Color schemes
REGIME_COLORS = {
    'Recovery': '#2ecc71',      # Green
    'Overheat': '#e74c3c',      # Red
    'Stagflation': '#9b59b6',   # Purple
    'Reflation': '#3498db',     # Blue
    'Rising': '#27ae60',        # Dark green
    'Falling': '#c0392b',       # Dark red
    'High': '#2980b9',          # Blue
    'Low': '#7f8c8d',           # Gray
    'Unknown': '#bdc3c7'        # Light gray
}

CHART_TEMPLATE = "plotly_white"


# =============================================================================
# Time Series Charts
# =============================================================================

def plot_timeseries(data: pd.DataFrame,
                   y_cols: List[str],
                   regime_col: Optional[str] = None,
                   title: str = "Time Series",
                   height: int = 400) -> go.Figure:
    """
    Create interactive time series chart with optional regime background.

    Args:
        data: DataFrame with datetime index
        y_cols: List of column names to plot
        regime_col: Optional column for regime coloring
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Add regime background if provided
    if regime_col and regime_col in data.columns:
        regimes = data[regime_col].dropna()
        for regime in regimes.unique():
            if pd.isna(regime):
                continue
            mask = data[regime_col] == regime
            periods = _get_contiguous_periods(mask)
            color = REGIME_COLORS.get(regime, '#bdc3c7')

            for start, end in periods:
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor=color, opacity=0.15,
                    layer="below", line_width=0
                )

    # Add traces for each y column
    colors = px.colors.qualitative.Set2
    for i, col in enumerate(y_cols):
        if col in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                name=col,
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f"{col}: %{{y:.4f}}<br>Date: %{{x}}<extra></extra>"
            ))

    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=height,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(title="Date", rangeslider=dict(visible=False)),
        yaxis=dict(title="Value")
    )

    return fig


def plot_dual_axis(data: pd.DataFrame,
                  y1_col: str,
                  y2_col: str,
                  title: str = "Dual Axis Chart",
                  y1_name: str = None,
                  y2_name: str = None,
                  height: int = 400) -> go.Figure:
    """
    Create dual-axis time series chart.

    Args:
        data: DataFrame with datetime index
        y1_col: Column for left y-axis
        y2_col: Column for right y-axis
        title: Chart title
        y1_name, y2_name: Display names for axes
        height: Chart height

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    y1_name = y1_name or y1_col
    y2_name = y2_name or y2_col

    fig.add_trace(
        go.Scatter(x=data.index, y=data[y1_col], name=y1_name,
                  line=dict(color='#3498db', width=2)),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=data.index, y=data[y2_col], name=y2_name,
                  line=dict(color='#e74c3c', width=2)),
        secondary_y=True
    )

    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=height,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    fig.update_yaxes(title_text=y1_name, secondary_y=False)
    fig.update_yaxes(title_text=y2_name, secondary_y=True)

    return fig


# =============================================================================
# Correlation Charts
# =============================================================================

def plot_heatmap(corr_matrix: pd.DataFrame,
                title: str = "Correlation Matrix",
                height: int = 500) -> go.Figure:
    """
    Create correlation heatmap.

    Args:
        corr_matrix: Correlation matrix DataFrame
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=height,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed")
    )

    return fig


def plot_scatter(data: pd.DataFrame,
                x_col: str,
                y_col: str,
                color_col: Optional[str] = None,
                title: str = "Scatter Plot",
                height: int = 400) -> go.Figure:
    """
    Create scatter plot with optional color coding.

    Args:
        data: DataFrame
        x_col: Column for x-axis
        y_col: Column for y-axis
        color_col: Optional column for color coding
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    if color_col and color_col in data.columns:
        fig = px.scatter(data, x=x_col, y=y_col, color=color_col,
                        color_discrete_map=REGIME_COLORS,
                        title=title, height=height,
                        template=CHART_TEMPLATE)
    else:
        fig = px.scatter(data, x=x_col, y=y_col,
                        title=title, height=height,
                        template=CHART_TEMPLATE)

    # Add regression line
    valid = data[[x_col, y_col]].dropna()
    if len(valid) > 2:
        z = np.polyfit(valid[x_col], valid[y_col], 1)
        p = np.poly1d(z)
        x_range = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)

        fig.add_trace(go.Scatter(
            x=x_range, y=p(x_range),
            mode='lines',
            name='Trend',
            line=dict(color='gray', dash='dash', width=1)
        ))

    return fig


def plot_rolling_correlation(data: pd.Series,
                            title: str = "Rolling Correlation",
                            height: int = 300) -> go.Figure:
    """
    Plot rolling correlation over time.

    Args:
        data: Series of rolling correlations
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#3498db', width=2),
        hovertemplate="Date: %{x}<br>Correlation: %{y:.3f}<extra></extra>"
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=height,
        yaxis=dict(title="Correlation", range=[-1, 1]),
        xaxis=dict(title="Date")
    )

    return fig


# =============================================================================
# Lead-Lag Charts
# =============================================================================

def plot_leadlag_bars(leadlag_results: pd.DataFrame,
                     title: str = "Lead-Lag Cross-Correlation",
                     height: int = 400) -> go.Figure:
    """
    Create bar chart of cross-correlations at different lags.

    Args:
        leadlag_results: DataFrame with 'lag' and 'correlation' columns
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    df = leadlag_results.copy()

    # Color by significance
    colors = ['#e74c3c' if p < 0.05 else '#bdc3c7'
              for p in df.get('pvalue', [1] * len(df))]

    fig = go.Figure(data=go.Bar(
        x=df['lag'],
        y=df['correlation'],
        marker_color=colors,
        hovertemplate="Lag: %{x}<br>Correlation: %{y:.3f}<extra></extra>"
    ))

    # Find optimal lag
    if not df.empty:
        idx = df['correlation'].abs().idxmax()
        opt_lag = df.loc[idx, 'lag']
        opt_corr = df.loc[idx, 'correlation']

        fig.add_annotation(
            x=opt_lag, y=opt_corr,
            text=f"Optimal: lag={opt_lag}",
            showarrow=True,
            arrowhead=2
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=height,
        xaxis=dict(title="Lag (months)", dtick=1),
        yaxis=dict(title="Correlation")
    )

    return fig


# =============================================================================
# Regime Charts
# =============================================================================

def plot_regime_timeline(data: pd.DataFrame,
                        regime_col: str,
                        title: str = "Regime Timeline",
                        height: int = 200) -> go.Figure:
    """
    Create regime timeline visualization.

    Args:
        data: DataFrame with datetime index and regime column
        regime_col: Column with regime labels
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    regimes = data[regime_col].dropna()
    for regime in regimes.unique():
        if pd.isna(regime):
            continue
        mask = data[regime_col] == regime
        periods = _get_contiguous_periods(mask)
        color = REGIME_COLORS.get(regime, '#bdc3c7')

        for start, end in periods:
            fig.add_trace(go.Scatter(
                x=[start, end, end, start, start],
                y=[0, 0, 1, 1, 0],
                fill='toself',
                fillcolor=color,
                line=dict(width=0),
                mode='lines',
                name=regime,
                showlegend=False,
                hoverinfo='skip'
            ))

    # Add legend entries
    for regime in regimes.unique():
        if pd.isna(regime):
            continue
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=REGIME_COLORS.get(regime, '#bdc3c7')),
            name=regime
        ))

    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(title="Date"),
        yaxis=dict(visible=False, range=[0, 1])
    )

    return fig


def plot_regime_boxplot(data: pd.DataFrame,
                       regime_col: str,
                       value_col: str,
                       title: str = "Returns by Regime",
                       height: int = 400) -> go.Figure:
    """
    Create box plot of values by regime.

    Args:
        data: DataFrame with regime and value columns
        regime_col: Column with regime labels
        value_col: Column with values to plot
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    for regime in data[regime_col].dropna().unique():
        values = data.loc[data[regime_col] == regime, value_col]
        color = REGIME_COLORS.get(regime, '#bdc3c7')

        fig.add_trace(go.Box(
            y=values,
            name=regime,
            marker_color=color,
            boxmean=True
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=height,
        yaxis=dict(title=value_col, tickformat='.2%'),
        showlegend=False
    )

    return fig


def plot_regime_performance_bars(regime_perf: pd.DataFrame,
                                metric: str = 'mean_return',
                                title: str = "Regime Performance",
                                height: int = 350) -> go.Figure:
    """
    Create bar chart of regime performance metrics.

    Args:
        regime_perf: DataFrame with regime performance stats
        metric: Column to plot ('mean_return', 'sharpe_ratio', etc.)
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    df = regime_perf.copy()
    colors = [REGIME_COLORS.get(r, '#bdc3c7') for r in df['regime']]

    fig = go.Figure(data=go.Bar(
        x=df['regime'],
        y=df[metric],
        marker_color=colors,
        text=df[metric].round(3),
        textposition='outside'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=height,
        xaxis=dict(title="Regime"),
        yaxis=dict(title=metric.replace('_', ' ').title())
    )

    return fig


# =============================================================================
# Backtest Charts
# =============================================================================

def plot_equity_curve(data: pd.DataFrame,
                     strategy_col: str = 'strategy_cumulative',
                     benchmark_col: str = 'benchmark_cumulative',
                     title: str = "Equity Curve",
                     height: int = 400) -> go.Figure:
    """
    Create equity curve comparison chart.

    Args:
        data: DataFrame with cumulative return columns
        strategy_col: Column with strategy cumulative returns
        benchmark_col: Column with benchmark cumulative returns
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[strategy_col],
        name='Strategy',
        mode='lines',
        line=dict(color='#27ae60', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[benchmark_col],
        name='Benchmark',
        mode='lines',
        line=dict(color='#3498db', width=2)
    ))

    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=height,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Cumulative Return", tickformat='.1%')
    )

    return fig


def plot_drawdown(data: pd.DataFrame,
                 returns_col: str,
                 title: str = "Drawdown",
                 height: int = 250) -> go.Figure:
    """
    Create drawdown chart.

    Args:
        data: DataFrame with returns column
        returns_col: Column with returns
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    cumulative = (1 + data[returns_col]).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = cumulative / rolling_max - 1

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#e74c3c', width=1),
        fillcolor='rgba(231, 76, 60, 0.3)'
    ))

    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=height,
        xaxis=dict(title="Date"),
        yaxis=dict(title="Drawdown", tickformat='.1%')
    )

    return fig


# =============================================================================
# KPI and Card Components
# =============================================================================

def render_kpi_row(metrics: List[Dict], columns: int = 4) -> None:
    """
    Render a row of KPI metrics using st.metric.

    Args:
        metrics: List of dicts with 'label', 'value', optional 'delta'
        columns: Number of columns
    """
    cols = st.columns(columns)
    for i, metric in enumerate(metrics[:columns]):
        with cols[i]:
            delta = metric.get('delta')
            delta_color = metric.get('delta_color', 'normal')
            st.metric(
                label=metric['label'],
                value=metric['value'],
                delta=delta,
                delta_color=delta_color
            )


def render_regime_badge(regime: str, size: str = "large") -> None:
    """
    Render a colored regime badge.

    Args:
        regime: Regime label
        size: 'small', 'medium', or 'large'
    """
    color = REGIME_COLORS.get(regime, '#bdc3c7')
    font_size = {'small': '12px', 'medium': '16px', 'large': '20px'}[size]
    padding = {'small': '4px 8px', 'medium': '6px 12px', 'large': '8px 16px'}[size]

    st.markdown(f"""
        <span style="
            background-color: {color};
            color: white;
            padding: {padding};
            border-radius: 4px;
            font-size: {font_size};
            font-weight: bold;
        ">{regime}</span>
    """, unsafe_allow_html=True)


def render_analysis_card(metadata: Dict) -> None:
    """
    Render an analysis summary card for the catalog.

    Args:
        metadata: Dict with analysis metadata:
            - id: Analysis ID
            - title: Display title
            - regime: Current regime
            - metrics: List of {label, value} dicts
            - finding: Key finding text
            - updated: Last updated date
            - sparkline_data: Optional series for sparkline
    """
    with st.container():
        st.markdown(f"### {metadata['title']}")

        # Regime badge
        regime = metadata.get('regime', 'Unknown')
        render_regime_badge(regime, size='medium')

        # Sparkline if provided
        if 'sparkline_data' in metadata and metadata['sparkline_data'] is not None:
            sparkline = metadata['sparkline_data']
            fig = go.Figure(data=go.Scatter(
                y=sparkline.values[-24:],  # Last 24 periods
                mode='lines',
                line=dict(color='#3498db', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.2)'
            ))
            fig.update_layout(
                height=60,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Metrics
        metrics = metadata.get('metrics', [])
        if metrics:
            cols = st.columns(len(metrics))
            for i, m in enumerate(metrics):
                cols[i].metric(m['label'], m['value'])

        # Key finding
        if 'finding' in metadata:
            st.caption(f"**Key Finding:** {metadata['finding']}")

        # Updated date
        if 'updated' in metadata:
            st.caption(f"_Updated: {metadata['updated']}_")

        # Link to full analysis
        if 'id' in metadata:
            st.page_link(f"pages/2_📊_Overview.py",
                        label="View Full Analysis →",
                        icon="📊")


# =============================================================================
# Helper Functions
# =============================================================================

def _get_contiguous_periods(mask: pd.Series) -> List[tuple]:
    """
    Get contiguous True periods from a boolean mask.

    Returns list of (start, end) tuples.
    """
    periods = []
    in_period = False
    start = None

    for idx, val in mask.items():
        if val and not in_period:
            in_period = True
            start = idx
        elif not val and in_period:
            in_period = False
            periods.append((start, idx))

    if in_period:
        periods.append((start, mask.index[-1]))

    return periods


def format_pct(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with specified decimals."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"
