# Dashboard Component Specifications

**Version:** 1.0
**Date:** 2026-01-24
**Author:** RA Cheryl
**Related:** [Unified Analysis SOP](./unified_analysis_sop.md)

---

## 1. Architecture Overview

### 1.1 Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend Framework** | Plotly Dash | Python-native reactive web framework |
| **Component Library** | Dash Bootstrap Components | Responsive layout, cards, navigation |
| **Charting** | Plotly.js | Interactive visualizations |
| **Data Layer** | Pandas + Parquet | Efficient data loading |
| **State Management** | Dash Callbacks | Reactive updates |
| **Deployment** | Docker + Gunicorn | Production-ready serving |

### 1.2 File Structure

```
src/dashboard/
├── app.py                    # Main application entry
├── layouts/
│   ├── __init__.py
│   ├── overview.py           # Overview page layout
│   ├── qualitative.py        # Qualitative analysis page
│   ├── correlation.py        # Correlation analysis page
│   ├── leadlag.py            # Lead-lag analysis page
│   ├── regimes.py            # Regime analysis page
│   ├── backtests.py          # Backtesting page
│   └── forecasts.py          # Forecasts page
├── components/
│   ├── __init__.py
│   ├── charts.py             # Reusable chart components
│   ├── cards.py              # KPI and data cards
│   ├── tables.py             # Interactive tables
│   └── controls.py           # Dropdowns, sliders, buttons
├── callbacks/
│   ├── __init__.py
│   ├── chart_callbacks.py    # Chart interactivity
│   ├── filter_callbacks.py   # Filter and control callbacks
│   └── export_callbacks.py   # Download functionality
├── utils/
│   ├── __init__.py
│   ├── data_loader.py        # Data loading utilities
│   ├── calculations.py       # Metrics calculations
│   └── formatters.py         # Display formatting
└── assets/
    ├── style.css             # Custom styles
    └── favicon.ico           # Browser icon
```

---

## 2. Core Components

### 2.1 Interactive Time Series Chart

**Component ID:** `interactive-timeseries`

**Purpose:** Display time series data with full interactivity (zoom, pan, crosshair, hover cards).

**Props:**
| Prop | Type | Required | Description |
|------|------|----------|-------------|
| `data` | DataFrame | Yes | Time series data with DatetimeIndex |
| `y_columns` | List[str] | Yes | Columns to plot |
| `title` | str | No | Chart title |
| `regime_column` | str | No | Column for background coloring |
| `show_range_slider` | bool | No | Show bottom range slider (default: True) |

**Implementation:**

```python
# src/dashboard/components/charts.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def create_interactive_timeseries(
    data,
    y_columns,
    title="",
    regime_column=None,
    show_range_slider=True,
    height=500
):
    """
    Create fully interactive time series chart.

    Features:
    - Mouse wheel zoom
    - Click and drag pan
    - Crosshair on hover
    - Data preview cards
    - Range selector buttons
    - Optional regime background coloring
    """
    fig = go.Figure()

    # Add regime background coloring if specified
    if regime_column is not None and regime_column in data.columns:
        _add_regime_backgrounds(fig, data, regime_column)

    # Add traces for each column
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

    for i, col in enumerate(y_columns):
        # Prepare custom data for hover
        pct_change = data[col].pct_change() * 100

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[col],
            name=col,
            mode='lines',
            line=dict(width=2, color=colors[i % len(colors)]),
            customdata=np.column_stack([
                pct_change.fillna(0),
                data[regime_column].fillna('N/A') if regime_column else ['N/A'] * len(data)
            ]),
            hovertemplate=(
                '<b>%{x|%Y-%m-%d}</b><br>'
                f'<b>{col}</b>: %{{y:.4f}}<br>'
                'Change: %{customdata[0]:.2f}%<br>'
                'Regime: %{customdata[1]}<br>'
                '<extra></extra>'
            )
        ))

    # Configure layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        height=height,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=60, r=40, t=80, b=60),

        # X-axis with range selector
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor='white',
                activecolor='#2E86AB'
            ),
            rangeslider=dict(visible=show_range_slider),
            type="date"
        ),

        # Y-axis
        yaxis=dict(fixedrange=False)
    )

    # Enable spikes (crosshair lines)
    fig.update_xaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikecolor='gray',
        spikedash='dot'
    )

    fig.update_yaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikecolor='gray',
        spikedash='dot'
    )

    # Configure interactivity
    config = {
        'scrollZoom': True,
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'chart_export',
            'height': 800,
            'width': 1200,
            'scale': 2
        }
    }

    return fig, config


def _add_regime_backgrounds(fig, data, regime_column):
    """Add background coloring for different regimes."""
    regime_colors = {
        'Recovery': 'rgba(46, 204, 113, 0.2)',    # Green
        'Overheat': 'rgba(241, 196, 15, 0.2)',    # Yellow
        'Stagflation': 'rgba(231, 76, 60, 0.2)',  # Red
        'Reflation': 'rgba(52, 152, 219, 0.2)',   # Blue
        'Rising': 'rgba(46, 204, 113, 0.2)',      # Green
        'Falling': 'rgba(231, 76, 60, 0.2)',      # Red
    }

    # Find regime change points
    regime_changes = data[regime_column] != data[regime_column].shift(1)
    change_points = data.index[regime_changes].tolist()
    change_points.append(data.index[-1])

    # Add shapes for each regime period
    for i in range(len(change_points) - 1):
        start = change_points[i]
        end = change_points[i + 1]
        regime = data.loc[start, regime_column]

        if regime in regime_colors:
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=regime_colors[regime],
                layer='below',
                line_width=0
            )
```

---

### 2.2 Correlation Heatmap

**Component ID:** `correlation-heatmap`

**Purpose:** Interactive heatmap showing correlations with click-to-drill-down.

**Props:**
| Prop | Type | Required | Description |
|------|------|----------|-------------|
| `correlation_matrix` | DataFrame | Yes | Square correlation matrix |
| `significance_matrix` | DataFrame | No | P-values for significance masking |
| `title` | str | No | Chart title |

**Implementation:**

```python
def create_correlation_heatmap(
    correlation_matrix,
    significance_matrix=None,
    title="Correlation Matrix",
    height=600
):
    """
    Create interactive correlation heatmap.

    Features:
    - Color-coded cells (-1 to +1)
    - Click cell to see scatter plot
    - Significance masking (optional)
    - Hover with exact values
    """
    import plotly.figure_factory as ff

    # Mask non-significant correlations if provided
    display_matrix = correlation_matrix.copy()
    if significance_matrix is not None:
        mask = significance_matrix > 0.05
        display_matrix = display_matrix.mask(mask, 0)

    # Create annotations with formatted values
    annotations = []
    for i, row in enumerate(correlation_matrix.index):
        for j, col in enumerate(correlation_matrix.columns):
            val = correlation_matrix.loc[row, col]
            sig = significance_matrix.loc[row, col] if significance_matrix is not None else 0

            # Format annotation
            if sig > 0.05:
                text = ""  # Non-significant
            elif abs(val) >= 0.5:
                text = f"<b>{val:.2f}</b>"
            else:
                text = f"{val:.2f}"

            annotations.append(dict(
                x=col, y=row, text=text,
                showarrow=False,
                font=dict(color='white' if abs(val) > 0.4 else 'black')
            ))

    fig = go.Figure(data=go.Heatmap(
        z=display_matrix.values,
        x=display_matrix.columns.tolist(),
        y=display_matrix.index.tolist(),
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(
            title='Correlation',
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1.0', '-0.5', '0', '0.5', '1.0']
        ),
        hovertemplate=(
            '<b>%{y}</b> vs <b>%{x}</b><br>'
            'Correlation: %{z:.4f}<br>'
            '<extra>Click for scatter plot</extra>'
        )
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=height,
        annotations=annotations,
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange='reversed')
    )

    return fig
```

---

### 2.3 Lead-Lag Bar Chart

**Component ID:** `leadlag-barchart`

**Purpose:** Display cross-correlation at different lags.

**Props:**
| Prop | Type | Required | Description |
|------|------|----------|-------------|
| `leadlag_results` | DataFrame | Yes | Results from lead_lag_analysis() |
| `highlight_optimal` | bool | No | Highlight peak correlation (default: True) |

**Implementation:**

```python
def create_leadlag_chart(
    leadlag_results,
    title="Cross-Correlation at Different Lags",
    highlight_optimal=True,
    height=400
):
    """
    Create lead-lag analysis bar chart.

    Features:
    - Bar chart of correlations vs lags
    - Highlighted optimal lag
    - Significance threshold line
    - Clear interpretation labels
    """
    df = leadlag_results.copy()

    # Determine optimal lag
    optimal_idx = df['abs_corr'].idxmax()
    optimal_lag = df.loc[optimal_idx, 'lag']
    optimal_corr = df.loc[optimal_idx, 'correlation']

    # Color bars based on significance and optimal
    colors = []
    for _, row in df.iterrows():
        if row['lag'] == optimal_lag and highlight_optimal:
            colors.append('#2E86AB')  # Highlight color
        elif row['p_value'] < 0.05:
            colors.append('#27AE60' if row['correlation'] > 0 else '#E74C3C')
        else:
            colors.append('#BDC3C7')  # Non-significant

    fig = go.Figure()

    # Add bars
    fig.add_trace(go.Bar(
        x=df['lag'],
        y=df['correlation'],
        marker_color=colors,
        hovertemplate=(
            '<b>Lag: %{x}</b><br>'
            'Correlation: %{y:.4f}<br>'
            '<extra></extra>'
        )
    ))

    # Add significance threshold lines
    n = len(df)
    sig_threshold = 1.96 / np.sqrt(n)
    fig.add_hline(y=sig_threshold, line_dash="dash", line_color="gray",
                  annotation_text="95% CI")
    fig.add_hline(y=-sig_threshold, line_dash="dash", line_color="gray")

    # Add interpretation annotation
    interpretation = "Contemporaneous"
    if optimal_lag < 0:
        interpretation = f"X leads Y by {abs(optimal_lag)} months"
    elif optimal_lag > 0:
        interpretation = f"Y leads X by {optimal_lag} months"

    fig.add_annotation(
        x=optimal_lag,
        y=optimal_corr,
        text=f"<b>Optimal: {interpretation}</b>",
        showarrow=True,
        arrowhead=2,
        yshift=30
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=height,
        xaxis_title="Lag (months)",
        yaxis_title="Correlation",
        xaxis=dict(dtick=1),
        showlegend=False
    )

    return fig
```

---

### 2.4 Regime Timeline

**Component ID:** `regime-timeline`

**Purpose:** Interactive timeline showing regime periods.

**Implementation:**

```python
def create_regime_timeline(
    data,
    regime_column,
    title="Regime Timeline",
    height=150
):
    """
    Create interactive regime timeline.

    Features:
    - Color-coded regime periods
    - Hover for duration and details
    - Click to filter other charts
    - Zoom and pan
    """
    regime_colors = {
        'Recovery': '#27AE60',
        'Overheat': '#F39C12',
        'Stagflation': '#E74C3C',
        'Reflation': '#3498DB'
    }

    fig = go.Figure()

    # Find regime periods
    regime_changes = data[regime_column] != data[regime_column].shift(1)
    change_idx = data.index[regime_changes].tolist()
    change_idx.append(data.index[-1])

    for i in range(len(change_idx) - 1):
        start = change_idx[i]
        end = change_idx[i + 1]
        regime = data.loc[start, regime_column]
        duration = (end - start).days // 30  # Months

        fig.add_trace(go.Scatter(
            x=[start, end, end, start, start],
            y=[0, 0, 1, 1, 0],
            fill='toself',
            fillcolor=regime_colors.get(regime, '#95A5A6'),
            line=dict(width=0),
            name=regime,
            showlegend=(i == 0 or data.loc[change_idx[i-1], regime_column] != regime),
            hovertemplate=(
                f'<b>{regime}</b><br>'
                f'Start: {start.strftime("%Y-%m")}<br>'
                f'End: {end.strftime("%Y-%m")}<br>'
                f'Duration: {duration} months<br>'
                '<extra></extra>'
            )
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=height,
        yaxis=dict(visible=False, range=[0, 1]),
        xaxis=dict(type='date'),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation='h', y=1.1)
    )

    return fig
```

---

### 2.5 KPI Cards

**Component ID:** `kpi-card`

**Purpose:** Display key performance indicators with trend arrows.

**Implementation:**

```python
# src/dashboard/components/cards.py

from dash import html
import dash_bootstrap_components as dbc

def create_kpi_card(
    title,
    value,
    change=None,
    change_period="vs last month",
    format_str="{:.2f}",
    suffix="",
    color=None
):
    """
    Create a KPI card with value and optional change indicator.

    Args:
        title: Card title
        value: Main numeric value
        change: Percentage change (optional)
        change_period: Description of comparison period
        format_str: Format string for value
        suffix: Unit suffix (%, $, etc.)
        color: Override card color
    """
    # Determine change styling
    if change is not None:
        if change > 0:
            change_color = "success"
            arrow = "↑"
        elif change < 0:
            change_color = "danger"
            arrow = "↓"
        else:
            change_color = "secondary"
            arrow = "→"

        change_badge = dbc.Badge(
            f"{arrow} {abs(change):.1f}%",
            color=change_color,
            className="ms-2"
        )
    else:
        change_badge = None

    # Determine card color based on value ranges
    if color is None:
        color = "light"

    card = dbc.Card([
        dbc.CardBody([
            html.H6(title, className="card-subtitle mb-2 text-muted"),
            html.Div([
                html.H3(
                    format_str.format(value) + suffix,
                    className="card-title mb-0 d-inline"
                ),
                change_badge
            ]),
            html.Small(
                change_period if change is not None else "",
                className="text-muted"
            )
        ])
    ], color=color, inverse=(color not in ["light", "white"]))

    return card


def create_kpi_row(metrics):
    """
    Create a row of KPI cards.

    Args:
        metrics: List of dicts with keys: title, value, change, suffix
    """
    cards = []
    for m in metrics:
        cards.append(
            dbc.Col(
                create_kpi_card(
                    title=m.get('title', ''),
                    value=m.get('value', 0),
                    change=m.get('change'),
                    suffix=m.get('suffix', ''),
                    format_str=m.get('format', '{:.2f}')
                ),
                width=12 // len(metrics)
            )
        )

    return dbc.Row(cards, className="mb-4")
```

---

### 2.6 Interactive Data Table

**Component ID:** `data-table`

**Purpose:** Sortable, filterable data table with export.

**Implementation:**

```python
from dash import dash_table

def create_data_table(
    data,
    columns=None,
    page_size=15,
    sortable=True,
    filterable=True,
    exportable=True,
    id_prefix="table"
):
    """
    Create interactive data table.

    Features:
    - Sorting by any column
    - Filtering with text input
    - Pagination
    - CSV/Excel export
    - Conditional formatting
    """
    if columns is None:
        columns = [{"name": c, "id": c} for c in data.columns]

    # Format numeric columns
    for col in columns:
        if data[col['id']].dtype in ['float64', 'float32']:
            col['type'] = 'numeric'
            col['format'] = {'specifier': '.4f'}

    table = dash_table.DataTable(
        id=f'{id_prefix}-datatable',
        columns=columns,
        data=data.to_dict('records'),

        # Sorting
        sort_action='native' if sortable else 'none',
        sort_mode='multi',

        # Filtering
        filter_action='native' if filterable else 'none',

        # Pagination
        page_action='native',
        page_size=page_size,

        # Export
        export_format='csv' if exportable else 'none',
        export_headers='display',

        # Styling
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '8px',
            'fontFamily': 'system-ui'
        },
        style_header={
            'backgroundColor': '#f8f9fa',
            'fontWeight': 'bold',
            'borderBottom': '2px solid #dee2e6'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f8f9fa'
            },
            {
                'if': {
                    'filter_query': '{p_value} < 0.05',
                    'column_id': 'p_value'
                },
                'backgroundColor': '#d4edda',
                'color': '#155724'
            },
            {
                'if': {
                    'filter_query': '{sharpe} > 0.5',
                    'column_id': 'sharpe'
                },
                'backgroundColor': '#d4edda',
                'color': '#155724'
            }
        ]
    )

    return table
```

---

### 2.7 Equity Curve with Drawdown

**Component ID:** `equity-curve`

**Purpose:** Backtest equity curve with drawdown overlay.

**Implementation:**

```python
def create_equity_curve(
    data,
    strategy_col='strategy_cumret',
    benchmark_col='benchmark_cumret',
    drawdown_col='drawdown',
    regime_col=None,
    title="Equity Curve",
    height=500
):
    """
    Create equity curve with drawdown subplot.

    Features:
    - Cumulative returns comparison
    - Drawdown area chart
    - Regime background coloring
    - Hover with detailed metrics
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Cumulative Returns", "Drawdown")
    )

    # Add regime backgrounds if provided
    if regime_col and regime_col in data.columns:
        _add_regime_backgrounds(fig, data, regime_col)

    # Strategy equity curve
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[strategy_col],
        name='Strategy',
        line=dict(color='#2E86AB', width=2),
        hovertemplate=(
            '<b>%{x|%Y-%m-%d}</b><br>'
            'Strategy: %{y:.2%}<br>'
            '<extra></extra>'
        )
    ), row=1, col=1)

    # Benchmark equity curve
    if benchmark_col in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[benchmark_col],
            name='Benchmark',
            line=dict(color='#95A5A6', width=1.5, dash='dash'),
            hovertemplate=(
                '<b>%{x|%Y-%m-%d}</b><br>'
                'Benchmark: %{y:.2%}<br>'
                '<extra></extra>'
            )
        ), row=1, col=1)

    # Drawdown area
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[drawdown_col],
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='#E74C3C', width=1),
        fillcolor='rgba(231, 76, 60, 0.3)',
        hovertemplate=(
            '<b>%{x|%Y-%m-%d}</b><br>'
            'Drawdown: %{y:.2%}<br>'
            '<extra></extra>'
        )
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=height,
        hovermode='x unified',
        legend=dict(orientation='h', y=1.08),
        xaxis2=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(count=10, label="10Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        )
    )

    # Enable crosshair
    fig.update_xaxes(showspikes=True, spikemode='across')
    fig.update_yaxes(showspikes=True, spikemode='across')

    return fig
```

---

### 2.8 Monte Carlo Distribution

**Component ID:** `montecarlo-dist`

**Purpose:** Display Monte Carlo simulation results.

**Implementation:**

```python
def create_montecarlo_histogram(
    simulated_values,
    actual_value,
    confidence=0.95,
    title="Monte Carlo Distribution",
    metric_name="Sharpe Ratio",
    height=400
):
    """
    Create Monte Carlo simulation histogram with confidence intervals.
    """
    import numpy as np

    # Calculate statistics
    mean_sim = np.mean(simulated_values)
    std_sim = np.std(simulated_values)
    ci_lower = np.percentile(simulated_values, (1 - confidence) / 2 * 100)
    ci_upper = np.percentile(simulated_values, (1 + confidence) / 2 * 100)
    p_value = (simulated_values >= actual_value).mean()

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=simulated_values,
        nbinsx=50,
        name='Simulated',
        marker_color='rgba(46, 134, 171, 0.6)',
        hovertemplate=(
            f'{metric_name}: %{{x:.4f}}<br>'
            'Count: %{y}<br>'
            '<extra></extra>'
        )
    ))

    # Actual value line
    fig.add_vline(
        x=actual_value,
        line=dict(color='#E74C3C', width=3),
        annotation_text=f"Actual: {actual_value:.4f}",
        annotation_position="top"
    )

    # Confidence interval
    fig.add_vrect(
        x0=ci_lower,
        x1=ci_upper,
        fillcolor='rgba(39, 174, 96, 0.2)',
        line_width=0,
        annotation_text=f"{confidence:.0%} CI",
        annotation_position="top left"
    )

    # Mean line
    fig.add_vline(
        x=mean_sim,
        line=dict(color='gray', width=2, dash='dash'),
        annotation_text=f"Mean: {mean_sim:.4f}"
    )

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>p-value: {p_value:.4f}</sup>",
            x=0.5
        ),
        height=height,
        xaxis_title=metric_name,
        yaxis_title="Frequency",
        showlegend=False,
        annotations=[
            dict(
                x=0.98,
                y=0.98,
                xref='paper',
                yref='paper',
                text=(
                    f"<b>Statistics:</b><br>"
                    f"Mean: {mean_sim:.4f}<br>"
                    f"Std: {std_sim:.4f}<br>"
                    f"{confidence:.0%} CI: [{ci_lower:.4f}, {ci_upper:.4f}]<br>"
                    f"p-value: {p_value:.4f}"
                ),
                showarrow=False,
                bgcolor='white',
                bordercolor='gray',
                borderwidth=1,
                align='left'
            )
        ]
    )

    return fig
```

---

## 3. Page Layouts

### 3.1 Overview Page

```python
# src/dashboard/layouts/overview.py

from dash import html, dcc
import dash_bootstrap_components as dbc
from ..components.charts import create_interactive_timeseries
from ..components.cards import create_kpi_row

def create_overview_layout(data, metrics):
    """Create the overview dashboard page."""

    return dbc.Container([
        # Header row
        dbc.Row([
            dbc.Col([
                html.H2("Analysis Overview"),
                html.P("Key metrics and current regime status", className="text-muted")
            ])
        ], className="mb-4"),

        # KPI Cards
        create_kpi_row([
            {'title': 'Sharpe Ratio', 'value': metrics['sharpe'], 'suffix': ''},
            {'title': 'CAGR', 'value': metrics['cagr'] * 100, 'suffix': '%'},
            {'title': 'Max Drawdown', 'value': metrics['max_dd'] * 100, 'suffix': '%'},
            {'title': 'Win Rate', 'value': metrics['win_rate'] * 100, 'suffix': '%'}
        ]),

        # Main chart
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='overview-main-chart',
                    figure=create_interactive_timeseries(
                        data,
                        y_columns=['strategy_cumret', 'benchmark_cumret'],
                        title="Strategy vs Benchmark",
                        regime_column='regime'
                    )[0],
                    config=create_interactive_timeseries(data, ['x'])[1]
                )
            ], width=8),

            dbc.Col([
                # Current regime card
                dbc.Card([
                    dbc.CardHeader("Current Regime"),
                    dbc.CardBody([
                        html.H3(metrics['current_regime'], className="text-primary"),
                        html.P([
                            html.Strong("Growth: "),
                            metrics['growth_signal']
                        ]),
                        html.P([
                            html.Strong("Inflation: "),
                            metrics['inflation_signal']
                        ])
                    ])
                ], className="mb-3"),

                # Recent signals table
                dbc.Card([
                    dbc.CardHeader("Recent Signals"),
                    dbc.CardBody([
                        create_data_table(
                            data.tail(5)[['date', 'signal', 'return']],
                            page_size=5,
                            sortable=False,
                            filterable=False
                        )
                    ])
                ])
            ], width=4)
        ])
    ], fluid=True)
```

---

## 4. Callbacks

### 4.1 Chart Interactivity Callbacks

```python
# src/dashboard/callbacks/chart_callbacks.py

from dash import callback, Input, Output, State
import plotly.express as px

@callback(
    Output('scatter-detail', 'figure'),
    Input('correlation-heatmap', 'clickData'),
    State('analysis-data', 'data')
)
def show_scatter_on_heatmap_click(clickData, data):
    """
    Show scatter plot when user clicks a correlation heatmap cell.
    """
    if clickData is None:
        return px.scatter(title="Click a cell to see scatter plot")

    x_col = clickData['points'][0]['x']
    y_col = clickData['points'][0]['y']

    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        trendline='ols',
        title=f"{x_col} vs {y_col}"
    )

    # Add correlation coefficient annotation
    corr = data[x_col].corr(data[y_col])
    fig.add_annotation(
        x=0.05, y=0.95,
        xref='paper', yref='paper',
        text=f"r = {corr:.4f}",
        showarrow=False,
        bgcolor='white'
    )

    return fig


@callback(
    Output('filtered-chart', 'figure'),
    Input('regime-timeline', 'clickData'),
    State('full-data', 'data')
)
def filter_chart_by_regime(clickData, data):
    """
    Filter main chart to show only selected regime period.
    """
    if clickData is None:
        return create_full_chart(data)

    regime = clickData['points'][0]['text']
    filtered = data[data['regime'] == regime]

    return create_filtered_chart(filtered, regime)
```

### 4.2 Export Callbacks

```python
# src/dashboard/callbacks/export_callbacks.py

from dash import callback, Input, Output
import io
import base64

@callback(
    Output('download-csv', 'data'),
    Input('export-csv-button', 'n_clicks'),
    State('current-data', 'data'),
    prevent_initial_call=True
)
def export_to_csv(n_clicks, data):
    """Export current data to CSV."""
    df = pd.DataFrame(data)
    return dcc.send_data_frame(df.to_csv, "analysis_export.csv")


@callback(
    Output('download-chart', 'data'),
    Input('export-chart-button', 'n_clicks'),
    State('main-chart', 'figure'),
    prevent_initial_call=True
)
def export_chart(n_clicks, figure):
    """Export chart as PNG."""
    import plotly.io as pio

    img_bytes = pio.to_image(figure, format='png', scale=2)
    encoded = base64.b64encode(img_bytes).decode()

    return dict(
        content=encoded,
        filename='chart_export.png',
        base64=True
    )
```

---

## 5. Deployment Configuration

### 5.1 Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/dashboard ./dashboard
COPY data ./data

EXPOSE 8050

CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "4", "dashboard.app:server"]
```

### 5.2 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  dashboard:
    build: .
    ports:
      - "8050:8050"
    volumes:
      - ./data:/app/data:ro
    environment:
      - DASH_DEBUG=false
    restart: unless-stopped
```

### 5.3 Requirements

```
# requirements.txt
dash==2.14.2
dash-bootstrap-components==1.5.0
plotly==5.18.0
pandas==2.1.4
numpy==1.26.2
gunicorn==21.2.0
pyarrow==14.0.1
scipy==1.11.4
statsmodels==0.14.0
```

---

## 6. Usage Examples

### 6.1 Creating a Complete Page

```python
# Example: Creating the Regime Analysis page

from dash import html, dcc
import dash_bootstrap_components as dbc
from .components.charts import (
    create_regime_timeline,
    create_interactive_timeseries,
    create_correlation_heatmap
)
from .components.cards import create_kpi_row

def create_regime_page(data, sector_data, metrics):
    """Create the regime analysis page."""

    return dbc.Container([
        # Page header
        html.H2("Regime Analysis"),
        html.Hr(),

        # Regime timeline (full width)
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='regime-timeline',
                    figure=create_regime_timeline(
                        data,
                        regime_column='phase',
                        title="Investment Clock Phases"
                    ),
                    config={'scrollZoom': True}
                )
            ])
        ], className="mb-4"),

        # Two-column layout
        dbc.Row([
            # Sector-Phase Heatmap
            dbc.Col([
                dcc.Graph(
                    id='sector-phase-heatmap',
                    figure=create_correlation_heatmap(
                        sector_data.pivot_table(
                            values='return',
                            index='sector',
                            columns='phase',
                            aggfunc='mean'
                        ),
                        title="Annualized Sector Returns by Phase"
                    )
                )
            ], width=6),

            # Performance boxplots
            dbc.Col([
                dcc.Graph(
                    id='regime-boxplots',
                    figure=create_regime_boxplots(data)
                )
            ], width=6)
        ]),

        # Phase statistics table
        dbc.Row([
            dbc.Col([
                html.H4("Phase Statistics"),
                create_data_table(
                    metrics['phase_stats'],
                    page_size=4
                )
            ])
        ])
    ], fluid=True)
```

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-24 | RA Cheryl | Initial component specifications |
