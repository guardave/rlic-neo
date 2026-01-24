# Dashboard Portal Catalog Design

**Version:** 2.0
**Date:** 2026-01-24
**Author:** RA Cheryl

---

## 1. Design Principles

### 1.1 Layout Philosophy

- **Catalog-first**: Primary purpose is navigation to individual dashboards
- **Rich cards**: Each card shows key insights specific to that analysis
- **2x2 grid**: Four cards visible on desktop (1920x1080)
- **Responsive**: Reflows to single column on mobile
- **Per-analysis context**: Regime, metrics, and insights are specific to each study

### 1.2 Target Resolutions

| Device | Resolution | Grid | Card Size |
|--------|------------|------|-----------|
| Desktop | 1920x1080 | 2x2 | ~900x450px |
| Laptop | 1440x900 | 2x2 | ~680x380px |
| Tablet | 1024x768 | 2x1 | ~480x400px |
| Mobile | 375x812 | 1x1 | ~350x450px |

---

## 2. Page Layout

### 2.1 Desktop View (1920x1080)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  RLIC Analysis Portal                              [🔍 Search] [Filter ▼]    │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  ┌────────────────────────────────────┐  ┌────────────────────────────────┐ │
│  │ Investment Clock Sector Analysis   │  │ SPY vs RETAILIRSA              │ │
│  │ ─────────────────────────────────  │  │ ─────────────────────────────  │ │
│  │                                    │  │                                │ │
│  │  ┌──────────────┐  CURRENT REGIME  │  │  ┌──────────────┐  REGIME      │ │
│  │  │              │  ┌────────────┐  │  │  │              │  ┌────────┐  │ │
│  │  │  [Sparkline] │  │  OVERHEAT  │  │  │  │  [Sparkline] │  │ RISING │  │ │
│  │  │   Chart      │  └────────────┘  │  │  │   Chart      │  └────────┘  │ │
│  │  │              │  Growth: ↑       │  │  │              │  Inv: ↑      │ │
│  │  └──────────────┘  Inflation: ↑    │  │  └──────────────┘              │ │
│  │                                    │  │                                │ │
│  │  ┌─────────┐ ┌─────────┐ ┌──────┐ │  │  ┌─────────┐ ┌─────────┐       │ │
│  │  │Sharpe   │ │Accuracy │ │Sectors│ │  │  │Corr     │ │Lead-Lag │       │ │
│  │  │  0.62   │ │  96.8%  │ │ 11   │ │  │  │  0.23   │ │ Lag=0   │       │ │
│  │  └─────────┘ └─────────┘ └──────┘ │  │  └─────────┘ └─────────┘       │ │
│  │                                    │  │                                │ │
│  │  Key: Best pair Orders/Inv + PPI   │  │  Finding: Contemporaneous,     │ │
│  │  ★ Primary Strategy                │  │  no predictive power           │ │
│  │                                    │  │                                │ │
│  │  Updated: 2026-01-03               │  │  Updated: 2025-12-21           │ │
│  └────────────────────────────────────┘  └────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────┐  ┌────────────────────────────────┐ │
│  │ SPY vs Industrial Production       │  │ ML Regime Detection            │ │
│  │ ─────────────────────────────────  │  │ ─────────────────────────────  │ │
│  │                                    │  │                                │ │
│  │  ┌──────────────┐  REGIME          │  │  ┌──────────────┐  MODEL       │ │
│  │  │              │  ┌────────┐      │  │  │              │  ┌────────┐  │ │
│  │  │  [Sparkline] │  │FALLING │      │  │  │  [Sparkline] │  │HMM Sup │  │ │
│  │  │   Chart      │  └────────┘      │  │  │   Chart      │  └────────┘  │ │
│  │  │              │  IP: ↓           │  │  │              │  States: 4   │ │
│  │  └──────────────┘                  │  │  └──────────────┘              │ │
│  │                                    │  │                                │ │
│  │  ┌─────────┐ ┌─────────┐          │  │  ┌─────────┐ ┌─────────┐       │ │
│  │  │Corr     │ │Granger  │          │  │  │Sharpe   │ │WFER     │       │ │
│  │  │  0.31   │ │ p>0.05  │          │  │  │  0.62   │ │  0.87   │       │ │
│  │  └─────────┘ └─────────┘          │  │  └─────────┘ └─────────┘       │ │
│  │                                    │  │                                │ │
│  │  Finding: Coincident indicator,    │  │  Key: HMM beats rule-based,    │ │
│  │  NBER Big Four                     │  │  40% fewer regime changes      │ │
│  │                                    │  │                                │ │
│  │  Updated: 2025-12-21               │  │  Updated: 2025-12-15           │ │
│  └────────────────────────────────────┘  └────────────────────────────────┘ │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Showing 4 of 8 analyses                                    [Load More ↓]   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Mobile View (375px)

```
┌─────────────────────────┐
│ RLIC Portal    [🔍] [≡] │
├─────────────────────────┤
│                         │
│ ┌─────────────────────┐ │
│ │ Investment Clock    │ │
│ │ Sector Analysis     │ │
│ │ ───────────────────│ │
│ │                     │ │
│ │ [Chart]  OVERHEAT   │ │
│ │          Growth: ↑  │ │
│ │          Infl: ↑    │ │
│ │                     │ │
│ │ Sharpe │ Acc │ Sect │ │
│ │  0.62  │96.8%│ 11   │ │
│ │                     │ │
│ │ ★ Primary Strategy  │ │
│ │ Updated: 2026-01-03 │ │
│ └─────────────────────┘ │
│                         │
│ ┌─────────────────────┐ │
│ │ SPY vs RETAILIRSA   │ │
│ │ ───────────────────│ │
│ │                     │ │
│ │ [Chart]  RISING     │ │
│ │          Inv: ↑     │ │
│ │                     │ │
│ │ Corr  │ Lag         │ │
│ │ 0.23  │ 0 (contemp) │ │
│ │                     │ │
│ │ No predictive power │ │
│ │ Updated: 2025-12-21 │ │
│ └─────────────────────┘ │
│                         │
│      [Load More ↓]      │
└─────────────────────────┘
```

---

## 3. Card Component Specification

### 3.1 Card Anatomy

```
┌────────────────────────────────────────────────────────────────┐
│  TITLE                                                         │  <- Header
│  ────────────────────────────────────────────────────────────  │
│                                                                │
│  ┌──────────────────┐   REGIME/STATUS                         │
│  │                  │   ┌─────────────────┐                   │  <- Status Area
│  │   [Sparkline     │   │   PHASE NAME    │                   │
│  │    or Mini       │   └─────────────────┘                   │
│  │    Chart]        │   Direction: ↑/↓                        │
│  │                  │   Secondary: value                      │
│  └──────────────────┘                                         │
│                                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                    │  <- Metrics Row
│  │ Metric 1 │  │ Metric 2 │  │ Metric 3 │                    │
│  │  Value   │  │  Value   │  │  Value   │                    │
│  └──────────┘  └──────────┘  └──────────┘                    │
│                                                                │
│  Key Finding: One-line insight text...                        │  <- Insight
│  ★ Badge (if applicable)                                      │  <- Badge
│                                                                │
│  Updated: YYYY-MM-DD                                          │  <- Footer
└────────────────────────────────────────────────────────────────┘
```

### 3.2 Card Content by Analysis Type

| Analysis Type | Regime/Status | Metric 1 | Metric 2 | Metric 3 | Key Finding |
|---------------|---------------|----------|----------|----------|-------------|
| **Investment Clock** | Phase (Recovery/Overheat/etc) | Sharpe | Classification % | # Sectors | Best indicator pair |
| **Indicator Study** | Direction (Rising/Falling) | Correlation | Lead-Lag | Granger p | Predictive power |
| **Sector Analysis** | Best Sector | Avg Return | Win Rate | # Phases | Allocation insight |
| **ML Model** | Model Type | Sharpe | WFER | # States | Model advantage |
| **Forecast** | Trend Direction | RMSE | MAPE | Horizon | Accuracy insight |

### 3.3 Card Implementation

```python
# src/dashboard/components/catalog_card.py

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

def create_analysis_card(analysis):
    """
    Create a rich analysis catalog card.

    Args:
        analysis: dict with:
            - id: str
            - title: str
            - type: str ('investment_clock', 'indicator', 'sector', 'ml', 'forecast')
            - sparkline_data: list of values for mini chart
            - regime: dict with name, direction, secondary
            - metrics: list of dicts with label, value
            - finding: str (key insight)
            - badge: str or None ('Primary', 'New', etc.)
            - updated: str (date)
    """

    # Regime/Status colors
    regime_colors = {
        'Recovery': '#27AE60',
        'Overheat': '#F39C12',
        'Stagflation': '#E74C3C',
        'Reflation': '#3498DB',
        'Rising': '#27AE60',
        'Falling': '#E74C3C',
        'Neutral': '#95A5A6'
    }

    regime = analysis.get('regime', {})
    regime_color = regime_colors.get(regime.get('name'), '#95A5A6')

    # Build card
    card = dbc.Card([
        dbc.CardBody([
            # Header
            html.H5(analysis['title'], className="card-title mb-2"),
            html.Hr(className="my-2"),

            # Main content row: Sparkline + Regime
            dbc.Row([
                # Sparkline chart
                dbc.Col([
                    dcc.Graph(
                        figure=_create_sparkline(analysis.get('sparkline_data', [])),
                        config={'displayModeBar': False},
                        style={'height': '80px'}
                    )
                ], width=5, className="pe-0"),

                # Regime/Status
                dbc.Col([
                    html.Div([
                        html.Small("REGIME", className="text-muted text-uppercase"),
                        html.Div(
                            regime.get('name', 'N/A'),
                            className="fw-bold px-2 py-1 rounded text-center",
                            style={
                                'backgroundColor': regime_color,
                                'color': 'white',
                                'fontSize': '0.9rem'
                            }
                        ),
                        html.Small([
                            html.Span(
                                f"{regime.get('direction_label', '')}: ",
                                className="text-muted"
                            ),
                            html.Span(
                                regime.get('direction', ''),
                                className="fw-bold"
                            )
                        ], className="d-block mt-1") if regime.get('direction') else None,
                        html.Small(
                            regime.get('secondary', ''),
                            className="text-muted d-block"
                        ) if regime.get('secondary') else None
                    ])
                ], width=7, className="ps-2")
            ], className="mb-3"),

            # Metrics row
            dbc.Row([
                dbc.Col([
                    _create_metric_chip(m['label'], m['value'])
                ], width=4) for m in analysis.get('metrics', [])[:3]
            ], className="mb-3"),

            # Key finding
            html.P([
                html.Strong("Key: "),
                analysis.get('finding', '')
            ], className="small mb-2 text-truncate",
               title=analysis.get('finding', '')),

            # Badge (if any)
            html.Div([
                dbc.Badge(
                    f"★ {analysis['badge']}",
                    color="warning",
                    className="me-1"
                )
            ], className="mb-2") if analysis.get('badge') else None,

            # Footer
            html.Small(
                f"Updated: {analysis.get('updated', '')}",
                className="text-muted"
            )
        ], className="p-3")
    ],
    className="catalog-card h-100",
    id={'type': 'catalog-card', 'index': analysis['id']},
    style={'cursor': 'pointer'})

    return card


def _create_sparkline(data):
    """Create a minimal sparkline chart."""
    if not data:
        data = [0]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=data,
        mode='lines',
        line=dict(color='#2E86AB', width=2),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.1)',
        hoverinfo='skip'
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )

    return fig


def _create_metric_chip(label, value):
    """Create a small metric display chip."""
    return html.Div([
        html.Small(label, className="text-muted d-block text-center"),
        html.Div(
            str(value),
            className="text-center fw-bold",
            style={'fontSize': '1.1rem'}
        )
    ], className="border rounded py-1 px-2 bg-light")
```

---

## 4. Page Layout Implementation

### 4.1 Main Layout

```python
# src/dashboard/layouts/catalog.py

from dash import html, dcc
import dash_bootstrap_components as dbc
from ..components.catalog_card import create_analysis_card

def create_catalog_layout(analyses, page=1, per_page=4):
    """
    Create the catalog landing page.

    Args:
        analyses: List of analysis metadata
        page: Current page number
        per_page: Cards per page (4 for 2x2 grid)
    """

    return dbc.Container([
        # Header
        _create_header(),

        html.Hr(className="my-3"),

        # Catalog grid
        _create_catalog_grid(analyses, page, per_page),

        # Pagination
        _create_pagination(len(analyses), page, per_page)

    ], fluid=True, className="py-3 catalog-container")


def _create_header():
    """Header with title, search, and filter."""
    return dbc.Row([
        dbc.Col([
            html.H3("RLIC Analysis Portal", className="mb-0")
        ], xs=12, md=6),
        dbc.Col([
            dbc.InputGroup([
                dbc.Input(
                    id="catalog-search",
                    placeholder="Search analyses...",
                    type="search"
                ),
                dbc.Select(
                    id="catalog-filter",
                    options=[
                        {"label": "All Types", "value": "all"},
                        {"label": "Indicators", "value": "indicator"},
                        {"label": "Sectors", "value": "sector"},
                        {"label": "ML Models", "value": "ml"},
                        {"label": "Forecasts", "value": "forecast"}
                    ],
                    value="all",
                    style={'maxWidth': '140px'}
                )
            ], size="sm")
        ], xs=12, md=6, className="mt-2 mt-md-0")
    ], className="align-items-center")


def _create_catalog_grid(analyses, page, per_page):
    """Create 2x2 grid of analysis cards."""
    start = (page - 1) * per_page
    end = start + per_page
    visible = analyses[start:end]

    cards = []
    for analysis in visible:
        card = create_analysis_card(analysis)
        # 2 columns on desktop, 1 on mobile
        cards.append(
            dbc.Col(card, xs=12, lg=6, className="mb-4")
        )

    return dbc.Row(cards, id="catalog-grid", className="g-4")


def _create_pagination(total, current_page, per_page):
    """Create pagination or load more button."""
    total_pages = (total + per_page - 1) // per_page

    if total_pages <= 1:
        return None

    return html.Div([
        html.Hr(className="my-3"),
        html.Div([
            html.Small(
                f"Showing {min(current_page * per_page, total)} of {total} analyses",
                className="text-muted me-3"
            ),
            dbc.Button(
                "Load More ↓",
                id="load-more",
                color="outline-primary",
                size="sm"
            ) if current_page < total_pages else None
        ], className="d-flex justify-content-center align-items-center")
    ])
```

---

## 5. Sample Data

```python
SAMPLE_ANALYSES = [
    {
        'id': 'investment-clock-sectors',
        'title': 'Investment Clock Sector Analysis',
        'type': 'investment_clock',
        'sparkline_data': [0.02, 0.03, 0.01, 0.04, 0.02, 0.05, 0.03, 0.06, 0.04, 0.05],
        'regime': {
            'name': 'Overheat',
            'direction_label': 'Growth',
            'direction': '↑',
            'secondary': 'Inflation: ↑'
        },
        'metrics': [
            {'label': 'Sharpe', 'value': '0.62'},
            {'label': 'Accuracy', 'value': '96.8%'},
            {'label': 'Sectors', 'value': '11'}
        ],
        'finding': 'Best pair: Orders/Inv Ratio + PPI',
        'badge': 'Primary Strategy',
        'updated': '2026-01-03'
    },
    {
        'id': 'spy-retailirsa',
        'title': 'SPY vs RETAILIRSA',
        'type': 'indicator',
        'sparkline_data': [1.2, 1.3, 1.1, 1.4, 1.3, 1.5, 1.4, 1.6, 1.5, 1.7],
        'regime': {
            'name': 'Rising',
            'direction_label': 'Inventories',
            'direction': '↑',
            'secondary': None
        },
        'metrics': [
            {'label': 'Corr', 'value': '0.23'},
            {'label': 'Lead-Lag', 'value': 'Lag=0'},
            {'label': 'Granger', 'value': 'p>0.05'}
        ],
        'finding': 'Contemporaneous, no predictive power',
        'badge': None,
        'updated': '2025-12-21'
    },
    {
        'id': 'spy-indpro',
        'title': 'SPY vs Industrial Production',
        'type': 'indicator',
        'sparkline_data': [102, 103, 101, 100, 99, 98, 97, 96, 95, 94],
        'regime': {
            'name': 'Falling',
            'direction_label': 'IP',
            'direction': '↓',
            'secondary': None
        },
        'metrics': [
            {'label': 'Corr', 'value': '0.31'},
            {'label': 'Lead-Lag', 'value': 'Lag=0'},
            {'label': 'Granger', 'value': 'p>0.05'}
        ],
        'finding': 'Coincident indicator, NBER Big Four',
        'badge': None,
        'updated': '2025-12-21'
    },
    {
        'id': 'ml-regime-detection',
        'title': 'ML Regime Detection',
        'type': 'ml',
        'sparkline_data': [0.55, 0.58, 0.56, 0.60, 0.59, 0.61, 0.60, 0.62, 0.61, 0.62],
        'regime': {
            'name': 'HMM Supervised',
            'direction_label': 'States',
            'direction': '4',
            'secondary': None
        },
        'metrics': [
            {'label': 'Sharpe', 'value': '0.62'},
            {'label': 'WFER', 'value': '0.87'},
            {'label': 'Transitions', 'value': '-40%'}
        ],
        'finding': 'HMM beats rule-based, smoother transitions',
        'badge': None,
        'updated': '2025-12-15'
    },
    {
        'id': 'xlre-orders-inv',
        'title': 'XLRE vs Orders/Inv Ratio',
        'type': 'sector',
        'sparkline_data': [45, 46, 44, 47, 48, 46, 49, 50, 48, 51],
        'regime': {
            'name': 'Rising',
            'direction_label': 'Ratio',
            'direction': '↑',
            'secondary': None
        },
        'metrics': [
            {'label': 'Corr', 'value': '0.42'},
            {'label': 'Lead', 'value': '-2mo'},
            {'label': 'Sharpe', 'value': '0.38'}
        ],
        'finding': 'Real estate leading indicator potential',
        'badge': 'New',
        'updated': '2026-01-24'
    },
    {
        'id': 'cass-freight-index',
        'title': 'Cass Freight Index',
        'type': 'forecast',
        'sparkline_data': [1.1, 1.15, 1.08, 1.12, 1.18, 1.14, 1.20, 1.16, 1.22, 1.19],
        'regime': {
            'name': 'Seasonal Peak',
            'direction_label': 'Trend',
            'direction': '↗',
            'secondary': None
        },
        'metrics': [
            {'label': 'RMSE', 'value': '0.023'},
            {'label': 'MAPE', 'value': '2.1%'},
            {'label': 'Horizon', 'value': '12mo'}
        ],
        'finding': 'Strong seasonality, SARIMA best method',
        'badge': None,
        'updated': '2025-12-15'
    },
    {
        'id': 'ppi-analysis',
        'title': 'PPI Inflation Analysis',
        'type': 'indicator',
        'sparkline_data': [2.1, 2.3, 2.5, 2.8, 3.0, 3.2, 3.1, 2.9, 2.7, 2.5],
        'regime': {
            'name': 'Falling',
            'direction_label': 'PPI MoM',
            'direction': '↓',
            'secondary': None
        },
        'metrics': [
            {'label': 'Current', 'value': '2.5%'},
            {'label': 'vs CPI', 'value': 'r=0.78'},
            {'label': 'Lead', 'value': '+2mo'}
        ],
        'finding': 'Leads CPI by 2 months, inflation proxy',
        'badge': None,
        'updated': '2025-11-20'
    },
    {
        'id': 'yield-curve-study',
        'title': 'Yield Curve Analysis',
        'type': 'indicator',
        'sparkline_data': [0.5, 0.3, 0.1, -0.1, -0.2, -0.1, 0.0, 0.2, 0.4, 0.6],
        'regime': {
            'name': 'Steepening',
            'direction_label': '10Y-2Y',
            'direction': '↑',
            'secondary': None
        },
        'metrics': [
            {'label': 'Spread', 'value': '+60bp'},
            {'label': 'Recession', 'value': 'p<15%'},
            {'label': 'Signal', 'value': 'Risk-on'}
        ],
        'finding': 'Recession probability declining',
        'badge': None,
        'updated': '2025-10-15'
    }
]
```

---

## 6. Responsive CSS

```css
/* assets/catalog.css */

/* Container sizing */
.catalog-container {
    max-width: 1800px;
    margin: 0 auto;
}

/* Card styling */
.catalog-card {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
}

.catalog-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    border-color: #2E86AB;
}

/* Card with badge */
.catalog-card.primary {
    border-left: 4px solid #F39C12;
}

/* Sparkline container */
.catalog-card .js-plotly-plot {
    pointer-events: none;
}

/* Metric chips */
.catalog-card .border {
    border-color: #e9ecef !important;
}

/* Responsive adjustments */
@media (max-width: 991px) {
    .catalog-card {
        margin-bottom: 1rem;
    }
}

@media (max-width: 575px) {
    .catalog-container {
        padding: 0.5rem;
    }

    .catalog-card .card-body {
        padding: 0.75rem;
    }

    .catalog-card h5 {
        font-size: 1rem;
    }
}

/* Search and filter */
#catalog-search {
    border-radius: 4px 0 0 4px;
}

#catalog-filter {
    border-radius: 0 4px 4px 0;
    border-left: none;
}

/* Hover effect on mobile - use active state instead */
@media (hover: none) {
    .catalog-card:active {
        transform: scale(0.98);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
}
```

---

## 7. Summary

| Aspect | Specification |
|--------|---------------|
| **Grid** | 2x2 on desktop, 1 column on mobile |
| **Card Height** | ~450px (flexible based on content) |
| **Per-card Content** | Sparkline, regime badge, 3 metrics, key finding, badge, date |
| **Regime Display** | Per-analysis, color-coded, with direction indicators |
| **Responsiveness** | Breakpoints at 992px (tablet), 576px (mobile) |
| **Interaction** | Click card → navigate to full dashboard |

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-24 | RA Cheryl | Initial simple catalog |
| 2.0 | 2026-01-24 | RA Cheryl | Rich cards with per-analysis regime and metrics |
