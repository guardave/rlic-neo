# Dashboard Portal Landing Page Design

**Version:** 1.0
**Date:** 2026-01-24
**Author:** RA Cheryl
**Related:** [Dashboard Component Specs](./dashboard_component_specs.md)

---

## 1. Portal Architecture

### 1.1 Site Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RLIC ANALYSIS PORTAL                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LANDING PAGE (Home)                                                    │
│  ├── Hero: Key Insights Summary                                         │
│  ├── Analysis Catalog Grid                                              │
│  ├── Performance Overview                                               │
│  └── Recent Activity                                                    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ANALYSIS PAGES                                │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                   │   │
│  │  /analysis/{study-id}                                            │   │
│  │  ├── Overview Tab                                                 │   │
│  │  ├── Qualitative Tab                                              │   │
│  │  ├── Correlation Tab                                              │   │
│  │  ├── Lead-Lag Tab                                                 │   │
│  │  ├── Regimes Tab                                                  │   │
│  │  ├── Backtests Tab                                                │   │
│  │  └── Forecasts Tab                                                │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    PORTFOLIO PAGES                               │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  /portfolio/investment-clock   - Current regime & allocation     │   │
│  │  /portfolio/sector-rotation    - Sector performance by phase     │   │
│  │  /portfolio/signals            - Active signals dashboard        │   │
│  │  /portfolio/performance        - Strategy performance tracking   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    REFERENCE PAGES                               │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  /reference/methodology        - Analysis SOP documentation      │   │
│  │  /reference/indicators         - Indicator definitions           │   │
│  │  /reference/backtest-methods   - Backtest methodology catalog    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Landing Page Layout

### 2.1 Full Page Wireframe

```
┌─────────────────────────────────────────────────────────────────────────┐
│  RLIC Analysis Portal                    [Search] [Notifications] [⚙️]  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     HERO SECTION                                 │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │  Current Regime: OVERHEAT                                  │  │   │
│  │  │  Growth: Rising ↑  |  Inflation: Rising ↑                  │  │   │
│  │  │  Since: November 2025 (3 months)                           │  │   │
│  │  │                                                             │  │   │
│  │  │  [View Investment Clock] [See Sector Allocation]           │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  │                                                                  │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│  │  │ Strategy │ │ Best     │ │ Win Rate │ │ Active   │           │   │
│  │  │ Sharpe   │ │ Sector   │ │ (12M)    │ │ Signals  │           │   │
│  │  │   0.62   │ │  XLE     │ │   58%    │ │   4/11   │           │   │
│  │  │ +0.04 ↑  │ │ +12.3%   │ │          │ │ Long     │           │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  KEY INSIGHTS                                              [View All →] │
│  ┌──────────────────────┐ ┌──────────────────────┐ ┌─────────────────┐ │
│  │ 🎯 Best Indicator    │ │ 📈 Top Finding       │ │ ⚠️ Latest Alert │ │
│  │                      │ │                      │ │                  │ │
│  │ Orders/Inv + PPI     │ │ HMM Supervised Init  │ │ Regime change   │ │
│  │ 96.8% classification │ │ beats rule-based     │ │ detected in     │ │
│  │ vs 66% benchmark     │ │ Sharpe: 0.62 vs 0.58 │ │ Nov 2025        │ │
│  │                      │ │                      │ │                  │ │
│  │ [View Analysis →]    │ │ [View Backtest →]    │ │ [View Details →]│ │
│  └──────────────────────┘ └──────────────────────┘ └─────────────────┘ │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  ANALYSIS CATALOG                          [Filter ▼] [Sort: Recent ▼] │
│                                                                         │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐│
│  │ ┌─────────────────┐ │ │ ┌─────────────────┐ │ │ ┌─────────────────┐ ││
│  │ │  [Thumbnail]    │ │ │ │  [Thumbnail]    │ │ │ │  [Thumbnail]    │ ││
│  │ │  Chart Preview  │ │ │ │  Chart Preview  │ │ │ │  Chart Preview  │ ││
│  │ └─────────────────┘ │ │ └─────────────────┘ │ │ └─────────────────┘ ││
│  │                     │ │                     │ │                     ││
│  │ Investment Clock    │ │ SPY vs RETAILIRSA   │ │ SPY vs Industrial   ││
│  │ Sector Analysis     │ │                     │ │ Production          ││
│  │                     │ │ Contemporaneous,    │ │                     ││
│  │ ★ Primary Strategy  │ │ no predictive power │ │ Coincident, NBER    ││
│  │ Sharpe: 0.62        │ │ Regime filter only  │ │ Big Four indicator  ││
│  │                     │ │                     │ │                     ││
│  │ [Qualitative]       │ │ [Qualitative]       │ │ [Qualitative]       ││
│  │ [Correlation]       │ │ [Correlation]       │ │ [Correlation]       ││
│  │ [Lead-Lag]          │ │ [Lead-Lag]          │ │ [Lead-Lag]          ││
│  │ [Backtest]          │ │ [Backtest]          │ │ [Backtest]          ││
│  │                     │ │                     │ │                     ││
│  │ Updated: 2026-01-03 │ │ Updated: 2025-12-21 │ │ Updated: 2025-12-21 ││
│  └─────────────────────┘ └─────────────────────┘ └─────────────────────┘│
│                                                                         │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐│
│  │ ┌─────────────────┐ │ │ ┌─────────────────┐ │ │ ┌─────────────────┐ ││
│  │ │  [Thumbnail]    │ │ │ │  [Thumbnail]    │ │ │ │  [Thumbnail]    │ ││
│  │ └─────────────────┘ │ │ └─────────────────┘ │ │ └─────────────────┘ ││
│  │                     │ │                     │ │                     ││
│  │ XLRE vs Orders/Inv  │ │ Cass Freight Index  │ │ ML Regime Detection ││
│  │                     │ │ Shipment            │ │                     ││
│  │ Real estate leading │ │                     │ │ HMM vs GMM vs       ││
│  │ indicator potential │ │ Seasonality study   │ │ Rule-based comparison││
│  │                     │ │ Forecasting methods │ │                     ││
│  │ [View Full →]       │ │ [View Full →]       │ │ [View Full →]       ││
│  └─────────────────────┘ └─────────────────────┘ └─────────────────────┘│
│                                                                         │
│  [Load More Analyses...]                                                │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  PERFORMANCE SUMMARY                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  [Equity curve chart - Strategy vs Benchmark - last 5 years]    │   │
│  │                                                                  │   │
│  │  ████████████████████████████████████████████████████████████   │   │
│  │                                                                  │   │
│  │  Regime backgrounds: Recovery(green) Overheat(yellow) etc.      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │Total Ret │ │ CAGR     │ │ Max DD   │ │ Sortino  │ │ Calmar   │     │
│  │ +127.3%  │ │  8.2%    │ │ -12.2%   │ │  0.89    │ │  0.67    │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  RECENT ACTIVITY                                           [View All →] │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  📊 New analysis added: XLRE vs Orders/Inv     2 hours ago      │   │
│  │  🔄 Regime change detected: Overheat → ?       1 day ago        │   │
│  │  ✅ Backtest completed: Walk-Forward 60/12     3 days ago       │   │
│  │  📈 Signal update: XLE Long, XLU Short         5 days ago       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Specifications

### 3.1 Hero Section - Current Regime Display

```python
# src/dashboard/components/hero.py

from dash import html, dcc
import dash_bootstrap_components as dbc

def create_hero_section(current_regime_data, portfolio_metrics):
    """
    Create the hero section with current regime and key metrics.

    Args:
        current_regime_data: Dict with regime info
            - phase: str ('Recovery', 'Overheat', 'Stagflation', 'Reflation')
            - growth_direction: str ('Rising', 'Falling')
            - inflation_direction: str ('Rising', 'Falling')
            - start_date: datetime
            - duration_months: int
        portfolio_metrics: Dict with key metrics
    """

    # Regime color mapping
    regime_colors = {
        'Recovery': {'bg': '#27AE60', 'text': 'white'},
        'Overheat': {'bg': '#F39C12', 'text': 'black'},
        'Stagflation': {'bg': '#E74C3C', 'text': 'white'},
        'Reflation': {'bg': '#3498DB', 'text': 'white'}
    }

    phase = current_regime_data['phase']
    colors = regime_colors.get(phase, {'bg': '#95A5A6', 'text': 'white'})

    # Direction arrows
    growth_arrow = '↑' if current_regime_data['growth_direction'] == 'Rising' else '↓'
    inflation_arrow = '↑' if current_regime_data['inflation_direction'] == 'Rising' else '↓'

    hero = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                # Regime Display
                dbc.Col([
                    html.Div([
                        html.H2("Current Regime", className="text-muted mb-2"),
                        html.Div([
                            html.H1(
                                phase.upper(),
                                className="display-4 mb-0",
                                style={
                                    'backgroundColor': colors['bg'],
                                    'color': colors['text'],
                                    'padding': '10px 20px',
                                    'borderRadius': '8px',
                                    'display': 'inline-block'
                                }
                            )
                        ]),
                        html.P([
                            html.Span(f"Growth: {current_regime_data['growth_direction']} {growth_arrow}",
                                     className="me-4"),
                            html.Span(f"Inflation: {current_regime_data['inflation_direction']} {inflation_arrow}")
                        ], className="mt-3 mb-2 lead"),
                        html.P(
                            f"Since {current_regime_data['start_date'].strftime('%B %Y')} "
                            f"({current_regime_data['duration_months']} months)",
                            className="text-muted"
                        ),
                        html.Div([
                            dbc.Button("View Investment Clock", color="primary",
                                      href="/portfolio/investment-clock", className="me-2"),
                            dbc.Button("See Sector Allocation", color="outline-primary",
                                      href="/portfolio/sector-rotation")
                        ], className="mt-3")
                    ])
                ], width=6),

                # Key Metrics
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            _create_metric_card(
                                "Strategy Sharpe",
                                f"{portfolio_metrics['sharpe']:.2f}",
                                change=portfolio_metrics.get('sharpe_change'),
                                icon="📈"
                            )
                        ], width=6),
                        dbc.Col([
                            _create_metric_card(
                                "Best Sector",
                                portfolio_metrics['best_sector'],
                                subtitle=f"+{portfolio_metrics['best_sector_return']:.1f}%",
                                icon="🏆"
                            )
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            _create_metric_card(
                                "Win Rate (12M)",
                                f"{portfolio_metrics['win_rate']:.0f}%",
                                icon="🎯"
                            )
                        ], width=6),
                        dbc.Col([
                            _create_metric_card(
                                "Active Signals",
                                f"{portfolio_metrics['long_signals']}/{portfolio_metrics['total_sectors']}",
                                subtitle="Long",
                                icon="📊"
                            )
                        ], width=6)
                    ])
                ], width=6)
            ])
        ])
    ], className="mb-4 shadow-sm")

    return hero


def _create_metric_card(title, value, change=None, subtitle=None, icon=None):
    """Create a single metric card for the hero section."""

    change_element = None
    if change is not None:
        change_color = "success" if change > 0 else "danger" if change < 0 else "secondary"
        change_arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        change_element = dbc.Badge(
            f"{change_arrow} {abs(change):.2f}",
            color=change_color,
            className="ms-2"
        )

    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={'fontSize': '24px'}) if icon else None,
                html.Small(title, className="text-muted ms-2")
            ], className="d-flex align-items-center mb-2"),
            html.Div([
                html.H3(value, className="mb-0 d-inline"),
                change_element
            ]),
            html.Small(subtitle, className="text-muted") if subtitle else None
        ], className="py-2")
    ], className="h-100")
```

---

### 3.2 Key Insights Cards

```python
def create_key_insights_section(insights):
    """
    Create the key insights section with highlight cards.

    Args:
        insights: List of dicts with:
            - icon: str (emoji)
            - title: str
            - headline: str
            - detail: str
            - link: str (href)
            - link_text: str
    """

    insight_cards = []
    for insight in insights[:3]:  # Show top 3
        card = dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.Span(insight['icon'], style={'fontSize': '28px'}),
                    html.H5(insight['title'], className="ms-2 mb-0 d-inline")
                ], className="d-flex align-items-center mb-3"),
                html.H4(insight['headline'], className="mb-2"),
                html.P(insight['detail'], className="text-muted mb-3"),
                dbc.Button(
                    insight.get('link_text', 'View Analysis →'),
                    href=insight['link'],
                    color="link",
                    className="p-0"
                )
            ])
        ], className="h-100 shadow-sm hover-shadow")

        insight_cards.append(dbc.Col(card, width=4))

    return html.Div([
        html.Div([
            html.H4("Key Insights", className="mb-0"),
            dbc.Button("View All →", color="link", href="/insights")
        ], className="d-flex justify-content-between align-items-center mb-3"),
        dbc.Row(insight_cards)
    ], className="mb-4")


# Example usage
SAMPLE_INSIGHTS = [
    {
        'icon': '🎯',
        'title': 'Best Indicator Pair',
        'headline': 'Orders/Inv + PPI',
        'detail': '96.8% classification accuracy vs 66% benchmark',
        'link': '/analysis/investment-clock',
        'link_text': 'View Analysis →'
    },
    {
        'icon': '📈',
        'title': 'Top Finding',
        'headline': 'HMM Supervised Init',
        'detail': 'Beats rule-based approach. Sharpe: 0.62 vs 0.58',
        'link': '/analysis/ml-regime-detection',
        'link_text': 'View Backtest →'
    },
    {
        'icon': '⚠️',
        'title': 'Latest Alert',
        'headline': 'Regime Change Detected',
        'detail': 'Transition to Overheat phase in November 2025',
        'link': '/portfolio/investment-clock',
        'link_text': 'View Details →'
    }
]
```

---

### 3.3 Analysis Catalog Grid

```python
def create_analysis_catalog(analyses, filters=None):
    """
    Create the analysis catalog grid with filterable cards.

    Args:
        analyses: List of analysis metadata dicts
        filters: Current filter state
    """

    # Filter controls
    filter_bar = dbc.Row([
        dbc.Col([
            html.H4("Analysis Catalog", className="mb-0")
        ], width=6),
        dbc.Col([
            dbc.InputGroup([
                dbc.Select(
                    id='catalog-filter-type',
                    options=[
                        {'label': 'All Types', 'value': 'all'},
                        {'label': 'Indicator Studies', 'value': 'indicator'},
                        {'label': 'Sector Analysis', 'value': 'sector'},
                        {'label': 'ML Models', 'value': 'ml'},
                        {'label': 'Forecasting', 'value': 'forecast'}
                    ],
                    value='all',
                    className="me-2"
                ),
                dbc.Select(
                    id='catalog-sort',
                    options=[
                        {'label': 'Most Recent', 'value': 'recent'},
                        {'label': 'Highest Sharpe', 'value': 'sharpe'},
                        {'label': 'Most Significant', 'value': 'significance'},
                        {'label': 'Alphabetical', 'value': 'alpha'}
                    ],
                    value='recent'
                )
            ], size="sm")
        ], width=6, className="text-end")
    ], className="mb-4 align-items-center")

    # Analysis cards
    cards = []
    for analysis in analyses:
        card = _create_analysis_card(analysis)
        cards.append(dbc.Col(card, width=4, className="mb-4"))

    catalog_grid = dbc.Row(cards, id='catalog-grid')

    # Load more button
    load_more = html.Div([
        dbc.Button(
            "Load More Analyses...",
            id='load-more-analyses',
            color="outline-secondary",
            className="w-100"
        )
    ], className="text-center mt-3")

    return html.Div([
        filter_bar,
        catalog_grid,
        load_more
    ])


def _create_analysis_card(analysis):
    """
    Create a single analysis catalog card.

    Args:
        analysis: Dict with:
            - id: str (unique identifier)
            - title: str
            - subtitle: str
            - summary: str
            - thumbnail: str (path to preview image)
            - sharpe: float (optional)
            - is_primary: bool
            - tags: list of str
            - updated: datetime
            - sections: list of available sections
    """

    # Badge for primary/featured analyses
    badges = []
    if analysis.get('is_primary'):
        badges.append(dbc.Badge("★ Primary Strategy", color="warning", className="me-1"))
    if analysis.get('sharpe'):
        sharpe_color = "success" if analysis['sharpe'] > 0.5 else "warning" if analysis['sharpe'] > 0 else "danger"
        badges.append(dbc.Badge(f"Sharpe: {analysis['sharpe']:.2f}", color=sharpe_color))

    # Section quick links
    section_links = []
    section_icons = {
        'qualitative': '📖',
        'correlation': '📊',
        'leadlag': '⏱️',
        'regimes': '🔄',
        'backtest': '📈',
        'forecast': '🔮'
    }

    for section in analysis.get('sections', []):
        section_links.append(
            dbc.Button(
                f"{section_icons.get(section, '📄')} {section.title()}",
                href=f"/analysis/{analysis['id']}/{section}",
                color="light",
                size="sm",
                className="me-1 mb-1"
            )
        )

    card = dbc.Card([
        # Thumbnail preview
        html.Div([
            html.Img(
                src=analysis.get('thumbnail', '/assets/default_chart.png'),
                className="card-img-top",
                style={'height': '150px', 'objectFit': 'cover'}
            ),
            # Overlay with view button on hover
            html.Div([
                dbc.Button(
                    "View Full Analysis",
                    href=f"/analysis/{analysis['id']}",
                    color="primary"
                )
            ], className="card-img-overlay d-flex justify-content-center align-items-center",
               style={'backgroundColor': 'rgba(0,0,0,0.5)', 'opacity': '0'},
               id={'type': 'card-overlay', 'index': analysis['id']})
        ], className="position-relative"),

        dbc.CardBody([
            # Title and badges
            html.H5(analysis['title'], className="card-title mb-1"),
            html.Div(badges, className="mb-2") if badges else None,

            # Subtitle/summary
            html.P(analysis.get('subtitle', ''), className="card-subtitle text-muted mb-2"),
            html.P(analysis.get('summary', ''), className="card-text small"),

            # Section quick links
            html.Div(section_links, className="mb-2"),

            # Footer with date
            html.Small(
                f"Updated: {analysis['updated'].strftime('%Y-%m-%d')}",
                className="text-muted"
            )
        ])
    ], className="h-100 shadow-sm catalog-card")

    return card


# Example analysis data
SAMPLE_ANALYSES = [
    {
        'id': 'investment-clock-sectors',
        'title': 'Investment Clock Sector Analysis',
        'subtitle': '',
        'summary': '★ Primary Strategy. Sharpe: 0.62. Sector rotation across 4 economic phases.',
        'thumbnail': '/assets/thumbnails/investment_clock.png',
        'sharpe': 0.62,
        'is_primary': True,
        'tags': ['sector', 'regime', 'ml'],
        'updated': datetime(2026, 1, 3),
        'sections': ['qualitative', 'correlation', 'leadlag', 'regimes', 'backtest']
    },
    {
        'id': 'spy-retailirsa',
        'title': 'SPY vs RETAILIRSA',
        'subtitle': '',
        'summary': 'Contemporaneous relationship, no predictive power. Use as regime filter only.',
        'thumbnail': '/assets/thumbnails/spy_retailirsa.png',
        'sharpe': None,
        'is_primary': False,
        'tags': ['indicator', 'regime'],
        'updated': datetime(2025, 12, 21),
        'sections': ['qualitative', 'correlation', 'leadlag', 'regimes']
    },
    {
        'id': 'spy-indpro',
        'title': 'SPY vs Industrial Production',
        'subtitle': '',
        'summary': 'Coincident indicator. NBER Big Four. No predictive power.',
        'thumbnail': '/assets/thumbnails/spy_indpro.png',
        'sharpe': None,
        'is_primary': False,
        'tags': ['indicator'],
        'updated': datetime(2025, 12, 21),
        'sections': ['qualitative', 'correlation', 'leadlag']
    }
]
```

---

### 3.4 Performance Summary Chart

```python
def create_performance_summary(performance_data):
    """
    Create the performance summary section with equity curve.

    Args:
        performance_data: DataFrame with:
            - date index
            - strategy_cumret: cumulative strategy returns
            - benchmark_cumret: cumulative benchmark returns
            - regime: current regime at each point
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Add regime background coloring
    regime_colors = {
        'Recovery': 'rgba(46, 204, 113, 0.2)',
        'Overheat': 'rgba(241, 196, 15, 0.2)',
        'Stagflation': 'rgba(231, 76, 60, 0.2)',
        'Reflation': 'rgba(52, 152, 219, 0.2)'
    }

    # Find regime change points
    regime_changes = performance_data['regime'] != performance_data['regime'].shift(1)
    change_points = performance_data.index[regime_changes].tolist()
    change_points.append(performance_data.index[-1])

    for i in range(len(change_points) - 1):
        start = change_points[i]
        end = change_points[i + 1]
        regime = performance_data.loc[start, 'regime']

        if regime in regime_colors:
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=regime_colors[regime],
                layer='below',
                line_width=0
            )

    # Strategy line
    fig.add_trace(go.Scatter(
        x=performance_data.index,
        y=performance_data['strategy_cumret'] * 100,
        name='Strategy',
        line=dict(color='#2E86AB', width=2.5),
        hovertemplate='<b>%{x|%Y-%m}</b><br>Strategy: %{y:.1f}%<extra></extra>'
    ))

    # Benchmark line
    fig.add_trace(go.Scatter(
        x=performance_data.index,
        y=performance_data['benchmark_cumret'] * 100,
        name='Benchmark (SPY)',
        line=dict(color='#95A5A6', width=1.5, dash='dash'),
        hovertemplate='<b>%{x|%Y-%m}</b><br>Benchmark: %{y:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title=None,
        height=350,
        margin=dict(l=40, r=40, t=20, b=40),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            type="date"
        ),
        yaxis=dict(title="Cumulative Return (%)")
    )

    # Enable crosshair
    fig.update_xaxes(showspikes=True, spikemode='across', spikethickness=1)
    fig.update_yaxes(showspikes=True, spikemode='across', spikethickness=1)

    return html.Div([
        html.H4("Performance Summary", className="mb-3"),
        dcc.Graph(
            id='performance-summary-chart',
            figure=fig,
            config={'scrollZoom': True, 'displayModeBar': True}
        ),
        _create_performance_metrics_row(performance_data)
    ])


def _create_performance_metrics_row(performance_data):
    """Create the performance metrics row below the chart."""

    # Calculate metrics
    strategy_ret = performance_data['strategy_cumret'].iloc[-1]
    years = (performance_data.index[-1] - performance_data.index[0]).days / 365.25
    cagr = (1 + strategy_ret) ** (1/years) - 1

    # Simple drawdown calculation
    cummax = (1 + performance_data['strategy_cumret']).cummax()
    drawdown = (1 + performance_data['strategy_cumret']) / cummax - 1
    max_dd = drawdown.min()

    metrics = [
        {'title': 'Total Return', 'value': f"+{strategy_ret*100:.1f}%"},
        {'title': 'CAGR', 'value': f"{cagr*100:.1f}%"},
        {'title': 'Max Drawdown', 'value': f"{max_dd*100:.1f}%"},
        {'title': 'Sortino', 'value': "0.89"},  # Would calculate from data
        {'title': 'Calmar', 'value': f"{cagr/abs(max_dd):.2f}"}
    ]

    cards = []
    for m in metrics:
        cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Small(m['title'], className="text-muted d-block"),
                        html.H5(m['value'], className="mb-0")
                    ], className="text-center py-2")
                ])
            ], width=True)
        )

    return dbc.Row(cards, className="mt-3")
```

---

### 3.5 Recent Activity Feed

```python
def create_activity_feed(activities, limit=5):
    """
    Create the recent activity feed.

    Args:
        activities: List of activity dicts with:
            - type: str ('analysis', 'regime', 'backtest', 'signal')
            - message: str
            - timestamp: datetime
            - link: str (optional)
    """

    type_icons = {
        'analysis': '📊',
        'regime': '🔄',
        'backtest': '✅',
        'signal': '📈',
        'alert': '⚠️'
    }

    items = []
    for activity in activities[:limit]:
        icon = type_icons.get(activity['type'], '📌')

        # Time ago calculation
        delta = datetime.now() - activity['timestamp']
        if delta.days > 0:
            time_ago = f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
        elif delta.seconds > 3600:
            hours = delta.seconds // 3600
            time_ago = f"{hours} hour{'s' if hours > 1 else ''} ago"
        else:
            minutes = delta.seconds // 60
            time_ago = f"{minutes} minute{'s' if minutes > 1 else ''} ago"

        item = dbc.ListGroupItem([
            html.Div([
                html.Span(icon, className="me-2"),
                html.Span(activity['message']),
                html.Small(time_ago, className="text-muted float-end")
            ])
        ], action=True, href=activity.get('link'))

        items.append(item)

    return html.Div([
        html.Div([
            html.H4("Recent Activity", className="mb-0"),
            dbc.Button("View All →", color="link", href="/activity")
        ], className="d-flex justify-content-between align-items-center mb-3"),
        dbc.ListGroup(items)
    ])


# Example activities
SAMPLE_ACTIVITIES = [
    {
        'type': 'analysis',
        'message': 'New analysis added: XLRE vs Orders/Inv',
        'timestamp': datetime.now() - timedelta(hours=2),
        'link': '/analysis/xlre-orders-inv'
    },
    {
        'type': 'regime',
        'message': 'Regime change detected: Overheat → ?',
        'timestamp': datetime.now() - timedelta(days=1),
        'link': '/portfolio/investment-clock'
    },
    {
        'type': 'backtest',
        'message': 'Backtest completed: Walk-Forward 60/12',
        'timestamp': datetime.now() - timedelta(days=3),
        'link': '/analysis/investment-clock-sectors/backtest'
    },
    {
        'type': 'signal',
        'message': 'Signal update: XLE Long, XLU Short',
        'timestamp': datetime.now() - timedelta(days=5),
        'link': '/portfolio/signals'
    }
]
```

---

## 4. Complete Landing Page Layout

```python
# src/dashboard/layouts/landing.py

from dash import html, dcc
import dash_bootstrap_components as dbc
from ..components.hero import create_hero_section
from ..components.catalog import create_analysis_catalog, create_key_insights_section
from ..components.performance import create_performance_summary
from ..components.activity import create_activity_feed

def create_landing_layout(data):
    """
    Create the complete landing page layout.

    Args:
        data: Dict containing all required data:
            - current_regime: regime data
            - portfolio_metrics: performance metrics
            - insights: key insights list
            - analyses: analysis catalog list
            - performance: performance history DataFrame
            - activities: recent activities list
    """

    return dbc.Container([
        # Hero Section
        create_hero_section(
            data['current_regime'],
            data['portfolio_metrics']
        ),

        html.Hr(className="my-4"),

        # Key Insights
        create_key_insights_section(data['insights']),

        html.Hr(className="my-4"),

        # Analysis Catalog
        create_analysis_catalog(data['analyses']),

        html.Hr(className="my-4"),

        # Performance Summary
        create_performance_summary(data['performance']),

        html.Hr(className="my-4"),

        # Recent Activity
        create_activity_feed(data['activities']),

        # Footer spacing
        html.Div(className="mb-5")

    ], fluid=True, className="py-4")
```

---

## 5. Styling

### 5.1 Custom CSS

```css
/* assets/style.css */

/* Card hover effects */
.catalog-card {
    transition: transform 0.2s, box-shadow 0.2s;
}

.catalog-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15) !important;
}

.catalog-card:hover .card-img-overlay {
    opacity: 1 !important;
    transition: opacity 0.3s;
}

/* Hero section */
.hero-regime-badge {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}

/* Insight cards */
.hover-shadow:hover {
    box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
}

/* Activity feed */
.list-group-item-action:hover {
    background-color: #f8f9fa;
}

/* Performance chart */
.js-plotly-plot .plotly .modebar {
    opacity: 0;
    transition: opacity 0.3s;
}

.js-plotly-plot:hover .plotly .modebar {
    opacity: 1;
}

/* Filter bar */
.filter-bar {
    position: sticky;
    top: 56px;
    background: white;
    z-index: 100;
    padding: 10px 0;
    border-bottom: 1px solid #dee2e6;
}

/* Section headers */
.section-header {
    border-left: 4px solid #2E86AB;
    padding-left: 12px;
}
```

---

## 6. Data Requirements

### 6.1 Landing Page Data Schema

```python
# Data structure for landing page

LANDING_PAGE_DATA = {
    # Current regime (updated daily/weekly)
    'current_regime': {
        'phase': 'Overheat',
        'growth_direction': 'Rising',
        'inflation_direction': 'Rising',
        'start_date': datetime(2025, 11, 1),
        'duration_months': 3,
        'confidence': 0.87
    },

    # Portfolio metrics (updated daily)
    'portfolio_metrics': {
        'sharpe': 0.62,
        'sharpe_change': 0.04,  # vs previous period
        'cagr': 0.082,
        'max_drawdown': -0.122,
        'win_rate': 0.58,
        'best_sector': 'XLE',
        'best_sector_return': 12.3,
        'long_signals': 4,
        'short_signals': 3,
        'total_sectors': 11
    },

    # Key insights (curated manually or auto-generated)
    'insights': [
        {
            'icon': '🎯',
            'title': 'Best Indicator Pair',
            'headline': 'Orders/Inv + PPI',
            'detail': '96.8% classification accuracy vs 66% benchmark',
            'link': '/analysis/investment-clock'
        },
        # ... more insights
    ],

    # Analysis catalog (from database/filesystem)
    'analyses': [
        {
            'id': 'investment-clock-sectors',
            'title': 'Investment Clock Sector Analysis',
            'summary': 'Primary strategy...',
            'sharpe': 0.62,
            'is_primary': True,
            'updated': datetime(2026, 1, 3),
            'sections': ['qualitative', 'correlation', 'leadlag', 'regimes', 'backtest']
        },
        # ... more analyses
    ],

    # Performance history (from parquet/database)
    'performance': pd.DataFrame({
        'strategy_cumret': [...],
        'benchmark_cumret': [...],
        'regime': [...]
    }, index=pd.DatetimeIndex([...])),

    # Recent activities (from activity log)
    'activities': [
        {
            'type': 'analysis',
            'message': 'New analysis added...',
            'timestamp': datetime.now() - timedelta(hours=2),
            'link': '/analysis/...'
        },
        # ... more activities
    ]
}
```

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-24 | RA Cheryl | Initial portal landing design |
