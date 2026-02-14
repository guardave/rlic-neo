# XLI vs ISM Services PMI (Non-Manufacturing)

## What is ISM Services PMI?

The **ISM Services PMI** (also known as Non-Manufacturing PMI) is a monthly survey of purchasing
managers at services-sector companies. Published by the Institute for Supply Management on the
**3rd business day of each month** (2 days after Manufacturing PMI).

**Key characteristics:**
- Covers **~80% of GDP** (services dominate the US economy)
- Surveys sectors like healthcare, finance, retail, transportation, professional services
- Composite index from 4 subindices: Business Activity, New Orders, Employment, Supplier Deliveries
- **PMI > 50** = services sector expansion; **PMI <= 50** = contraction

## Economic Signal Interpretation

| PMI Range | Signal | Economic Meaning |
|---|---|---|
| > 55 | Strong Expansion | Services sector growing robustly, broad-based expansion |
| 50-55 | Moderate Growth | Growth continuing but at a slower pace |
| < 50 | Contraction | Services sector shrinking, potential recession risk |
| < 45 | Severe Contraction | Deep contraction, typically recession territory |

## Why Compare with XLI (Industrials)?

**Hypothesis**: Services PMI may influence Industrials because:
1. **Input-output linkages**: Services firms purchase manufactured goods
2. **GDP proxy**: Services = ~80% of GDP, so broad health affects all sectors
3. **Employment signal**: Services employment trends affect consumer spending on industrial goods

**Counter-hypothesis**: The relationship may be weak because:
- XLI holds **manufacturing** companies, not services
- Manufacturing PMI is a more direct indicator for XLI
- Services PMI may be better suited for consumer/financial sector ETFs

## Key Finding

::: warning
**CONFIRMATORY, NOT PREDICTIVE**: ISM Services PMI does NOT predict XLI returns.

All 8 significant lags are **negative** (lag -1 through -11), meaning XLI returns
move FIRST, then Services PMI follows 1-11 months later.

**Best lag: -1 month** (r=0.317, p<0.0001) — XLI leads Services PMI by 1 month.
:::

**Despite reverse causality, regime filtering is useful:**
- **Svc Expansion** (PMI > 50): Mean +1.05%/mo, Sharpe 0.79
- **Svc Contraction** (PMI <= 50): Mean -3.94%/mo, Sharpe -1.38

The regime difference is highly significant (p<0.0001).

## Data Limitation

::: info
**Data Period**: December 1999 to April 2020 (245 months)

The ISM Services data was assembled from historical sources (forecasts.org) and
hardcoded ISM press release values. Recent data (2020-05+) is not available because
FRED discontinued the NMFBAI series and alternative web sources returned errors.
:::

## Comparison with Manufacturing PMI

| Metric | Manufacturing | Services |
|---|---|---|
| Best Lag | -4 | -1 |
| Best r | 0.241 | 0.317 |
| Significant Lags | 11 | 8 |
| Direction | All negative | All negative |
| Expansion Sharpe | 0.93 | 0.79 |
| Contraction Sharpe | -0.12 | -1.38 |
| Observations | 314 | 245 |

**Key differences:**
- Services PMI shows **stronger concurrent correlation** (r=0.317 at lag -1 vs r=0.241 at lag -4)
- Both are confirmatory (all significant lags negative)
- Services has **more extreme contraction penalty** (Sharpe -1.38 vs -0.12)
- Manufacturing has more data (314 vs 245 observations)
