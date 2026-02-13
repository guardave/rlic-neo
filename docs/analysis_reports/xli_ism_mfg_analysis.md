# XLI vs ISM Manufacturing PMI Analysis

## Overview

This analysis explores the relationship between XLI (Industrial Select Sector ETF) and ISM Manufacturing PMI.

**Data Period**: December 1999 to January 2026 (314 months)

**Analysis conducted following SOP v1.3** with random_seed=42 for reproducibility.

**Data Source Note**: The FRED NAPM series is DISCONTINUED. ISM Manufacturing PMI data was assembled from 4 alternative sources: forecasts.org (historical), ISM press releases (2014-09 to 2020-04), DBnomics ISM/pmi/pm, and ycharts.

---

## Qualitative Analysis

### ISM Manufacturing PMI as Potential Leading Indicator

**ISM Manufacturing PMI** (formerly NAPM) measures the prevailing direction of economic trends in the manufacturing sector. It is a survey-based diffusion index where:

1. **PMI > 50**: Manufacturing sector is expanding
2. **PMI = 50**: No change from prior month
3. **PMI < 50**: Manufacturing sector is contracting

**Hypothesis for XLI Relationship:**
- Rising manufacturing activity (PMI above 50, trending up) should benefit industrial companies
- PMI is a widely-watched leading indicator for the broader economy
- If PMI leads XLI, it would be actionable for trade timing

**Economic Rationale:**
- Higher manufacturing PMI signals rising orders, production, and employment
- Industrial companies (XLI constituents) directly benefit from manufacturing expansion
- PMI is released on the first business day of each month (timely)

---

## Key Findings Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Concurrent Correlation (Level, lag=0) | 0.114 | Weak but significant |
| Best Concurrent Transform | QoQ (r=0.169) | Weak but significant |
| **Best Lead-Lag Correlation** | **0.241 at lag -4** | **Moderate** |
| **P-value at Best Lag** | **0.000018** | **Highly Significant** |
| Direction of Causality | **Reverse** | **XLI leads PMI, NOT the other way** |
| Number of Significant Positive Lags | **0** | **No predictive power** |

**Actionable for Trade Timing**: NO - ISM Mfg PMI does NOT predict XLI returns

**Actionable for Risk Management**: YES - PMI > 50 regime filtering improves risk-adjusted returns

---

## Correlation Analysis (Concurrent)

| Transform | Correlation | P-value | N | Significant |
|-----------|-------------|---------|---|-------------|
| ISM_Mfg_PMI_Level vs XLI_Returns | 0.114 | 0.043 | 314 | YES |
| ISM_Mfg_PMI_MoM vs XLI_Returns | 0.167 | 0.003 | 314 | YES |
| ISM_Mfg_PMI_QoQ vs XLI_Returns | 0.169 | 0.003 | 314 | YES |
| ISM_Mfg_PMI_YoY vs XLI_Returns | 0.111 | 0.049 | 314 | YES |
| ISM_Mfg_PMI_Direction vs XLI_Returns | 0.102 | 0.071 | 314 | NO |

Concurrent correlations are statistically significant for most transforms but weak in magnitude (r < 0.2). The Direction transform narrowly misses significance (p=0.071).

---

## Lead-Lag Analysis (Extended Range: -18 to +18 months)

### Critical Finding: ALL Significant Lags Are Negative

Out of the 37 lags tested (-18 to +18), **11 lags are significant** and **ALL are negative or zero**. A negative lag means XLI returns LEAD ISM Mfg PMI changes -- the opposite of the desired predictive relationship.

### Significant Lags Table

| Lag (months) | Correlation | P-value | Significant |
|--------------|-------------|---------|-------------|
| -11 | 0.116 | 0.043 | YES |
| -9 | 0.131 | 0.022 | YES |
| -8 | 0.136 | 0.018 | YES |
| -7 | 0.168 | 0.003 | YES |
| -6 | 0.172 | 0.002 | YES |
| -5 | 0.173 | 0.002 | YES |
| **-4** | **0.241** | **0.00002** | **YES** |
| **-3** | **0.233** | **0.00003** | **YES** |
| **-2** | **0.241** | **0.00002** | **YES** |
| -1 | 0.204 | 0.0003 | YES |
| 0 | 0.114 | 0.043 | YES |

### Predictive Lags (Positive Lags: Indicator Would Lead XLI)

**NONE are significant.** No positive lag from +1 to +18 achieves p < 0.05.

This conclusively shows ISM Mfg PMI has **no predictive power** for XLI returns.

---

## Economic Interpretation

### Why XLI Leads PMI (Reverse Causality)

The finding that XLI leads ISM Mfg PMI by 1-4 months is consistent with well-established financial theory:

1. **Markets Are Forward-Looking**: Stock prices reflect expected future conditions, including manufacturing activity
2. **Information Efficiency**: Equity markets aggregate information from thousands of participants faster than survey-based indicators
3. **PMI Is Backward-Looking**: Despite being called a "leading indicator" for GDP, PMI reflects current/recent conditions reported by purchasing managers
4. **Industrial Stocks Anticipate Orders**: XLI components (Caterpillar, 3M, Honeywell, etc.) see their stock prices move on forward order expectations before PMI surveys capture the change

### Peak at Lag -4 and -2

The strongest correlations at lag -4 and lag -2 suggest:
- Stock market returns from 2-4 months ago are the best predictor of current PMI readings
- This aligns with the typical 1-3 month delay between economic expectations (priced into stocks) and survey responses (captured in PMI)

### Why This Is Not Surprising

ISM Manufacturing PMI is often cited as a leading indicator for GDP, but this does not mean it leads equity markets. The stock market is itself a leading indicator, and typically leads survey-based indicators by several months.

---

## Regime Analysis

### Manufacturing Expansion vs Contraction (PMI > 50 Threshold)

| Regime | Mean Return (Monthly) | Std Dev | Sharpe (Annualized) | Count |
|--------|----------------------|---------|---------------------|-------|
| Mfg Expansion (PMI > 50) | +1.34% | 4.99% | 0.93 | 214 |
| Mfg Contraction (PMI <= 50) | -0.22% | 6.15% | -0.12 | 96 |

### Regime Analysis Interpretation

Despite the lack of predictive lead-lag relationship, the regime segmentation IS statistically meaningful:

1. **Large Sharpe Differential**: 0.93 vs -0.12 is a substantial gap
2. **Return Difference**: +1.34% vs -0.22% per month (1.56 percentage points)
3. **Volatility Difference**: Contraction regimes have higher volatility (6.15% vs 4.99%)
4. **Practical Significance**: Avoiding contraction periods would improve risk-adjusted returns

### Backtest Results (PMI > 50 Regime Filter)

| Metric | Strategy (Long Expansion Only) | Benchmark (Buy-and-Hold XLI) |
|--------|-------------------------------|------------------------------|
| Final Cumulative Return | 11.69x | 7.89x |
| Annualized Sharpe Ratio | 0.767 | 0.559 |

**Important Caveat**: This strategy uses the CURRENT PMI reading (lagged 1 month for execution feasibility). It does NOT rely on PMI predicting future returns -- it simply avoids holding XLI during confirmed contraction periods.

---

## Conclusion

### PARTIALLY NEGATIVE RESULT: ISM Mfg PMI Is Confirmatory, Not Predictive

### Key Findings

1. **No Predictive Power**: Zero significant positive lags from +1 to +18 months
2. **Reverse Causality Confirmed**: All 11 significant lags are negative (XLI leads PMI by 1-11 months)
3. **Peak Reverse Correlation**: r=0.241 at lag -4 and lag -2 (XLI returns predict PMI 2-4 months later)
4. **Concurrent Correlations Weak**: r=0.11 to 0.17, statistically significant but low magnitude

### What IS Useful

1. **Regime Filter**: PMI > 50 regime yields Sharpe 0.93 vs -0.12 during contraction
2. **Risk Management**: Avoiding XLI during ISM Mfg contraction improves risk-adjusted returns
3. **Confirmation Signal**: PMI confirms what markets have already priced in

### Practical Recommendations

1. **DO NOT USE ISM Mfg PMI as a trade timing signal for XLI** -- it has no predictive power
2. **DO USE PMI > 50 as a risk filter** -- reduce/avoid XLI exposure during manufacturing contraction
3. **Combine with genuinely leading indicators** for a complete signal (e.g., Orders/Inventories Ratio)
4. **Recognize market efficiency**: Equity markets lead survey-based indicators, not the other way around

### Comparison with Other Indicators

| Indicator | Relationship to Sector ETF | Predictive? | Best Lag |
|-----------|---------------------------|-------------|----------|
| New Home Sales vs XLRE | Positive | YES (+8 months) | +8 |
| ISM Mfg PMI vs XLI | Reverse | NO (all lags negative) | -4 (reverse) |

This result reinforces that not all economic indicators lead equity markets. The market's own price action is often the first signal.

---

## Files Created

| File | Description |
|------|-------------|
| `data/xli_ism_mfg_full.parquet` | Full analysis dataset (314 rows x 12 cols) |
| `data/xli_ism_mfg_leadlag.parquet` | Lead-lag correlation results (37 rows) |
| `data/xli_ism_mfg_correlation.parquet` | Concurrent correlation results (5 rows) |
| `data/xli_ism_mfg_regimes.parquet` | Regime analysis results (2 rows) |
| `script/analyze_xli_ism_mfg.py` | Analysis script |
| `docs/analysis_reports/xli_ism_mfg_analysis.md` | This report |

---

## Appendix: Audit Trail

- **Analysis Date**: 2026-02-13
- **Random Seed**: 42
- **SOP Version**: 1.3
- **Data Period**: 1999-12-31 to 2026-01-31
- **Observations**: 314
- **Lead-Lag Range**: -18 to +18 months
- **Data Sources**: forecasts.org (historical), ISM press releases (2014-09 to 2020-04), DBnomics ISM/pmi/pm, ycharts
- **FRED NAPM Status**: Discontinued (alternative sources assembled)
