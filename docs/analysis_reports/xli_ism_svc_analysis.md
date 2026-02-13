# XLI vs ISM Services PMI Analysis Report

## Overview

| Field | Value |
|-------|-------|
| **Analysis** | XLI (Industrials ETF) vs ISM Services PMI (Non-Manufacturing) |
| **Data Period** | December 1999 to April 2020 (245 months) |
| **SOP Version** | 1.3 |
| **Random Seed** | 42 |
| **Data Sources** | forecasts.org (historical), ISM press releases (2014-09 to 2020-04) |
| **Note** | FRED NMFBAI/NMFCI discontinued; DBnomics/ycharts returned 404 |

## Phase 0: Data Assembly

ISM Services PMI data was assembled from multiple sources:
- **Historical (1997-2014)**: forecasts.org archived ISM Non-Manufacturing data
- **Recent (2014-09 to 2020-04)**: Hardcoded values from ISM press releases
- **Post-2020**: Not available (FRED discontinued NMFBAI series, alternative web sources returned errors)

Final dataset: 245 monthly observations from 1999-12-31 to 2020-04-30.

## Phase 2: Concurrent Correlation

| Indicator Variant | r | p-value | Significant? |
|-------------------|------|---------|-------------|
| ISM_Svc_PMI_Level | 0.124 | 0.052 | NO |
| ISM_Svc_PMI_MoM | 0.160 | 0.012 | YES |
| ISM_Svc_PMI_QoQ | 0.064 | 0.319 | NO |
| ISM_Svc_PMI_YoY | 0.094 | 0.143 | NO |
| ISM_Svc_PMI_Direction | 0.027 | 0.677 | NO |

Only MoM change is significant at the concurrent level, with a weak positive correlation (r=0.160).

## Phase 3: Lead-Lag Analysis (-18 to +18)

- **Total significant lags**: 8
- **Direction of all significant lags**: ALL NEGATIVE (lag -1, -2, -3, -4, -5, -6, -9, -11)
- **No positive lags significant**

| Lag | r | p-value |
|-----|-------|---------|
| -1 | 0.317 | <0.0001 |
| -2 | 0.268 | <0.0001 |
| -3 | 0.267 | <0.0001 |
| -4 | 0.244 | <0.001 |
| -5 | 0.211 | <0.01 |
| -6 | 0.180 | <0.01 |
| -9 | 0.145 | <0.05 |
| -11 | 0.136 | <0.05 |

**Interpretation**: XLI returns lead ISM Services PMI by 1-11 months. This is **reverse causality** -- the stock market moves first, then Services PMI follows. ISM Services PMI has NO predictive power for XLI returns.

## Phase 4: Regime Analysis

Regime definition: PMI > 50 = Svc Expansion, PMI <= 50 = Svc Contraction

| Regime | Mean Return (%/mo) | Std Dev (%) | Sharpe | n |
|--------|-------------------|-------------|--------|---|
| Svc Expansion | +1.05 | 4.57 | 0.79 | 223 |
| Svc Contraction | -3.94 | 9.91 | -1.38 | 21 |

- **t-test**: p < 0.0001 (highly significant regime difference)
- Contraction periods are rare (21/245 = 8.6%) but extremely damaging
- Contraction Sharpe of -1.38 indicates severe risk during services contraction

## Phase 5: Backtest

| Metric | Strategy | Benchmark |
|--------|----------|-----------|
| Sharpe Ratio | 0.53 | 0.40 |
| Cumulative Return | 3.15x | 2.16x |
| Exposure | ~91% | 100% |

Strategy: Long XLI during Svc Expansion (PMI > 50), flat during Svc Contraction.

## Phase 6: Conclusion

### PARTIALLY NEGATIVE RESULT: Confirmatory, Not Predictive

1. **Reverse causality confirmed**: All 8 significant lags are negative, meaning XLI returns move FIRST, then ISM Services PMI follows 1-11 months later.

2. **No predictive power**: ISM Services PMI cannot be used as a leading indicator for XLI returns.

3. **Regime filtering useful for risk management**: Despite lack of predictive power, the PMI > 50 / PMI <= 50 regime split shows highly significant performance differences (p<0.0001). Avoiding Services contraction periods would have prevented severe drawdowns.

4. **Services PMI shows more extreme contraction penalty than Manufacturing**: Contraction Sharpe of -1.38 (Services) vs -0.12 (Manufacturing), though Services contraction periods are rarer (21 vs ~30 months).

### Comparison with Manufacturing PMI

| Metric | Manufacturing | Services |
|--------|--------------|----------|
| Best Lag | -4 | -1 |
| Best r | 0.241 | 0.317 |
| Significant Lags | 11 | 8 |
| Direction | All negative | All negative |
| Expansion Sharpe | 0.93 | 0.79 |
| Contraction Sharpe | -0.12 | -1.38 |
| Observations | 314 | 245 |

Key differences:
- Services PMI shows stronger concurrent correlation (r=0.317 at lag -1 vs r=0.241 at lag -4)
- Both are confirmatory (all significant lags negative)
- Services has more extreme contraction penalty (Sharpe -1.38 vs -0.12)
- Manufacturing has more data (314 vs 245 observations)

## Phase 7: Files Created

| File | Description |
|------|-------------|
| `data/xli_ism_svc_full.parquet` | 245 rows x 12 cols - Full analysis dataset |
| `data/xli_ism_svc_leadlag.parquet` | 37 rows - Lead-lag correlation results |
| `data/xli_ism_svc_correlation.parquet` | 5 rows - Concurrent correlation results |
| `data/xli_ism_svc_regimes.parquet` | 2 rows - Regime performance summary |
| `script/analyze_xli_ism_svc.py` | Analysis script |
| `docs/analysis_reports/xli_ism_svc_analysis.md` | This report |

## Audit Trail

| Field | Value |
|-------|-------|
| Analysis Date | 2026-02-13 |
| Random Seed | 42 |
| SOP Version | 1.3 |
| Data Period | 1999-12-31 to 2020-04-30 |
| Observations | 245 |
| Lead-Lag Range | -18 to +18 months |
| Data Sources | forecasts.org (historical), ISM press releases (2014-09 to 2020-04) |
