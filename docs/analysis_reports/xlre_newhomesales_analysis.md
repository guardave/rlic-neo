# XLRE vs NewHomeSales Analysis

## Overview

This analysis explores the relationship between XLRE (Real Estate Select Sector ETF) and NewHomeSales.

**Data Period**: October 2016 to October 2025 (~109 months)

**Analysis conducted following SOP v1.2** with random_seed=42 for reproducibility.

---

## Qualitative Analysis

### New Home Sales as Leading Indicator

**New Home Sales** measures the number of newly constructed homes sold each month (FRED: HSN1F). It is a leading indicator because:

1. **Forward-Looking**: New home sales require mortgage applications, credit checks, and planning - all occurring before actual purchase
2. **Construction Pipeline**: Sales drive future construction activity and employment
3. **Consumer Confidence**: New home purchases are major financial decisions reflecting consumer outlook
4. **Real Estate Connection**: Directly impacts XLRE holdings (residential REITs, home-related companies)

**Economic Rationale for XLRE Relationship:**
- Rising new home sales → increased real estate activity → positive for XLRE
- Home sales lead construction spending by 2-6 months
- Strong home sales indicate healthy housing demand benefiting real estate sector

---

## Key Findings Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Returns Correlation | 0.059 | Weak |
| P-value | 0.541 | Not significant |
| Optimal Lag | 0 months | Concurrent |
| Regime Difference | Not significant | p=nan |

**Fast-Fail**: Yes - relationship too weak for practical use

---

## Correlation Analysis

- **NewHomeSales_MoM_vs_XLRE_Returns**: r=0.059, p=0.541, n=109
- **NewHomeSales_QoQ_vs_XLRE_Returns**: r=0.055, p=0.573, n=109
- **NewHomeSales_YoY_vs_XLRE_Returns**: r=0.051, p=0.595, n=109

---

## Lead-Lag Analysis

| Lag (months) | Correlation | P-value | Significant |
|--------------|-------------|---------|-------------|

**Optimal Lag**: 0 months (r=0.059, p=0.541)

---

## Regime Analysis


**Regime Difference Test**: p=nan (Not significant)

---

## Backtest Results

| Metric | Strategy | Benchmark |
|--------|----------|-----------|
| Sharpe Ratio | 0.00 | 0.00 |
| Total Return | 0.0% | 0.0% |
| Annualized Return | 0.0% | 0.0% |

**Strategy**: Long XLRE when NewHomeSales is falling (YoY), cash otherwise.

---

## Conclusion

**No meaningful relationship found between NewHomeSales and XLRE**

### Practical Implications

1. **Do NOT use NewHomeSales as a trading signal for XLRE**
2. The relationship is too weak to provide actionable information
3. Consider other housing indicators or sector-specific data

---

## Files Created

| File | Description |
|------|-------------|
| `data/xlre_newhomesales.parquet` | Analysis data |
| `docs/analysis_reports/xlre_newhomesales_analysis.md` | This document |

---

## Appendix: Audit Trail

- **Analysis Date**: 2026-01-26 22:15:45
- **Random Seed**: 42
- **SOP Version**: 1.2
- **Data Period**: 2016-10-31 to 2025-10-31
- **Observations**: 109
