# XLRE vs BuildingPermits Analysis

## Overview

This analysis explores the relationship between XLRE (Real Estate Select Sector ETF) and BuildingPermits.

**Data Period**: October 2016 to August 2025 (~107 months)

**Analysis conducted following SOP v1.2** with random_seed=42 for reproducibility.

---

## Qualitative Analysis

### BuildingPermits as Leading Indicator

**Building Permits** measures the number of new privately-owned housing units authorized by building permits. It is a leading indicator because:

1. **Most Forward-Looking Housing Indicator**: Permits precede construction starts by weeks/months
2. **Developer Confidence**: Reflects builder outlook on future demand
3. **Pipeline Indicator**: Strong permits signal robust future housing supply
4. **Planning Horizon**: Permits obtained 3-6 months before construction completes

**Economic Rationale for XLRE Relationship:**
- Rising permits → future construction → real estate sector growth → positive for XLRE
- Permits lead new home sales by 2-4 months
- Strong permit activity indicates expanding real estate market

---

## Key Findings Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Returns Correlation | 0.033 | Weak |
| P-value | 0.735 | Not significant |
| Optimal Lag | 0 months | Concurrent |
| Regime Difference | Not significant | p=nan |

**Fast-Fail**: Yes - relationship too weak for practical use

---

## Correlation Analysis

- **BuildingPermits_MoM_vs_XLRE_Returns**: r=0.033, p=0.735, n=107
- **BuildingPermits_QoQ_vs_XLRE_Returns**: r=-0.036, p=0.714, n=107
- **BuildingPermits_YoY_vs_XLRE_Returns**: r=0.075, p=0.441, n=107

---

## Lead-Lag Analysis

| Lag (months) | Correlation | P-value | Significant |
|--------------|-------------|---------|-------------|

**Optimal Lag**: 0 months (r=0.075, p=0.441)

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

**Strategy**: Long XLRE when BuildingPermits is falling (YoY), cash otherwise.

---

## Conclusion

**No meaningful relationship found between BuildingPermits and XLRE**

### Practical Implications

1. **Do NOT use BuildingPermits as a trading signal for XLRE**
2. The relationship is too weak to provide actionable information
3. Consider other housing indicators or sector-specific data

---

## Files Created

| File | Description |
|------|-------------|
| `data/xlre_buildingpermits.parquet` | Analysis data |
| `docs/analysis_reports/xlre_buildingpermits_analysis.md` | This document |

---

## Appendix: Audit Trail

- **Analysis Date**: 2026-01-26 22:15:45
- **Random Seed**: 42
- **SOP Version**: 1.2
- **Data Period**: 2016-10-31 to 2025-08-31
- **Observations**: 107
