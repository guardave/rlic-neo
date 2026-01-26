# XLRE vs Building Permits Analysis

## Overview

This analysis explores the relationship between XLRE (Real Estate Select Sector ETF) and Building Permits.

**Data Period**: October 2016 to August 2025 (~107 months)

**Analysis conducted following SOP v1.3** with random_seed=42 for reproducibility.

---

## Qualitative Analysis

### Building Permits as Leading Indicator

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
| Concurrent Correlation (lag=0) | 0.033 | Weak |
| Best Predictive Correlation | -0.188 at lag +16 | Weak (wrong sign) |
| P-value at Best Predictive Lag | 0.075 | Marginally significant |
| Best Overall Correlation | **0.216 at lag -1** | **XLRE leads Permits** |

**Actionable**: ❌ NO - Only significant relationship is reverse causality (XLRE leads Permits)

---

## Correlation Analysis (Concurrent)

- **BuildingPermits_MoM_vs_XLRE_Returns**: r=0.033, p=0.735, n=107
- **BuildingPermits_QoQ_vs_XLRE_Returns**: r=-0.036, p=0.714, n=107
- **BuildingPermits_YoY_vs_XLRE_Returns**: r=0.075, p=0.441, n=107

---

## Lead-Lag Analysis (Extended Range: -18 to +18 months)

### Predictive Lags (Indicator Leads XLRE - Would Be Useful for Trading)

| Lag (months) | Correlation | P-value | Significant |
|--------------|-------------|---------|-------------|
| +16 | -0.188 | 0.075 | Marginal (wrong sign) |
| +12 | -0.139 | 0.179 | No |
| +13 | -0.125 | 0.230 | No |
| +4 | +0.097 | 0.327 | No |

**Finding**: No significant predictive relationships at positive lags. The marginally significant lag +16 has a *negative* correlation, which is counter-intuitive and likely spurious.

### Reverse Lags (XLRE Leads Indicator - Market Anticipation)

| Lag (months) | Correlation | P-value | Significant |
|--------------|-------------|---------|-------------|
| **-1** | **+0.216** | **0.026** | **✅ YES** |
| -2 | +0.154 | 0.118 | No |
| -4 | +0.132 | 0.183 | No |
| -7 | +0.129 | 0.200 | No |

**Interpretation**: XLRE returns predict next month's Building Permits (lag=-1). This is **reverse causality** - the market anticipates permit data before it's released.

---

## Economic Interpretation

### Why Only Reverse Causality Exists

1. **Market Efficiency**: Building permits data is widely followed and quickly incorporated into prices
2. **XLRE Composition**: XLRE is dominated by REITs (commercial real estate), not homebuilders
3. **Permits vs REITs Mismatch**: Building permits measure residential construction; XLRE holds commercial REITs
4. **Forward-Looking Markets**: Stock returns reflect future expectations; by the time permits are released, the information is already priced in

### Why This Differs from New Home Sales

| Factor | New Home Sales | Building Permits |
|--------|----------------|------------------|
| Predictive for XLRE? | ✅ Yes (lag +8) | ❌ No |
| Market anticipates? | Partially | Fully |
| Timing | Later in pipeline | Earlier in pipeline |
| Signal quality | Actionable | Not actionable |

---

## Conclusion

**NEGATIVE RESULT: No actionable predictive relationship found between Building Permits and XLRE**

### Key Findings

1. **No significant predictive lags**: All positive lags show p > 0.05
2. **Reverse causality only**: XLRE returns predict permits at lag -1 (p=0.026)
3. **Market efficiency**: Permit data appears fully priced into XLRE by release date
4. **Composition mismatch**: Building permits (residential) vs XLRE (commercial REITs)

### Practical Recommendations

1. **Do NOT use Building Permits as a trading signal for XLRE**
2. The only significant relationship is reverse causality (not useful for trading)
3. For housing-related XLRE timing, use **New Home Sales at lag +8** instead

### Comparison with New Home Sales

This analysis confirms that not all housing indicators are equally predictive for XLRE:
- **New Home Sales**: Significant at lag +8 (actionable)
- **Building Permits**: Only reverse causality (not actionable)

---

## Files Created

| File | Description |
|------|-------------|
| `data/xlre_buildingpermits.parquet` | Analysis data |
| `docs/analysis_reports/xlre_buildingpermits_analysis.md` | This document |

---

## Appendix: Audit Trail

- **Analysis Date**: 2026-01-26 (Updated)
- **Random Seed**: 42
- **SOP Version**: 1.3 (updated from 1.2)
- **Data Period**: 2016-10-31 to 2025-08-31
- **Observations**: 107
- **Lead-Lag Range**: -18 to +18 months
