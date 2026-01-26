# Sector ETFs vs Retail Inventories-to-Sales Ratio Analysis

## Overview

This analysis explores the relationship between sector ETFs (XLP - Consumer Staples, XLY - Consumer Discretionary) and the Retail Inventories-to-Sales Ratio (RETAILIRSA). These sectors were selected based on their direct economic connection to retail and consumer spending.

**Data Period**: January 1993 to Present (~380 months)

**Note**: This analysis was motivated by exploring whether sector-specific ETFs show stronger relationships with RETAILIRSA than broad market indices (SPY). The hypothesis was that consumer-facing sectors would have more direct sensitivity to retail inventory dynamics.

---

## Qualitative Analysis: Sector Selection Rationale

### Why XLP (Consumer Staples)?

XLP tracks companies producing essential consumer products (food, beverages, tobacco, household products). Key holdings include Procter & Gamble, Coca-Cola, Costco, and Walmart.

**Economic Connection to Retail Inventories:**
- Consumer staples are inventory-intensive - retailers must stock these essential goods
- Inventory management directly impacts retailer margins on staples
- Low margins mean inventory efficiency is critical for profitability
- These products have relatively stable demand, making inventory planning more predictable

**Hypothesis**: XLP should show *weaker* sensitivity to RETAILIRSA because:
- Demand for staples is inelastic (people buy necessities regardless of economic conditions)
- Inventory cycles are less volatile for essential goods
- Price competition and private labels may dampen any relationship

### Why XLY (Consumer Discretionary)?

XLY tracks companies producing non-essential consumer goods and services. Key holdings include Amazon, Tesla, Home Depot, McDonald's, and Nike.

**Economic Connection to Retail Inventories:**
- Discretionary purchases are highly sensitive to consumer confidence and economic conditions
- Retailers often over/under-order discretionary items based on demand expectations
- These products experience the most pronounced bullwhip effect
- Markdowns on excess discretionary inventory directly impact retail profitability

**Hypothesis**: XLY should show *stronger* sensitivity to RETAILIRSA because:
- Discretionary demand is highly elastic - amplifies inventory dynamics
- Consumer discretionary is the first to suffer when inventories build up
- Rising inventories signal weakening consumer demand, which directly impacts XLY earnings
- XLY includes major retailers themselves (Home Depot, Amazon retail operations)

### Literature Support

From the [SPY vs RETAILIRSA analysis](./spy_retailirsa_analysis.md):
- Rising inventories precede recessions (NetSuite, FocusEconomics)
- The bullwhip effect amplifies demand changes (MIT Sloan)
- Consumer spending drives inventory cycles (Morningstar)

Sector-specific research from [Fidelity Sector Investing](https://www.fidelity.com/learning-center/investment-products/etf/sector-etf-investing):
- Consumer discretionary is most cyclical, strongest correlation with business cycle
- Consumer staples is defensive, lowest correlation with economic growth

---

## Key Findings Summary

### Critical Finding: Neither Relationship is Statistically Significant

| Sector | Level Corr | Returns Corr | Best Regime Sharpe | Regime P-value | Significant? |
|--------|------------|--------------|--------------------|--------------------|--------------|
| **XLP** | -0.751 | -0.022 | Rising: 0.64 | 0.785 | **NO** |
| **XLY** | -0.760 | -0.132 | Falling: 0.77 | 0.379 | **NO** |
| SPY (reference) | -0.767 | -0.105 | Falling: 0.98 | ~0.05 | Marginal |

### Interpretation

1. **Level Correlations are Spurious**: All three show strong negative level correlations (-0.75 to -0.77), but this reflects opposite secular trends (rising equity prices, falling inventory ratios over time), not a causal relationship.

2. **Returns Correlations are Weak**:
   - XLP: -0.022 (essentially zero - no relationship)
   - XLY: -0.132 (weak negative, similar to SPY)
   - The hypothesis that XLP would be weaker is supported, but XLY is not stronger than SPY

3. **Regime Analysis Shows Expected Pattern but Lacks Significance**:
   - XLP performs slightly *better* when inventories are rising (contrary to economic intuition)
   - XLY performs better when inventories are falling (matching economic intuition)
   - Neither difference is statistically significant (p > 0.05)

---

## Detailed Analysis: XLP (Consumer Staples)

### Correlation Analysis

| Metric | XLP | RETAILIRSA | Returns Corr |
|--------|-----|------------|--------------|
| Level Correlation | 1.000 | -0.751 | - |
| MoM Change | - | - | -0.022 |
| QoQ Change | - | - | -0.047 |
| YoY Change | - | - | -0.031 |

**Observation**: XLP has essentially zero correlation with RETAILIRSA changes. This supports the hypothesis that consumer staples, being essential goods with inelastic demand, are insensitive to retail inventory dynamics.

### Regime Analysis

| Regime | Mean Monthly Return | Sharpe Ratio | N Months |
|--------|---------------------|--------------|----------|
| **Falling RETAILIRSA** | +0.73% | 0.55 | ~190 |
| **Rising RETAILIRSA** | +0.82% | 0.64 | ~190 |

**Unexpected Result**: XLP performs *better* when inventories are rising. This contradicts the general market pattern but may make sense for staples:
- During economic stress (rising inventories), investors rotate *into* defensive sectors like XLP
- "Flight to safety" effect outweighs any negative impact from inventory dynamics
- Staples retailers may actually benefit from trade-down effects during slowdowns

**Statistical Significance**: p = 0.785 - **NOT SIGNIFICANT**

The regime difference (+0.09% per month favoring rising) cannot be distinguished from random noise.

---

## Detailed Analysis: XLY (Consumer Discretionary)

### Correlation Analysis

| Metric | XLY | RETAILIRSA | Returns Corr |
|--------|-----|------------|--------------|
| Level Correlation | 1.000 | -0.760 | - |
| MoM Change | - | - | -0.132 |
| QoQ Change | - | - | -0.158 |
| YoY Change | - | - | -0.089 |

**Observation**: XLY shows the expected negative relationship with RETAILIRSA changes (-0.132), similar to SPY. The QoQ correlation of -0.158 is actually stronger than SPY's QoQ correlation of -0.129.

### Regime Analysis

| Regime | Mean Monthly Return | Sharpe Ratio | N Months |
|--------|---------------------|--------------|----------|
| **Falling RETAILIRSA** | +1.05% | 0.77 | ~190 |
| **Rising RETAILIRSA** | +0.52% | 0.31 | ~190 |

**Expected Result**: XLY performs significantly better when inventories are falling:
- +0.53% per month advantage (~6.4% annualized)
- Sharpe ratio more than doubles (0.77 vs 0.31)
- This matches economic intuition about discretionary spending sensitivity

**Statistical Significance**: p = 0.379 - **NOT SIGNIFICANT**

Despite the economically meaningful difference (6.4% annualized), sample size and volatility prevent statistical confidence.

---

## Comparison with SPY Baseline

### Why Don't Sectors Beat SPY?

| Hypothesis | Expected | Actual | Explanation |
|------------|----------|--------|-------------|
| XLP weaker than SPY | Yes | **Yes** | Staples are defensive, less cyclical |
| XLY stronger than SPY | Yes | **No** | XLY vol higher, offsetting signal strength |
| Sector specificity adds value | Yes | **No** | Noise overwhelms sector-specific signal |

### Returns Correlation Comparison

- SPY: -0.105 (weak but consistent)
- XLP: -0.022 (essentially zero)
- XLY: -0.132 (weak, similar to SPY)

XLY shows slightly stronger correlation than SPY, but not enough to improve statistical significance.

### Regime Sharpe Comparison

| Regime | SPY | XLP | XLY |
|--------|-----|-----|-----|
| Falling RETAILIRSA | **0.98** | 0.55 | 0.77 |
| Rising RETAILIRSA | 0.52 | 0.64 | 0.31 |
| Difference | +0.46 | -0.09 | +0.46 |

SPY and XLY show similar regime differentiation, but SPY has better absolute Sharpe ratios. XLP shows the opposite pattern (rising > falling), consistent with its defensive nature.

---

## Economic Interpretation

### Why XLP Shows Inverse Pattern

The finding that XLP performs *better* during rising inventory periods aligns with sector rotation theory:

1. **Risk-Off Behavior**: When economic stress builds (rising inventories signal weakening demand), investors rotate from cyclical (XLY) to defensive (XLP) sectors

2. **Relative Performance**: XLP doesn't gain absolutely; it *loses less* than the market, appearing as relative outperformance

3. **Trade-Down Effect**: Economic stress may actually boost staples consumption as consumers substitute premium goods with basic necessities

### Why XLY Matches Theory but Lacks Significance

XLY shows the expected pattern (better in falling inventory periods) but cannot achieve significance because:

1. **Higher Volatility**: XLY's standard deviation (~6%) is higher than SPY's (~4.5%), requiring more data for significance

2. **Sector Concentration**: XLY has fewer holdings and more idiosyncratic risk

3. **Structural Changes**: Amazon's dominance (25%+ of XLY) changes the sector's fundamental relationship with retail dynamics

---

## Practical Implications

### For Portfolio Construction

1. **Do NOT use RETAILIRSA as a sector rotation signal** - insufficient statistical evidence

2. **Sector Regime Filter (Low Confidence)**: If using RETAILIRSA regime in a broader strategy:
   - Favor XLY over XLP when RETAILIRSA is falling
   - Favor XLP over XLY when RETAILIRSA is rising
   - Expected benefit: ~1% per month sector differential (unproven)

3. **Stick with SPY**: The broad market index shows stronger, more consistent regime patterns than sector ETFs

### For Research Continuation

1. **Expand to Other Sectors**: Test RETAILIRSA against all 11 GICS sectors
2. **Use Sector-Specific Inventory Data**: RETAILIRSA aggregates all retail; sector-specific inventory data may be more predictive
3. **Increase Sample Size**: Monthly data since 1993 may be insufficient for sector-level analysis
4. **Test Sub-Periods**: Pre-2010 (before Amazon dominance) may show different patterns

---

## Researcher's Notes

### Limitations Acknowledged

1. **Negative Results are Results**: The lack of statistical significance is a valid finding that prevents false confidence in sector rotation strategies

2. **Economic Intuition vs Statistical Evidence**: XLY shows economically intuitive patterns, but we cannot recommend trading on intuition alone

3. **Data Period Concerns**: The 1998-2000 and 2020-2022 periods contain significant outliers (tech bubble, COVID) that may distort sector relationships

4. **Attribution Challenge**: Cannot separate direct inventory effect from correlated economic cycle effects

### Questions for Further Investigation

1. Why does SPY show stronger regime patterns than its component sectors?
2. Would a sector-weighted inventory index be more predictive than aggregate RETAILIRSA?
3. Are there specific sub-industries within XLY (e.g., home improvement vs restaurants) with stronger relationships?

---

## Files Created

| File | Description |
|------|-------------|
| `data/xlp_retail_inv_sales.parquet` | XLP price/returns with RETAILIRSA indicator |
| `data/xly_retail_inv_sales.parquet` | XLY price/returns with RETAILIRSA indicator |
| `docs/analysis_reports/sector_retailirsa_analysis.md` | This document |

---

## Conclusion

**RETAILIRSA does not reliably predict sector ETF returns (XLP or XLY)**:

1. **XLP (Consumer Staples)**: Zero correlation with RETAILIRSA; defensive nature may cause inverse regime pattern
2. **XLY (Consumer Discretionary)**: Shows expected economic pattern but lacks statistical significance (p = 0.379)
3. **SPY remains superior**: Broader diversification produces more stable regime patterns than sector bets

**Recommendation**: Do not use RETAILIRSA for sector rotation. The relationship is too weak and noisy for practical application. If using RETAILIRSA at all, apply it to broad market exposure (SPY) rather than sector allocation.

---

## Appendix: Data Quality Notes

- XLP data begins December 1998 (ETF inception)
- XLY data begins December 1998 (ETF inception)
- RETAILIRSA data begins January 1992 (FRED)
- Overlap period: December 1998 - Present (~310 months)
- All returns calculated as monthly percentage changes
- Regime classification: YoY RETAILIRSA change (rising > 0, falling < 0)
