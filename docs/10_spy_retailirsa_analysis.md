# SPY vs Retail Inventories-to-Sales Ratio Analysis

## Overview

This analysis explores the relationship between S&P 500 (SPY) price/returns and the Retail Inventories-to-Sales Ratio (RETAILIRSA) using correlation analysis, lead-lag testing, Granger causality, ML predictive models, and regime analysis.

**Data Period**: January 1993 to September 2025 (393 months)

## Key Findings Summary

### 1. Strong Inverse Level Relationship
- **Correlation: -0.767** between SPY price and RETAILIRSA level
- As SPY rises over time, the Inventories-to-Sales ratio has fallen (supply chain efficiency)
- This is a **spurious correlation** due to opposite secular trends, not a causal relationship

### 2. Weak Contemporaneous Relationship in Changes
- RETAILIRSA changes have **weak negative correlation** with SPY returns:
  - MoM: -0.105
  - QoQ: -0.129 (strongest)
  - YoY: -0.072
- Interpretation: When retailers build inventory faster, SPY tends to underperform slightly

### 3. No Predictive Power
- **Granger Causality**: No significant causality in either direction (all p > 0.05)
- **ML Models**: Negative R² values indicate RETAILIRSA cannot predict SPY returns
- Best performing model (Lasso, 1-month horizon): R² = -0.18 (worse than naive mean)

### 4. Regime-Based Insights (Most Valuable Finding)

| Regime | Mean Monthly Return | Sharpe Ratio | Win Rate |
|--------|---------------------|--------------|----------|
| **Low Inv/Sales** | +1.14% | **1.03** | 68.2% |
| High Inv/Sales | +0.70% | 0.51 | 61.1% |
| **Falling Inv/Sales** | +1.09% | **0.98** | 64.9% |
| Rising Inv/Sales | +0.72% | 0.52 | 65.2% |
| **Recession** | -1.36% | -0.64 | 39.3% |

**Key Insight**: SPY performs significantly better when:
- Inventories-to-Sales ratio is **low** (efficient supply chains)
- Inventories-to-Sales ratio is **falling** (improving efficiency)

## Detailed Analysis

### Correlation Matrix

```
                          SPY    RETAILIRSA   MoM     QoQ     YoY   SPY_MoM  SPY_QoQ  SPY_YoY
SPY                     1.000       -0.767  0.007   0.012   0.006    0.069    0.106    0.175
Retail_Inv_Sales_Ratio -0.767        1.000  0.102   0.149   0.240   -0.033   -0.105   -0.118
RETAILIRSA_MoM          0.007        0.102  1.000   0.537   0.315   -0.105   -0.189   -0.066
RETAILIRSA_QoQ          0.012        0.149  0.537   1.000   0.531   -0.129   -0.340   -0.113
RETAILIRSA_YoY          0.006        0.240  0.315   0.531   1.000   -0.072   -0.264   -0.355
```

**Observations**:
- Strong negative correlation between SPY price and RETAILIRSA level (-0.767)
- Weak negative correlation between RETAILIRSA changes and SPY returns
- RETAILIRSA_QoQ has strongest relationship with SPY_QoQ (-0.340)

### Lead-Lag Analysis

Tested lags from -12 to +12 months:

| Relationship | Lag | Correlation | P-value | Interpretation |
|--------------|-----|-------------|---------|----------------|
| RETAILIRSA_QoQ_Dir | 0 | -0.138 | 0.006 | Contemporaneous |
| RETAILIRSA_QoQ | 0 | -0.129 | 0.011 | Contemporaneous |
| RETAILIRSA_MoM | 0 | -0.105 | 0.039 | Contemporaneous |

**Conclusion**: The relationship is primarily **contemporaneous**, not predictive. No significant lead-lag relationships suggest RETAILIRSA doesn't predict future SPY returns.

### Granger Causality Tests

Tested if RETAILIRSA Granger-causes SPY returns at lags 1-6 months:

| Feature | Best Lag | F-statistic | P-value | Significant? |
|---------|----------|-------------|---------|--------------|
| RETAILIRSA_MoM | 2 | 1.619 | 0.199 | No |
| RETAILIRSA_QoQ | 1 | 2.101 | 0.148 | No |
| RETAILIRSA_YoY | 4 | 1.293 | 0.272 | No |

**Conclusion**: RETAILIRSA does **not** Granger-cause SPY returns at any lag tested.

### ML Predictive Model Results

Models trained to predict 1-month and 3-month forward SPY returns using lagged RETAILIRSA features:

| Horizon | Model | CV R² | RMSE | MAE |
|---------|-------|-------|------|-----|
| 1m | Linear Regression | -0.54 | 5.25 | 3.88 |
| 1m | Ridge | -0.31 | 4.88 | 3.69 |
| 1m | Lasso | -0.18 | 4.64 | 3.50 |
| 1m | Random Forest | -0.25 | 4.79 | 3.67 |
| 1m | Gradient Boosting | -0.50 | 5.23 | 4.08 |

**Top Features (Random Forest)**:
1. RETAILIRSA_MoM_lag6: 11.2%
2. RETAILIRSA_QoQ_lag3: 9.3%
3. RETAILIRSA_YoY_lag6: 8.8%
4. RETAILIRSA_YoY_lag1: 8.7%
5. RETAILIRSA_QoQ_lag12: 8.5%

**Conclusion**: All models have **negative R²**, meaning RETAILIRSA features perform worse than simply predicting the mean return. There is no exploitable predictive relationship.

### Regime Analysis

SPY monthly returns segmented by RETAILIRSA regimes:

#### By Level (High vs Low)
- **Low Inv/Sales** (ratio < median 1.49): Mean return +1.14%, Sharpe 1.03
- **High Inv/Sales** (ratio > median 1.49): Mean return +0.70%, Sharpe 0.51
- **Difference**: +0.44% per month, ~5.3% annualized

#### By Direction (Rising vs Falling)
- **Falling Inv/Sales** (YoY < 0): Mean return +1.09%, Sharpe 0.98
- **Rising Inv/Sales** (YoY > 0): Mean return +0.72%, Sharpe 0.52
- **Difference**: +0.37% per month, ~4.5% annualized

#### Combined with Recession Indicator
- **Expansion + Falling**: Mean +1.16%, Sharpe 1.06 (best regime)
- **Expansion + Rising**: Mean +1.05%, Sharpe 0.89
- **Recession**: Mean -1.36%, Sharpe -0.64 (worst regime)

## Economic Interpretation

### Why the Inverse Relationship?

The negative correlation between RETAILIRSA changes and SPY returns makes economic sense:

1. **Rising Inventories = Demand Weakness**
   - When retailers build inventory (ratio rising), it often signals weakening consumer demand
   - Products aren't selling as expected, leading to inventory buildup
   - This is bearish for stocks

2. **Falling Inventories = Demand Strength**
   - When inventory-to-sales falls, retailers are selling through inventory faster
   - Indicates strong consumer demand
   - This is bullish for stocks

3. **Supply Chain Efficiency**
   - The secular decline in RETAILIRSA (from 1.75 to 1.28) reflects just-in-time inventory management
   - This structural improvement has coincided with the long-term bull market

### Why No Predictive Power?

Despite the contemporaneous correlation, RETAILIRSA lacks predictive power because:

1. **Data is Monthly and Lagged**: RETAILIRSA is released with a 6-week delay
2. **Markets are Efficient**: The information is priced in by release time
3. **Relationship is Weak**: Even contemporaneous correlation is only -0.13
4. **Many Other Factors**: Stock returns are driven by many variables beyond retail inventories

## Practical Applications

### For Investment Clock Integration

While RETAILIRSA cannot predict SPY returns directly, it can be useful as:

1. **Economic Regime Confirmation**: Rising RETAILIRSA often precedes or coincides with economic slowdowns
2. **Risk Indicator**: High RETAILIRSA levels (>1.5) have historically been associated with lower SPY returns
3. **Recession Warning**: Rapidly rising RETAILIRSA can signal demand destruction

### Trading Rules (Based on Regime Analysis)

**Simple Rule**: Favor stocks when RETAILIRSA YoY < 0 (falling inventories)
- Historical advantage: +4.5% annualized over rising periods
- Win rate: 65% vs 65% (similar)
- Sharpe improvement: 0.98 vs 0.52

**Caution**: This is a filter, not a timing signal. Use with other indicators.

## Files Created

| File | Description |
|------|-------------|
| `src/ml/retail_spy_analysis/__init__.py` | Package initialization |
| `src/ml/retail_spy_analysis/relationship_analysis.py` | Analysis functions |
| `docs/10_spy_retailirsa_analysis.md` | This document |

## Conclusion

**RETAILIRSA does not predict SPY returns**, but it provides valuable regime context:

1. **Low/Falling inventory ratios** are associated with better SPY performance
2. **The relationship is contemporaneous**, not leading
3. **Use as a filter**, not a signal - combine with other indicators
4. **Recession indicator** adds significant value to the regime analysis

The most actionable insight: Favor equity exposure when RETAILIRSA is falling (YoY < 0) and avoid during recessions.
