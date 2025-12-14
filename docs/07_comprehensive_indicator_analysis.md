# Comprehensive Indicator Analysis

## Objective

Find the optimal combination of **leading Growth indicator** + **leading Inflation indicator** to replace the traditional lagging indicators (Industrial Production YoY + CPI YoY) in the Investment Clock framework.

## Methodology

### Indicators Tested

**14 Growth Indicators:**
| Indicator | Source | Signal Method |
|-----------|--------|---------------|
| IP YoY (Benchmark) | FRED INDPRO | Momentum |
| LEI 3M/6M Change | FRED USSLIND | Threshold (0) |
| CFNAI | FRED CFNAI | Threshold (0) |
| CFNAI 3MA | FRED CFNAI | Threshold (0) |
| Yield Curve 10Y-3M | FRED T10Y3M | Threshold (0) |
| Yield Curve 10Y-2Y | FRED T10Y2Y | Threshold (0) |
| Orders/Inv Ratio YoY | FRED AMTMNO/AMTMTI | Threshold (0) |
| Orders/Inv Ratio MoM | FRED AMTMNO/AMTMTI | Direction |
| Building Permits YoY | FRED PERMIT | Threshold (0) |
| Initial Claims YoY (inverted) | FRED ICSA | Threshold (0) |
| Capacity Utilization | FRED TCU | Momentum |
| Durable Orders YoY | FRED DGORDER | Threshold (0) |
| OECD CLI | FRED USALOLITONOSTSAM | Threshold (100) |
| Unemployment (inverted) | FRED UNRATE | Momentum |

**19 Inflation Indicators:**
| Indicator | Source | Signal Method |
|-----------|--------|---------------|
| CPI YoY (Benchmark) | FRED CPIAUCSL | Momentum |
| Core CPI YoY | FRED CPILFESL | Momentum |
| PPI YoY | FRED PPIACO | Momentum |
| PPI MoM | FRED PPIACO | Direction |
| PPI 3M Annualized | FRED PPIACO | Momentum |
| Breakeven 10Y | FRED T10YIE | Momentum |
| Breakeven 10Y MoM | FRED T10YIE | Direction |
| Breakeven 5Y | FRED T5YIE | Momentum |
| M2 YoY | FRED M2SL | Momentum |
| M2 YoY Lag12 | FRED M2SL | Momentum |
| M2 YoY Lag18 | FRED M2SL | Momentum |
| Commodity Index YoY | FRED PALLFNFINDEXM | Momentum |
| Commodity Index MoM | FRED PALLFNFINDEXM | Direction |
| Oil YoY | FRED DCOILWTICO | Momentum |
| Oil MoM | FRED DCOILWTICO | Direction |
| Import Prices YoY | FRED IR | Momentum |
| Wage Growth YoY | FRED CES0500000003 | Momentum |
| PCE YoY | FRED PCEPI | Momentum |
| Inflation Expectations | FRED MICH | Momentum |

### Signal Generation Methods

1. **Momentum Signal**:
   - Rising (+1): Value > 6M MA AND Value > 12M MA
   - Falling (-1): Value < 6M MA AND Value < 12M MA
   - Unknown (0): Otherwise

2. **Threshold Signal**:
   - Rising (+1): Value > threshold
   - Falling (-1): Value < threshold

3. **Direction Signal**:
   - Rising (+1): 3M MA > 6M MA
   - Falling (-1): 3M MA < 6M MA

### Evaluation Metrics

1. **Classification Rate**: % of months with a defined phase (not "Unknown")
2. **Quality Score**: How well asset returns match theoretical expectations by phase
3. **Combined Score**: Quality Score × Classification Rate

---

## Results

### Best Indicator Combination

| Metric | Value |
|--------|-------|
| **Growth Indicator** | Orders/Inv MoM |
| **Inflation Indicator** | PPI MoM |
| **Classification Rate** | 96.8% |
| **Quality Score** | 83.3% |
| **Combined Score** | 80.7 |

### Comparison to Benchmark

| Metric | Benchmark (IP YoY + CPI YoY) | Best Combination | Improvement |
|--------|------------------------------|------------------|-------------|
| Classification Rate | 66.0% | 96.8% | **+30.8 pp** |
| Quality Score | ~100% | 83.3% | -16.7 pp |
| Combined Score | 66.0 | 80.7 | **+14.7** |

---

## Top Combinations by Combined Score

| Rank | Growth Indicator | Inflation Indicator | Class Rate | Quality | Combined |
|------|------------------|---------------------|------------|---------|----------|
| 1 | **Orders/Inv MoM** | **PPI MoM** | 96.8% | 83.3% | **80.7** |
| 2 | Yield Curve 10Y-3M | Commodity MoM | 97.8% | 66.7% | 65.2 |
| 3 | Yield Curve 10Y-2Y | Commodity MoM | 97.1% | 66.7% | 64.7 |
| 4 | OECD CLI | Commodity MoM | 100.0% | 50.0% | 50.0 |
| 5 | CFNAI 3MA | Oil MoM | 100.0% | 50.0% | 50.0 |

---

## Best Individual Indicators

### Top 5 Growth Indicators
(Average classification rate across all inflation indicators)

| Rank | Indicator | Avg Classification Rate |
|------|-----------|-------------------------|
| 1 | **Orders/Inv YoY** | 85.5% |
| 2 | Orders/Inv MoM | 84.6% |
| 3 | CFNAI 3MA | 84.2% |
| 4 | Capacity Utilization | 84.1% |
| 5 | OECD CLI | 84.1% |

### Top 5 Inflation Indicators
(Average classification rate across all growth indicators)

| Rank | Indicator | Avg Classification Rate |
|------|-----------|-------------------------|
| 1 | **Oil MoM** | 93.3% |
| 2 | Commodity MoM | 91.2% |
| 3 | PPI MoM | 90.6% |
| 4 | Breakeven 10Y MoM | 90.4% |
| 5 | M2 YoY | 79.1% |

---

## Key Findings

### 1. MoM Indicators Are Superior for Classification

Month-over-month (MoM) indicators using direction signals consistently outperform YoY momentum-based indicators for classification rate:
- **Oil MoM**: 93.3% vs Oil YoY: ~70%
- **PPI MoM**: 90.6% vs PPI YoY: ~75%
- **Orders/Inv MoM**: 84.6% vs Orders/Inv YoY: 85.5%

This is because MoM direction signals are more decisive (always +1 or -1), while YoY momentum signals can be ambiguous.

### 2. PPI is a Better Inflation Indicator Than CPI

PPI (Producer Price Index) leads CPI by 2-4 months because:
- PPI measures input costs that flow through to consumer prices
- PPI is more volatile and responsive to supply shocks
- PPI captures commodity/energy price changes earlier

### 3. Orders/Inventories Ratio is the Best Growth Indicator

The Manufacturing New Orders / Inventories ratio is highly effective because:
- **Leading**: Rising orders relative to inventories signals future production increases
- **Logical**: Economic pressure builds when orders exceed inventory capacity
- **Timely**: Monthly data with minimal revision

### 4. Oil/Commodity Prices Are Leading Inflation Indicators

Oil and commodity price movements lead official inflation measures:
- Direct input to PPI and eventually CPI
- Respond immediately to supply/demand shifts
- Capture global inflationary pressures

---

## Recommended Implementation

### Primary Recommendation: Orders/Inv MoM + PPI MoM

```python
# Growth Signal (Orders/Inventories MoM Direction)
orders_inv_ratio = new_orders / inventories
growth_3ma = orders_inv_ratio.rolling(3).mean()
growth_6ma = orders_inv_ratio.rolling(6).mean()
growth_signal = 1 if growth_3ma > growth_6ma else -1

# Inflation Signal (PPI MoM Direction)
ppi_3ma = ppi.rolling(3).mean()
ppi_6ma = ppi.rolling(6).mean()
inflation_signal = 1 if ppi_3ma > ppi_6ma else -1

# Phase Classification
if growth_signal == 1 and inflation_signal == 1:
    phase = "Overheat"
elif growth_signal == 1 and inflation_signal == -1:
    phase = "Recovery"
elif growth_signal == -1 and inflation_signal == -1:
    phase = "Reflation"
else:  # growth == -1, inflation == 1
    phase = "Stagflation"
```

### Alternative Recommendations

1. **Yield Curve 10Y-3M + Commodity MoM** (Combined: 65.2)
   - Simpler growth indicator (single series vs ratio)
   - 12-month leading property for recessions
   - Commodity prices capture global inflation

2. **CFNAI 3MA + Oil MoM** (Combined: 50.0, but 100% classification)
   - Broadest growth indicator (85 underlying series)
   - Very high classification rate
   - Lower quality score suggests some phase mismatches

---

## Data Files Generated

| File | Description |
|------|-------------|
| `all_indicator_combinations.csv` | Full results of 285 combinations tested |
| `indicator_quality_scores_full.csv` | Quality scores for top combinations |
| `monthly_all_indicators.parquet` | Monthly data with all computed indicators |

---

## Next Steps

1. **Backtest** the recommended combinations against sector returns
2. **Fetch Fama-French** industry portfolios for long-history sector analysis
3. **ML Enhancement**: Use indicator combinations as features for regime detection
4. **Out-of-sample testing** to validate robustness

---

*Analysis Date: 2025-12-13*
