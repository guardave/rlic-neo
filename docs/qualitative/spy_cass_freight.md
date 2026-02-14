# SPY vs Cass Freight Index

## What is the Cass Freight Index?

The **Cass Freight Index** is a monthly measure of North American freight activity
published by Cass Information Systems. It tracks actual freight bills processed
through Cass's payment systems, covering hundreds of large shippers across all
modes (truck, rail, intermodal, air, ocean).

**Two Components:**
- **Shipments Index** (FRGSHPUSM649NCIS): Measures the *volume* of freight
  shipments. Indexed to January 1990 = 1.0
- **Expenditures Index** (FRGEXPUSM649NCIS): Measures the *total spend* on
  freight (volume x price). Indexed to January 1990 = 1.0

**Key Characteristics:**
- **Real Activity Data**: Based on actual freight bills, not surveys or estimates
- **Monthly Frequency**: Published monthly via FRED
- **Broad Coverage**: Captures freight across all major transportation modes
- **Leading Property**: Freight activity often leads industrial production and GDP

## Economic Signal Interpretation

| Shipments YoY | Interpretation |
|---|---|
| > +5% | Strong expansion, robust goods demand |
| 0% to +5% | Moderate growth, normal conditions |
| -5% to 0% | Mild contraction, demand softening |
| < -5% | Sharp contraction, recessionary signal |

## Why Compare with SPY?

**Goods Economy Proxy**: The Cass Freight Index captures the physical movement
of goods across the economy. Rising freight volumes indicate increasing
economic activity, which supports corporate earnings and equity valuations.

**Transmission Mechanism**:
1. Rising freight shipments signal growing consumer and industrial demand
2. Higher volumes support revenue growth across multiple sectors
3. Freight trends tend to precede official industrial production data
4. Sustained freight declines often precede or accompany recessions

## Key Research Finding: Confirmatory Signal

::: warning
**Our Analysis Found**: Cass Freight Shipments has a confirmatory (not predictive)
relationship with SPY returns. All 12 significant lags are at **negative values**
(best: lag -8, r=+0.195, p=0.0008), meaning the stock market moves FIRST and
freight activity confirms later.

**Expenditures** show a similar pattern (best: lag -12, r=+0.209, p=0.0004),
also confirmatory.

**Regime analysis is statistically significant:**
- **Freight Rising** (YoY > 0): Mean return +1.34%/mo, Sharpe 1.11
- **Freight Falling** (YoY <= 0): Mean return +0.22%/mo, Sharpe 0.17
- Regime difference: t=2.19, p=0.029

**Strategy performance** (long during rising, cash during falling):
- Strategy Sharpe: 0.77 vs Benchmark Sharpe: 0.63
- Exposure: 45.8% — in market less than half the time

This is a **contemporaneous regime indicator** for risk management, but NOT a
leading signal for timing entries.
:::

## Academic and Professional Research

| Finding | Source | Implication |
|---|---|---|
| Freight volumes are procyclical and lead industrial production | Lahiri & Yao (2004) | Freight data captures real economic momentum |
| Transportation indices correlate with GDP growth | Bougheas et al. (2000) | Physical goods movement proxies aggregate demand |
| Cass Freight Index tracks closely with ISM Manufacturing | Cass Information Systems | Validates freight as manufacturing activity measure |
| Equity markets are forward-looking and price in real activity changes early | Fama (1981) | Explains why SPY leads freight data |
| Freight expenditures capture both volume and pricing pressure | DAT Solutions | Expenditures add inflation/pricing dimension |

## Limitations

1. **Not Predictive for Equities**: Stock market moves 8+ months before freight
   data confirms the trend
2. **Goods-Only Coverage**: Does not capture services sector (~70% of GDP)
3. **Seasonal Patterns**: Strong seasonal effects require YoY comparison
4. **Composition Bias**: Weighted toward large shippers using Cass payment systems
5. **Index vs Absolute**: As an index (base = Jan 1990), cannot directly compare
   freight levels across different economic eras
