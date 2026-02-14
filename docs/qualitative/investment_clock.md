# Investment Clock Framework

## Overview

The Investment Clock framework evaluates **sector performance across four economic regimes** using:
- **Growth Signal**: Orders/Inventories Ratio (3MA vs 6MA direction)
- **Inflation Signal**: PPI (3MA vs 6MA direction)

Together these achieve a 96.8% classification rate vs 66% with traditional indicators.

## Growth Dimension

**What Does "Growth" Mean?**

In the Investment Clock context, "Growth" refers to the direction of economic activity—whether
GDP, employment, and corporate earnings are accelerating or decelerating.

**How Growth Affects Sectors:**

| Growth Direction | Sector Impact | Mechanism |
|---|---|---|
| Rising | Cyclicals outperform | Increased consumer spending, capital investment, hiring |
| Falling | Defensives outperform | Stable demand for necessities; flight to safety |

**Growth-Sensitive Sectors** (High Beta to Growth):
- Technology: Discretionary IT spending expands/contracts with growth
- Consumer Discretionary: Durable goods, travel, entertainment
- Industrials: Capital expenditure, manufacturing orders
- Financials: Loan demand, credit quality

**Growth-Defensive Sectors** (Low Beta to Growth):
- Utilities: Regulated returns, inelastic demand
- Consumer Staples: Food, beverages, household products
- Healthcare: Non-discretionary spending

## Inflation Dimension

**What Does "Inflation" Mean?**

In the Investment Clock context, "Inflation" refers to the direction of price pressure—whether
prices are accelerating or decelerating.

**How Inflation Affects Sectors:**

| Inflation Direction | Sector Impact | Mechanism |
|---|---|---|
| Rising | Real assets outperform | Commodity producers benefit; pricing power matters |
| Falling | Rate-sensitive sectors outperform | Lower rates boost valuations; borrowing costs fall |

**Inflation-Beneficiary Sectors** (Positive Beta to Inflation):
- Energy: Direct commodity exposure; oil/gas price correlation
- Materials: Mining, chemicals, commodity producers

**Inflation-Hurt Sectors** (Negative Beta to Inflation):
- Utilities: Regulated prices lag inflation; rising rates hurt
- Consumer Discretionary: Purchasing power erosion

## Interaction Effects: Why Four Phases Matter

The Investment Clock framework recognizes that growth and inflation **interact**:

| Growth | Inflation | Phase | Combined Effect |
|---|---|---|---|
| Rising | Falling | Recovery | Best for cyclicals - Growth boosts earnings; low inflation allows Fed accommodation |
| Rising | Rising | Overheat | Real assets - Growth supports demand; inflation boosts commodity prices |
| Falling | Rising | Stagflation | Worst combo - No growth + price pressure = margin compression |
| Falling | Falling | Reflation | Rate-sensitive recovery - Fed eases; rate-sensitive sectors benefit |

## Sector Sensitivity Matrix

| Sector | Growth Sensitivity | Inflation Sensitivity | Best Phase | Worst Phase |
|---|---|---|---|---|
| Technology | High (+) | Moderate (-) | Recovery | Stagflation |
| Financials | High (+) | Mixed | Recovery | Stagflation |
| Healthcare | Low | Low | Stagflation | — |
| Energy | Moderate (+) | High (+) | Overheat | Reflation |
| Industrials | High (+) | Moderate (+) | Overheat | Stagflation |
| Consumer Disc. | High (+) | Moderate (-) | Recovery/Reflation | Stagflation |
| Consumer Staples | Low (-) | Low | Stagflation | Recovery |
| Utilities | Low (-) | High (-) | Stagflation | Overheat |
| Materials | Moderate (+) | High (+) | Overheat | Reflation |

## Key Literature

- **Fama (1981)** established the relationship between real economic activity and stock returns
- **Chen, Roll & Ross (1986)** identified industrial production growth as a priced factor
- **Boudoukh & Richardson (1993)** found inflation hedging varies by sector
- **Invesco Inflation Research** documents sector rotation strategies
