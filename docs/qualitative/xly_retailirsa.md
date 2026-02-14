# XLY (Consumer Discretionary) vs Retail Inv/Sales

## Sector Overview: Consumer Discretionary (XLY)

XLY tracks companies producing **non-essential consumer goods and services**. Top holdings
include Amazon, Tesla, Home Depot, McDonald's, and Nike.

**Key Characteristics:**
- **Cyclical Sector**: High beta, performance tied to economic growth
- **Elastic Demand**: Discretionary purchases expand/contract with consumer confidence
- **Growth-Oriented**: Many holdings are growth stocks (Amazon, Tesla)
- **ETF Inception**: December 1998 (SPDR Select Sector)

## Economic Connection to Retail Inventories

**Hypothesis**: XLY should show *stronger* sensitivity to RETAILIRSA because:

1. **Discretionary Demand is Elastic**: First to suffer when consumers cut back
2. **Bullwhip Effect**: Discretionary items experience most pronounced inventory swings
3. **Markup Sensitivity**: Excess inventory leads to markdowns on discretionary goods
4. **Retailer Exposure**: XLY includes major retailers (Home Depot, Amazon retail)

## Consumer Confidence Link

Consumer discretionary spending is directly tied to confidence and inventory dynamics:

> *"Rising retail inventories signal weakening consumer demand—discretionary purchases
are the first to be cut when consumers feel uncertain."* - Morningstar

> *"Consumer discretionary is the most cyclical sector, showing the strongest correlation
with the business cycle."* - Fidelity Sector Research

## Research Finding: Expected Pattern but Lacks Significance

::: info
**Our Analysis Found**: XLY shows the economically expected pattern—performs better when
retail inventories are falling (Sharpe 0.77 vs 0.31 for rising). The difference is
substantial: +0.53% monthly (~6.4% annualized).

**However**: p = 0.379 — **NOT STATISTICALLY SIGNIFICANT** at conventional thresholds.
While the pattern matches economic intuition, the sample size and volatility prevent
statistical confidence.
:::

## Why XLY Doesn't Beat SPY

| Metric | XLY | SPY | Observation |
|---|---|---|---|
| Returns Correlation | -0.132 | -0.105 | XLY slightly stronger correlation |
| Regime Sharpe (Falling) | 0.77 | 0.98 | SPY has better absolute performance |
| Regime Sharpe (Rising) | 0.31 | 0.52 | SPY performs better even in "bad" regime |
| Regime Difference | +0.46 | +0.46 | Similar regime differentiation |

**Why SPY is superior for RETAILIRSA analysis:**
1. SPY has lower volatility, making patterns more detectable
2. SPY diversification reduces idiosyncratic noise
3. SPY has better absolute Sharpe ratios in both regimes

## Key Insights

| Finding | Implication |
|---|---|
| XLY shows expected economic pattern | Economic intuition is correct |
| Performs better when inventories falling | Falling inventories = strong consumer demand = good for XLY |
| Pattern matches bullwhip effect theory | Discretionary items most sensitive to demand swings |
| Statistical significance not achieved | Cannot recommend trading strategy with confidence |
| Higher volatility requires more data | Need longer history or lower-vol instruments |

## Limitations

1. **Amazon Concentration**: Amazon is 25%+ of XLY, distorting sector dynamics
2. **Statistical Insignificance**: p = 0.379 means pattern could be random
3. **Higher Volatility**: XLY's ~6% monthly std requires more data for significance
4. **Aggregation Issues**: RETAILIRSA covers all retail, not just discretionary
5. **Structural Changes**: E-commerce has changed retail inventory dynamics
