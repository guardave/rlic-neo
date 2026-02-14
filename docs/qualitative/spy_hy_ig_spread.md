# SPY vs HY-IG Credit Spread

## What is the HY-IG Credit Spread?

The **HY-IG Credit Spread** is the difference between the ICE BofA US High Yield
Option-Adjusted Spread (OAS) and the ICE BofA US Corporate (Investment Grade) OAS.
Both are measured in basis points (bps) relative to Treasuries.

**Formula**: HY-IG Spread = BAMLH0A0HYM2 - BAMLC0A0CM

**Key Characteristics:**
- **Risk Premium**: Measures the extra yield investors demand for holding junk bonds
  over investment-grade corporate bonds
- **Risk Barometer**: Widens during market stress (risk-off), tightens during calm
  (risk-on)
- **Daily Data**: Available from FRED, published daily by ICE/BofA
- **History**: Available from December 1996 (~27+ years)

## Economic Signal Interpretation

| Spread Level | Interpretation |
|---|---|
| < 2 bps | Extremely tight, excessive risk-taking, potential complacency |
| 2-4 bps | Normal range, healthy credit conditions |
| 4-6 bps | Elevated stress, risk-off sentiment building |
| > 6 bps | Crisis territory (GFC peaked ~15 bps, COVID ~7 bps) |

## Why Compare with SPY?

**Direct Link**: Credit spreads reflect the same risk appetite that drives equity
valuations. When investors flee high-yield bonds for safety, they typically also sell
equities.

**Transmission Mechanism**:
1. Rising HY-IG spreads signal deteriorating credit conditions
2. Companies face higher borrowing costs, compressing margins
3. Default risk rises, reducing equity terminal values
4. Flight-to-quality rotation moves capital from equities to Treasuries

## Key Research Finding: Concurrent/Confirmatory Signal

::: warning
**Our Analysis Found**: The HY-IG spread has a very strong concurrent relationship
with SPY returns. The MoM change shows r=-0.667 (p<0.0001) — one of the strongest
concurrent correlations in the RLIC suite.

All 12 significant lags are at 0 or negative (target leads), with best at lag -1
(r=-0.278). This means the spread moves slightly AFTER equity markets.

**Regime analysis is highly significant:**
- **Spread Tightening** (YoY < 0): Mean return +1.66%/mo, Sharpe 1.89
- **Spread Widening** (YoY >= 0): Mean return -0.09%/mo, Sharpe -0.06
- Regime difference: t=3.71, p=0.0002

**Strategy performance** (long during tightening, cash during widening):
- Strategy Sharpe: 1.07 vs Benchmark Sharpe: 0.65
- Exposure: 52.7% — only in market half the time with near-equal total returns

This is a **powerful contemporaneous regime indicator** for risk management, but
NOT a leading signal for timing entries.
:::

## Academic and Professional Research

| Finding | Source | Implication |
|---|---|---|
| Credit spreads are countercyclical and widen before recessions | Gilchrist & Zakrajsek (2012) | Potential leading indicator for the real economy |
| High-yield spread changes contain equity-relevant information | Collin-Dufresne et al. (2001) | Credit markets process risk information |
| OAS contains information beyond default risk (liquidity, sentiment) | Longstaff et al. (2005) | Captures broad risk appetite, not just defaults |
| Credit markets often lead equity markets at turning points | Various practitioner research | Useful at regime changes, less so mid-regime |
| HY spreads are among top macro risk indicators | Goldman Sachs, JPM Research | Widely used in institutional risk management |

## Limitations

1. **Mostly Concurrent**: Best correlation at lag -1 suggests the spread confirms
   rather than predicts equity moves
2. **Liquidity Premium**: Spread changes partly reflect bond market liquidity, not
   just credit risk
3. **Composition Changes**: Index composition changes over time as bonds mature and
   new ones are issued
4. **Crowded Signal**: HY-IG spread is widely watched; may already be priced into
   equity markets
5. **Regime Persistence**: Tightening/widening regimes can persist for years,
   reducing the number of independent regime transitions
