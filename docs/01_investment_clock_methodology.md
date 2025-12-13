# Investment Clock Methodology

## Overview

The Investment Clock is a macroeconomic analysis and asset allocation framework that relates asset class performance to stages of the economic cycle. This document covers both the original Merrill Lynch Investment Clock and the Royal London adaptation.

## Historical Background

- **Origin**: The concept of positioning investments around a clock face representing the business cycle traces back to 1937
- **Modern Development**: Trevor Greetham developed the comprehensive "Investment Clock" model while at Merrill Lynch
- **Seminal Publication**: "The Investment Clock - Making Money from Macro" (Merrill Lynch, November 10, 2004)
- **Authors**: Trevor Greetham, Michael Hartnett
- **Current Implementation**: Trevor Greetham now heads Multi-Asset at Royal London Asset Management (RLAM), managing approximately £105 billion in multi-asset funds

## Core Framework

### The Two Key Indicators

The Investment Clock uses two primary macroeconomic variables:

1. **Output Gap (Growth)**: The gap between actual economic output and potential output
   - Positive output gap: Economy operating above capacity (overheating)
   - Negative output gap: Economy operating below capacity (slack)
   - Source: OECD output gap estimates, GDP data

2. **Inflation (CPI)**: Consumer Price Index year-over-year change
   - Rising inflation: Prices accelerating
   - Falling inflation: Prices decelerating or deflation
   - Source: Bureau of Labor Statistics CPI data

### The Four Economic Phases

The clock divides the economic cycle into four quadrants:

| Phase | Growth | Inflation | Best Asset Class | Equity Style |
|-------|--------|-----------|------------------|--------------|
| **Reflation** | Below trend (falling) | Falling | Bonds | Defensive Growth |
| **Recovery** | Above trend (rising) | Low/Falling | Stocks | Cyclical Growth |
| **Overheat** | Above trend | Rising | Commodities | Cyclical Value |
| **Stagflation** | Below trend (falling) | Rising | Cash | Defensive Value |

### Phase Characteristics

#### Phase 1: Reflation (Bottom-Left)
- GDP growth is sluggish
- Excess capacity drives down commodity prices and inflation
- Profits are weak, real yields drop
- Central banks cut short rates to stimulate growth
- Yield curves steepen
- **Best Asset**: Bonds

#### Phase 2: Recovery (Top-Left)
- Central bank easing takes effect
- Growth rebounds while inflation remains low
- "Goldilocks" phase: best environment for stocks
- Companies making profits, share prices rise
- Loose monetary policy supports risk assets
- **Best Asset**: Stocks

#### Phase 3: Overheat (Top-Right)
- Productivity growth slows
- GDP gap closes, economy hits supply constraints
- Inflation rises
- Central banks hike rates
- Bond market enters bear market
- Commodity prices rise sharply
- **Best Asset**: Commodities

#### Phase 4: Stagflation (Bottom-Right)
- GDP growth slows while inflation persists
- Productivity falling, wage-price spiral develops
- Companies raise prices to protect margins
- Central banks maintain high rates
- Yield curve inverts
- **Best Asset**: Cash

## Asset Allocation by Phase

### Asset Returns Pattern

The original Merrill Lynch research (1973-2004 data) found:

| Phase | Asset Return Ranking |
|-------|---------------------|
| Reflation | Bonds > Cash > Stocks > Commodities |
| Recovery | Stocks > Bonds > Cash > Commodities |
| Overheat | Commodities > Stocks > Cash > Bonds |
| Stagflation | Cash > Commodities > Bonds > Stocks |

### Sector Rotation

The clock also guides equity sector allocation:

| Phase | Favored Sectors |
|-------|----------------|
| Reflation | Consumer Staples, Healthcare, Utilities |
| Recovery | Consumer Discretionary, Technology, Financials |
| Overheat | Energy, Materials, Industrials |
| Stagflation | Utilities, Healthcare, Consumer Staples |

## Quantitative Methodology

### Original Merrill Lynch Approach

1. **Data Period**: 30+ years of U.S. data (1973-2004)
2. **Phase Classification**:
   - Growth: Direction relative to trend (OECD output gap)
   - Inflation: Direction of CPI YoY change
3. **Statistical Testing**: One-way ANOVA and paired T-tests
4. **Average Phase Duration**: ~20 months (6-year full cycle)

### Scoring System (ML Growth/Inflation Scorecard)

**Output Gap Scoring**:
- +1 if growth is 1% or more above trend
- -1 if growth is 1% or more below trend
- 0 otherwise

**Inflation Scoring**:
- +1 if indicator above 6-month and 12-month moving average
- -1 if below both moving averages

### Key Indicators Used by RLAM

Royal London's implementation incorporates:
- Earnings data
- Housing indicators
- Money supply metrics
- Survey data (ISM, PMI)
- Employment statistics
- Wage data
- OECD leading indicators
- Consensus GDP forecasts

## Statistical Significance

The original Merrill Lynch research found:
- All four asset classes showed statistically significant return differences across phases
- Less than 0.1% probability that results occurred by chance
- Model works best for broad asset rotation
- Some sectors (Oil & Gas, Consumer Discretionary) explained more consistently than others (Telecoms, Utilities)

## Sources

### Primary Sources
- [Royal London Investment Clock](https://www.rlam.com/uk/intermediaries/our-capabilities/multi-asset/investment-clock/)
- [Royal London Adviser - Investment Clock](https://adviser.royallondon.com/investment/our-investment-options/governed-range/governed-portfolios/investment-clock/)
- [The Investment Clock - Trevor Greetham (Original ML Report)](https://silo.tips/download/the-investment-clock)

### Secondary Analysis
- [Dr Wealth - Investment Clock by Trevor Greetham](https://drwealth.com/investment-clock-by-trevor-greetham/)
- [Macro Ops - The Merrill Lynch Investment Clock](https://macro-ops.com/the-investment-clock/)
- [Moomoo - What is Merrill Lynch's Investment Clock?](https://www.moomoo.com/us/learn/detail-what-is-merrill-lynch-s-investment-clock-59567-220659016)

### Academic Research
- [UC Berkeley - Investment based on Merrill Lynch Investment Cycle](https://www.stat.berkeley.edu/~aldous/Research/Ugrad/Tiantian_first_draft.pdf)
- [Investment Clock Feasibility Study - China Market](https://webofproceedings.org/proceedings_series/ESSP/ASSAH%202021/DAS25146.pdf)
