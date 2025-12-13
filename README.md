# RLIC Enhancement Project

Quantitative-based analysis to improve the Royal London Investment Clock (RLIC).

## Project Type

Research (experimentation, POC, data analysis)

## Overview

The Investment Clock is a macroeconomic framework developed by Trevor Greetham (originally at Merrill Lynch, now at Royal London Asset Management) that relates asset class performance to the economic cycle. This project aims to enhance the traditional Investment Clock methodology using machine learning and quantitative analysis.

## Project Structure

```
rlic/
├── docs/           # Documentation
├── src/            # Source code
├── notebooks/      # Jupyter notebooks for analysis
├── data/           # Data storage
├── cache/          # Cache files
├── temp/           # Temporary files
├── script/         # Utility scripts
└── build/          # Build artifacts
```

## Documentation

- [01_investment_clock_methodology.md](docs/01_investment_clock_methodology.md) - Core RLIC methodology and framework
- [02_limitations_and_criticisms.md](docs/02_limitations_and_criticisms.md) - Known limitations and enhancement opportunities
- [03_outstanding_possibilities.md](docs/03_outstanding_possibilities.md) - Potential deliverables, approaches, and research directions

## Data Sources

### Price Data
- **Yahoo Finance**: Equity indices, bonds, commodities, ETFs

### Economic Data
- **FRED (Federal Reserve Economic Data)**: GDP, CPI, unemployment, interest rates, etc.

## Methodology

### Traditional Investment Clock

The clock divides the economic cycle into four phases based on growth and inflation:

| Phase | Growth | Inflation | Best Asset |
|-------|--------|-----------|------------|
| Reflation | Below trend | Falling | Bonds |
| Recovery | Above trend | Falling | Stocks |
| Overheat | Above trend | Rising | Commodities |
| Stagflation | Below trend | Rising | Cash |

### Enhancement Approaches

1. **Machine Learning Regime Detection**
   - Hidden Markov Models
   - Clustering (K-means, GMM, Hierarchical)
   - Supervised classification

2. **Expanded Feature Set**
   - Yield curve metrics
   - Credit spreads
   - Market volatility
   - Sentiment indicators

3. **Dynamic Phase Boundaries**
   - Data-driven thresholds
   - Probabilistic phase assignment

## Getting Started

*To be developed*

## License

Private research project

## References

- [Royal London Investment Clock](https://www.rlam.com/uk/intermediaries/our-capabilities/multi-asset/investment-clock/)
- [Merrill Lynch - The Investment Clock (2004)](https://silo.tips/download/the-investment-clock)
