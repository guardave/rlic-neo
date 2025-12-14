# Free Data Sources for RLIC Enhancement

## Overview

This document summarizes publicly available free data sources for economic indicators, sector/industry returns, commodities, and bonds - essential for the Investment Clock enhancement project.

---

## 1. Economic Indicators

### Tier 1: Best Free Sources

| Source | Description | Access Method | Coverage |
|--------|-------------|---------------|----------|
| **[FRED](https://fred.stlouisfed.org/)** | Federal Reserve Economic Data | `pandas-datareader`, API | 500,000+ series, US & global |
| **[OECD](https://data.oecd.org/)** | OECD Statistics | `pandas-datareader`, API | CLI, GDP, inflation, 40+ countries |
| **[World Bank](https://data.worldbank.org/)** | World Development Indicators | `pandas-datareader`, API | 200+ countries, development data |
| **[IMF](https://www.imf.org/en/data)** | International Financial Statistics | API, download | Global macro, WEO projections |

### FRED Key Series (Already Using)

```python
# Growth indicators
'INDPRO'      # Industrial Production
'GDPC1'       # Real GDP
'TCU'         # Capacity Utilization
'CFNAI'       # Chicago Fed National Activity Index
'USSLIND'     # Leading Economic Index

# Inflation
'CPIAUCSL'    # CPI All Urban
'CPILFESL'    # Core CPI
'PCEPI'       # PCE Price Index

# Yield curve / Rates
'T10Y3M'      # 10Y-3M Spread
'T10Y2Y'      # 10Y-2Y Spread
'FEDFUNDS'    # Fed Funds Rate
'GS10'        # 10Y Treasury

# Credit spreads
'BAA10Y'      # BAA Corporate Spread
'AAA10Y'      # AAA Corporate Spread
'BAMLC0A0CM'  # ICE BofA US Corporate OAS

# Labor
'UNRATE'      # Unemployment Rate
'ICSA'        # Initial Claims
'PAYEMS'      # Nonfarm Payrolls

# Housing
'PERMIT'      # Building Permits
'HOUST'       # Housing Starts

# Money
'M2SL'        # M2 Money Supply
```

### pandas-datareader Supported Sources

From [official documentation](https://pandas-datareader.readthedocs.io/en/latest/remote_data.html):
- `fred` - Federal Reserve Economic Data
- `famafrench` - Fama-French Data Library
- `oecd` - OECD Statistics
- `eurostat` - Eurostat
- `yahoo` - Yahoo Finance (may be unreliable)
- `stooq` - Stooq
- `tiingo` - Tiingo (requires API key)
- `av-*` - Alpha Vantage (requires API key)
- `econdb` - EconDB
- `quandl` - Quandl/Nasdaq Data Link

---

## 2. Sector/Industry Returns

### Tier 1: Kenneth French Data Library (BEST for Long History)

**Source**: [https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

| Dataset | Industries | History |
|---------|------------|---------|
| **5 Industry Portfolios** | Cnsmr, Manuf, HiTec, Hlth, Other | 1926-present |
| **10 Industry Portfolios** | NoDur, Durbl, Manuf, Enrgy, HiTec, Telcm, Shops, Hlth, Utils, Other | 1926-present |
| **12 Industry Portfolios** | More granular | 1926-present |
| **17 Industry Portfolios** | More granular | 1926-present |
| **30 Industry Portfolios** | More granular | 1926-present |
| **38 Industry Portfolios** | More granular | 1926-present |
| **48 Industry Portfolios** | Most granular | 1926-present |
| **49 Industry Portfolios** | Most granular | 1926-present |

**Access via pandas-datareader**:
```python
from pandas_datareader.famafrench import get_available_datasets
import pandas_datareader.data as web

# List all 297 available datasets
datasets = get_available_datasets()

# Get 12 industry portfolios
ds = web.DataReader('12_Industry_Portfolios', 'famafrench')
# ds[0] = Value-weighted returns
# ds[1] = Equal-weighted returns
# ds[2] = Value-weighted returns (annual)
# etc.
```

**12 Industry Portfolio Mapping**:
| Code | Industry | Comparable S&P Sector |
|------|----------|----------------------|
| NoDur | Consumer NonDurables | Consumer Staples |
| Durbl | Consumer Durables | Consumer Discretionary |
| Manuf | Manufacturing | Industrials |
| Enrgy | Oil, Gas, Coal | Energy |
| Chems | Chemicals | Materials |
| BusEq | Business Equipment | Technology |
| Telcm | Telecom | Communication Services |
| Utils | Utilities | Utilities |
| Shops | Retail | Consumer Discretionary |
| Hlth  | Healthcare | Healthcare |
| Money | Finance | Financials |
| Other | Other | - |

### Tier 2: Sector ETFs (Shorter History)

**Source**: Yahoo Finance via `yfinance`

| ETF | Sector | Inception |
|-----|--------|-----------|
| XLK | Technology | 1998-12 |
| XLF | Financials | 1998-12 |
| XLE | Energy | 1998-12 |
| XLV | Healthcare | 1998-12 |
| XLY | Consumer Discretionary | 1998-12 |
| XLP | Consumer Staples | 1998-12 |
| XLI | Industrials | 1998-12 |
| XLB | Materials | 1998-12 |
| XLU | Utilities | 1998-12 |
| XLRE | Real Estate | 2015-10 |
| XLC | Communication Services | 2018-06 |

### Tier 3: Other Factor Data (Fama-French)

```python
# Also available from Fama-French:
'F-F_Research_Data_Factors'        # Market, SMB, HML
'F-F_Research_Data_5_Factors_2x3'  # Mkt, SMB, HML, RMW, CMA
'F-F_Momentum_Factor'              # Momentum factor
'Portfolios_Formed_on_BE-ME'       # Book-to-Market portfolios
'Portfolios_Formed_on_ME'          # Size portfolios
```

---

## 3. Commodities

### Tier 1: Yahoo Finance (via yfinance)

| Ticker | Commodity | Type |
|--------|-----------|------|
| `GC=F` | Gold | Futures |
| `SI=F` | Silver | Futures |
| `CL=F` | Crude Oil (WTI) | Futures |
| `BZ=F` | Brent Crude | Futures |
| `NG=F` | Natural Gas | Futures |
| `HG=F` | Copper | Futures |
| `ZC=F` | Corn | Futures |
| `ZW=F` | Wheat | Futures |
| `ZS=F` | Soybeans | Futures |

**ETF Alternatives** (longer history for some):
| ETF | Commodity | Inception |
|-----|-----------|-----------|
| GLD | Gold | 2004-11 |
| SLV | Silver | 2006-04 |
| USO | Oil | 2006-04 |
| DBC | Commodity Index | 2006-02 |
| PDBC | Commodity Index | 2014-11 |

### Tier 2: FRED Commodity Prices

```python
# Available on FRED
'GOLDAMGBD228NLBM'  # Gold Price (London PM Fix)
'DCOILWTICO'        # WTI Crude Oil
'DCOILBRENTEU'      # Brent Crude Oil
'PNRGINDEXM'        # Global Energy Price Index
'PALLFNFINDEXM'     # All Commodity Price Index
```

### Tier 3: API Services (Free Tiers)

| Service | Free Tier | Coverage |
|---------|-----------|----------|
| [Commodities-API](https://commodities-api.com/) | 100 req/month | Oil, Gold, Agri |
| [Metals-API](https://metals-api.com/) | 50 req/month | Precious metals |
| [Nasdaq Data Link](https://data.nasdaq.com/) | Limited | 100+ commodities |
| [Alpha Vantage](https://www.alphavantage.co/) | 25 req/day | Limited commodities |

---

## 4. Bonds

### Tier 1: FRED (Yields & Spreads)

```python
# Treasury Yields
'DGS1'    # 1-Year Treasury
'DGS2'    # 2-Year Treasury
'DGS5'    # 5-Year Treasury
'DGS10'   # 10-Year Treasury
'DGS30'   # 30-Year Treasury
'DFII10'  # 10-Year TIPS

# Corporate Bond Yields
'AAA'            # Moody's AAA Corporate
'BAA'            # Moody's BAA Corporate
'BAMLC0A0CM'     # ICE BofA US Corp OAS (since 1996)
'BAMLH0A0HYM2'   # ICE BofA High Yield OAS
'BAMLC4A0C710YEY' # 7-10Y Corp Effective Yield

# Spreads
'BAA10Y'   # BAA spread over 10Y Treasury
'AAA10Y'   # AAA spread over 10Y Treasury
'T10Y2Y'   # 10Y-2Y Treasury Spread
'T10Y3M'   # 10Y-3M Treasury Spread
```

### Tier 2: Bond ETFs (Yahoo Finance)

| ETF | Description | Inception |
|-----|-------------|-----------|
| TLT | 20+ Year Treasury | 2002-07 |
| IEF | 7-10 Year Treasury | 2002-07 |
| SHY | 1-3 Year Treasury | 2002-07 |
| TIP | TIPS | 2003-12 |
| LQD | Investment Grade Corp | 2002-07 |
| HYG | High Yield Corp | 2007-04 |
| AGG | US Aggregate Bond | 2003-09 |
| BND | Total Bond Market | 2007-04 |

---

## 5. Additional Useful Sources

### Stock Market Indices

```python
# Yahoo Finance tickers
'^GSPC'   # S&P 500
'^DJI'    # Dow Jones Industrial
'^IXIC'   # NASDAQ Composite
'^RUT'    # Russell 2000
'^VIX'    # CBOE Volatility Index
'^TNX'    # 10-Year Treasury Yield
'DX-Y.NYB' # US Dollar Index
```

### International

```python
# Yahoo Finance
'^FTSE'   # FTSE 100 (UK)
'^GDAXI'  # DAX (Germany)
'^N225'   # Nikkei 225 (Japan)
'^HSI'    # Hang Seng (Hong Kong)
'EFA'     # MSCI EAFE ETF
'EEM'     # MSCI Emerging Markets ETF
```

---

## Recommended Data Architecture

### For This Project

| Data Type | Primary Source | Backup Source |
|-----------|----------------|---------------|
| **Economic Indicators** | FRED (pandas-datareader) | OECD |
| **Sector Returns** | Fama-French (pandas-datareader) | Sector ETFs (yfinance) |
| **Commodities** | Yahoo Finance (yfinance) | FRED |
| **Bonds** | FRED | Bond ETFs (yfinance) |
| **Stock Indices** | Yahoo Finance (yfinance) | - |

### Python Access Pattern

```python
import pandas_datareader as pdr
import yfinance as yf

# FRED economic data
fred_data = pdr.get_data_fred(['INDPRO', 'CPIAUCSL', 'T10Y3M'], start, end)

# Fama-French industry portfolios (BEST for sectors)
ff_data = pdr.DataReader('12_Industry_Portfolios', 'famafrench')
sector_returns = ff_data[0]  # Value-weighted monthly returns

# Yahoo Finance prices
prices = yf.download(['GC=F', 'CL=F', 'TLT', '^GSPC'], start=start, end=end)
```

---

## Key Recommendations

### 1. Use Fama-French for Sector Analysis
- **Why**: 100 years of history (1926-present)
- **Limitation**: US only, 12-49 industries (not exact S&P GICS sectors)
- **Good for**: Long-term Investment Clock validation

### 2. Use FRED Extensively
- **Why**: Comprehensive, reliable, free, well-documented
- **Coverage**: 500,000+ series
- **Good for**: All economic indicators, yields, spreads

### 3. Use yfinance for Recent Price Data
- **Why**: Easy, free, covers futures and ETFs
- **Limitation**: Some reliability issues, shorter history
- **Good for**: Recent backtesting, commodities

### 4. Combine Sources for Robustness
- Cross-validate with multiple sources
- Use longest available history for each asset class
- Document data sources clearly

---

## References

- [FRED Data](https://fred.stlouisfed.org/)
- [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- [pandas-datareader Documentation](https://pandas-datareader.readthedocs.io/en/latest/remote_data.html)
- [yfinance Documentation](https://ranaroussi.github.io/yfinance/)
- [World Bank Data](https://data.worldbank.org/)
- [IMF Data](https://www.imf.org/en/data)
- [OECD Data](https://data.oecd.org/)

---
*Document Date: 2025-12-13*
