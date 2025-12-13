#!/usr/bin/env python3
"""
RLIC Enhancement: Data Fetching and Exploratory Analysis

This script:
1. Fetches price data from Yahoo Finance
2. Fetches economic data from FRED
3. Computes Investment Clock indicators
4. Classifies phases using traditional methodology
5. Analyzes asset performance by phase
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)

START_DATE = '1990-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

print("=" * 60)
print("RLIC Enhancement: Data Fetching and Analysis")
print("=" * 60)
print(f"Date range: {START_DATE} to {END_DATE}")
print()


# =============================================================================
# 1. Fetch Price Data from Yahoo Finance
# =============================================================================
print("1. FETCHING PRICE DATA FROM YAHOO FINANCE")
print("-" * 40)

PRICE_TICKERS = {
    'sp500': '^GSPC',
    'treasury_10y': '^TNX',
    'gold': 'GC=F',
    'crude_oil': 'CL=F',
    'spy': 'SPY',
    'tlt': 'TLT',
    'gld': 'GLD',
    'dbc': 'DBC',
    'nasdaq': '^IXIC',
    'russell2000': '^RUT',
    'vix': '^VIX',
}

ticker_symbols = list(PRICE_TICKERS.values())
print(f"Downloading {len(ticker_symbols)} tickers...")

raw_data = yf.download(ticker_symbols, start=START_DATE, end=END_DATE, progress=False)

# Extract Close prices
if isinstance(raw_data.columns, pd.MultiIndex):
    prices = raw_data['Close'].copy()
else:
    prices = raw_data.copy()

# Rename columns
reverse_map = {v: k for k, v in PRICE_TICKERS.items()}
prices.columns = [reverse_map.get(c, c) for c in prices.columns]

print(f"Downloaded {len(prices)} days of price data")
print("First available dates:")
for col in prices.columns:
    first_valid = prices[col].first_valid_index()
    if first_valid:
        print(f"  {col}: {first_valid.date()}")

# Save
prices.to_parquet(DATA_DIR / 'prices.parquet')
print(f"\nSaved to {DATA_DIR / 'prices.parquet'}")
print()


# =============================================================================
# 2. Fetch Economic Data from FRED
# =============================================================================
print("2. FETCHING ECONOMIC DATA FROM FRED")
print("-" * 40)

FRED_SERIES = {
    'gdp_real': 'GDPC1',
    'industrial_prod': 'INDPRO',
    'capacity_util': 'TCU',
    'cpi_all': 'CPIAUCSL',
    'cpi_core': 'CPILFESL',
    'pce_price': 'PCEPI',
    'unemployment': 'UNRATE',
    'nonfarm_payrolls': 'PAYEMS',
    'fed_funds': 'FEDFUNDS',
    'treasury_10y_rate': 'GS10',
    'treasury_2y_rate': 'GS2',
    'spread_10y2y': 'T10Y2Y',
    'm2': 'M2SL',
    'consumer_sentiment': 'UMCSENT',
    'leading_index': 'USSLIND',
}

fred_data = {}
for name, series_id in FRED_SERIES.items():
    try:
        data = pdr.get_data_fred(series_id, start=START_DATE, end=END_DATE)
        fred_data[name] = data[series_id]
        print(f"  ✓ {name}: {len(data)} obs")
    except Exception as e:
        print(f"  ✗ {name}: {e}")

fred_df = pd.DataFrame(fred_data)
fred_df.to_parquet(DATA_DIR / 'fred_data.parquet')
print(f"\nSaved to {DATA_DIR / 'fred_data.parquet'}")
print()


# =============================================================================
# 3. Build Monthly Dataset with Investment Clock Indicators
# =============================================================================
print("3. BUILDING MONTHLY DATASET")
print("-" * 40)

# Resample to monthly
fred_monthly = fred_df.resample('M').last()
prices_monthly = prices.resample('M').last()

monthly = pd.DataFrame()

# Growth indicators
if 'cpi_all' in fred_monthly.columns:
    monthly['cpi_yoy'] = fred_monthly['cpi_all'].pct_change(12) * 100
    print("  ✓ CPI YoY computed")

if 'industrial_prod' in fred_monthly.columns:
    monthly['ip_yoy'] = fred_monthly['industrial_prod'].pct_change(12) * 100
    print("  ✓ Industrial Production YoY computed")

if 'gdp_real' in fred_monthly.columns:
    monthly['gdp_yoy'] = fred_monthly['gdp_real'].pct_change(4) * 100
    print("  ✓ GDP YoY computed")

# Other indicators
for col in ['unemployment', 'fed_funds', 'spread_10y2y', 'consumer_sentiment',
            'capacity_util', 'leading_index']:
    if col in fred_monthly.columns:
        monthly[col] = fred_monthly[col]

# Asset returns
if 'sp500' in prices_monthly.columns:
    monthly['sp500_ret'] = prices_monthly['sp500'].pct_change() * 100
if 'gold' in prices_monthly.columns:
    monthly['gold_ret'] = prices_monthly['gold'].pct_change() * 100
if 'crude_oil' in prices_monthly.columns:
    monthly['oil_ret'] = prices_monthly['crude_oil'].pct_change() * 100
if 'treasury_10y' in prices_monthly.columns:
    monthly['bond_ret'] = -prices_monthly['treasury_10y'].diff()

print(f"\nMonthly dataset: {monthly.shape[0]} rows, {monthly.shape[1]} columns")
print()


# =============================================================================
# 4. Investment Clock Phase Classification
# =============================================================================
print("4. CLASSIFYING INVESTMENT CLOCK PHASES")
print("-" * 40)

def compute_momentum_signal(series, short_window=6, long_window=12):
    """Returns 1 if rising, -1 if falling, 0 otherwise."""
    short_ma = series.rolling(short_window).mean()
    long_ma = series.rolling(long_window).mean()

    signal = pd.Series(0, index=series.index)
    signal[(series > short_ma) & (series > long_ma)] = 1
    signal[(series < short_ma) & (series < long_ma)] = -1

    return signal


def classify_phase(growth_signal, inflation_signal):
    """Classify Investment Clock phase."""
    phase = pd.Series('Unknown', index=growth_signal.index)

    phase[(growth_signal == -1) & (inflation_signal == -1)] = 'Reflation'
    phase[(growth_signal == 1) & (inflation_signal == -1)] = 'Recovery'
    phase[(growth_signal == 1) & (inflation_signal == 1)] = 'Overheat'
    phase[(growth_signal == -1) & (inflation_signal == 1)] = 'Stagflation'

    return phase


# Compute signals
growth_signal = compute_momentum_signal(monthly['ip_yoy'].dropna())
inflation_signal = compute_momentum_signal(monthly['cpi_yoy'].dropna())

# Align and classify
common_idx = growth_signal.index.intersection(inflation_signal.index)
growth_signal = growth_signal.loc[common_idx]
inflation_signal = inflation_signal.loc[common_idx]
phases = classify_phase(growth_signal, inflation_signal)

monthly['growth_signal'] = growth_signal
monthly['inflation_signal'] = inflation_signal
monthly['phase'] = phases

print("Phase distribution:")
phase_counts = phases.value_counts()
for phase, count in phase_counts.items():
    pct = count / len(phases) * 100
    print(f"  {phase}: {count} months ({pct:.1f}%)")
print()


# =============================================================================
# 5. Asset Performance by Phase
# =============================================================================
print("5. ASSET PERFORMANCE BY PHASE")
print("-" * 40)

valid_data = monthly[['phase', 'sp500_ret', 'gold_ret', 'oil_ret', 'bond_ret']].dropna(subset=['phase'])
valid_data = valid_data[valid_data['phase'] != 'Unknown']

# Mean returns by phase
mean_returns = valid_data.groupby('phase')[['sp500_ret', 'gold_ret', 'oil_ret', 'bond_ret']].mean()
mean_returns.columns = ['Stocks', 'Gold', 'Oil', 'Bonds']

phase_order = ['Reflation', 'Recovery', 'Overheat', 'Stagflation']
mean_returns = mean_returns.reindex(phase_order)

print("Average Monthly Returns by Phase (%):\n")
print(mean_returns.round(3).to_string())
print()

# Ranking
print("Asset Ranking by Phase (best to worst):")
for phase in phase_order:
    if phase in mean_returns.index:
        ranked = mean_returns.loc[phase].sort_values(ascending=False)
        print(f"  {phase}: {' > '.join(ranked.index)}")

print()
print("Expected (from theory):")
print("  Reflation: Bonds > Cash > Stocks > Commodities")
print("  Recovery: Stocks > Bonds > Cash > Commodities")
print("  Overheat: Commodities > Stocks > Cash > Bonds")
print("  Stagflation: Cash > Commodities > Bonds > Stocks")
print()


# =============================================================================
# 6. Phase Duration Analysis
# =============================================================================
print("6. PHASE DURATION ANALYSIS")
print("-" * 40)

phase_changes = phases[phases != phases.shift(1)]
phase_durations = []

for i in range(len(phase_changes) - 1):
    start = phase_changes.index[i]
    end = phase_changes.index[i + 1]
    duration = (end - start).days / 30
    phase_durations.append({
        'phase': phase_changes.iloc[i],
        'start': start,
        'end': end,
        'duration_months': duration
    })

duration_df = pd.DataFrame(phase_durations)
duration_df = duration_df[duration_df['phase'] != 'Unknown']

print("Average Phase Duration (months):\n")
duration_stats = duration_df.groupby('phase')['duration_months'].agg(['mean', 'std', 'min', 'max', 'count'])
print(duration_stats.round(1).to_string())
print()


# =============================================================================
# 7. Create Visualizations
# =============================================================================
print("7. CREATING VISUALIZATIONS")
print("-" * 40)

# Figure 1: Investment Clock Phases Over Time
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

phase_colors = {
    'Reflation': 'blue',
    'Recovery': 'green',
    'Overheat': 'red',
    'Stagflation': 'orange',
    'Unknown': 'gray'
}

# Growth indicator
ax = axes[0]
ax.plot(monthly.index, monthly['ip_yoy'], 'b-', alpha=0.7, linewidth=1)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.fill_between(monthly.index, 0, monthly['ip_yoy'],
                where=monthly['growth_signal']==1, color='green', alpha=0.3, label='Rising')
ax.fill_between(monthly.index, 0, monthly['ip_yoy'],
                where=monthly['growth_signal']==-1, color='red', alpha=0.3, label='Falling')
ax.set_ylabel('IP YoY %')
ax.set_title('Growth Indicator (Industrial Production YoY)', fontweight='bold')
ax.legend(loc='upper right')

# Inflation indicator
ax = axes[1]
ax.plot(monthly.index, monthly['cpi_yoy'], 'r-', alpha=0.7, linewidth=1)
ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5)
ax.fill_between(monthly.index, 0, monthly['cpi_yoy'],
                where=monthly['inflation_signal']==1, color='red', alpha=0.3, label='Rising')
ax.fill_between(monthly.index, 0, monthly['cpi_yoy'],
                where=monthly['inflation_signal']==-1, color='blue', alpha=0.3, label='Falling')
ax.set_ylabel('CPI YoY %')
ax.set_title('Inflation Indicator (CPI YoY)', fontweight='bold')
ax.legend(loc='upper right')

# Phase bands
ax = axes[2]
for phase_name, color in phase_colors.items():
    mask = monthly['phase'] == phase_name
    if mask.any():
        ax.fill_between(monthly.index, 0, 1, where=mask, color=color, alpha=0.7, label=phase_name)
ax.set_ylabel('Phase')
ax.set_title('Investment Clock Phase', fontweight='bold')
ax.legend(loc='upper right', ncol=5)
ax.set_yticks([])

# S&P 500
ax = axes[3]
sp500_plot = prices_monthly['sp500'].loc[monthly.index[0]:monthly.index[-1]]
ax.plot(sp500_plot.index, sp500_plot, 'k-', linewidth=1)
ax.set_ylabel('S&P 500')
ax.set_title('S&P 500 Index', fontweight='bold')
ax.set_xlabel('Date')

plt.tight_layout()
plt.savefig(DATA_DIR / 'investment_clock_phases.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved investment_clock_phases.png")


# Figure 2: Returns by Phase
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(phase_order))
width = 0.2

colors = ['blue', 'green', 'gold', 'brown']
labels = ['Stocks', 'Bonds', 'Gold', 'Oil']

for i, (label, color) in enumerate(zip(labels, colors)):
    if label in mean_returns.columns:
        vals = [mean_returns.loc[p, label] if p in mean_returns.index else 0 for p in phase_order]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color)

ax.set_ylabel('Average Monthly Return (%)')
ax.set_title('Asset Class Performance by Investment Clock Phase', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(phase_order)
ax.legend()
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig(DATA_DIR / 'returns_by_phase.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved returns_by_phase.png")


# Figure 3: Economic Indicators Dashboard
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

indicators = [
    ('gdp_yoy', 'Real GDP YoY %', 'blue'),
    ('cpi_yoy', 'CPI YoY %', 'red'),
    ('unemployment', 'Unemployment Rate %', 'orange'),
    ('fed_funds', 'Fed Funds Rate %', 'purple'),
    ('spread_10y2y', '10Y-2Y Spread %', 'green'),
    ('ip_yoy', 'Industrial Production YoY %', 'brown'),
]

for ax, (col, title, color) in zip(axes.flat, indicators):
    if col in monthly.columns:
        data = monthly[col].dropna()
        ax.plot(data.index, data, color=color, linewidth=1)
        ax.set_title(title, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(DATA_DIR / 'economic_indicators.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved economic_indicators.png")

plt.close('all')
print()


# =============================================================================
# 8. Save Final Dataset
# =============================================================================
print("8. SAVING FINAL DATASET")
print("-" * 40)

monthly.to_parquet(DATA_DIR / 'monthly_with_phases.parquet')
duration_df.to_csv(DATA_DIR / 'phase_durations.csv', index=False)

print(f"  ✓ {DATA_DIR / 'monthly_with_phases.parquet'}")
print(f"  ✓ {DATA_DIR / 'phase_durations.csv'}")
print()


# =============================================================================
# Summary
# =============================================================================
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Data period: {monthly.index[0].date()} to {monthly.index[-1].date()}")
print(f"Total months: {len(monthly)}")
print(f"Phases classified: {len(phases[phases != 'Unknown'])}")
print()
print("Files saved in data/:")
print("  - prices.parquet")
print("  - fred_data.parquet")
print("  - monthly_with_phases.parquet")
print("  - phase_durations.csv")
print("  - investment_clock_phases.png")
print("  - returns_by_phase.png")
print("  - economic_indicators.png")
print()
print("Ready for ML enhancement (Phase 3)!")
