#!/usr/bin/env python3
"""
RLIC Enhancement: Leading Indicators Analysis

This script:
1. Fetches leading indicators from FRED
2. Computes ISM New Orders minus Inventories spread
3. Compares leading indicators vs Industrial Production for phase classification
4. Analyzes which indicator provides better Investment Clock signals
"""

import pandas as pd
import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

START_DATE = '1990-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

print("=" * 70)
print("RLIC Enhancement: Leading Indicators Analysis")
print("=" * 70)
print(f"Date range: {START_DATE} to {END_DATE}")
print()


# =============================================================================
# 1. Fetch Leading Indicators from FRED
# =============================================================================
print("1. FETCHING LEADING INDICATORS FROM FRED")
print("-" * 50)

LEADING_INDICATORS = {
    # Yield Curve spreads (excellent leading indicators)
    'spread_10y3m': 'T10Y3M',             # 10Y-3M spread (best recession predictor)
    'spread_10y2y': 'T10Y2Y',             # 10Y-2Y spread

    # Conference Board Leading Index
    'lei': 'USSLIND',                     # Leading Economic Index

    # Manufacturing New Orders (leading indicator)
    'mfg_new_orders': 'AMTMNO',           # Manufacturers New Orders: Total Manufacturing
    'mfg_new_orders_durable': 'DGORDER',  # Manufacturers New Orders: Durable Goods
    'mfg_inventories': 'AMTMTI',          # Manufacturers Total Inventories

    # Other leading indicators
    'building_permits': 'PERMIT',          # Housing Building Permits
    'initial_claims': 'ICSA',             # Initial Jobless Claims
    'consumer_expectations': 'UMCSENT',   # Consumer Sentiment
    'capacity_util': 'TCU',               # Capacity Utilization (proxy for output gap)

    # Chicago Fed National Activity Index (leading)
    'cfnai': 'CFNAI',                     # Chicago Fed National Activity Index

    # S&P 500 (leading indicator for economy)
    'sp500': 'SP500',                     # S&P 500 Index

    # Money supply (leading)
    'm2': 'M2SL',                         # M2 Money Supply

    # For comparison - our current growth indicator
    'industrial_prod': 'INDPRO',          # Industrial Production (current)

    # Inflation indicator
    'cpi_all': 'CPIAUCSL',                # CPI for inflation signal
}

# Additional OECD CLI if available
OECD_SERIES = {
    'oecd_cli': 'USALOLITONOSTSAM',       # OECD CLI for US (amplitude adjusted)
}

def fetch_fred_data(series_dict, start, end):
    """Fetch multiple FRED series."""
    results = {}
    for name, series_id in series_dict.items():
        try:
            data = pdr.get_data_fred(series_id, start=start, end=end)
            results[name] = data[series_id]
            print(f"  ✓ {name} ({series_id}): {len(data)} obs")
        except Exception as e:
            print(f"  ✗ {name} ({series_id}): {e}")
    return pd.DataFrame(results)

# Fetch main indicators
lead_data = fetch_fred_data(LEADING_INDICATORS, START_DATE, END_DATE)

# Try to fetch OECD CLI
print("\nFetching OECD CLI...")
oecd_data = fetch_fred_data(OECD_SERIES, START_DATE, END_DATE)
if not oecd_data.empty:
    lead_data = pd.concat([lead_data, oecd_data], axis=1)

print(f"\nTotal series fetched: {len(lead_data.columns)}")
print()


# =============================================================================
# 2. Compute Derived Leading Indicators
# =============================================================================
print("2. COMPUTING DERIVED LEADING INDICATORS")
print("-" * 50)

# Resample to monthly (end of month)
monthly = lead_data.resample('M').last()

# Manufacturing New Orders / Inventories ratio (key leading indicator)
if 'mfg_new_orders' in monthly.columns and 'mfg_inventories' in monthly.columns:
    # Ratio of new orders to inventories (>1 = demand exceeding supply)
    monthly['orders_inv_ratio'] = monthly['mfg_new_orders'] / monthly['mfg_inventories']
    # YoY change in the ratio
    monthly['orders_inv_ratio_yoy'] = monthly['orders_inv_ratio'].pct_change(12) * 100
    print("  ✓ New Orders / Inventories ratio computed")

# Chicago Fed National Activity Index (already centered at 0)
if 'cfnai' in monthly.columns:
    monthly['cfnai_3ma'] = monthly['cfnai'].rolling(3).mean()  # 3-month moving average
    print("  ✓ CFNAI 3-month MA computed")

# LEI month-over-month change
if 'lei' in monthly.columns:
    monthly['lei_mom'] = monthly['lei'].pct_change() * 100
    monthly['lei_6m_change'] = monthly['lei'].pct_change(6) * 100
    print("  ✓ LEI momentum indicators computed")

# Initial claims (inverted - lower claims = better economy)
if 'initial_claims' in monthly.columns:
    monthly['initial_claims_inv'] = -monthly['initial_claims']
    monthly['initial_claims_yoy'] = monthly['initial_claims'].pct_change(12) * 100
    print("  ✓ Initial claims indicators computed")

# Building permits YoY
if 'building_permits' in monthly.columns:
    monthly['permits_yoy'] = monthly['building_permits'].pct_change(12) * 100
    print("  ✓ Building permits YoY computed")

# OECD CLI deviation from 100
if 'oecd_cli' in monthly.columns:
    monthly['oecd_cli_deviation'] = monthly['oecd_cli'] - 100
    print("  ✓ OECD CLI deviation computed")

# Industrial Production YoY (our current benchmark)
if 'industrial_prod' in monthly.columns:
    monthly['ip_yoy'] = monthly['industrial_prod'].pct_change(12) * 100
    print("  ✓ Industrial Production YoY computed")

# CPI YoY for inflation
if 'cpi_all' in monthly.columns:
    monthly['cpi_yoy'] = monthly['cpi_all'].pct_change(12) * 100
    print("  ✓ CPI YoY computed")

print()


# =============================================================================
# 3. Define Growth Signal Functions
# =============================================================================
print("3. DEFINING GROWTH SIGNAL METHODS")
print("-" * 50)

def momentum_signal(series, short_window=6, long_window=12):
    """
    Original method: Returns 1 if rising (above both MAs), -1 if falling, 0 otherwise.
    """
    short_ma = series.rolling(short_window).mean()
    long_ma = series.rolling(long_window).mean()

    signal = pd.Series(0, index=series.index)
    signal[(series > short_ma) & (series > long_ma)] = 1
    signal[(series < short_ma) & (series < long_ma)] = -1

    return signal


def threshold_signal(series, upper=0, lower=0):
    """
    Simple threshold method: Above upper = 1, below lower = -1.
    Useful for indicators with natural thresholds (e.g., ISM 50, CLI 100).
    """
    signal = pd.Series(0, index=series.index)
    signal[series > upper] = 1
    signal[series < lower] = -1
    return signal


def direction_signal(series, lookback=3):
    """
    Direction method: Based on recent direction of change.
    1 if trending up, -1 if trending down.
    """
    diff = series.diff(lookback)
    signal = pd.Series(0, index=series.index)
    signal[diff > 0] = 1
    signal[diff < 0] = -1
    return signal


def classify_phase(growth_signal, inflation_signal):
    """Classify Investment Clock phase."""
    phase = pd.Series('Unknown', index=growth_signal.index)

    phase[(growth_signal == -1) & (inflation_signal == -1)] = 'Reflation'
    phase[(growth_signal == 1) & (inflation_signal == -1)] = 'Recovery'
    phase[(growth_signal == 1) & (inflation_signal == 1)] = 'Overheat'
    phase[(growth_signal == -1) & (inflation_signal == 1)] = 'Stagflation'

    return phase

print("Signal methods defined:")
print("  - momentum_signal: Above/below 6M and 12M moving averages")
print("  - threshold_signal: Above/below fixed threshold")
print("  - direction_signal: Recent direction of change")
print()


# =============================================================================
# 4. Compare Growth Indicators for Phase Classification
# =============================================================================
print("4. COMPARING GROWTH INDICATORS")
print("-" * 50)

# Define growth indicators to test
GROWTH_INDICATORS = {
    'IP YoY (Current)': ('ip_yoy', 'momentum'),
    'LEI 6M Change': ('lei_6m_change', 'threshold_0'),
    'Yield Curve 10Y-3M': ('spread_10y3m', 'threshold_0'),
    'Yield Curve 10Y-2Y': ('spread_10y2y', 'threshold_0'),
    'CFNAI': ('cfnai', 'threshold_0'),
    'CFNAI 3MA': ('cfnai_3ma', 'threshold_0'),
    'Orders/Inv Ratio YoY': ('orders_inv_ratio_yoy', 'threshold_0'),
    'Building Permits YoY': ('permits_yoy', 'momentum'),
    'Capacity Utilization': ('capacity_util', 'threshold_80'),  # 80% is approx. average
    'OECD CLI': ('oecd_cli_deviation', 'threshold_0'),
}

# Compute inflation signal (common for all)
inflation_signal = momentum_signal(monthly['cpi_yoy'].dropna())

# Store results
results = {}
phase_data = {}

for name, (col, method) in GROWTH_INDICATORS.items():
    if col not in monthly.columns:
        print(f"  ✗ {name}: Column '{col}' not available")
        continue

    series = monthly[col].dropna()

    # Compute growth signal based on method
    if method == 'momentum':
        growth_sig = momentum_signal(series)
    elif method == 'threshold_50':
        growth_sig = threshold_signal(series, upper=50, lower=50)
    elif method == 'threshold_0':
        growth_sig = threshold_signal(series, upper=0, lower=0)
    elif method == 'threshold_80':
        growth_sig = threshold_signal(series, upper=80, lower=80)
    else:
        growth_sig = momentum_signal(series)

    # Align with inflation signal
    common_idx = growth_sig.index.intersection(inflation_signal.index)
    growth_aligned = growth_sig.loc[common_idx]
    inflation_aligned = inflation_signal.loc[common_idx]

    # Classify phases
    phases = classify_phase(growth_aligned, inflation_aligned)

    # Calculate statistics
    phase_counts = phases.value_counts()
    unknown_pct = phase_counts.get('Unknown', 0) / len(phases) * 100

    # Store results
    results[name] = {
        'total_months': len(phases),
        'unknown_pct': unknown_pct,
        'reflation': phase_counts.get('Reflation', 0),
        'recovery': phase_counts.get('Recovery', 0),
        'overheat': phase_counts.get('Overheat', 0),
        'stagflation': phase_counts.get('Stagflation', 0),
    }

    phase_data[name] = phases

    print(f"  ✓ {name}: {unknown_pct:.1f}% unknown, {len(phases)} months")

print()


# =============================================================================
# 5. Analyze Phase Distribution
# =============================================================================
print("5. PHASE DISTRIBUTION COMPARISON")
print("-" * 50)

results_df = pd.DataFrame(results).T
results_df['classified_pct'] = 100 - results_df['unknown_pct']
results_df = results_df.sort_values('unknown_pct')

print("\nIndicator Comparison (sorted by % classified):\n")
print(results_df[['total_months', 'classified_pct', 'unknown_pct',
                   'reflation', 'recovery', 'overheat', 'stagflation']].round(1).to_string())
print()


# =============================================================================
# 6. Analyze Asset Returns by Phase for Each Indicator
# =============================================================================
print("6. ASSET RETURNS BY PHASE FOR EACH INDICATOR")
print("-" * 50)

# Load price data
prices = pd.read_parquet(DATA_DIR / 'prices.parquet')
prices_monthly = prices.resample('M').last()

# Calculate returns
returns = pd.DataFrame()
returns['sp500_ret'] = prices_monthly['sp500'].pct_change() * 100
returns['gold_ret'] = prices_monthly['gold'].pct_change() * 100
returns['oil_ret'] = prices_monthly['crude_oil'].pct_change() * 100
if 'treasury_10y' in prices_monthly.columns:
    returns['bond_ret'] = -prices_monthly['treasury_10y'].diff()

# Analyze returns by phase for each indicator
return_analysis = {}

for name, phases in phase_data.items():
    # Align returns with phases
    common_idx = phases.index.intersection(returns.index)
    aligned_returns = returns.loc[common_idx].copy()
    aligned_returns['phase'] = phases.loc[common_idx]

    # Filter out Unknown
    valid = aligned_returns[aligned_returns['phase'] != 'Unknown']

    if len(valid) > 0:
        mean_returns = valid.groupby('phase')[['sp500_ret', 'gold_ret', 'oil_ret', 'bond_ret']].mean()
        return_analysis[name] = mean_returns

# Print comparison for key phases
print("\nStocks Performance by Phase (should be: Recovery > Overheat > Reflation > Stagflation):\n")
stock_comparison = pd.DataFrame({
    name: data['sp500_ret'] if 'sp500_ret' in data.columns else pd.Series()
    for name, data in return_analysis.items()
}).T
print(stock_comparison.round(2).to_string())

print("\nOil Performance by Phase (should be: Overheat >> others):\n")
oil_comparison = pd.DataFrame({
    name: data['oil_ret'] if 'oil_ret' in data.columns else pd.Series()
    for name, data in return_analysis.items()
}).T
print(oil_comparison.round(2).to_string())
print()


# =============================================================================
# 7. Calculate Phase Quality Score
# =============================================================================
print("7. PHASE QUALITY SCORE")
print("-" * 50)

def calculate_quality_score(returns_by_phase):
    """
    Calculate a quality score based on how well the phases match theoretical expectations.

    Expected patterns:
    - Stocks: Recovery > Overheat > Reflation > Stagflation
    - Oil: Overheat >> others
    - Bonds: Reflation > Stagflation > Recovery > Overheat
    - Stagflation: Stocks negative, Gold positive
    """
    score = 0
    max_score = 0

    try:
        # Check if Overheat has best Oil returns
        if 'Overheat' in returns_by_phase.index and 'oil_ret' in returns_by_phase.columns:
            oil_overheat = returns_by_phase.loc['Overheat', 'oil_ret']
            oil_others = returns_by_phase.loc[returns_by_phase.index != 'Overheat', 'oil_ret'].mean()
            if oil_overheat > oil_others:
                score += 2
            max_score += 2

        # Check if Stagflation has worst Stock returns
        if 'Stagflation' in returns_by_phase.index and 'sp500_ret' in returns_by_phase.columns:
            stock_stagflation = returns_by_phase.loc['Stagflation', 'sp500_ret']
            stock_others = returns_by_phase.loc[returns_by_phase.index != 'Stagflation', 'sp500_ret'].mean()
            if stock_stagflation < stock_others:
                score += 2
            if stock_stagflation < 0:  # Negative returns in stagflation
                score += 1
            max_score += 3

        # Check if Recovery/Overheat have positive stock returns
        for phase in ['Recovery', 'Overheat']:
            if phase in returns_by_phase.index and 'sp500_ret' in returns_by_phase.columns:
                if returns_by_phase.loc[phase, 'sp500_ret'] > 0:
                    score += 1
                max_score += 1

        # Check if Reflation has positive bond returns
        if 'Reflation' in returns_by_phase.index and 'bond_ret' in returns_by_phase.columns:
            if returns_by_phase.loc['Reflation', 'bond_ret'] > 0:
                score += 1
            max_score += 1

    except Exception as e:
        pass

    return score, max_score

print("Quality Score (how well phases match theoretical expectations):\n")
quality_scores = {}
for name, returns_by_phase in return_analysis.items():
    score, max_score = calculate_quality_score(returns_by_phase)
    pct_classified = results_df.loc[name, 'classified_pct'] if name in results_df.index else 0

    # Combined score: quality * coverage
    combined = (score / max_score * 100) * (pct_classified / 100) if max_score > 0 else 0

    quality_scores[name] = {
        'quality_score': score,
        'max_score': max_score,
        'quality_pct': score / max_score * 100 if max_score > 0 else 0,
        'classified_pct': pct_classified,
        'combined_score': combined
    }

quality_df = pd.DataFrame(quality_scores).T.sort_values('combined_score', ascending=False)
print(quality_df.round(1).to_string())
print()


# =============================================================================
# 8. Visualizations
# =============================================================================
print("8. CREATING VISUALIZATIONS")
print("-" * 50)

# Figure 1: Leading Indicators Time Series
fig, axes = plt.subplots(4, 2, figsize=(14, 12))

indicators_to_plot = [
    ('ip_yoy', 'Industrial Production YoY % (Current)', 'blue'),
    ('cfnai', 'Chicago Fed National Activity Index', 'green'),
    ('lei_6m_change', 'LEI 6-Month Change %', 'purple'),
    ('spread_10y3m', 'Yield Curve 10Y-3M Spread', 'orange'),
    ('orders_inv_ratio_yoy', 'Orders/Inventories Ratio YoY %', 'brown'),
    ('permits_yoy', 'Building Permits YoY %', 'teal'),
    ('capacity_util', 'Capacity Utilization %', 'red'),
    ('cpi_yoy', 'CPI YoY % (Inflation)', 'magenta'),
]

for ax, (col, title, color) in zip(axes.flat, indicators_to_plot):
    if col in monthly.columns:
        data = monthly[col].dropna()
        ax.plot(data.index, data, color=color, linewidth=1)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Add threshold line for Capacity Utilization
        if col == 'capacity_util':
            ax.axhline(y=80, color='gray', linestyle='--', alpha=0.7, label='80% threshold')

        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_ylabel(col.split('_')[0].upper())
    else:
        ax.set_title(f'{title} (N/A)', fontweight='bold', fontsize=10)
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig(DATA_DIR / 'leading_indicators_timeseries.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved leading_indicators_timeseries.png")


# Figure 2: Indicator Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Classification rate
ax = axes[0]
sorted_results = results_df.sort_values('classified_pct', ascending=True)
colors = plt.cm.RdYlGn(sorted_results['classified_pct'] / 100)
bars = ax.barh(range(len(sorted_results)), sorted_results['classified_pct'], color=colors)
ax.set_yticks(range(len(sorted_results)))
ax.set_yticklabels(sorted_results.index)
ax.set_xlabel('% of Months Classified (not Unknown)')
ax.set_title('Phase Classification Rate by Indicator', fontweight='bold')
ax.axvline(x=66, color='red', linestyle='--', alpha=0.7, label='Current (IP YoY): 66%')
ax.legend()

# Quality score
ax = axes[1]
sorted_quality = quality_df.sort_values('combined_score', ascending=True)
colors = plt.cm.RdYlGn(sorted_quality['combined_score'] / sorted_quality['combined_score'].max())
bars = ax.barh(range(len(sorted_quality)), sorted_quality['combined_score'], color=colors)
ax.set_yticks(range(len(sorted_quality)))
ax.set_yticklabels(sorted_quality.index)
ax.set_xlabel('Combined Score (Quality × Coverage)')
ax.set_title('Overall Indicator Quality Score', fontweight='bold')

plt.tight_layout()
plt.savefig(DATA_DIR / 'indicator_comparison.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved indicator_comparison.png")


# Figure 3: Phase comparison for top indicators
top_indicators = quality_df.head(3).index.tolist()
if 'IP YoY (Current)' not in top_indicators:
    top_indicators.append('IP YoY (Current)')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, indicator_name in zip(axes.flat, top_indicators[:4]):
    if indicator_name in phase_data:
        phases = phase_data[indicator_name]

        # Create phase bands
        phase_colors = {
            'Reflation': 'blue',
            'Recovery': 'green',
            'Overheat': 'red',
            'Stagflation': 'orange',
            'Unknown': 'gray'
        }

        for phase_name, color in phase_colors.items():
            mask = phases == phase_name
            if mask.any():
                ax.fill_between(phases.index, 0, 1, where=mask,
                              color=color, alpha=0.7, label=phase_name)

        ax.set_title(f'{indicator_name}', fontweight='bold')
        ax.set_yticks([])
        ax.legend(loc='upper right', ncol=3, fontsize=8)

plt.suptitle('Phase Classification Comparison', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(DATA_DIR / 'phase_comparison.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved phase_comparison.png")

plt.close('all')
print()


# =============================================================================
# 9. Save Results
# =============================================================================
print("9. SAVING RESULTS")
print("-" * 50)

# Save monthly data with all indicators
monthly.to_parquet(DATA_DIR / 'monthly_leading_indicators.parquet')
print(f"  ✓ {DATA_DIR / 'monthly_leading_indicators.parquet'}")

# Save comparison results
results_df.to_csv(DATA_DIR / 'indicator_comparison_results.csv')
print(f"  ✓ {DATA_DIR / 'indicator_comparison_results.csv'}")

quality_df.to_csv(DATA_DIR / 'indicator_quality_scores.csv')
print(f"  ✓ {DATA_DIR / 'indicator_quality_scores.csv'}")

# Save best indicator phases
best_indicator = quality_df.index[0]
if best_indicator in phase_data:
    best_phases = phase_data[best_indicator]
    monthly['best_indicator_phase'] = best_phases
    monthly.to_parquet(DATA_DIR / 'monthly_with_best_phases.parquet')
    print(f"  ✓ {DATA_DIR / 'monthly_with_best_phases.parquet'}")

print()


# =============================================================================
# Summary
# =============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nBest Performing Indicator: {quality_df.index[0]}")
print(f"  - Combined Score: {quality_df.iloc[0]['combined_score']:.1f}")
print(f"  - Classification Rate: {quality_df.iloc[0]['classified_pct']:.1f}%")
print(f"  - Quality Score: {quality_df.iloc[0]['quality_pct']:.1f}%")

print(f"\nCurrent Indicator (IP YoY):")
if 'IP YoY (Current)' in quality_df.index:
    current = quality_df.loc['IP YoY (Current)']
    print(f"  - Combined Score: {current['combined_score']:.1f}")
    print(f"  - Classification Rate: {current['classified_pct']:.1f}%")
    print(f"  - Quality Score: {current['quality_pct']:.1f}%")

print("\nTop 3 Indicators by Combined Score:")
for i, (name, row) in enumerate(quality_df.head(3).iterrows(), 1):
    print(f"  {i}. {name}: {row['combined_score']:.1f}")

print("\nRecommendation:")
best = quality_df.index[0]
print(f"  Consider replacing Industrial Production YoY with '{best}'")
print(f"  for better leading properties and phase classification.")
print()

print("Files saved in data/:")
print("  - monthly_leading_indicators.parquet")
print("  - indicator_comparison_results.csv")
print("  - indicator_quality_scores.csv")
print("  - leading_indicators_timeseries.png")
print("  - indicator_comparison.png")
print("  - phase_comparison.png")
