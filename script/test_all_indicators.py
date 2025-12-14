#!/usr/bin/env python3
"""
RLIC Enhancement: Comprehensive Leading Indicators Analysis

This script tests both GROWTH and INFLATION leading indicators to find
the best combination for Investment Clock phase classification.

Growth Leading Indicators:
- Orders/Inventories ratio, CFNAI, Yield Curve, LEI, etc.

Inflation Leading Indicators:
- Breakeven inflation, PPI, Oil prices, M2, Commodity prices, etc.
"""

import pandas as pd
import numpy as np
import pandas_datareader as pdr
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from itertools import product
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

print("=" * 70)
print("RLIC Enhancement: Growth & Inflation Leading Indicators Analysis")
print("=" * 70)
print(f"Date range: {START_DATE} to {END_DATE}")
print()


# =============================================================================
# 1. Define All Indicators to Fetch
# =============================================================================

# Growth-related indicators
GROWTH_SERIES = {
    # Current benchmark
    'industrial_prod': 'INDPRO',          # Industrial Production (current benchmark)

    # Leading indicators for growth
    'lei': 'USSLIND',                     # Conference Board Leading Index
    'cfnai': 'CFNAI',                     # Chicago Fed National Activity Index
    'spread_10y3m': 'T10Y3M',             # Yield Curve 10Y-3M
    'spread_10y2y': 'T10Y2Y',             # Yield Curve 10Y-2Y
    'mfg_new_orders': 'AMTMNO',           # Manufacturers New Orders
    'mfg_inventories': 'AMTMTI',          # Manufacturers Inventories
    'building_permits': 'PERMIT',          # Building Permits
    'initial_claims': 'ICSA',             # Initial Jobless Claims
    'consumer_sentiment': 'UMCSENT',      # Consumer Sentiment
    'capacity_util': 'TCU',               # Capacity Utilization
    'durable_orders': 'DGORDER',          # Durable Goods Orders
    'retail_sales': 'RSXFS',              # Retail Sales ex Food Services
    'unemployment': 'UNRATE',             # Unemployment Rate
}

# Inflation-related indicators
INFLATION_SERIES = {
    # Current benchmark
    'cpi_all': 'CPIAUCSL',                # CPI All Urban (current benchmark)
    'cpi_core': 'CPILFESL',               # Core CPI

    # Leading indicators for inflation
    'ppi_all': 'PPIACO',                  # PPI All Commodities
    'ppi_finished': 'WPSFD4111',          # PPI Finished Goods
    'breakeven_10y': 'T10YIE',            # 10Y Breakeven Inflation
    'breakeven_5y': 'T5YIE',              # 5Y Breakeven Inflation
    'breakeven_5y5y': 'T5YIFR',           # 5Y5Y Forward Inflation
    'm2': 'M2SL',                         # M2 Money Supply
    'commodity_index': 'PALLFNFINDEXM',   # All Commodity Price Index
    'oil_wti': 'DCOILWTICO',              # WTI Crude Oil
    'import_prices': 'IR',                # Import Price Index
    'wage_growth': 'CES0500000003',       # Average Hourly Earnings
    'pce_price': 'PCEPI',                 # PCE Price Index
    'inflation_expect': 'MICH',           # Michigan Inflation Expectations
}

# Additional from OECD
OECD_SERIES = {
    'oecd_cli': 'USALOLITONOSTSAM',       # OECD CLI
}


# =============================================================================
# 2. Fetch Data
# =============================================================================
print("1. FETCHING DATA FROM FRED")
print("-" * 50)

def fetch_fred_data(series_dict, start, end):
    """Fetch multiple FRED series."""
    results = {}
    for name, series_id in series_dict.items():
        try:
            data = pdr.get_data_fred(series_id, start=start, end=end)
            results[name] = data[series_id]
            print(f"  ✓ {name} ({series_id}): {len(data)} obs")
        except Exception as e:
            print(f"  ✗ {name} ({series_id}): {str(e)[:50]}")
    return pd.DataFrame(results)

print("\nGrowth indicators:")
growth_data = fetch_fred_data(GROWTH_SERIES, START_DATE, END_DATE)

print("\nInflation indicators:")
inflation_data = fetch_fred_data(INFLATION_SERIES, START_DATE, END_DATE)

print("\nOECD indicators:")
oecd_data = fetch_fred_data(OECD_SERIES, START_DATE, END_DATE)

# Combine all data
all_data = pd.concat([growth_data, inflation_data, oecd_data], axis=1)
print(f"\nTotal series fetched: {len(all_data.columns)}")


# =============================================================================
# 3. Compute Derived Indicators
# =============================================================================
print("\n2. COMPUTING DERIVED INDICATORS")
print("-" * 50)

# Resample to monthly
monthly = all_data.resample('M').last()

# --- Growth Derivatives ---

# Industrial Production YoY (benchmark)
if 'industrial_prod' in monthly.columns:
    monthly['ip_yoy'] = monthly['industrial_prod'].pct_change(12) * 100
    print("  ✓ IP YoY")

# LEI derivatives
if 'lei' in monthly.columns:
    monthly['lei_mom'] = monthly['lei'].pct_change() * 100
    monthly['lei_3m'] = monthly['lei'].pct_change(3) * 100
    monthly['lei_6m'] = monthly['lei'].pct_change(6) * 100
    print("  ✓ LEI MoM, 3M, 6M")

# CFNAI derivatives
if 'cfnai' in monthly.columns:
    monthly['cfnai_3ma'] = monthly['cfnai'].rolling(3).mean()
    print("  ✓ CFNAI 3MA")

# Orders/Inventories ratio
if 'mfg_new_orders' in monthly.columns and 'mfg_inventories' in monthly.columns:
    monthly['orders_inv_ratio'] = monthly['mfg_new_orders'] / monthly['mfg_inventories']
    monthly['orders_inv_yoy'] = monthly['orders_inv_ratio'].pct_change(12) * 100
    monthly['orders_inv_mom'] = monthly['orders_inv_ratio'].pct_change() * 100
    print("  ✓ Orders/Inv ratio, YoY, MoM")

# Building Permits YoY
if 'building_permits' in monthly.columns:
    monthly['permits_yoy'] = monthly['building_permits'].pct_change(12) * 100
    print("  ✓ Permits YoY")

# Initial Claims (inverted YoY - lower is better)
if 'initial_claims' in monthly.columns:
    monthly['claims_yoy'] = monthly['initial_claims'].pct_change(12) * 100
    monthly['claims_yoy_inv'] = -monthly['claims_yoy']  # Inverted
    print("  ✓ Claims YoY (inverted)")

# Durable Orders YoY
if 'durable_orders' in monthly.columns:
    monthly['durable_yoy'] = monthly['durable_orders'].pct_change(12) * 100
    print("  ✓ Durable Orders YoY")

# OECD CLI deviation
if 'oecd_cli' in monthly.columns:
    monthly['oecd_cli_dev'] = monthly['oecd_cli'] - 100
    print("  ✓ OECD CLI deviation")

# Unemployment (inverted - lower is better for growth)
if 'unemployment' in monthly.columns:
    monthly['unemp_inv'] = -monthly['unemployment']
    monthly['unemp_yoy_inv'] = -monthly['unemployment'].diff(12)
    print("  ✓ Unemployment (inverted)")


# --- Inflation Derivatives ---

# CPI YoY (benchmark)
if 'cpi_all' in monthly.columns:
    monthly['cpi_yoy'] = monthly['cpi_all'].pct_change(12) * 100
    monthly['cpi_mom'] = monthly['cpi_all'].pct_change() * 100
    monthly['cpi_3m_ann'] = monthly['cpi_all'].pct_change(3) * 400  # Annualized
    print("  ✓ CPI YoY, MoM, 3M annualized")

# Core CPI YoY
if 'cpi_core' in monthly.columns:
    monthly['core_cpi_yoy'] = monthly['cpi_core'].pct_change(12) * 100
    print("  ✓ Core CPI YoY")

# PPI derivatives
if 'ppi_all' in monthly.columns:
    monthly['ppi_yoy'] = monthly['ppi_all'].pct_change(12) * 100
    monthly['ppi_mom'] = monthly['ppi_all'].pct_change() * 100
    monthly['ppi_3m_ann'] = monthly['ppi_all'].pct_change(3) * 400
    print("  ✓ PPI YoY, MoM, 3M annualized")

# Breakeven inflation (already in % terms)
if 'breakeven_10y' in monthly.columns:
    monthly['be10y'] = monthly['breakeven_10y']
    monthly['be10y_mom'] = monthly['breakeven_10y'].diff()
    print("  ✓ Breakeven 10Y, MoM change")

if 'breakeven_5y' in monthly.columns:
    monthly['be5y'] = monthly['breakeven_5y']
    print("  ✓ Breakeven 5Y")

# M2 YoY (leads inflation by 12-18 months)
if 'm2' in monthly.columns:
    monthly['m2_yoy'] = monthly['m2'].pct_change(12) * 100
    # Lagged versions for inflation prediction
    monthly['m2_yoy_lag12'] = monthly['m2_yoy'].shift(12)
    monthly['m2_yoy_lag18'] = monthly['m2_yoy'].shift(18)
    print("  ✓ M2 YoY, lagged 12M, lagged 18M")

# Commodity Index YoY
if 'commodity_index' in monthly.columns:
    monthly['commodity_yoy'] = monthly['commodity_index'].pct_change(12) * 100
    monthly['commodity_mom'] = monthly['commodity_index'].pct_change() * 100
    print("  ✓ Commodity Index YoY, MoM")

# Oil YoY
if 'oil_wti' in monthly.columns:
    monthly['oil_yoy'] = monthly['oil_wti'].pct_change(12) * 100
    monthly['oil_mom'] = monthly['oil_wti'].pct_change() * 100
    print("  ✓ Oil YoY, MoM")

# Import Prices YoY
if 'import_prices' in monthly.columns:
    monthly['import_yoy'] = monthly['import_prices'].pct_change(12) * 100
    print("  ✓ Import Prices YoY")

# Wage Growth YoY
if 'wage_growth' in monthly.columns:
    monthly['wage_yoy'] = monthly['wage_growth'].pct_change(12) * 100
    print("  ✓ Wage Growth YoY")

# PCE Price YoY
if 'pce_price' in monthly.columns:
    monthly['pce_yoy'] = monthly['pce_price'].pct_change(12) * 100
    print("  ✓ PCE YoY")

print(f"\nTotal indicators available: {len(monthly.columns)}")


# =============================================================================
# 4. Define Signal Generation Functions
# =============================================================================
print("\n3. DEFINING SIGNAL GENERATION METHODS")
print("-" * 50)

def momentum_signal(series, short_window=6, long_window=12):
    """
    Momentum signal: 1 if above both MAs, -1 if below both, 0 otherwise.
    """
    short_ma = series.rolling(short_window).mean()
    long_ma = series.rolling(long_window).mean()

    signal = pd.Series(0, index=series.index)
    signal[(series > short_ma) & (series > long_ma)] = 1
    signal[(series < short_ma) & (series < long_ma)] = -1
    return signal


def threshold_signal(series, threshold=0):
    """
    Simple threshold signal: 1 if above threshold, -1 if below.
    """
    signal = pd.Series(0, index=series.index)
    signal[series > threshold] = 1
    signal[series < threshold] = -1
    return signal


def direction_signal(series, lookback=3):
    """
    Direction signal: 1 if trending up, -1 if trending down.
    """
    diff = series.diff(lookback)
    signal = pd.Series(0, index=series.index)
    signal[diff > 0] = 1
    signal[diff < 0] = -1
    return signal


def classify_phase(growth_signal, inflation_signal):
    """
    Classify Investment Clock phase based on growth and inflation signals.
    """
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


# =============================================================================
# 5. Define Indicator Configurations
# =============================================================================

# Growth indicators to test
GROWTH_INDICATORS = {
    # Benchmark
    'IP YoY (Benchmark)': ('ip_yoy', 'momentum'),

    # Leading indicators
    'LEI 6M Change': ('lei_6m', 'threshold', 0),
    'LEI 3M Change': ('lei_3m', 'threshold', 0),
    'CFNAI': ('cfnai', 'threshold', 0),
    'CFNAI 3MA': ('cfnai_3ma', 'threshold', 0),
    'Yield Curve 10Y-3M': ('spread_10y3m', 'threshold', 0),
    'Yield Curve 10Y-2Y': ('spread_10y2y', 'threshold', 0),
    'Orders/Inv YoY': ('orders_inv_yoy', 'threshold', 0),
    'Orders/Inv MoM': ('orders_inv_mom', 'threshold', 0),
    'Building Permits YoY': ('permits_yoy', 'momentum'),
    'Initial Claims YoY (inv)': ('claims_yoy_inv', 'threshold', 0),
    'Durable Orders YoY': ('durable_yoy', 'momentum'),
    'Capacity Utilization': ('capacity_util', 'threshold', 80),
    'OECD CLI': ('oecd_cli_dev', 'threshold', 0),
    'Unemployment YoY (inv)': ('unemp_yoy_inv', 'threshold', 0),
}

# Inflation indicators to test
INFLATION_INDICATORS = {
    # Benchmark
    'CPI YoY (Benchmark)': ('cpi_yoy', 'momentum'),

    # Leading indicators
    'PPI YoY': ('ppi_yoy', 'momentum'),
    'PPI MoM': ('ppi_mom', 'threshold', 0),
    'PPI 3M Ann': ('ppi_3m_ann', 'momentum'),
    'Breakeven 10Y': ('be10y', 'momentum'),
    'Breakeven 10Y MoM': ('be10y_mom', 'threshold', 0),
    'Breakeven 5Y': ('be5y', 'momentum'),
    'M2 YoY': ('m2_yoy', 'momentum'),
    'M2 YoY Lag12': ('m2_yoy_lag12', 'momentum'),
    'M2 YoY Lag18': ('m2_yoy_lag18', 'momentum'),
    'Commodity YoY': ('commodity_yoy', 'momentum'),
    'Commodity MoM': ('commodity_mom', 'threshold', 0),
    'Oil YoY': ('oil_yoy', 'momentum'),
    'Oil MoM': ('oil_mom', 'threshold', 0),
    'Import Prices YoY': ('import_yoy', 'momentum'),
    'Wage YoY': ('wage_yoy', 'momentum'),
    'Core CPI YoY': ('core_cpi_yoy', 'momentum'),
    'PCE YoY': ('pce_yoy', 'momentum'),
    'Inflation Expectations': ('inflation_expect', 'momentum'),
}


# =============================================================================
# 6. Test All Indicator Combinations
# =============================================================================
print("\n4. TESTING ALL INDICATOR COMBINATIONS")
print("-" * 50)

def get_signal(series, method, threshold=0):
    """Generate signal based on method."""
    if method == 'momentum':
        return momentum_signal(series)
    elif method == 'threshold':
        return threshold_signal(series, threshold)
    elif method == 'direction':
        return direction_signal(series)
    else:
        return momentum_signal(series)


def evaluate_indicator_pair(growth_name, growth_config, infl_name, infl_config, monthly_data):
    """Evaluate a growth-inflation indicator pair."""

    # Extract config
    if len(growth_config) == 2:
        g_col, g_method = growth_config
        g_thresh = 0
    else:
        g_col, g_method, g_thresh = growth_config

    if len(infl_config) == 2:
        i_col, i_method = infl_config
        i_thresh = 0
    else:
        i_col, i_method, i_thresh = infl_config

    # Check if columns exist
    if g_col not in monthly_data.columns or i_col not in monthly_data.columns:
        return None

    # Get signals
    g_series = monthly_data[g_col].dropna()
    i_series = monthly_data[i_col].dropna()

    g_signal = get_signal(g_series, g_method, g_thresh)
    i_signal = get_signal(i_series, i_method, i_thresh)

    # Align
    common_idx = g_signal.index.intersection(i_signal.index)
    if len(common_idx) < 100:  # Need enough data
        return None

    g_aligned = g_signal.loc[common_idx]
    i_aligned = i_signal.loc[common_idx]

    # Classify phases
    phases = classify_phase(g_aligned, i_aligned)

    # Calculate statistics
    phase_counts = phases.value_counts()
    unknown_pct = phase_counts.get('Unknown', 0) / len(phases) * 100
    classified_pct = 100 - unknown_pct

    return {
        'growth_indicator': growth_name,
        'inflation_indicator': infl_name,
        'total_months': len(phases),
        'classified_pct': classified_pct,
        'unknown_pct': unknown_pct,
        'reflation': phase_counts.get('Reflation', 0),
        'recovery': phase_counts.get('Recovery', 0),
        'overheat': phase_counts.get('Overheat', 0),
        'stagflation': phase_counts.get('Stagflation', 0),
        'phases': phases,
    }


# Test all combinations
results = []
all_phases = {}

total_combos = len(GROWTH_INDICATORS) * len(INFLATION_INDICATORS)
print(f"Testing {total_combos} combinations...")

for (g_name, g_config), (i_name, i_config) in product(
    GROWTH_INDICATORS.items(), INFLATION_INDICATORS.items()
):
    result = evaluate_indicator_pair(g_name, g_config, i_name, i_config, monthly)
    if result:
        phases = result.pop('phases')
        results.append(result)
        all_phases[(g_name, i_name)] = phases

print(f"Successfully tested: {len(results)} combinations")


# =============================================================================
# 7. Analyze Results
# =============================================================================
print("\n5. ANALYZING RESULTS")
print("-" * 50)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('classified_pct', ascending=False)

# Top combinations by classification rate
print("\nTop 15 Indicator Combinations by Classification Rate:\n")
top_15 = results_df.head(15)[['growth_indicator', 'inflation_indicator', 'classified_pct', 'total_months']]
print(top_15.to_string(index=False))

# Best for each growth indicator
print("\n\nBest Inflation Indicator for Each Growth Indicator:\n")
best_per_growth = results_df.loc[results_df.groupby('growth_indicator')['classified_pct'].idxmax()]
best_per_growth = best_per_growth.sort_values('classified_pct', ascending=False)
print(best_per_growth[['growth_indicator', 'inflation_indicator', 'classified_pct']].to_string(index=False))

# Best for each inflation indicator
print("\n\nBest Growth Indicator for Each Inflation Indicator:\n")
best_per_infl = results_df.loc[results_df.groupby('inflation_indicator')['classified_pct'].idxmax()]
best_per_infl = best_per_infl.sort_values('classified_pct', ascending=False)
print(best_per_infl[['inflation_indicator', 'growth_indicator', 'classified_pct']].to_string(index=False))


# =============================================================================
# 8. Fetch Asset Returns for Quality Testing
# =============================================================================
print("\n6. FETCHING ASSET RETURNS FOR QUALITY TESTING")
print("-" * 50)

# Load or fetch price data
try:
    prices = pd.read_parquet(DATA_DIR / 'prices.parquet')
    print("Loaded existing price data")
except:
    print("Fetching price data from Yahoo Finance...")
    tickers = ['^GSPC', 'GC=F', 'CL=F', '^TNX']
    prices = yf.download(tickers, start=START_DATE, end=END_DATE, progress=False)['Close']
    prices.columns = ['sp500', 'gold', 'oil', 'treasury_10y']
    prices.to_parquet(DATA_DIR / 'prices.parquet')

prices_monthly = prices.resample('M').last()

# Calculate returns
returns = pd.DataFrame()
if 'sp500' in prices_monthly.columns:
    returns['stock_ret'] = prices_monthly['sp500'].pct_change() * 100
if 'gold' in prices_monthly.columns:
    returns['gold_ret'] = prices_monthly['gold'].pct_change() * 100
if 'oil' in prices_monthly.columns:
    returns['oil_ret'] = prices_monthly['oil'].pct_change() * 100
if 'treasury_10y' in prices_monthly.columns:
    returns['bond_ret'] = -prices_monthly['treasury_10y'].diff()  # Simplified proxy

print(f"Asset returns calculated: {list(returns.columns)}")


# =============================================================================
# 9. Calculate Quality Scores
# =============================================================================
print("\n7. CALCULATING QUALITY SCORES")
print("-" * 50)

def calculate_quality_score(phases, returns_df):
    """
    Calculate quality score based on theoretical expectations.

    Expected:
    - Overheat: Oil best
    - Stagflation: Stocks worst (negative)
    - Recovery/Overheat: Stocks positive
    - Reflation: Bonds positive
    """
    # Align phases with returns
    common_idx = phases.index.intersection(returns_df.index)
    aligned_phases = phases.loc[common_idx]
    aligned_returns = returns_df.loc[common_idx].copy()
    aligned_returns['phase'] = aligned_phases

    # Filter out Unknown
    valid = aligned_returns[aligned_returns['phase'] != 'Unknown']
    if len(valid) < 50:
        return 0, 0

    # Group by phase
    phase_returns = valid.groupby('phase').mean()

    score = 0
    max_score = 0

    # Check Oil in Overheat
    if 'Overheat' in phase_returns.index and 'oil_ret' in phase_returns.columns:
        oil_overheat = phase_returns.loc['Overheat', 'oil_ret']
        oil_others = phase_returns.loc[phase_returns.index != 'Overheat', 'oil_ret'].mean()
        if oil_overheat > oil_others:
            score += 2
        max_score += 2

    # Check Stocks in Stagflation (should be worst/negative)
    if 'Stagflation' in phase_returns.index and 'stock_ret' in phase_returns.columns:
        stock_stag = phase_returns.loc['Stagflation', 'stock_ret']
        stock_others = phase_returns.loc[phase_returns.index != 'Stagflation', 'stock_ret'].mean()
        if stock_stag < stock_others:
            score += 2
        if stock_stag < 0:
            score += 1
        max_score += 3

    # Check Stocks in Recovery/Overheat (should be positive)
    for phase in ['Recovery', 'Overheat']:
        if phase in phase_returns.index and 'stock_ret' in phase_returns.columns:
            if phase_returns.loc[phase, 'stock_ret'] > 0:
                score += 1
            max_score += 1

    # Check Bonds in Reflation (should be positive)
    if 'Reflation' in phase_returns.index and 'bond_ret' in phase_returns.columns:
        if phase_returns.loc['Reflation', 'bond_ret'] > 0:
            score += 1
        max_score += 1

    return score, max_score


# Calculate quality scores for top combinations
print("Calculating quality scores for top 30 combinations...")
quality_results = []

for idx, row in results_df.head(30).iterrows():
    g_name = row['growth_indicator']
    i_name = row['inflation_indicator']

    if (g_name, i_name) in all_phases:
        phases = all_phases[(g_name, i_name)]
        score, max_score = calculate_quality_score(phases, returns)

        quality_pct = (score / max_score * 100) if max_score > 0 else 0
        combined_score = quality_pct * (row['classified_pct'] / 100)

        quality_results.append({
            'growth_indicator': g_name,
            'inflation_indicator': i_name,
            'classified_pct': row['classified_pct'],
            'quality_score': score,
            'max_score': max_score,
            'quality_pct': quality_pct,
            'combined_score': combined_score,
        })

quality_df = pd.DataFrame(quality_results)
quality_df = quality_df.sort_values('combined_score', ascending=False)

print("\nTop 15 Combinations by Combined Score (Quality × Coverage):\n")
print(quality_df.head(15).to_string(index=False))


# =============================================================================
# 10. Save Results
# =============================================================================
print("\n8. SAVING RESULTS")
print("-" * 50)

# Save all results
results_df.to_csv(DATA_DIR / 'all_indicator_combinations.csv', index=False)
print(f"  ✓ {DATA_DIR / 'all_indicator_combinations.csv'}")

quality_df.to_csv(DATA_DIR / 'indicator_quality_scores_full.csv', index=False)
print(f"  ✓ {DATA_DIR / 'indicator_quality_scores_full.csv'}")

# Save monthly data with all indicators
monthly.to_parquet(DATA_DIR / 'monthly_all_indicators.parquet')
print(f"  ✓ {DATA_DIR / 'monthly_all_indicators.parquet'}")


# =============================================================================
# 11. Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if len(quality_df) > 0:
    best = quality_df.iloc[0]
    print(f"\nBEST INDICATOR COMBINATION:")
    print(f"  Growth:    {best['growth_indicator']}")
    print(f"  Inflation: {best['inflation_indicator']}")
    print(f"  Classification Rate: {best['classified_pct']:.1f}%")
    print(f"  Quality Score: {best['quality_pct']:.1f}%")
    print(f"  Combined Score: {best['combined_score']:.1f}")

    # Compare to benchmark
    benchmark = results_df[
        (results_df['growth_indicator'] == 'IP YoY (Benchmark)') &
        (results_df['inflation_indicator'] == 'CPI YoY (Benchmark)')
    ]
    if len(benchmark) > 0:
        bench = benchmark.iloc[0]
        print(f"\nBENCHMARK (IP YoY + CPI YoY):")
        print(f"  Classification Rate: {bench['classified_pct']:.1f}%")
        print(f"\nIMPROVEMENT:")
        print(f"  Classification: +{best['classified_pct'] - bench['classified_pct']:.1f} pp")

print("\n\nTop 5 Growth Indicators (across all inflation indicators):")
growth_avg = results_df.groupby('growth_indicator')['classified_pct'].mean().sort_values(ascending=False)
for i, (name, val) in enumerate(growth_avg.head(5).items(), 1):
    print(f"  {i}. {name}: {val:.1f}%")

print("\nTop 5 Inflation Indicators (across all growth indicators):")
infl_avg = results_df.groupby('inflation_indicator')['classified_pct'].mean().sort_values(ascending=False)
for i, (name, val) in enumerate(infl_avg.head(5).items(), 1):
    print(f"  {i}. {name}: {val:.1f}%")

print("\n" + "=" * 70)
