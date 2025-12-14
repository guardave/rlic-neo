#!/usr/bin/env python3
"""
Analyze sector performance by Investment Clock phase.

Tests the theoretical sector preferences:
- Recovery: Cyclicals, Technology, Industrials
- Overheat: Energy, Materials, Industrials
- Stagflation: Defensives, Healthcare, Utilities, Consumer Staples
- Reflation: Financials, Consumer Discretionary
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Sector ETFs (SPDR Select Sector ETFs - inception 1998)
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLV': 'Healthcare',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLU': 'Utilities',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',  # 2015+
}

# Theoretical sector preferences by phase
THEORY_SECTORS = {
    'Recovery': ['Technology', 'Industrials', 'Consumer Discretionary', 'Financials'],
    'Overheat': ['Energy', 'Materials', 'Industrials'],
    'Stagflation': ['Healthcare', 'Utilities', 'Consumer Staples'],
    'Reflation': ['Financials', 'Consumer Discretionary', 'Real Estate']
}

# Sector characteristics
SECTOR_TRAITS = {
    'Technology': 'Growth/Cyclical',
    'Financials': 'Cyclical/Rate-sensitive',
    'Healthcare': 'Defensive',
    'Energy': 'Commodity/Cyclical',
    'Industrials': 'Cyclical',
    'Consumer Discretionary': 'Cyclical',
    'Consumer Staples': 'Defensive',
    'Utilities': 'Defensive/Rate-sensitive',
    'Materials': 'Commodity/Cyclical',
    'Real Estate': 'Rate-sensitive',
}


def load_data():
    """Load indicator and price data."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')

    indicators = pd.read_parquet(os.path.join(data_dir, 'monthly_all_indicators.parquet'))
    indicators.index = pd.to_datetime(indicators.index)

    return indicators, data_dir


def fetch_sector_data(start_date='1999-01-01'):
    """Fetch sector ETF price data."""
    print("\nFetching sector ETF data...")

    end_date = datetime.now().strftime('%Y-%m-%d')
    prices = pd.DataFrame()

    for ticker, name in SECTOR_ETFS.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                # Handle both old and new yfinance column formats
                if 'Adj Close' in data.columns:
                    prices[ticker] = data['Adj Close']
                elif 'Close' in data.columns:
                    prices[ticker] = data['Close']
                elif isinstance(data.columns, pd.MultiIndex):
                    # New yfinance format with MultiIndex columns
                    if ('Adj Close', ticker) in data.columns:
                        prices[ticker] = data[('Adj Close', ticker)]
                    elif ('Close', ticker) in data.columns:
                        prices[ticker] = data[('Close', ticker)]
                    else:
                        # Try to get any price column
                        for col in data.columns:
                            if 'Close' in str(col):
                                prices[ticker] = data[col]
                                break
                print(f"  ✓ {ticker} ({name}): {len(data)} obs from {data.index[0].strftime('%Y-%m')}")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")

    return prices


def generate_signals(indicators):
    """Generate growth and inflation signals using Orders/Inv MoM + PPI MoM."""
    signals = pd.DataFrame(index=indicators.index)

    # Growth Signal: Orders/Inv MoM Direction
    if 'orders_inv_ratio' in indicators.columns:
        ratio = indicators['orders_inv_ratio']
        growth_3ma = ratio.rolling(3).mean()
        growth_6ma = ratio.rolling(6).mean()
        signals['growth_signal'] = np.where(growth_3ma > growth_6ma, 1, -1)

    # Inflation Signal: PPI MoM Direction
    if 'ppi_all' in indicators.columns:
        ppi = indicators['ppi_all']
        ppi_3ma = ppi.rolling(3).mean()
        ppi_6ma = ppi.rolling(6).mean()
        signals['inflation_signal'] = np.where(ppi_3ma > ppi_6ma, 1, -1)

    # Classify phases
    def classify_phase(row):
        g, i = row['growth_signal'], row['inflation_signal']
        if pd.isna(g) or pd.isna(i):
            return 'Unknown'
        if g == 1 and i == -1:
            return 'Recovery'
        elif g == 1 and i == 1:
            return 'Overheat'
        elif g == -1 and i == 1:
            return 'Stagflation'
        elif g == -1 and i == -1:
            return 'Reflation'
        return 'Unknown'

    signals['phase'] = signals.apply(classify_phase, axis=1)
    return signals


def calculate_sector_returns(prices):
    """Calculate monthly returns for each sector."""
    monthly_prices = prices.resample('ME').last()
    returns = monthly_prices.pct_change()

    # Rename columns to sector names
    returns.columns = [SECTOR_ETFS.get(col, col) for col in returns.columns]

    return returns


def analyze_sector_performance_by_phase(signals, returns, lag=1):
    """Analyze sector returns by phase with specified lag."""

    results = []

    # Align data
    common_idx = signals.index.intersection(returns.index)
    signals = signals.loc[common_idx].copy()
    returns = returns.loc[common_idx].copy()

    # Apply lag
    signals['phase_lagged'] = signals['phase'].shift(lag)

    # Remove invalid rows
    valid_mask = (signals['phase_lagged'].notna()) & \
                 (signals['phase_lagged'] != 'Unknown')

    signals = signals[valid_mask]
    returns = returns[valid_mask]

    for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
        mask = signals['phase_lagged'] == phase
        phase_returns = returns[mask]

        if len(phase_returns) > 0:
            for sector in returns.columns:
                sector_ret = phase_returns[sector].dropna()
                if len(sector_ret) > 0:
                    results.append({
                        'phase': phase,
                        'sector': sector,
                        'months': len(sector_ret),
                        'ann_return': sector_ret.mean() * 12 * 100,
                        'ann_vol': sector_ret.std() * np.sqrt(12) * 100,
                        'sharpe': (sector_ret.mean() * 12 - 0.02) / (sector_ret.std() * np.sqrt(12)) if sector_ret.std() > 0 else 0,
                        'win_rate': (sector_ret > 0).mean() * 100,
                        'is_theory_pick': sector in THEORY_SECTORS.get(phase, [])
                    })

    return pd.DataFrame(results)


def rank_sectors_by_phase(results_df):
    """Rank sectors within each phase by annualized return."""
    ranked = []

    for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
        phase_data = results_df[results_df['phase'] == phase].copy()
        phase_data = phase_data.sort_values('ann_return', ascending=False)
        phase_data['rank'] = range(1, len(phase_data) + 1)
        ranked.append(phase_data)

    return pd.concat(ranked, ignore_index=True)


def print_phase_sector_analysis(ranked_df, phase):
    """Print sector analysis for a specific phase."""

    phase_data = ranked_df[ranked_df['phase'] == phase]
    theory_picks = THEORY_SECTORS.get(phase, [])

    print(f"\n{'=' * 85}")
    print(f"PHASE: {phase.upper()}")
    print(f"Theory recommends: {', '.join(theory_picks)}")
    print("=" * 85)
    print(f"{'Rank':>4} {'Sector':<25} {'Ann Ret':>10} {'Vol':>8} {'Sharpe':>8} {'Win%':>7} {'Theory':>8}")
    print("-" * 85)

    for _, row in phase_data.iterrows():
        theory_mark = '✓' if row['is_theory_pick'] else ''
        print(f"{row['rank']:>4} {row['sector']:<25} {row['ann_return']:>9.1f}% "
              f"{row['ann_vol']:>7.1f}% {row['sharpe']:>8.2f} {row['win_rate']:>6.1f}% {theory_mark:>8}")


def calculate_theory_performance(ranked_df):
    """Calculate how well theory picks performed vs non-theory picks."""

    results = []

    for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
        phase_data = ranked_df[ranked_df['phase'] == phase]

        theory_picks = phase_data[phase_data['is_theory_pick']]
        non_theory = phase_data[~phase_data['is_theory_pick']]

        if len(theory_picks) > 0 and len(non_theory) > 0:
            theory_avg_ret = theory_picks['ann_return'].mean()
            theory_avg_rank = theory_picks['rank'].mean()
            non_theory_avg_ret = non_theory['ann_return'].mean()

            # Count how many theory picks are in top 3
            top3_theory = len(theory_picks[theory_picks['rank'] <= 3])

            results.append({
                'phase': phase,
                'theory_avg_return': theory_avg_ret,
                'non_theory_avg_return': non_theory_avg_ret,
                'theory_advantage': theory_avg_ret - non_theory_avg_ret,
                'theory_avg_rank': theory_avg_rank,
                'theory_in_top3': top3_theory,
                'total_theory_picks': len(theory_picks)
            })

    return pd.DataFrame(results)


def analyze_across_lags(signals, returns, max_lag=6):
    """Analyze sector performance across different lag periods."""

    all_results = []

    for lag in range(1, max_lag + 1):
        results = analyze_sector_performance_by_phase(signals.copy(), returns.copy(), lag=lag)
        results['lag'] = lag
        all_results.append(results)

    return pd.concat(all_results, ignore_index=True)


def main():
    print("=" * 85)
    print("SECTOR PERFORMANCE BY INVESTMENT CLOCK PHASE")
    print("=" * 85)

    # Load data
    indicators, data_dir = load_data()
    signals = generate_signals(indicators)

    # Fetch sector data
    sector_prices = fetch_sector_data()

    if sector_prices.empty:
        print("Failed to fetch sector data")
        return

    # Save sector prices
    sector_prices.to_parquet(os.path.join(data_dir, 'sector_prices.parquet'))

    # Calculate returns
    sector_returns = calculate_sector_returns(sector_prices)

    # Analyze with 1-month lag (baseline)
    print("\n" + "=" * 85)
    print("ANALYSIS WITH 1-MONTH LAG")
    print("=" * 85)

    results_lag1 = analyze_sector_performance_by_phase(signals.copy(), sector_returns.copy(), lag=1)
    ranked_lag1 = rank_sectors_by_phase(results_lag1)

    for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
        print_phase_sector_analysis(ranked_lag1, phase)

    # Theory performance summary
    theory_perf = calculate_theory_performance(ranked_lag1)

    print("\n" + "=" * 85)
    print("THEORY PERFORMANCE SUMMARY (1-Month Lag)")
    print("=" * 85)
    print(f"{'Phase':<15} {'Theory Avg':>12} {'Other Avg':>12} {'Advantage':>12} {'Avg Rank':>10} {'In Top 3':>10}")
    print("-" * 85)

    for _, row in theory_perf.iterrows():
        print(f"{row['phase']:<15} {row['theory_avg_return']:>11.1f}% {row['non_theory_avg_return']:>11.1f}% "
              f"{row['theory_advantage']:>+11.1f}% {row['theory_avg_rank']:>10.1f} "
              f"{row['theory_in_top3']}/{row['total_theory_picks']:>8}")

    # Analyze across multiple lags
    print("\n" + "=" * 85)
    print("LAG SENSITIVITY: THEORY ADVANTAGE BY LAG")
    print("=" * 85)

    all_lag_results = analyze_across_lags(signals.copy(), sector_returns.copy(), max_lag=6)

    print(f"\n{'Lag':>4} {'Recovery':>12} {'Overheat':>12} {'Stagflation':>12} {'Reflation':>12} {'Avg':>10}")
    print("-" * 70)

    for lag in range(1, 7):
        lag_data = all_lag_results[all_lag_results['lag'] == lag]
        ranked = rank_sectors_by_phase(lag_data)
        theory_perf = calculate_theory_performance(ranked)

        advantages = {}
        for _, row in theory_perf.iterrows():
            advantages[row['phase']] = row['theory_advantage']

        avg_advantage = np.mean(list(advantages.values()))

        print(f"{lag:>4} {advantages.get('Recovery', 0):>+11.1f}% {advantages.get('Overheat', 0):>+11.1f}% "
              f"{advantages.get('Stagflation', 0):>+11.1f}% {advantages.get('Reflation', 0):>+11.1f}% "
              f"{avg_advantage:>+9.1f}%")

    # Best sectors by phase across all lags (average)
    print("\n" + "=" * 85)
    print("AVERAGE SECTOR RETURNS BY PHASE (across lags 1-6)")
    print("=" * 85)

    avg_by_phase_sector = all_lag_results.groupby(['phase', 'sector']).agg({
        'ann_return': 'mean',
        'ann_vol': 'mean',
        'sharpe': 'mean',
        'is_theory_pick': 'first'
    }).reset_index()

    for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
        phase_data = avg_by_phase_sector[avg_by_phase_sector['phase'] == phase]
        phase_data = phase_data.sort_values('ann_return', ascending=False)

        theory_picks = THEORY_SECTORS.get(phase, [])
        print(f"\n{phase} (Theory: {', '.join(theory_picks)})")
        print("-" * 60)

        for i, (_, row) in enumerate(phase_data.iterrows(), 1):
            theory_mark = '✓' if row['is_theory_pick'] else ''
            print(f"  {i}. {row['sector']:<22} {row['ann_return']:>+7.1f}% (Sharpe: {row['sharpe']:.2f}) {theory_mark}")

    # Save results
    all_lag_results.to_csv(os.path.join(data_dir, 'sector_phase_analysis.csv'), index=False)
    print(f"\n✓ Saved to data/sector_phase_analysis.csv")


if __name__ == '__main__':
    main()
