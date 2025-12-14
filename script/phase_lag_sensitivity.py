#!/usr/bin/env python3
"""
Analyze how phase performance varies across different lag periods.

This script examines whether the theoretical phase-asset relationships
hold at different prediction horizons (1-6 months).
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load indicator and price data."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')

    indicators = pd.read_parquet(os.path.join(data_dir, 'monthly_all_indicators.parquet'))
    indicators.index = pd.to_datetime(indicators.index)

    prices = pd.read_parquet(os.path.join(data_dir, 'prices.parquet'))
    prices.index = pd.to_datetime(prices.index)

    return indicators, prices, data_dir


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


def calculate_asset_returns(prices):
    """Calculate monthly returns for each asset class."""
    monthly_prices = prices.resample('ME').last()
    returns = pd.DataFrame(index=monthly_prices.index)

    # Stock returns
    stock_cols = ['SPY', 'spy', 'sp500', 'SP500']
    for col in stock_cols:
        if col in monthly_prices.columns:
            returns['stocks'] = monthly_prices[col].pct_change()
            break

    # Bond returns
    bond_cols = ['TLT', 'tlt', 'treasury_10y']
    for col in bond_cols:
        if col in monthly_prices.columns:
            returns['bonds'] = monthly_prices[col].pct_change()
            break

    # Commodity returns
    commodity_cols = ['GLD', 'gld', 'gold', 'DBC', 'dbc']
    for col in commodity_cols:
        if col in monthly_prices.columns:
            returns['commodities'] = monthly_prices[col].pct_change()
            break

    returns['cash'] = 0.002
    return returns


def analyze_phase_performance_by_lag(signals, returns, max_lag=6):
    """Analyze phase-specific asset returns across different lag periods."""

    results = []

    # Align data
    common_idx = signals.index.intersection(returns.index)
    signals = signals.loc[common_idx].copy()
    returns = returns.loc[common_idx].copy()

    for lag in range(1, max_lag + 1):
        # Apply lag
        signals_lagged = signals.copy()
        signals_lagged['phase_lagged'] = signals_lagged['phase'].shift(lag)

        # Remove invalid rows
        valid_mask = (signals_lagged['phase_lagged'].notna()) & \
                     (signals_lagged['phase_lagged'] != 'Unknown') & \
                     returns.notna().all(axis=1)

        sig_valid = signals_lagged[valid_mask]
        ret_valid = returns[valid_mask]

        for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
            mask = sig_valid['phase_lagged'] == phase
            phase_returns = ret_valid[mask]

            if len(phase_returns) > 0:
                results.append({
                    'lag': lag,
                    'phase': phase,
                    'months': len(phase_returns),
                    'stocks_ann': phase_returns['stocks'].mean() * 12 * 100,
                    'bonds_ann': phase_returns['bonds'].mean() * 12 * 100,
                    'commodities_ann': phase_returns['commodities'].mean() * 12 * 100,
                    'cash_ann': phase_returns['cash'].mean() * 12 * 100,
                    'stocks_std': phase_returns['stocks'].std() * np.sqrt(12) * 100,
                    'bonds_std': phase_returns['bonds'].std() * np.sqrt(12) * 100,
                    'commodities_std': phase_returns['commodities'].std() * np.sqrt(12) * 100,
                })

    return pd.DataFrame(results)


def identify_best_asset(row):
    """Identify which asset performed best in this phase/lag."""
    assets = {
        'stocks': row['stocks_ann'],
        'bonds': row['bonds_ann'],
        'commodities': row['commodities_ann'],
        'cash': row['cash_ann']
    }
    return max(assets, key=assets.get)


def check_theory_match(phase, best_asset):
    """Check if the best asset matches theoretical expectation."""
    theory = {
        'Recovery': 'stocks',
        'Overheat': 'commodities',
        'Stagflation': 'cash',
        'Reflation': 'bonds'
    }
    return '✓' if theory.get(phase) == best_asset else '✗'


def main():
    print("=" * 80)
    print("PHASE PERFORMANCE SENSITIVITY TO LAG PERIOD")
    print("=" * 80)

    # Load data
    indicators, prices, data_dir = load_data()
    signals = generate_signals(indicators)
    returns = calculate_asset_returns(prices)

    # Analyze
    results = analyze_phase_performance_by_lag(signals, returns, max_lag=6)

    # Add best asset and theory match columns
    results['best_asset'] = results.apply(identify_best_asset, axis=1)
    results['theory_match'] = results.apply(
        lambda r: check_theory_match(r['phase'], r['best_asset']), axis=1
    )

    # Print results by phase
    for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
        phase_data = results[results['phase'] == phase]

        theory_asset = {
            'Recovery': 'Stocks',
            'Overheat': 'Commodities',
            'Stagflation': 'Cash',
            'Reflation': 'Bonds'
        }[phase]

        print(f"\n{'=' * 80}")
        print(f"PHASE: {phase.upper()} (Theory expects: {theory_asset})")
        print("=" * 80)
        print(f"{'Lag':>4} {'Months':>7} {'Stocks':>10} {'Bonds':>10} {'Cmdty':>10} {'Cash':>8} {'Best':>12} {'Match':>6}")
        print("-" * 80)

        for _, row in phase_data.iterrows():
            print(f"{row['lag']:>4} {row['months']:>7} "
                  f"{row['stocks_ann']:>9.1f}% {row['bonds_ann']:>9.1f}% "
                  f"{row['commodities_ann']:>9.1f}% {row['cash_ann']:>7.1f}% "
                  f"{row['best_asset']:>12} {row['theory_match']:>6}")

    # Summary: Theory match rate by lag
    print("\n" + "=" * 80)
    print("SUMMARY: THEORY MATCH RATE BY LAG")
    print("=" * 80)
    print(f"{'Lag':>4} {'Matches':>10} {'Total':>8} {'Match Rate':>12}")
    print("-" * 40)

    for lag in range(1, 7):
        lag_data = results[results['lag'] == lag]
        matches = (lag_data['theory_match'] == '✓').sum()
        total = len(lag_data)
        rate = matches / total * 100 if total > 0 else 0
        print(f"{lag:>4} {matches:>10} {total:>8} {rate:>11.1f}%")

    # Summary: Average returns by lag (across all phases)
    print("\n" + "=" * 80)
    print("AVERAGE ANNUALIZED RETURNS BY LAG (across all phases)")
    print("=" * 80)
    print(f"{'Lag':>4} {'Stocks':>10} {'Bonds':>10} {'Cmdty':>10} {'Cash':>8}")
    print("-" * 50)

    for lag in range(1, 7):
        lag_data = results[results['lag'] == lag]
        # Weighted average by months
        total_months = lag_data['months'].sum()
        stocks_avg = (lag_data['stocks_ann'] * lag_data['months']).sum() / total_months
        bonds_avg = (lag_data['bonds_ann'] * lag_data['months']).sum() / total_months
        cmdty_avg = (lag_data['commodities_ann'] * lag_data['months']).sum() / total_months
        cash_avg = (lag_data['cash_ann'] * lag_data['months']).sum() / total_months

        print(f"{lag:>4} {stocks_avg:>9.1f}% {bonds_avg:>9.1f}% {cmdty_avg:>9.1f}% {cash_avg:>7.1f}%")

    # Pivot table for easy visualization
    print("\n" + "=" * 80)
    print("STOCKS ANNUALIZED RETURN BY PHASE AND LAG")
    print("=" * 80)
    pivot_stocks = results.pivot(index='phase', columns='lag', values='stocks_ann')
    pivot_stocks = pivot_stocks.reindex(['Recovery', 'Overheat', 'Stagflation', 'Reflation'])
    print(pivot_stocks.round(1).to_string())

    print("\n" + "=" * 80)
    print("BONDS ANNUALIZED RETURN BY PHASE AND LAG")
    print("=" * 80)
    pivot_bonds = results.pivot(index='phase', columns='lag', values='bonds_ann')
    pivot_bonds = pivot_bonds.reindex(['Recovery', 'Overheat', 'Stagflation', 'Reflation'])
    print(pivot_bonds.round(1).to_string())

    print("\n" + "=" * 80)
    print("COMMODITIES ANNUALIZED RETURN BY PHASE AND LAG")
    print("=" * 80)
    pivot_cmdty = results.pivot(index='phase', columns='lag', values='commodities_ann')
    pivot_cmdty = pivot_cmdty.reindex(['Recovery', 'Overheat', 'Stagflation', 'Reflation'])
    print(pivot_cmdty.round(1).to_string())

    # Save results
    results.to_csv(os.path.join(data_dir, 'phase_lag_sensitivity.csv'), index=False)
    print(f"\n✓ Saved to data/phase_lag_sensitivity.csv")


if __name__ == '__main__':
    main()
