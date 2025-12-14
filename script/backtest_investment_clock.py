#!/usr/bin/env python3
"""
Backtest Investment Clock Strategy using Orders/Inv MoM + PPI MoM signals.

This script tests the effectiveness of the enhanced Investment Clock
by comparing phase-based asset allocation to buy-and-hold benchmarks.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Investment Clock Phase -> Recommended Asset Allocation
# Based on traditional Investment Clock theory
PHASE_ALLOCATIONS = {
    'Recovery': {
        'description': 'Growth rising, Inflation falling',
        'favored': ['stocks'],
        'allocation': {'stocks': 1.0, 'bonds': 0.0, 'commodities': 0.0, 'cash': 0.0}
    },
    'Overheat': {
        'description': 'Growth rising, Inflation rising',
        'favored': ['commodities'],
        'allocation': {'stocks': 0.0, 'bonds': 0.0, 'commodities': 1.0, 'cash': 0.0}
    },
    'Stagflation': {
        'description': 'Growth falling, Inflation rising',
        'favored': ['cash'],
        'allocation': {'stocks': 0.0, 'bonds': 0.0, 'commodities': 0.0, 'cash': 1.0}
    },
    'Reflation': {
        'description': 'Growth falling, Inflation falling',
        'favored': ['bonds'],
        'allocation': {'stocks': 0.0, 'bonds': 1.0, 'commodities': 0.0, 'cash': 0.0}
    }
}

# Alternative: Diversified allocation per phase
PHASE_ALLOCATIONS_DIVERSIFIED = {
    'Recovery': {'stocks': 0.7, 'bonds': 0.2, 'commodities': 0.0, 'cash': 0.1},
    'Overheat': {'stocks': 0.3, 'bonds': 0.0, 'commodities': 0.5, 'cash': 0.2},
    'Stagflation': {'stocks': 0.0, 'bonds': 0.2, 'commodities': 0.3, 'cash': 0.5},
    'Reflation': {'stocks': 0.2, 'bonds': 0.6, 'commodities': 0.0, 'cash': 0.2}
}


def load_data():
    """Load indicator and price data."""
    print("Loading data...")

    # Get script directory for relative paths
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')

    # Load indicators
    indicators_path = os.path.join(data_dir, 'monthly_all_indicators.parquet')
    indicators = pd.read_parquet(indicators_path)
    indicators.index = pd.to_datetime(indicators.index)

    # Load price data
    prices_path = os.path.join(data_dir, 'prices.parquet')
    if os.path.exists(prices_path):
        prices = pd.read_parquet(prices_path)
        prices.index = pd.to_datetime(prices.index)
    else:
        prices = pd.DataFrame()

    return indicators, prices, data_dir


def generate_signals(indicators):
    """Generate growth and inflation signals using Orders/Inv MoM + PPI MoM."""

    signals = pd.DataFrame(index=indicators.index)

    # Growth Signal: Orders/Inv MoM Direction
    # Rising if 3M MA > 6M MA
    if 'orders_inv_ratio' in indicators.columns:
        ratio = indicators['orders_inv_ratio']
        growth_3ma = ratio.rolling(3).mean()
        growth_6ma = ratio.rolling(6).mean()
        signals['growth_signal'] = np.where(growth_3ma > growth_6ma, 1, -1)
    else:
        print("Warning: orders_inv_ratio not found, using alternative")
        # Fallback to pre-computed if available
        signals['growth_signal'] = np.nan

    # Inflation Signal: PPI MoM Direction
    # Rising if 3M MA > 6M MA
    if 'ppi_all' in indicators.columns:
        ppi = indicators['ppi_all']
        ppi_3ma = ppi.rolling(3).mean()
        ppi_6ma = ppi.rolling(6).mean()
        signals['inflation_signal'] = np.where(ppi_3ma > ppi_6ma, 1, -1)
    else:
        print("Warning: ppi_all not found")
        signals['inflation_signal'] = np.nan

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

    # Resample to month-end
    monthly_prices = prices.resample('ME').last()

    returns = pd.DataFrame(index=monthly_prices.index)

    # Stock returns (S&P 500) - check multiple column names
    stock_cols = ['SPY', 'spy', 'sp500', 'SP500']
    for col in stock_cols:
        if col in monthly_prices.columns:
            returns['stocks'] = monthly_prices[col].pct_change()
            break

    # Bond returns (TLT - Long-term Treasury)
    bond_cols = ['TLT', 'tlt', 'treasury_10y']
    for col in bond_cols:
        if col in monthly_prices.columns:
            returns['bonds'] = monthly_prices[col].pct_change()
            break

    # Commodity returns (Gold or DBC)
    commodity_cols = ['GLD', 'gld', 'gold', 'DBC', 'dbc']
    for col in commodity_cols:
        if col in monthly_prices.columns:
            returns['commodities'] = monthly_prices[col].pct_change()
            break

    # Cash returns (approximate with 0.2% monthly = ~2.4% annual)
    returns['cash'] = 0.002

    return returns


def fetch_additional_assets():
    """Fetch additional asset data for more comprehensive backtest."""
    import yfinance as yf

    print("\nFetching additional asset data...")

    # Assets to fetch - using older ETFs where possible for longer history
    tickers = {
        'SPY': 'S&P 500',           # 1993+
        'TLT': 'Long-term Treasury Bonds',  # 2002+
        'IEF': 'Intermediate Treasury Bonds',  # 2002+
        'GLD': 'Gold',              # 2004+
        'DBC': 'Commodities Index', # 2006+
        'SHY': 'Short-term Treasury (Cash proxy)'  # 2002+
    }

    start_date = '1990-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    prices = pd.DataFrame()

    for ticker, name in tickers.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                prices[ticker] = data['Adj Close']
                print(f"  ✓ {ticker} ({name}): {len(data)} obs")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")

    return prices


def run_backtest(signals, returns, allocation_type='concentrated', signal_lag=1):
    """
    Run the Investment Clock backtest.

    Args:
        signals: DataFrame with 'phase' column
        returns: DataFrame with asset returns
        allocation_type: 'concentrated' or 'diversified'
        signal_lag: Number of months to lag signal (1 = use last month's signal for this month's allocation)
                   This is realistic: signal at month-end T determines allocation for month T+1
    """

    allocations = PHASE_ALLOCATIONS if allocation_type == 'concentrated' else PHASE_ALLOCATIONS_DIVERSIFIED

    # Align data
    common_idx = signals.index.intersection(returns.index)
    signals = signals.loc[common_idx].copy()
    returns = returns.loc[common_idx].copy()

    # Apply signal lag: shift signals forward so signal at T applies to returns at T+lag
    # This means: phase determined at end of month T-1 → trade in month T → capture returns of month T
    if signal_lag > 0:
        signals['phase_lagged'] = signals['phase'].shift(signal_lag)
    else:
        signals['phase_lagged'] = signals['phase']

    # Remove periods with Unknown phase or missing returns
    valid_mask = (signals['phase_lagged'].notna()) & (signals['phase_lagged'] != 'Unknown') & returns.notna().all(axis=1)
    signals = signals[valid_mask]
    returns = returns[valid_mask]

    print(f"\nBacktest period: {signals.index[0].strftime('%Y-%m')} to {signals.index[-1].strftime('%Y-%m')}")
    print(f"Total months: {len(signals)}")
    print(f"Signal lag: {signal_lag} month(s) (signal at T-{signal_lag} → returns at T)")

    # Calculate strategy returns
    strategy_returns = pd.Series(index=signals.index, dtype=float)

    for date in signals.index:
        phase = signals.loc[date, 'phase_lagged']
        if phase in allocations:
            alloc = allocations[phase]['allocation'] if allocation_type == 'concentrated' else allocations[phase]

            month_return = 0
            for asset, weight in alloc.items():
                if asset in returns.columns and weight > 0:
                    month_return += weight * returns.loc[date, asset]

            strategy_returns[date] = month_return

    return strategy_returns, signals, returns


def calculate_metrics(returns, name="Strategy"):
    """Calculate performance metrics for a return series."""

    returns = returns.dropna()

    if len(returns) == 0:
        return {}

    # Basic metrics
    total_return = (1 + returns).prod() - 1
    cagr = (1 + total_return) ** (12 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(12)
    sharpe = (returns.mean() * 12 - 0.02) / volatility if volatility > 0 else 0

    # Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Win rate
    win_rate = (returns > 0).mean()

    # Best/worst months
    best_month = returns.max()
    worst_month = returns.min()

    return {
        'name': name,
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'best_month': best_month,
        'worst_month': worst_month,
        'n_months': len(returns)
    }


def analyze_phase_performance(signals, returns):
    """Analyze returns by Investment Clock phase (using lagged phase if available)."""

    phase_stats = {}

    # Use lagged phase if available (from backtest), otherwise use original phase
    phase_col = 'phase_lagged' if 'phase_lagged' in signals.columns else 'phase'

    for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
        mask = signals[phase_col] == phase
        phase_returns = returns[mask]

        if len(phase_returns) > 0:
            phase_stats[phase] = {
                'count': mask.sum(),
                'pct_time': mask.mean() * 100,
                'stocks_avg': phase_returns['stocks'].mean() * 12 * 100 if 'stocks' in phase_returns else np.nan,
                'bonds_avg': phase_returns['bonds'].mean() * 12 * 100 if 'bonds' in phase_returns else np.nan,
                'commodities_avg': phase_returns['commodities'].mean() * 12 * 100 if 'commodities' in phase_returns else np.nan,
            }

    return phase_stats


def print_report(strategy_metrics, benchmark_metrics, phase_stats):
    """Print comprehensive backtest report."""

    print("\n" + "=" * 70)
    print("INVESTMENT CLOCK BACKTEST RESULTS")
    print("Strategy: Orders/Inv MoM + PPI MoM")
    print("=" * 70)

    # Performance comparison
    print("\n1. PERFORMANCE COMPARISON")
    print("-" * 50)
    print(f"{'Metric':<25} {'Strategy':>12} {'Buy & Hold':>12}")
    print("-" * 50)

    for metric, label in [
        ('total_return', 'Total Return'),
        ('cagr', 'CAGR'),
        ('volatility', 'Volatility'),
        ('sharpe', 'Sharpe Ratio'),
        ('max_drawdown', 'Max Drawdown'),
        ('win_rate', 'Win Rate')
    ]:
        strat_val = strategy_metrics.get(metric, np.nan)
        bench_val = benchmark_metrics.get(metric, np.nan)

        if metric in ['total_return', 'cagr', 'volatility', 'max_drawdown', 'win_rate']:
            print(f"{label:<25} {strat_val:>11.1%} {bench_val:>11.1%}")
        else:
            print(f"{label:<25} {strat_val:>12.2f} {bench_val:>12.2f}")

    # Phase analysis
    print("\n2. PHASE ANALYSIS")
    print("-" * 70)
    print(f"{'Phase':<15} {'Months':>8} {'% Time':>8} {'Stocks':>10} {'Bonds':>10} {'Cmdty':>10}")
    print("-" * 70)

    for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
        if phase in phase_stats:
            stats = phase_stats[phase]
            print(f"{phase:<15} {stats['count']:>8} {stats['pct_time']:>7.1f}% "
                  f"{stats['stocks_avg']:>9.1f}% {stats['bonds_avg']:>9.1f}% "
                  f"{stats['commodities_avg']:>9.1f}%")

    # Theoretical expectations
    print("\n3. THEORETICAL EXPECTATIONS")
    print("-" * 50)
    print("Phase          | Expected Best Asset")
    print("-" * 50)
    print("Recovery       | Stocks (growth up, inflation down)")
    print("Overheat       | Commodities (growth up, inflation up)")
    print("Stagflation    | Cash (growth down, inflation up)")
    print("Reflation      | Bonds (growth down, inflation down)")


def run_lag_sensitivity_analysis(signals, returns):
    """Run backtest across multiple lag values to test signal persistence."""

    print("\n" + "=" * 70)
    print("LAG SENSITIVITY ANALYSIS (1-6 months)")
    print("=" * 70)

    results = []

    for lag in range(1, 7):
        # Concentrated
        strat_ret_conc, _, _ = run_backtest(signals.copy(), returns.copy(),
                                             allocation_type='concentrated', signal_lag=lag)
        metrics_conc = calculate_metrics(strat_ret_conc, f"Concentrated Lag-{lag}")

        # Diversified
        strat_ret_div, _, _ = run_backtest(signals.copy(), returns.copy(),
                                            allocation_type='diversified', signal_lag=lag)
        metrics_div = calculate_metrics(strat_ret_div, f"Diversified Lag-{lag}")

        results.append({
            'lag': lag,
            'conc_cagr': metrics_conc.get('cagr', np.nan),
            'conc_vol': metrics_conc.get('volatility', np.nan),
            'conc_sharpe': metrics_conc.get('sharpe', np.nan),
            'conc_maxdd': metrics_conc.get('max_drawdown', np.nan),
            'div_cagr': metrics_div.get('cagr', np.nan),
            'div_vol': metrics_div.get('volatility', np.nan),
            'div_sharpe': metrics_div.get('sharpe', np.nan),
            'div_maxdd': metrics_div.get('max_drawdown', np.nan),
        })

    # Print summary table
    print("\n" + "-" * 90)
    print("CONCENTRATED STRATEGY BY LAG")
    print("-" * 90)
    print(f"{'Lag':>4} {'CAGR':>10} {'Volatility':>12} {'Sharpe':>10} {'Max DD':>12}")
    print("-" * 90)
    for r in results:
        print(f"{r['lag']:>4} {r['conc_cagr']:>9.1%} {r['conc_vol']:>11.1%} {r['conc_sharpe']:>10.2f} {r['conc_maxdd']:>11.1%}")

    print("\n" + "-" * 90)
    print("DIVERSIFIED STRATEGY BY LAG")
    print("-" * 90)
    print(f"{'Lag':>4} {'CAGR':>10} {'Volatility':>12} {'Sharpe':>10} {'Max DD':>12}")
    print("-" * 90)
    for r in results:
        print(f"{r['lag']:>4} {r['div_cagr']:>9.1%} {r['div_vol']:>11.1%} {r['div_sharpe']:>10.2f} {r['div_maxdd']:>11.1%}")

    return pd.DataFrame(results)


def main():
    """Main backtest execution."""

    print("=" * 70)
    print("RLIC Enhancement: Backtest Analysis")
    print("=" * 70)

    # Load data
    indicators, prices, data_dir = load_data()

    # Check if we have sufficient asset data
    if prices.empty or len(prices.columns) < 2:
        print("\nInsufficient price data. Fetching additional assets...")
        prices = fetch_additional_assets()
        import os
        prices.to_parquet(os.path.join(data_dir, 'price_data_extended.parquet'))

    # Generate signals
    print("\nGenerating signals using Orders/Inv MoM + PPI MoM...")
    signals = generate_signals(indicators)

    # Phase distribution
    phase_counts = signals['phase'].value_counts()
    print("\nPhase Distribution:")
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count} months ({count/len(signals)*100:.1f}%)")

    # Calculate asset returns
    returns = calculate_asset_returns(prices)

    # Run backtest - Concentrated allocation
    print("\n" + "-" * 50)
    print("Running backtest with CONCENTRATED allocation...")
    strategy_returns, aligned_signals, aligned_returns = run_backtest(
        signals, returns, allocation_type='concentrated'
    )

    # Calculate metrics
    strategy_metrics = calculate_metrics(strategy_returns, "IC Strategy")

    # Benchmark: Buy and hold stocks
    benchmark_returns = aligned_returns['stocks'] if 'stocks' in aligned_returns.columns else pd.Series()
    benchmark_metrics = calculate_metrics(benchmark_returns, "Buy & Hold SPY")

    # Phase performance analysis
    phase_stats = analyze_phase_performance(aligned_signals, aligned_returns)

    # Print report
    print_report(strategy_metrics, benchmark_metrics, phase_stats)

    # Run backtest - Diversified allocation
    print("\n" + "=" * 70)
    print("Running backtest with DIVERSIFIED allocation...")
    strategy_returns_div, _, _ = run_backtest(
        signals, returns, allocation_type='diversified'
    )

    strategy_metrics_div = calculate_metrics(strategy_returns_div, "IC Diversified")

    print("\n4. DIVERSIFIED STRATEGY COMPARISON")
    print("-" * 50)
    print(f"{'Metric':<25} {'Concentrated':>12} {'Diversified':>12} {'Buy&Hold':>12}")
    print("-" * 50)

    for metric, label in [
        ('cagr', 'CAGR'),
        ('volatility', 'Volatility'),
        ('sharpe', 'Sharpe Ratio'),
        ('max_drawdown', 'Max Drawdown'),
    ]:
        conc_val = strategy_metrics.get(metric, np.nan)
        div_val = strategy_metrics_div.get(metric, np.nan)
        bench_val = benchmark_metrics.get(metric, np.nan)

        if metric in ['cagr', 'volatility', 'max_drawdown']:
            print(f"{label:<25} {conc_val:>11.1%} {div_val:>11.1%} {bench_val:>11.1%}")
        else:
            print(f"{label:<25} {conc_val:>12.2f} {div_val:>12.2f} {bench_val:>12.2f}")

    # Run lag sensitivity analysis (1-6 months)
    lag_results = run_lag_sensitivity_analysis(signals, returns)

    # Save results
    print("\n5. SAVING RESULTS")
    print("-" * 50)

    # Save strategy returns
    results = pd.DataFrame({
        'phase': aligned_signals['phase'],
        'growth_signal': aligned_signals['growth_signal'],
        'inflation_signal': aligned_signals['inflation_signal'],
        'strategy_return': strategy_returns,
        'benchmark_return': benchmark_returns
    })
    import os
    results.to_csv(os.path.join(data_dir, 'backtest_results.csv'))
    print("  ✓ data/backtest_results.csv")

    # Save phase statistics
    phase_df = pd.DataFrame(phase_stats).T
    phase_df.to_csv(os.path.join(data_dir, 'phase_statistics.csv'))
    print("  ✓ data/phase_statistics.csv")

    # Save lag sensitivity results
    lag_results.to_csv(os.path.join(data_dir, 'lag_sensitivity.csv'), index=False)
    print("  ✓ data/lag_sensitivity.csv")

    print("\n" + "=" * 70)
    print("Backtest complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
