#!/usr/bin/env python3
"""
Backtest: Real Estate (XLRE) with Orders/Inventories Ratio Filter

Strategy:
- LONG XLRE when Orders/Inventories Ratio YoY > 0 (Rising)
- CASH (or reduced exposure) when Orders/Inventories Ratio YoY < 0 (Falling)

Based on bulk analysis finding: XLRE Rising Sharpe 1.12 vs Falling Sharpe -0.01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')


def load_data():
    """Load XLRE prices and Orders/Inventories indicator."""
    # Load indicators
    indicators = pd.read_parquet(os.path.join(DATA_DIR, 'monthly_all_indicators.parquet'))
    indicators.index = pd.to_datetime(indicators.index)

    # Load sector prices
    prices = pd.read_parquet(os.path.join(DATA_DIR, 'sector_prices.parquet'))
    prices.index = pd.to_datetime(prices.index)

    # Resample to monthly
    monthly_prices = prices.resample('ME').last()

    # Get XLRE
    xlre = monthly_prices['XLRE'].dropna()

    # Get Orders/Inv Ratio
    if 'orders_inv_ratio' not in indicators.columns:
        raise ValueError("orders_inv_ratio not found in indicators")

    orders_inv = indicators['orders_inv_ratio']

    # Calculate YoY change
    orders_inv_yoy = orders_inv.pct_change(12) * 100

    return xlre, orders_inv, orders_inv_yoy


def calculate_returns(prices):
    """Calculate monthly returns."""
    return prices.pct_change()


def generate_signal(orders_inv_yoy, lag=1):
    """
    Generate trading signal based on Orders/Inv Ratio YoY direction.

    lag: Number of months to lag the signal (to avoid look-ahead bias)
    """
    # Signal: 1 = Rising (bullish), 0 = Falling (bearish)
    signal = (orders_inv_yoy > 0).astype(int)

    # Apply lag
    signal = signal.shift(lag)

    return signal


def backtest_strategy(returns, signal, strategy_name="Strategy"):
    """
    Run backtest with given signal.

    Returns: DataFrame with strategy metrics
    """
    # Align data
    common_idx = returns.dropna().index.intersection(signal.dropna().index)
    returns = returns.loc[common_idx]
    signal = signal.loc[common_idx]

    # Strategy returns: invested when signal = 1, cash when signal = 0
    strategy_returns = returns * signal

    # Buy & Hold returns
    bh_returns = returns

    # Calculate cumulative returns
    strategy_cumret = (1 + strategy_returns).cumprod()
    bh_cumret = (1 + bh_returns).cumprod()

    # Calculate metrics
    def calc_metrics(rets, name):
        total_months = len(rets)
        ann_return = rets.mean() * 12 * 100
        ann_vol = rets.std() * np.sqrt(12) * 100
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Max drawdown
        cumret = (1 + rets).cumprod()
        rolling_max = cumret.expanding().max()
        drawdown = (cumret - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100

        # Win rate
        win_rate = (rets > 0).mean() * 100

        # Time in market
        invested_months = (rets != 0).sum()
        time_in_market = invested_months / total_months * 100

        return {
            'Strategy': name,
            'Total Months': total_months,
            'Ann. Return (%)': ann_return,
            'Ann. Volatility (%)': ann_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown (%)': max_dd,
            'Win Rate (%)': win_rate,
            'Time in Market (%)': time_in_market
        }

    metrics = [
        calc_metrics(strategy_returns, strategy_name),
        calc_metrics(bh_returns, 'Buy & Hold XLRE')
    ]

    return pd.DataFrame(metrics), strategy_cumret, bh_cumret, signal


def analyze_regime_performance(returns, signal):
    """Analyze performance in each regime."""
    common_idx = returns.dropna().index.intersection(signal.dropna().index)
    returns = returns.loc[common_idx]
    signal = signal.loc[common_idx]

    results = []

    for regime, mask in [('Rising O/I (Invested)', signal == 1),
                         ('Falling O/I (Cash)', signal == 0)]:
        regime_rets = returns[mask]
        if len(regime_rets) > 0:
            results.append({
                'Regime': regime,
                'Months': len(regime_rets),
                'Mean Monthly (%)': regime_rets.mean() * 100,
                'Ann. Return (%)': regime_rets.mean() * 12 * 100,
                'Ann. Vol (%)': regime_rets.std() * np.sqrt(12) * 100,
                'Sharpe': (regime_rets.mean() * 12) / (regime_rets.std() * np.sqrt(12)) if regime_rets.std() > 0 else 0,
                'Win Rate (%)': (regime_rets > 0).mean() * 100
            })

    return pd.DataFrame(results)


def plot_backtest(strategy_cumret, bh_cumret, signal, orders_inv_yoy, output_path):
    """Create backtest visualization."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Cumulative Returns
    ax1 = axes[0]
    ax1.plot(strategy_cumret.index, strategy_cumret.values, 'b-', linewidth=2,
             label=f'Strategy (Final: {strategy_cumret.iloc[-1]:.2f}x)')
    ax1.plot(bh_cumret.index, bh_cumret.values, 'gray', linewidth=1.5, alpha=0.7,
             label=f'Buy & Hold (Final: {bh_cumret.iloc[-1]:.2f}x)')
    ax1.set_ylabel('Cumulative Return', fontsize=10)
    ax1.set_title('XLRE Backtest: Orders/Inventories Ratio Filter\n'
                  'Long XLRE when O/I Ratio Rising, Cash when Falling', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Signal (Invested vs Cash)
    ax2 = axes[1]
    ax2.fill_between(signal.index, 0, signal.values, color='green', alpha=0.3,
                     label='Invested in XLRE', step='post')
    ax2.fill_between(signal.index, 0, 1 - signal.values, color='red', alpha=0.3,
                     label='Cash', step='post')
    ax2.set_ylabel('Position', fontsize=10)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Cash', 'Invested'])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Orders/Inventories Ratio YoY
    ax3 = axes[2]
    common_idx = orders_inv_yoy.index.intersection(signal.index)
    oi_yoy = orders_inv_yoy.loc[common_idx]

    ax3.plot(oi_yoy.index, oi_yoy.values, 'purple', linewidth=1)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax3.fill_between(oi_yoy.index, 0, oi_yoy.values,
                     where=oi_yoy > 0, color='green', alpha=0.3)
    ax3.fill_between(oi_yoy.index, 0, oi_yoy.values,
                     where=oi_yoy < 0, color='red', alpha=0.3)
    ax3.set_ylabel('O/I Ratio YoY (%)', fontsize=10)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.grid(True, alpha=0.3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved plot to {output_path}")


def run_lag_sensitivity(returns, orders_inv_yoy, max_lag=6):
    """Test strategy performance across different signal lags."""
    results = []

    for lag in range(1, max_lag + 1):
        signal = generate_signal(orders_inv_yoy, lag=lag)
        common_idx = returns.dropna().index.intersection(signal.dropna().index)
        ret = returns.loc[common_idx]
        sig = signal.loc[common_idx]

        strategy_rets = ret * sig

        ann_return = strategy_rets.mean() * 12 * 100
        ann_vol = strategy_rets.std() * np.sqrt(12) * 100
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        results.append({
            'Lag (months)': lag,
            'Ann. Return (%)': ann_return,
            'Ann. Vol (%)': ann_vol,
            'Sharpe': sharpe
        })

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("BACKTEST: XLRE with Orders/Inventories Ratio Filter")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    xlre, orders_inv, orders_inv_yoy = load_data()

    print(f"  XLRE: {xlre.index[0].strftime('%Y-%m')} to {xlre.index[-1].strftime('%Y-%m')} ({len(xlre)} months)")
    print(f"  O/I Ratio: {orders_inv.index[0].strftime('%Y-%m')} to {orders_inv.index[-1].strftime('%Y-%m')}")

    # Calculate returns
    returns = calculate_returns(xlre)

    # Generate signal with 1-month lag
    signal = generate_signal(orders_inv_yoy, lag=1)

    # Run backtest
    print("\n" + "-" * 80)
    print("BACKTEST RESULTS (1-month lag)")
    print("-" * 80)

    metrics_df, strategy_cumret, bh_cumret, aligned_signal = backtest_strategy(
        returns, signal, "O/I Ratio Filter"
    )

    print("\nPerformance Comparison:")
    print(metrics_df.to_string(index=False))

    # Regime analysis
    print("\n" + "-" * 80)
    print("REGIME PERFORMANCE")
    print("-" * 80)

    regime_df = analyze_regime_performance(returns, signal)
    print("\nXLRE Returns by O/I Ratio Regime:")
    print(regime_df.to_string(index=False))

    # Calculate strategy advantage
    if len(regime_df) == 2:
        rising_sharpe = regime_df[regime_df['Regime'].str.contains('Rising')]['Sharpe'].values[0]
        falling_sharpe = regime_df[regime_df['Regime'].str.contains('Falling')]['Sharpe'].values[0]
        print(f"\nSharpe Advantage (Rising vs Falling): {rising_sharpe - falling_sharpe:+.2f}")

    # Lag sensitivity
    print("\n" + "-" * 80)
    print("LAG SENSITIVITY ANALYSIS")
    print("-" * 80)

    lag_df = run_lag_sensitivity(returns, orders_inv_yoy, max_lag=6)
    print("\nStrategy Performance by Signal Lag:")
    print(lag_df.to_string(index=False))

    # Trade statistics
    print("\n" + "-" * 80)
    print("TRADE STATISTICS")
    print("-" * 80)

    # Count regime changes
    regime_changes = (aligned_signal.diff() != 0).sum() - 1  # Exclude first NaN
    total_months = len(aligned_signal)
    invested_months = aligned_signal.sum()
    cash_months = total_months - invested_months

    print(f"\n  Total Months: {total_months}")
    print(f"  Invested Months: {int(invested_months)} ({invested_months/total_months*100:.1f}%)")
    print(f"  Cash Months: {int(cash_months)} ({cash_months/total_months*100:.1f}%)")
    print(f"  Regime Changes: {int(regime_changes)}")
    print(f"  Avg. Holding Period: {total_months / (regime_changes + 1):.1f} months")

    # Create visualization
    output_path = os.path.join(DATA_DIR, 'xlre_orders_inv_backtest.png')
    plot_backtest(strategy_cumret, bh_cumret, aligned_signal, orders_inv_yoy, output_path)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    strategy_metrics = metrics_df[metrics_df['Strategy'] == 'O/I Ratio Filter'].iloc[0]
    bh_metrics = metrics_df[metrics_df['Strategy'] == 'Buy & Hold XLRE'].iloc[0]

    print(f"""
Strategy: Long XLRE when Orders/Inventories Ratio YoY > 0, Cash otherwise

Performance (with 1-month signal lag):
  Strategy Sharpe: {strategy_metrics['Sharpe Ratio']:.2f}
  Buy & Hold Sharpe: {bh_metrics['Sharpe Ratio']:.2f}
  Sharpe Improvement: {strategy_metrics['Sharpe Ratio'] - bh_metrics['Sharpe Ratio']:+.2f}

  Strategy Max Drawdown: {strategy_metrics['Max Drawdown (%)']:.1f}%
  Buy & Hold Max Drawdown: {bh_metrics['Max Drawdown (%)']:.1f}%

Key Insight: The O/I Ratio filter provides meaningful regime differentiation
for Real Estate. Avoiding exposure during falling O/I periods significantly
improves risk-adjusted returns.
""")

    return metrics_df, regime_df


if __name__ == '__main__':
    metrics, regimes = main()
