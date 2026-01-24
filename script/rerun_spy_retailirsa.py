#!/usr/bin/env python3
"""
Re-run SPY vs RETAILIRSA Analysis.

Independently reproduces the analysis from docs/analysis_reports/spy_retailirsa_analysis.pdf
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime
import json

from src.dashboard.data_loader import load_analysis_data
from src.dashboard.analysis_engine import (
    create_derivatives, correlation_analysis, correlation_with_pvalues,
    leadlag_analysis, find_optimal_lag, granger_causality_test,
    define_regimes_direction, regime_performance, simple_backtest,
    backtest_metrics
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "rerun" / "spy_retailirsa"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_analysis():
    """Run the SPY vs RETAILIRSA analysis."""
    print("=" * 60)
    print("SPY vs RETAILIRSA Analysis - Re-run")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load data
    print("Loading data...")
    data = load_analysis_data('spy_retailirsa')
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Columns: {data.columns.tolist()}")
    print()

    # Identify columns
    indicator_cols = [c for c in data.columns if 'retail' in c.lower() and not c.endswith('_return')]
    return_cols = [c for c in data.columns if c.endswith('_return')]

    # Handle case where columns don't have expected names
    if not indicator_cols:
        indicator_cols = [c for c in data.columns if c != 'SPY']
    if not return_cols:
        # Compute returns from SPY price
        if 'SPY' in data.columns:
            data['SPY_return'] = data['SPY'].pct_change()
            return_cols = ['SPY_return']

    indicator_col = indicator_cols[0] if indicator_cols else data.columns[0]
    return_col = return_cols[0] if return_cols else 'SPY_return'
    print(f"Indicator: {indicator_col}")
    print(f"Target: {return_col}")

    # Create derivatives
    print("\nCreating derivative series...")
    for suffix, periods in [('MoM', 1), ('QoQ', 3), ('YoY', 12)]:
        col_name = f"{indicator_col}_{suffix}"
        if col_name not in data.columns:
            data[col_name] = data[indicator_col].pct_change(periods)

    # Correlation analysis
    print("\nCorrelation Analysis:")
    x_cols = [indicator_col] + [f"{indicator_col}_{s}" for s in ['MoM', 'QoQ', 'YoY']
                                if f"{indicator_col}_{s}" in data.columns]
    y_cols = [return_col]

    corr_matrix = correlation_analysis(data, x_cols, y_cols)
    print(corr_matrix)
    corr_matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    # Lead-lag analysis
    print("\nLead-Lag Analysis:")
    mom_col = f"{indicator_col}_MoM"
    leadlag_results = leadlag_analysis(data, mom_col, return_col, max_lag=12)
    optimal = find_optimal_lag(leadlag_results)
    print(f"Optimal lag: {optimal['optimal_lag']} months")
    print(f"Correlation at optimal: {optimal['correlation']:.4f}")
    leadlag_results.to_csv(OUTPUT_DIR / "leadlag_results.csv", index=False)

    # Granger causality
    print("\nGranger Causality Test:")
    granger_results = granger_causality_test(data, mom_col, return_col, max_lag=6)
    if not granger_results.empty:
        sig_lags = granger_results[granger_results['pvalue'] < 0.05]['lag'].tolist()
        print(f"Significant at lags: {sig_lags}")
        granger_results.to_csv(OUTPUT_DIR / "granger_results.csv", index=False)
    else:
        print("Granger test could not be performed")

    # Regime analysis
    print("\nRegime Analysis:")
    data['regime'] = define_regimes_direction(data, indicator_col)
    regime_perf = regime_performance(data, 'regime', return_col)
    print(regime_perf)
    regime_perf.to_csv(OUTPUT_DIR / "regime_performance.csv", index=False)

    # Backtest
    print("\nBacktest (lag=1):")
    backtest_data = simple_backtest(data, mom_col, return_col, lag=1)
    strat_metrics = backtest_metrics(backtest_data['strategy_return'])
    bench_metrics = backtest_metrics(backtest_data['benchmark_return'])

    print(f"Strategy Sharpe: {strat_metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Benchmark Sharpe: {bench_metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Strategy Total Return: {strat_metrics.get('total_return', 0):.2%}")
    print(f"Benchmark Total Return: {bench_metrics.get('total_return', 0):.2%}")

    # Save summary
    summary = {
        'analysis_id': 'spy_retailirsa',
        'run_timestamp': datetime.now().isoformat(),
        'data_start': str(data.index.min()),
        'data_end': str(data.index.max()),
        'n_periods': len(data),
        'indicator': indicator_col,
        'target': return_col,
        'correlation_level': float(corr_matrix.loc[indicator_col, return_col]),
        'optimal_lag': optimal['optimal_lag'],
        'optimal_lag_correlation': float(optimal['correlation']),
        'granger_significant_lags': sig_lags if 'sig_lags' in dir() else [],
        'strategy_sharpe': strat_metrics.get('sharpe_ratio', 0),
        'benchmark_sharpe': bench_metrics.get('sharpe_ratio', 0),
        'strategy_total_return': strat_metrics.get('total_return', 0),
        'benchmark_total_return': bench_metrics.get('total_return', 0)
    }

    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to: {OUTPUT_DIR / 'summary.json'}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    run_analysis()
