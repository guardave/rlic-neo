#!/usr/bin/env python3
"""
Re-run Investment Clock Sector Analysis.

Independently reproduces the analysis from docs/analysis_reports/investment_clock_sector_analysis.pdf
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

from src.dashboard.data_loader import load_investment_clock_data
from src.dashboard.analysis_engine import (
    create_derivatives, define_investment_clock_phases,
    regime_performance, correlation_analysis, create_direction_signal
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "rerun" / "investment_clock"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_analysis():
    """Run the Investment Clock sector analysis."""
    print("=" * 60)
    print("Investment Clock Sector Analysis - Re-run")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load data
    print("Loading data...")
    data = load_investment_clock_data()
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print()

    # Define phases
    print("Defining Investment Clock phases...")
    # Check for existing phase column first
    if 'best_indicator_phase' in data.columns:
        data['phase'] = data['best_indicator_phase']
        print("Using existing 'best_indicator_phase' column")
    elif 'orders_inv_ratio' in data.columns and 'cpi_yoy' in data.columns:
        # Use CPI YoY as inflation proxy
        data['phase'] = define_investment_clock_phases(
            data,
            growth_col='orders_inv_ratio',
            inflation_col='cpi_yoy'
        )
    elif 'orders_inv_ratio' in data.columns and 'ppi_all' in data.columns:
        data['phase'] = define_investment_clock_phases(
            data,
            growth_col='orders_inv_ratio',
            inflation_col='ppi_all'
        )
    else:
        print("Warning: Required columns not found for phase classification")
        print(f"Available columns: {data.columns.tolist()}")
        return None

    # Phase distribution
    phase_counts = data['phase'].value_counts()
    print("\nPhase Distribution:")
    for phase, count in phase_counts.items():
        pct = count / len(data) * 100
        print(f"  {phase}: {count} months ({pct:.1f}%)")

    # Identify sector return columns
    return_cols = [c for c in data.columns if c.endswith('_return')]
    print(f"\nSector return columns: {len(return_cols)}")

    # Calculate sector performance by phase
    print("\nCalculating sector performance by phase...")
    results = []

    for sector_col in return_cols:
        sector_name = sector_col.replace('_return', '')

        for lag in [0, 1]:
            # Apply lag to phase signal
            if lag > 0:
                phase_lagged = data['phase'].shift(lag)
            else:
                phase_lagged = data['phase']

            for phase in data['phase'].dropna().unique():
                mask = phase_lagged == phase
                returns = data.loc[mask, sector_col].dropna()

                if len(returns) >= 12:  # At least 1 year of data
                    mean_ret = returns.mean()
                    std_ret = returns.std()
                    sharpe = mean_ret / std_ret * np.sqrt(12) if std_ret > 0 else 0

                    results.append({
                        'sector': sector_name,
                        'phase': phase,
                        'lag': lag,
                        'mean_return': mean_ret,
                        'std_return': std_ret,
                        'sharpe_ratio': sharpe,
                        'n_periods': len(returns),
                        'pct_positive': (returns > 0).mean()
                    })

    results_df = pd.DataFrame(results)
    print(f"Results shape: {results_df.shape}")

    # Save results
    print("\nSaving results...")

    # Full results
    results_df.to_csv(OUTPUT_DIR / "sector_phase_performance.csv", index=False)
    print(f"  Saved: sector_phase_performance.csv")

    # Pivot table for heatmap (lag=1 results)
    lag1_results = results_df[results_df['lag'] == 1]
    heatmap_data = lag1_results.pivot_table(
        values='mean_return',
        index='sector',
        columns='phase',
        aggfunc='mean'
    )
    heatmap_data.to_csv(OUTPUT_DIR / "sector_phase_heatmap.csv")
    print(f"  Saved: sector_phase_heatmap.csv")

    # Summary statistics
    summary = {
        'analysis_id': 'investment_clock',
        'run_timestamp': datetime.now().isoformat(),
        'data_start': str(data.index.min()),
        'data_end': str(data.index.max()),
        'n_periods': len(data),
        'n_sectors': len(return_cols),
        'phases': phase_counts.to_dict()
    }

    # Best sector per phase (lag=1)
    print("\nBest Sector by Phase (lag=1):")
    for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
        phase_data = lag1_results[lag1_results['phase'] == phase]
        if not phase_data.empty:
            best = phase_data.loc[phase_data['mean_return'].idxmax()]
            print(f"  {phase}: {best['sector']} ({best['mean_return']:.4f})")

    # Save summary
    import json
    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: summary.json")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    return results_df


if __name__ == "__main__":
    run_analysis()
