#!/usr/bin/env python3
"""
ML Regime Detection Baseline Experiments.

Compares unsupervised regime detection (GMM, HMM) against rule-based phases.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.ml.feature_engineering import FeatureEngineer, create_rule_based_targets, PHASE_NAMES
from src.ml.regime_detection import GMMRegimeDetector, HMMRegimeDetector, KMeansRegimeDetector, map_regimes_to_phases
from src.ml.validation import TimeSeriesBacktester


# Asset allocations per phase (for backtesting)
PHASE_ALLOCATIONS_DIVERSIFIED = {
    0: {'stocks': 0.7, 'bonds': 0.2, 'commodities': 0.0, 'cash': 0.1},  # Recovery
    1: {'stocks': 0.3, 'bonds': 0.0, 'commodities': 0.5, 'cash': 0.2},  # Overheat
    2: {'stocks': 0.0, 'bonds': 0.2, 'commodities': 0.3, 'cash': 0.5},  # Stagflation
    3: {'stocks': 0.2, 'bonds': 0.6, 'commodities': 0.0, 'cash': 0.2},  # Reflation
}


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


def main():
    print("=" * 80)
    print("ML REGIME DETECTION BASELINE EXPERIMENTS")
    print("=" * 80)

    # 1. Load Data
    print("\n1. LOADING DATA")
    print("-" * 40)
    indicators, prices, data_dir = load_data()
    print(f"  Indicators: {indicators.shape}")
    print(f"  Prices: {prices.shape}")

    # 2. Feature Engineering
    print("\n2. FEATURE ENGINEERING")
    print("-" * 40)
    fe = FeatureEngineer(lookback_window=60)
    features = fe.create_all_features(indicators, prices)
    print(f"  Created {len(features.columns)} features")
    print(f"  Date range: {features.index[0]} to {features.index[-1]}")

    # Prepare ML data (drop NaN, scale)
    X, _, valid_index = fe.prepare_ml_data(features, lag=0, dropna=True)
    print(f"  Valid samples after dropna: {len(X)}")

    # Scale features
    X_scaled = fe.scale_features(X, fit=True)
    print(f"  Scaled features shape: {X_scaled.shape}")

    # 3. Rule-Based Baseline
    print("\n3. RULE-BASED BASELINE (Orders/Inv MoM + PPI MoM)")
    print("-" * 40)
    rule_targets = create_rule_based_targets(indicators)
    rule_targets = rule_targets.loc[valid_index]

    phase_counts = rule_targets['phase_name'].value_counts()
    print("  Phase Distribution:")
    for phase, count in phase_counts.items():
        print(f"    {phase}: {count} ({count/len(rule_targets)*100:.1f}%)")

    # 4. GMM Regime Detection
    print("\n4. GMM REGIME DETECTION")
    print("-" * 40)

    # Find optimal number of regimes
    gmm_temp = GMMRegimeDetector(n_regimes=4)
    optimal_n, bic_results = gmm_temp.select_optimal_n_regimes(X_scaled, min_regimes=2, max_regimes=6)
    print(f"  Optimal regimes by BIC: {optimal_n}")
    print("\n  BIC Analysis:")
    print(bic_results.to_string(index=False))

    # Fit GMM with 4 regimes (to match Investment Clock)
    gmm = GMMRegimeDetector(n_regimes=4)
    gmm.fit(X_scaled)
    gmm_labels = gmm.predict(X_scaled)

    print(f"\n  GMM Regime Distribution:")
    for regime in range(4):
        count = (gmm_labels == regime).sum()
        print(f"    Regime {regime}: {count} ({count/len(gmm_labels)*100:.1f}%)")

    # Map GMM regimes to phases
    gmm_mapping = map_regimes_to_phases(gmm_labels, features.loc[valid_index])
    print("\n  GMM Regime to Phase Mapping:")
    if gmm_mapping is not None:
        print(gmm_mapping.to_string(index=False))

    # 5. HMM Regime Detection
    print("\n5. HMM REGIME DETECTION")
    print("-" * 40)

    try:
        hmm = HMMRegimeDetector(n_regimes=4, n_iter=100)
        hmm.fit(X_scaled)
        hmm_labels = hmm.predict(X_scaled)

        print(f"  HMM Regime Distribution:")
        for regime in range(4):
            count = (hmm_labels == regime).sum()
            print(f"    Regime {regime}: {count} ({count/len(hmm_labels)*100:.1f}%)")

        # Transition matrix
        trans_matrix = hmm.get_transition_matrix()
        print("\n  Transition Matrix:")
        print(trans_matrix.round(3).to_string())

        # Regime persistence
        persistence = hmm.get_regime_persistence()
        print("\n  Expected Regime Duration (months):")
        for regime, duration in persistence.items():
            print(f"    {regime}: {duration:.1f}")

        # Map HMM regimes to phases
        hmm_mapping = map_regimes_to_phases(hmm_labels, features.loc[valid_index])
        print("\n  HMM Regime to Phase Mapping:")
        if hmm_mapping is not None:
            print(hmm_mapping.to_string(index=False))

    except Exception as e:
        print(f"  HMM failed: {e}")
        hmm_labels = None

    # 6. K-Means Baseline
    print("\n6. K-MEANS BASELINE")
    print("-" * 40)
    kmeans = KMeansRegimeDetector(n_regimes=4)
    kmeans.fit(X_scaled)
    kmeans_labels = kmeans.predict(X_scaled)

    print(f"  K-Means Regime Distribution:")
    for regime in range(4):
        count = (kmeans_labels == regime).sum()
        print(f"    Regime {regime}: {count} ({count/len(kmeans_labels)*100:.1f}%)")

    # 7. Compare Agreement Between Methods
    print("\n7. METHOD AGREEMENT ANALYSIS")
    print("-" * 40)

    # Agreement with rule-based
    rule_phases = rule_targets['phase'].values

    # GMM agreement
    gmm_agreement = (gmm_labels == rule_phases).mean() * 100
    print(f"  GMM vs Rule-based agreement: {gmm_agreement:.1f}%")

    # HMM agreement
    if hmm_labels is not None:
        hmm_agreement = (hmm_labels == rule_phases).mean() * 100
        print(f"  HMM vs Rule-based agreement: {hmm_agreement:.1f}%")

    # K-Means agreement
    kmeans_agreement = (kmeans_labels == rule_phases).mean() * 100
    print(f"  K-Means vs Rule-based agreement: {kmeans_agreement:.1f}%")

    # 8. Backtest Comparison
    print("\n8. BACKTEST COMPARISON")
    print("-" * 40)

    asset_returns = calculate_asset_returns(prices)
    asset_returns = asset_returns.loc[valid_index]

    backtester = TimeSeriesBacktester(signal_lag=1)

    # Rule-based backtest
    rule_results = backtester.backtest_regimes(
        regime_labels=rule_phases.astype(int),
        asset_returns=asset_returns,
        regime_allocations=PHASE_ALLOCATIONS_DIVERSIFIED,
        index=valid_index
    )
    rule_metrics = backtester.calculate_metrics(rule_results['strategy_return'], "Rule-Based")

    # GMM backtest
    gmm_results = backtester.backtest_regimes(
        regime_labels=gmm_labels,
        asset_returns=asset_returns,
        regime_allocations=PHASE_ALLOCATIONS_DIVERSIFIED,
        index=valid_index
    )
    gmm_metrics = backtester.calculate_metrics(gmm_results['strategy_return'], "GMM")

    # HMM backtest
    if hmm_labels is not None:
        hmm_results = backtester.backtest_regimes(
            regime_labels=hmm_labels,
            asset_returns=asset_returns,
            regime_allocations=PHASE_ALLOCATIONS_DIVERSIFIED,
            index=valid_index
        )
        hmm_metrics = backtester.calculate_metrics(hmm_results['strategy_return'], "HMM")
    else:
        hmm_metrics = {}

    # K-Means backtest
    kmeans_results = backtester.backtest_regimes(
        regime_labels=kmeans_labels,
        asset_returns=asset_returns,
        regime_allocations=PHASE_ALLOCATIONS_DIVERSIFIED,
        index=valid_index
    )
    kmeans_metrics = backtester.calculate_metrics(kmeans_results['strategy_return'], "K-Means")

    # Benchmark
    benchmark_metrics = backtester.calculate_metrics(asset_returns['stocks'].dropna(), "Buy & Hold SPY")

    # Print comparison table
    print("\n  PERFORMANCE COMPARISON (Diversified Allocation, 1-Month Lag)")
    print("  " + "-" * 75)
    print(f"  {'Method':<15} {'CAGR':>10} {'Volatility':>12} {'Sharpe':>10} {'Max DD':>12}")
    print("  " + "-" * 75)

    for metrics in [rule_metrics, gmm_metrics, hmm_metrics, kmeans_metrics, benchmark_metrics]:
        if metrics:
            print(f"  {metrics['name']:<15} {metrics['cagr']:>9.1%} {metrics['volatility']:>11.1%} "
                  f"{metrics['sharpe']:>10.2f} {metrics['max_drawdown']:>11.1%}")

    # 9. Save Results
    print("\n9. SAVING RESULTS")
    print("-" * 40)

    # Create results DataFrame
    results_df = pd.DataFrame({
        'date': valid_index,
        'rule_based_phase': rule_phases,
        'gmm_regime': gmm_labels,
        'hmm_regime': hmm_labels if hmm_labels is not None else np.nan,
        'kmeans_regime': kmeans_labels,
    })
    results_df.set_index('date', inplace=True)

    # Add phase names
    results_df['rule_based_name'] = results_df['rule_based_phase'].map(PHASE_NAMES)

    results_df.to_csv(os.path.join(data_dir, 'ml_regime_comparison.csv'))
    print(f"  ✓ Saved to data/ml_regime_comparison.csv")

    # Save metrics
    metrics_df = pd.DataFrame([
        rule_metrics, gmm_metrics, hmm_metrics if hmm_metrics else {},
        kmeans_metrics, benchmark_metrics
    ])
    metrics_df.to_csv(os.path.join(data_dir, 'ml_regime_metrics.csv'), index=False)
    print(f"  ✓ Saved to data/ml_regime_metrics.csv")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    # Summary
    print("\nSUMMARY:")
    print(f"  - Rule-based (Orders/Inv MoM + PPI MoM) Sharpe: {rule_metrics.get('sharpe', 0):.2f}")
    print(f"  - GMM Sharpe: {gmm_metrics.get('sharpe', 0):.2f}")
    if hmm_metrics:
        print(f"  - HMM Sharpe: {hmm_metrics.get('sharpe', 0):.2f}")
    print(f"  - K-Means Sharpe: {kmeans_metrics.get('sharpe', 0):.2f}")
    print(f"  - Buy & Hold Sharpe: {benchmark_metrics.get('sharpe', 0):.2f}")


if __name__ == '__main__':
    main()
