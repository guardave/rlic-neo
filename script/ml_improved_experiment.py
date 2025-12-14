#!/usr/bin/env python3
"""
Improved ML Regime Detection Experiment.

Implements the three improvements discussed:
1. Composite features (reduce from 49 to 2 dimensions)
2. Direction-only features (binary signals matching Investment Clock logic)
3. Supervised learning with rule-based labels as ground truth
4. HMM with supervised initialization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.ml.feature_engineering import (
    FeatureEngineer,
    create_rule_based_targets,
    create_composite_features,
    create_direction_only_features,
    PHASE_NAMES
)
from src.ml.regime_detection import GMMRegimeDetector, HMMRegimeDetector
from src.ml.supervised_regime import SupervisedRegimeClassifier, HMMWithSupervisedInit
from src.ml.validation import TimeSeriesBacktester, PurgedWalkForwardCV


# Asset allocations per phase
PHASE_ALLOCATIONS = {
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
    for col in ['SPY', 'spy', 'sp500']:
        if col in monthly_prices.columns:
            returns['stocks'] = monthly_prices[col].pct_change()
            break

    # Bond returns
    for col in ['TLT', 'tlt', 'treasury_10y']:
        if col in monthly_prices.columns:
            returns['bonds'] = monthly_prices[col].pct_change()
            break

    # Commodity returns
    for col in ['GLD', 'gld', 'gold', 'DBC', 'dbc']:
        if col in monthly_prices.columns:
            returns['commodities'] = monthly_prices[col].pct_change()
            break

    returns['cash'] = 0.002
    return returns


def main():
    print("=" * 80)
    print("IMPROVED ML REGIME DETECTION EXPERIMENT")
    print("=" * 80)

    # 1. Load Data
    print("\n1. LOADING DATA")
    print("-" * 40)
    indicators, prices, data_dir = load_data()
    print(f"  Indicators: {indicators.shape}")
    print(f"  Date range: {indicators.index[0]} to {indicators.index[-1]}")

    # 2. Create Rule-Based Baseline
    print("\n2. RULE-BASED BASELINE (Ground Truth)")
    print("-" * 40)
    rule_targets = create_rule_based_targets(indicators)

    phase_counts = rule_targets['phase_name'].value_counts()
    print("  Phase Distribution:")
    for phase, count in phase_counts.items():
        if pd.notna(phase):
            print(f"    {phase}: {count} ({count/len(rule_targets)*100:.1f}%)")

    # Get valid index (non-NaN phases)
    valid_mask = rule_targets['phase'].notna()
    valid_index = rule_targets.index[valid_mask]
    rule_phases = rule_targets.loc[valid_index, 'phase'].values.astype(int)
    print(f"  Valid samples: {len(valid_index)}")

    # 3. Feature Sets
    print("\n3. CREATING FEATURE SETS")
    print("-" * 40)

    # A. Full features (49 features)
    fe = FeatureEngineer(lookback_window=60)
    full_features = fe.create_all_features(indicators)
    full_features = full_features.replace([np.inf, -np.inf], np.nan)
    print(f"  A. Full features: {len(full_features.columns)} features")

    # B. Composite features (2 dimensions)
    composite_features = create_composite_features(indicators)
    print(f"  B. Composite features: {len(composite_features.columns)} features")
    print(f"     Growth composite range: [{composite_features['growth_composite'].min():.2f}, {composite_features['growth_composite'].max():.2f}]")
    print(f"     Inflation composite range: [{composite_features['inflation_composite'].min():.2f}, {composite_features['inflation_composite'].max():.2f}]")

    # C. Direction-only features (binary)
    direction_features = create_direction_only_features(indicators)
    print(f"  C. Direction features: {len(direction_features.columns)} features")

    # 4. Prepare Data
    print("\n4. PREPARING DATA")
    print("-" * 40)

    # Common valid index across all feature sets
    common_idx = valid_index.intersection(composite_features.dropna().index)
    common_idx = common_idx.intersection(direction_features.dropna().index)

    # Align all data to common index
    X_composite = composite_features[['growth_composite', 'inflation_composite']].loc[common_idx]
    X_direction = direction_features.loc[common_idx].dropna()
    common_idx = X_composite.index.intersection(X_direction.index)

    X_composite = X_composite.loc[common_idx]
    X_direction = X_direction.loc[common_idx]
    y = rule_targets.loc[common_idx, 'phase'].values.astype(int)

    print(f"  Common samples: {len(common_idx)}")
    print(f"  X_composite shape: {X_composite.shape}")
    print(f"  X_direction shape: {X_direction.shape}")

    # Scale for clustering
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_composite_scaled = pd.DataFrame(
        scaler.fit_transform(X_composite),
        index=X_composite.index,
        columns=X_composite.columns
    )

    # 5. Approach 1: GMM on Composite Features
    print("\n5. APPROACH 1: GMM ON 2D COMPOSITE FEATURES")
    print("-" * 40)

    gmm_2d = GMMRegimeDetector(n_regimes=4)
    gmm_2d.fit(X_composite_scaled)
    gmm_2d_labels = gmm_2d.predict(X_composite_scaled)

    print("  GMM Regime Distribution:")
    for regime in range(4):
        count = (gmm_2d_labels == regime).sum()
        print(f"    Regime {regime}: {count} ({count/len(gmm_2d_labels)*100:.1f}%)")

    gmm_2d_agreement = (gmm_2d_labels == y).mean() * 100
    print(f"\n  Agreement with rule-based: {gmm_2d_agreement:.1f}%")

    # Analyze regime characteristics
    print("\n  Regime Characteristics (Growth, Inflation composites):")
    for regime in range(4):
        mask = gmm_2d_labels == regime
        g_mean = X_composite.loc[mask, 'growth_composite'].mean()
        i_mean = X_composite.loc[mask, 'inflation_composite'].mean()

        # Map to expected phase
        if g_mean > 0 and i_mean <= 0:
            expected = "Recovery"
        elif g_mean > 0 and i_mean > 0:
            expected = "Overheat"
        elif g_mean <= 0 and i_mean > 0:
            expected = "Stagflation"
        else:
            expected = "Reflation"

        print(f"    Regime {regime}: G={g_mean:+.2f}, I={i_mean:+.2f} -> {expected}")

    # 6. Approach 2: Supervised Random Forest
    print("\n6. APPROACH 2: SUPERVISED RANDOM FOREST")
    print("-" * 40)

    # Train/test split (time-based)
    split_idx = int(len(common_idx) * 0.7)
    train_idx = common_idx[:split_idx]
    test_idx = common_idx[split_idx:]

    X_train_dir = X_direction.loc[train_idx]
    X_test_dir = X_direction.loc[test_idx]
    y_train = rule_targets.loc[train_idx, 'phase'].values.astype(int)
    y_test = rule_targets.loc[test_idx, 'phase'].values.astype(int)

    print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    rf = SupervisedRegimeClassifier(model_type='random_forest')
    rf.fit(X_train_dir, y_train)

    # Evaluate
    rf_eval = rf.evaluate(X_test_dir, y_test)
    print(f"\n  Test Accuracy: {rf_eval['accuracy']:.1%}")
    print("\n  Classification Report:")
    for phase_name, metrics in rf_eval['classification_report'].items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            print(f"    {phase_name}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}")

    print("\n  Top 5 Important Features:")
    top_features = rf.get_top_features(5)
    if top_features is not None:
        for feat, imp in top_features.items():
            print(f"    {feat}: {imp:.3f}")

    # Full predictions for backtesting
    rf_labels = rf.predict(X_direction)

    # 7. Approach 3: HMM with Supervised Initialization
    print("\n7. APPROACH 3: HMM WITH SUPERVISED INITIALIZATION")
    print("-" * 40)

    try:
        hmm_sup = HMMWithSupervisedInit(n_regimes=4, covariance_type='diag')
        hmm_sup.fit(X_composite_scaled, rule_targets.loc[common_idx, 'phase'])
        hmm_sup_labels = hmm_sup.predict(X_composite_scaled)

        print("  HMM Regime Distribution:")
        for regime in range(4):
            count = (hmm_sup_labels == regime).sum()
            print(f"    Regime {regime}: {count} ({count/len(hmm_sup_labels)*100:.1f}%)")

        # Compare with initial labels
        comparison = hmm_sup.compare_with_initial(X_composite_scaled)
        print(f"\n  Agreement with rule-based: {comparison['agreement_rate']:.1%}")
        print(f"  Rule-based regime changes: {comparison['initial_regime_changes']}")
        print(f"  HMM regime changes: {comparison['hmm_regime_changes']}")
        print(f"  Smoothing ratio: {comparison['smoothing_ratio']:.2f}")

        # Transition matrix
        print("\n  Learned Transition Matrix:")
        trans = hmm_sup.get_transition_matrix()
        print(trans.round(2).to_string())

        # Persistence
        print("\n  Expected Regime Duration (months):")
        persistence = hmm_sup.get_regime_persistence()
        for phase, duration in persistence.items():
            print(f"    {phase}: {duration:.1f}")

    except Exception as e:
        print(f"  HMM with supervised init failed: {e}")
        hmm_sup_labels = None

    # 8. Backtest Comparison
    print("\n8. BACKTEST COMPARISON")
    print("-" * 40)

    asset_returns = calculate_asset_returns(prices)
    asset_returns = asset_returns.loc[common_idx]

    backtester = TimeSeriesBacktester(signal_lag=1)

    all_metrics = []

    # Rule-based
    rule_results = backtester.backtest_regimes(
        regime_labels=y,
        asset_returns=asset_returns,
        regime_allocations=PHASE_ALLOCATIONS,
        index=common_idx
    )
    rule_metrics = backtester.calculate_metrics(rule_results['strategy_return'], "Rule-Based")
    all_metrics.append(rule_metrics)

    # GMM 2D
    gmm_results = backtester.backtest_regimes(
        regime_labels=gmm_2d_labels,
        asset_returns=asset_returns,
        regime_allocations=PHASE_ALLOCATIONS,
        index=common_idx
    )
    gmm_metrics = backtester.calculate_metrics(gmm_results['strategy_return'], "GMM-2D")
    all_metrics.append(gmm_metrics)

    # Random Forest
    rf_results = backtester.backtest_regimes(
        regime_labels=rf_labels,
        asset_returns=asset_returns,
        regime_allocations=PHASE_ALLOCATIONS,
        index=common_idx
    )
    rf_metrics = backtester.calculate_metrics(rf_results['strategy_return'], "RandomForest")
    all_metrics.append(rf_metrics)

    # HMM Supervised
    if hmm_sup_labels is not None:
        hmm_results = backtester.backtest_regimes(
            regime_labels=hmm_sup_labels,
            asset_returns=asset_returns,
            regime_allocations=PHASE_ALLOCATIONS,
            index=common_idx
        )
        hmm_metrics = backtester.calculate_metrics(hmm_results['strategy_return'], "HMM-Supervised")
        all_metrics.append(hmm_metrics)

    # Benchmark
    benchmark_metrics = backtester.calculate_metrics(asset_returns['stocks'].dropna(), "Buy & Hold SPY")
    all_metrics.append(benchmark_metrics)

    # Print comparison
    print("\n  PERFORMANCE COMPARISON (1-Month Lag)")
    print("  " + "-" * 75)
    print(f"  {'Method':<15} {'CAGR':>10} {'Volatility':>12} {'Sharpe':>10} {'Max DD':>12}")
    print("  " + "-" * 75)

    for metrics in all_metrics:
        if metrics:
            print(f"  {metrics['name']:<15} {metrics['cagr']:>9.1%} {metrics['volatility']:>11.1%} "
                  f"{metrics['sharpe']:>10.2f} {metrics['max_drawdown']:>11.1%}")

    # 9. Walk-Forward Validation for Random Forest
    print("\n9. WALK-FORWARD VALIDATION (Random Forest)")
    print("-" * 40)

    cv = PurgedWalkForwardCV(n_splits=5, test_size=12, purge_gap=3)

    fold_results = []
    all_predictions = pd.Series(index=common_idx, dtype=float)

    for fold_idx, (train_idx_arr, test_idx_arr) in enumerate(cv.split(X_direction)):
        X_fold_train = X_direction.iloc[train_idx_arr]
        X_fold_test = X_direction.iloc[test_idx_arr]
        y_fold_train = y[train_idx_arr]
        y_fold_test = y[test_idx_arr]

        rf_fold = SupervisedRegimeClassifier(model_type='random_forest')
        rf_fold.fit(X_fold_train, y_fold_train)

        y_pred = rf_fold.predict(X_fold_test)
        accuracy = (y_pred == y_fold_test).mean()

        fold_results.append({
            'fold': fold_idx + 1,
            'train_size': len(train_idx_arr),
            'test_size': len(test_idx_arr),
            'accuracy': accuracy
        })

        # Store predictions
        test_dates = X_direction.index[test_idx_arr]
        all_predictions.loc[test_dates] = y_pred

        print(f"  Fold {fold_idx + 1}: Train={len(train_idx_arr)}, Test={len(test_idx_arr)}, Accuracy={accuracy:.1%}")

    # Overall CV metrics
    fold_df = pd.DataFrame(fold_results)
    print(f"\n  Mean CV Accuracy: {fold_df['accuracy'].mean():.1%} (+/- {fold_df['accuracy'].std():.1%})")

    # Backtest on OOS predictions
    valid_pred_mask = all_predictions.notna()
    if valid_pred_mask.sum() > 0:
        cv_results = backtester.backtest_regimes(
            regime_labels=all_predictions[valid_pred_mask].values.astype(int),
            asset_returns=asset_returns.loc[valid_pred_mask],
            regime_allocations=PHASE_ALLOCATIONS,
            index=common_idx[valid_pred_mask]
        )
        cv_metrics = backtester.calculate_metrics(cv_results['strategy_return'], "RF Walk-Forward")
        print(f"\n  Walk-Forward Backtest:")
        print(f"    CAGR: {cv_metrics['cagr']:.1%}")
        print(f"    Sharpe: {cv_metrics['sharpe']:.2f}")
        print(f"    Max DD: {cv_metrics['max_drawdown']:.1%}")

    # 10. Save Results
    print("\n10. SAVING RESULTS")
    print("-" * 40)

    results_df = pd.DataFrame({
        'date': common_idx,
        'rule_based': y,
        'gmm_2d': gmm_2d_labels,
        'random_forest': rf_labels,
        'hmm_supervised': hmm_sup_labels if hmm_sup_labels is not None else np.nan,
        'growth_composite': X_composite['growth_composite'].values,
        'inflation_composite': X_composite['inflation_composite'].values
    })
    results_df.set_index('date', inplace=True)
    results_df['rule_based_name'] = results_df['rule_based'].map(PHASE_NAMES)

    results_df.to_csv(os.path.join(data_dir, 'ml_improved_results.csv'))
    print(f"  Saved to data/ml_improved_results.csv")

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(data_dir, 'ml_improved_metrics.csv'), index=False)
    print(f"  Saved to data/ml_improved_metrics.csv")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    # Summary
    print("\nSUMMARY:")
    print("-" * 40)
    print("Approach Comparison (Sharpe Ratio):")
    for metrics in all_metrics:
        if metrics:
            print(f"  {metrics['name']:<20}: {metrics.get('sharpe', 0):.2f}")

    print("\nKey Findings:")
    print(f"  - GMM on 2D composites agreement: {gmm_2d_agreement:.1f}%")
    print(f"  - Random Forest test accuracy: {rf_eval['accuracy']:.1%}")
    if hmm_sup_labels is not None:
        print(f"  - HMM smoothing ratio: {comparison['smoothing_ratio']:.2f}x fewer regime changes")


if __name__ == '__main__':
    main()
