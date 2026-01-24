#!/usr/bin/env python3
"""
Bulk Regime Analysis Script

Applies the Time Series Relationship Analysis Framework to analyze
multiple target assets (SPY + 11 sector ETFs) against multiple indicators
(Orders/Inventories Ratio + PPI).

Based on: docs/11_time_series_relationship_framework.md
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'bulk_analysis')

# Target assets
TARGETS = {
    'SPY': 'S&P 500',
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLV': 'Health Care',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLU': 'Utilities',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services'
}

# Indicators to analyze
INDICATORS = {
    'orders_inv_ratio': {
        'name': 'Orders/Inventories Ratio',
        'interpretation': 'Rising = Growth accelerating, Falling = Growth decelerating',
        'direction': 'positive'  # Rising is bullish
    },
    'ppi_yoy': {
        'name': 'PPI YoY',
        'interpretation': 'Rising = Inflation accelerating, Falling = Inflation decelerating',
        'direction': 'negative'  # Rising is bearish (for most assets)
    }
}


def load_data():
    """Load all required datasets."""
    # Load indicators
    indicators = pd.read_parquet(os.path.join(DATA_DIR, 'monthly_all_indicators.parquet'))
    indicators.index = pd.to_datetime(indicators.index)

    # Load sector prices (contains SPY + all sector ETFs)
    sector_prices = pd.read_parquet(os.path.join(DATA_DIR, 'sector_prices.parquet'))
    sector_prices.index = pd.to_datetime(sector_prices.index)

    # Resample to monthly (end of month)
    monthly_prices = sector_prices.resample('ME').last()

    # Load recession indicator
    recession = pd.read_parquet(os.path.join(DATA_DIR, 'recession_indicator.parquet'))
    recession.index = pd.to_datetime(recession.index)

    return indicators, monthly_prices, recession


def calculate_returns(prices):
    """Calculate monthly returns for all assets."""
    returns = prices.pct_change() * 100
    return returns


def prepare_indicator_features(indicators, indicator_col):
    """Prepare indicator derivatives for analysis."""
    if indicator_col not in indicators.columns:
        return None

    ind = indicators[indicator_col].copy()

    features = pd.DataFrame(index=indicators.index)
    features['level'] = ind
    features['mom'] = ind.pct_change() * 100
    features['qoq'] = ind.pct_change(3) * 100
    features['yoy'] = ind.pct_change(12) * 100

    # Direction indicators
    features['mom_dir'] = np.sign(features['mom'])
    features['qoq_dir'] = np.sign(features['qoq'])
    features['yoy_dir'] = np.sign(features['yoy'])

    # Momentum crossovers
    ma3 = ind.rolling(3).mean()
    ma6 = ind.rolling(6).mean()
    features['ma_crossover'] = np.where(ma3 > ma6, 1, -1)

    return features


def correlation_analysis(target_returns, indicator_features):
    """Step 2: Correlation Analysis."""
    # Align indices
    common_idx = target_returns.dropna().index.intersection(
        indicator_features.dropna(how='all').index
    )

    if len(common_idx) < 50:
        return None

    target = target_returns.loc[common_idx]
    ind_feats = indicator_features.loc[common_idx]

    results = {}
    for col in ind_feats.columns:
        valid = ind_feats[col].dropna()
        common = target.index.intersection(valid.index)
        if len(common) < 50:
            continue

        corr, pval = stats.pearsonr(target.loc[common], valid.loc[common])
        results[col] = {'correlation': corr, 'p_value': pval}

    return results


def lead_lag_analysis(target_returns, indicator_features, max_lag=12):
    """Step 3: Lead-Lag Analysis."""
    # Use the YoY direction as primary feature
    if 'yoy' not in indicator_features.columns:
        return None

    ind = indicator_features['yoy']

    best_result = {'lag': 0, 'correlation': 0, 'p_value': 1}

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # Indicator leads target
            shifted_ind = ind.shift(-lag)
        else:
            # Target leads indicator
            shifted_ind = ind.shift(lag)

        common = target_returns.dropna().index.intersection(shifted_ind.dropna().index)
        if len(common) < 50:
            continue

        corr, pval = stats.pearsonr(
            target_returns.loc[common],
            shifted_ind.loc[common]
        )

        if abs(corr) > abs(best_result['correlation']):
            best_result = {
                'lag': lag,
                'correlation': corr,
                'p_value': pval,
                'interpretation': 'Indicator leads' if lag < 0 else 'Contemporaneous' if lag == 0 else 'Target leads'
            }

    return best_result


def granger_causality_analysis(target_returns, indicator_features, max_lag=6):
    """Step 4: Granger Causality Test."""
    from statsmodels.tsa.stattools import grangercausalitytests

    if 'yoy' not in indicator_features.columns:
        return None

    ind = indicator_features['yoy']

    # Prepare data
    data = pd.DataFrame({
        'target': target_returns,
        'indicator': ind
    }).dropna()

    if len(data) < 50:
        return None

    try:
        gc_results = grangercausalitytests(
            data[['target', 'indicator']],
            maxlag=max_lag,
            verbose=False
        )

        # Find best lag
        best_lag = 1
        best_pval = 1
        for lag in range(1, max_lag + 1):
            pval = gc_results[lag][0]['ssr_ftest'][1]
            if pval < best_pval:
                best_pval = pval
                best_lag = lag

        return {
            'best_lag': best_lag,
            'p_value': best_pval,
            'significant': best_pval < 0.05
        }
    except:
        return None


def predictive_model_analysis(target_returns, indicator_features, lags=[1, 3, 6, 12]):
    """Step 5: ML Predictive Models."""
    # Create lagged features
    lagged_features = pd.DataFrame(index=indicator_features.index)

    for col in ['mom', 'qoq', 'yoy']:
        if col not in indicator_features.columns:
            continue
        for lag in lags:
            lagged_features[f'{col}_lag{lag}'] = indicator_features[col].shift(lag)

    # Create forward target
    forward_target = target_returns.shift(-1)

    # Align and clean
    common = lagged_features.dropna().index.intersection(forward_target.dropna().index)
    if len(common) < 100:
        return None

    X = lagged_features.loc[common]
    y = forward_target.loc[common]

    # Time series CV
    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
    }

    results = {}
    for name, model in models.items():
        cv_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            cv_scores.append(r2_score(y_test, y_pred))

        results[name] = np.mean(cv_scores)

    return results


def regime_analysis(target_returns, indicator_features, recession):
    """Step 6: Regime Analysis."""
    if 'yoy' not in indicator_features.columns:
        return None

    # Create regime based on YoY direction
    regime = pd.DataFrame(index=indicator_features.index)
    regime['rising'] = indicator_features['yoy'] > 0

    # Merge with recession
    regime = regime.join(recession[['Recession']], how='left')
    regime['Recession'] = regime['Recession'].fillna(0)

    # Align with returns
    common = target_returns.dropna().index.intersection(regime.dropna().index)
    if len(common) < 50:
        return None

    returns = target_returns.loc[common]
    reg = regime.loc[common]

    results = {}

    # Rising vs Falling
    rising_ret = returns[reg['rising'] == True]
    falling_ret = returns[reg['rising'] == False]

    if len(rising_ret) > 10 and len(falling_ret) > 10:
        results['rising'] = {
            'months': len(rising_ret),
            'mean_return': rising_ret.mean(),
            'sharpe': rising_ret.mean() / rising_ret.std() * np.sqrt(12) if rising_ret.std() > 0 else 0,
            'win_rate': (rising_ret > 0).mean() * 100
        }
        results['falling'] = {
            'months': len(falling_ret),
            'mean_return': falling_ret.mean(),
            'sharpe': falling_ret.mean() / falling_ret.std() * np.sqrt(12) if falling_ret.std() > 0 else 0,
            'win_rate': (falling_ret > 0).mean() * 100
        }

    # Recession
    recession_ret = returns[reg['Recession'] == 1]
    expansion_ret = returns[reg['Recession'] == 0]

    if len(recession_ret) > 5:
        results['recession'] = {
            'months': len(recession_ret),
            'mean_return': recession_ret.mean(),
            'sharpe': recession_ret.mean() / recession_ret.std() * np.sqrt(12) if recession_ret.std() > 0 else 0,
            'win_rate': (recession_ret > 0).mean() * 100
        }

    if len(expansion_ret) > 10:
        results['expansion'] = {
            'months': len(expansion_ret),
            'mean_return': expansion_ret.mean(),
            'sharpe': expansion_ret.mean() / expansion_ret.std() * np.sqrt(12) if expansion_ret.std() > 0 else 0,
            'win_rate': (expansion_ret > 0).mean() * 100
        }

    return results


def create_regime_plot(target_returns, indicator_features, recession, target_name, indicator_name, output_path):
    """Step 7: Visualization."""
    if 'yoy' not in indicator_features.columns:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Align data
    common = target_returns.dropna().index.intersection(indicator_features.dropna().index)
    returns = target_returns.loc[common]
    ind = indicator_features.loc[common]

    # Cumulative returns
    cum_ret = (1 + returns / 100).cumprod()

    # Get regime for background coloring
    rising = ind['yoy'] > 0
    rec = recession.reindex(common)['Recession'].fillna(0)

    # Plot cumulative returns with regime background
    ax1.set_title(f'{target_name} vs {indicator_name}', fontsize=14)

    # Color background by regime
    for i in range(len(common) - 1):
        date = common[i]
        next_date = common[i + 1]

        if rec.loc[date] == 1:
            color = 'lightgray'
            alpha = 0.5
        elif rising.loc[date]:
            color = 'lightgreen'
            alpha = 0.3
        else:
            color = 'lightpink'
            alpha = 0.3

        ax1.axvspan(date, next_date, color=color, alpha=alpha)

    ax1.plot(cum_ret.index, cum_ret.values, 'b-', linewidth=1.5, label=target_name)
    ax1.set_ylabel('Cumulative Return', fontsize=10)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot indicator
    ax2.plot(ind.index, ind['yoy'].values, 'g-', linewidth=1, label=f'{indicator_name} YoY')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.fill_between(ind.index, 0, ind['yoy'].values,
                     where=ind['yoy'] > 0, color='green', alpha=0.3)
    ax2.fill_between(ind.index, 0, ind['yoy'].values,
                     where=ind['yoy'] < 0, color='red', alpha=0.3)
    ax2.set_ylabel(indicator_name, fontsize=10)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_pair(target_ticker, target_name, indicator_col, indicator_info,
                 returns, indicators, recession, output_dir):
    """Run full framework analysis on a single target-indicator pair."""

    target_returns = returns[target_ticker] if target_ticker in returns.columns else None
    if target_returns is None:
        return None

    indicator_features = prepare_indicator_features(indicators, indicator_col)
    if indicator_features is None:
        return None

    result = {
        'target': target_ticker,
        'target_name': target_name,
        'indicator': indicator_col,
        'indicator_name': indicator_info['name']
    }

    # Step 2: Correlation
    corr = correlation_analysis(target_returns, indicator_features)
    if corr and 'yoy' in corr:
        result['correlation_yoy'] = corr['yoy']['correlation']
        result['correlation_pval'] = corr['yoy']['p_value']

    # Step 3: Lead-Lag
    lead_lag = lead_lag_analysis(target_returns, indicator_features)
    if lead_lag:
        result['optimal_lag'] = lead_lag['lag']
        result['lag_correlation'] = lead_lag['correlation']

    # Step 4: Granger Causality
    granger = granger_causality_analysis(target_returns, indicator_features)
    if granger:
        result['granger_lag'] = granger['best_lag']
        result['granger_pval'] = granger['p_value']
        result['granger_significant'] = granger['significant']

    # Step 5: ML Predictive
    ml = predictive_model_analysis(target_returns, indicator_features)
    if ml:
        result['ml_ridge_r2'] = ml['Ridge']
        result['ml_lasso_r2'] = ml['Lasso']
        result['ml_rf_r2'] = ml['RandomForest']

    # Step 6: Regime
    regime = regime_analysis(target_returns, indicator_features, recession)
    if regime:
        if 'rising' in regime:
            result['rising_sharpe'] = regime['rising']['sharpe']
            result['rising_mean'] = regime['rising']['mean_return']
        if 'falling' in regime:
            result['falling_sharpe'] = regime['falling']['sharpe']
            result['falling_mean'] = regime['falling']['mean_return']
        if 'recession' in regime:
            result['recession_sharpe'] = regime['recession']['sharpe']

    # Step 7: Visualization
    plot_path = os.path.join(output_dir, f'{target_ticker}_{indicator_col}_regime.png')
    create_regime_plot(target_returns, indicator_features, recession,
                       target_name, indicator_info['name'], plot_path)
    result['plot_path'] = plot_path

    return result


def run_bulk_analysis():
    """Run full framework analysis on all target-indicator pairs."""
    print("=" * 80)
    print("BULK REGIME ANALYSIS")
    print("Time Series Relationship Framework Application")
    print("=" * 80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("\nLoading data...")
    indicators, prices, recession = load_data()
    returns = calculate_returns(prices)

    print(f"  Indicators: {indicators.shape}")
    print(f"  Prices: {prices.shape}")
    print(f"  Period: {indicators.index[0].strftime('%Y-%m')} to {indicators.index[-1].strftime('%Y-%m')}")

    # Run analysis for each pair
    all_results = []
    total = len(TARGETS) * len(INDICATORS)
    current = 0

    print(f"\nAnalyzing {total} target-indicator pairs...")
    print("-" * 80)

    for target_ticker, target_name in TARGETS.items():
        for indicator_col, indicator_info in INDICATORS.items():
            current += 1
            print(f"  [{current}/{total}] {target_name} vs {indicator_info['name']}...", end=' ')

            result = analyze_pair(
                target_ticker, target_name,
                indicator_col, indicator_info,
                returns, indicators, recession,
                OUTPUT_DIR
            )

            if result:
                all_results.append(result)
                sharpe_diff = result.get('rising_sharpe', 0) - result.get('falling_sharpe', 0)
                print(f"Sharpe diff: {sharpe_diff:+.2f}")
            else:
                print("SKIPPED (insufficient data)")

    # Create summary DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'bulk_analysis_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n Saved results to {results_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: REGIME SHARPE RATIOS")
    print("=" * 80)

    for indicator_col, indicator_info in INDICATORS.items():
        print(f"\n{indicator_info['name']}:")
        print("-" * 60)
        print(f"{'Target':<25} {'Rising':>10} {'Falling':>10} {'Diff':>10} {'Significant':>12}")
        print("-" * 60)

        ind_results = results_df[results_df['indicator'] == indicator_col]
        ind_results = ind_results.sort_values('rising_sharpe', ascending=False)

        for _, row in ind_results.iterrows():
            rising = row.get('rising_sharpe', np.nan)
            falling = row.get('falling_sharpe', np.nan)
            diff = rising - falling if pd.notna(rising) and pd.notna(falling) else np.nan
            sig = 'Yes' if abs(diff) > 0.3 else 'No' if pd.notna(diff) else '-'

            print(f"{row['target_name']:<25} {rising:>10.2f} {falling:>10.2f} {diff:>+10.2f} {sig:>12}")

    # Best pairs summary
    print("\n" + "=" * 80)
    print("TOP 10 TARGET-INDICATOR PAIRS BY REGIME SHARPE DIFFERENCE")
    print("=" * 80)

    results_df['sharpe_diff'] = results_df['rising_sharpe'] - results_df['falling_sharpe']
    top10 = results_df.nlargest(10, 'sharpe_diff')

    print(f"\n{'Target':<20} {'Indicator':<25} {'Rising':>10} {'Falling':>10} {'Diff':>10}")
    print("-" * 80)
    for _, row in top10.iterrows():
        print(f"{row['target_name']:<20} {row['indicator_name']:<25} "
              f"{row['rising_sharpe']:>10.2f} {row['falling_sharpe']:>10.2f} "
              f"{row['sharpe_diff']:>+10.2f}")

    # ML Predictive Power summary
    print("\n" + "=" * 80)
    print("ML PREDICTIVE POWER (R^2 Scores)")
    print("=" * 80)

    has_positive_r2 = results_df[results_df['ml_rf_r2'] > 0]
    if len(has_positive_r2) > 0:
        print("\nPairs with positive R^2:")
        for _, row in has_positive_r2.iterrows():
            print(f"  {row['target_name']} vs {row['indicator_name']}: R^2 = {row['ml_rf_r2']:.3f}")
    else:
        print("\nNo pairs show positive ML predictive power (R^2 > 0)")
        print("This confirms: Indicators are useful as FILTERS, not SIGNALS")

    return results_df


if __name__ == '__main__':
    results = run_bulk_analysis()
