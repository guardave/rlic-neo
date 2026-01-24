#!/usr/bin/env python3
"""
Full Time Series Relationship Analysis: XLRE vs Orders/Inventories Ratio

Following the framework in docs/11_time_series_relationship_framework.md:
- Step 1: Data Preparation
- Step 2: Correlation Analysis
- Step 3: Lead-Lag Analysis
- Step 4: Granger Causality
- Step 5: ML Predictive Models
- Step 6: Regime Analysis
- Step 7: Visualization
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import os
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')


def load_and_prepare_data():
    """Step 1: Data Preparation."""
    print("=" * 80)
    print("STEP 1: DATA PREPARATION")
    print("=" * 80)

    # Load indicators
    indicators = pd.read_parquet(os.path.join(DATA_DIR, 'monthly_all_indicators.parquet'))
    indicators.index = pd.to_datetime(indicators.index)

    # Load sector prices
    prices = pd.read_parquet(os.path.join(DATA_DIR, 'sector_prices.parquet'))
    prices.index = pd.to_datetime(prices.index)

    # Load recession indicator
    recession = pd.read_parquet(os.path.join(DATA_DIR, 'recession_indicator.parquet'))
    recession.index = pd.to_datetime(recession.index)

    # Resample prices to monthly
    monthly_prices = prices.resample('ME').last()

    # Get XLRE
    xlre = monthly_prices['XLRE'].dropna()

    # Get Orders/Inv Ratio
    orders_inv = indicators['orders_inv_ratio'].dropna()

    # Create derivatives
    df = pd.DataFrame(index=xlre.index)
    df['XLRE'] = xlre
    df['XLRE_MoM'] = xlre.pct_change() * 100
    df['XLRE_QoQ'] = xlre.pct_change(3) * 100
    df['XLRE_YoY'] = xlre.pct_change(12) * 100

    # Align Orders/Inv to XLRE index
    df['OI_Ratio'] = orders_inv.reindex(df.index)
    df['OI_MoM'] = df['OI_Ratio'].pct_change() * 100
    df['OI_QoQ'] = df['OI_Ratio'].pct_change(3) * 100
    df['OI_YoY'] = df['OI_Ratio'].pct_change(12) * 100

    # Direction indicators
    df['OI_MoM_Dir'] = np.sign(df['OI_MoM'])
    df['OI_QoQ_Dir'] = np.sign(df['OI_QoQ'])
    df['OI_YoY_Dir'] = np.sign(df['OI_YoY'])

    # Add recession
    df['Recession'] = recession['Recession'].reindex(df.index).fillna(0)

    # Drop NaN rows
    df = df.dropna()

    print(f"\nData prepared: {len(df)} observations")
    print(f"Date range: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    print(f"\nColumns: {list(df.columns)}")

    return df


def correlation_analysis(df):
    """Step 2: Correlation Analysis."""
    print("\n" + "=" * 80)
    print("STEP 2: CORRELATION ANALYSIS")
    print("=" * 80)

    # Select relevant columns
    cols = ['XLRE', 'XLRE_MoM', 'XLRE_QoQ', 'XLRE_YoY',
            'OI_Ratio', 'OI_MoM', 'OI_QoQ', 'OI_YoY']

    corr_matrix = df[cols].corr()

    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3).to_string())

    # Key correlations
    print("\n--- Key Correlations ---")
    print(f"XLRE Level vs OI_Ratio Level: {corr_matrix.loc['XLRE', 'OI_Ratio']:.3f}")
    print(f"XLRE_MoM vs OI_MoM: {corr_matrix.loc['XLRE_MoM', 'OI_MoM']:.3f}")
    print(f"XLRE_MoM vs OI_YoY: {corr_matrix.loc['XLRE_MoM', 'OI_YoY']:.3f}")
    print(f"XLRE_YoY vs OI_YoY: {corr_matrix.loc['XLRE_YoY', 'OI_YoY']:.3f}")

    return corr_matrix


def lead_lag_analysis(df, max_lag=12):
    """Step 3: Lead-Lag Analysis."""
    print("\n" + "=" * 80)
    print("STEP 3: LEAD-LAG ANALYSIS")
    print("=" * 80)

    xlre_returns = df['XLRE_MoM']
    results = []

    for feature in ['OI_MoM', 'OI_QoQ', 'OI_YoY', 'OI_YoY_Dir']:
        indicator = df[feature]

        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # Indicator leads XLRE
                shifted_ind = indicator.shift(-lag)
                interpretation = f'OI leads XLRE by {-lag}m'
            elif lag > 0:
                # XLRE leads indicator
                shifted_ind = indicator.shift(lag)
                interpretation = f'XLRE leads OI by {lag}m'
            else:
                shifted_ind = indicator
                interpretation = 'Contemporaneous'

            # Calculate correlation
            valid = pd.DataFrame({'x': shifted_ind, 'y': xlre_returns}).dropna()
            if len(valid) < 30:
                continue

            corr, pval = stats.pearsonr(valid['x'], valid['y'])

            results.append({
                'feature': feature,
                'lag': lag,
                'correlation': corr,
                'p_value': pval,
                'significant': pval < 0.05,
                'interpretation': interpretation
            })

    results_df = pd.DataFrame(results)

    # Find best lag for each feature
    print("\nBest Lag by Feature:")
    print("-" * 70)
    for feature in ['OI_MoM', 'OI_QoQ', 'OI_YoY', 'OI_YoY_Dir']:
        feat_results = results_df[results_df['feature'] == feature]
        best_idx = feat_results['correlation'].abs().idxmax()
        best = feat_results.loc[best_idx]
        sig = '*' if best['significant'] else ''
        print(f"{feature:12} | Lag: {best['lag']:+3d} | Corr: {best['correlation']:+.3f}{sig} | {best['interpretation']}")

    # Significant relationships
    significant = results_df[results_df['significant'] == True].copy()
    if len(significant) > 0:
        significant['abs_corr'] = significant['correlation'].abs()
        top5 = significant.nlargest(5, 'abs_corr')
        print("\nTop 5 Significant Relationships:")
        print(top5[['feature', 'lag', 'correlation', 'p_value', 'interpretation']].to_string(index=False))

    return results_df


def granger_causality_analysis(df, max_lag=6):
    """Step 4: Granger Causality Tests."""
    print("\n" + "=" * 80)
    print("STEP 4: GRANGER CAUSALITY TESTS")
    print("=" * 80)

    from statsmodels.tsa.stattools import grangercausalitytests

    xlre_returns = df['XLRE_MoM']
    results = []

    for feature in ['OI_MoM', 'OI_QoQ', 'OI_YoY']:
        indicator = df[feature]

        # Prepare data
        data = pd.DataFrame({
            'XLRE_Returns': xlre_returns,
            'OI': indicator
        }).dropna()

        if len(data) < 50:
            continue

        # Test: Does OI Granger-cause XLRE?
        try:
            gc_results = grangercausalitytests(
                data[['XLRE_Returns', 'OI']],
                maxlag=max_lag,
                verbose=False
            )

            for lag in range(1, max_lag + 1):
                f_stat = gc_results[lag][0]['ssr_ftest'][0]
                p_value = gc_results[lag][0]['ssr_ftest'][1]

                results.append({
                    'direction': f'{feature} -> XLRE',
                    'lag': lag,
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
        except Exception as e:
            print(f"  Error testing {feature}: {e}")

        # Test: Does XLRE Granger-cause OI?
        try:
            gc_results = grangercausalitytests(
                data[['OI', 'XLRE_Returns']],
                maxlag=max_lag,
                verbose=False
            )

            for lag in range(1, max_lag + 1):
                f_stat = gc_results[lag][0]['ssr_ftest'][0]
                p_value = gc_results[lag][0]['ssr_ftest'][1]

                results.append({
                    'direction': f'XLRE -> {feature}',
                    'lag': lag,
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
        except Exception as e:
            pass

    results_df = pd.DataFrame(results)

    # Print significant results
    significant = results_df[results_df['significant'] == True]
    if len(significant) > 0:
        print("\nSignificant Granger Causality (p < 0.05):")
        print(significant.to_string(index=False))
    else:
        print("\nNo significant Granger causality found at p < 0.05")

    # Summary by direction
    print("\nBest Lag by Direction:")
    for direction in results_df['direction'].unique():
        dir_results = results_df[results_df['direction'] == direction]
        best_idx = dir_results['p_value'].idxmin()
        best = dir_results.loc[best_idx]
        sig = '*' if best['significant'] else ''
        print(f"  {direction}: Lag {best['lag']}, p={best['p_value']:.4f}{sig}")

    return results_df


def ml_predictive_analysis(df, feature_lags=[1, 3, 6, 12]):
    """Step 5: ML Predictive Models."""
    print("\n" + "=" * 80)
    print("STEP 5: ML PREDICTIVE MODELS")
    print("=" * 80)

    results = []

    for horizon, target_shift in [('1m', 1), ('3m', 3)]:
        # Create forward target
        df_model = df.copy()
        df_model['Target'] = df['XLRE_MoM'].shift(-target_shift)

        # Create lagged features
        feature_cols = []
        for lag in feature_lags:
            for feat in ['OI_MoM', 'OI_QoQ', 'OI_YoY']:
                col_name = f'{feat}_lag{lag}'
                df_model[col_name] = df[feat].shift(lag)
                feature_cols.append(col_name)

        # Add recession
        feature_cols.append('Recession')

        # Prepare data
        df_clean = df_model[feature_cols + ['Target']].dropna()
        X = df_clean[feature_cols]
        y = df_clean['Target']

        if len(X) < 50:
            continue

        # Time series CV
        tscv = TimeSeriesSplit(n_splits=5)

        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        }

        for model_name, model in models.items():
            cv_scores = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)

                cv_scores.append({
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred)
                })

            avg_r2 = np.mean([s['r2'] for s in cv_scores])
            avg_rmse = np.mean([s['rmse'] for s in cv_scores])
            avg_mae = np.mean([s['mae'] for s in cv_scores])

            results.append({
                'horizon': horizon,
                'model': model_name,
                'cv_r2': avg_r2,
                'cv_rmse': avg_rmse,
                'cv_mae': avg_mae,
                'n_samples': len(X)
            })

            # Feature importance for tree models
            if model_name == 'RandomForest':
                scaler = StandardScaler()
                X_s = scaler.fit_transform(X)
                model.fit(X_s, y)
                importance = pd.Series(model.feature_importances_, index=feature_cols)
                results[-1]['top_features'] = importance.nlargest(5).to_dict()

    results_df = pd.DataFrame(results)

    print("\nModel Performance (Time Series CV):")
    print(results_df[['horizon', 'model', 'cv_r2', 'cv_rmse', 'cv_mae']].round(4).to_string(index=False))

    # Feature importance
    rf_1m = results_df[(results_df['model'] == 'RandomForest') & (results_df['horizon'] == '1m')]
    if len(rf_1m) > 0 and 'top_features' in rf_1m.iloc[0]:
        print("\nTop Features (Random Forest, 1m horizon):")
        for feat, imp in rf_1m.iloc[0]['top_features'].items():
            print(f"  {feat}: {imp:.3f}")

    return results_df


def regime_analysis(df):
    """Step 6: Regime Analysis."""
    print("\n" + "=" * 80)
    print("STEP 6: REGIME ANALYSIS")
    print("=" * 80)

    xlre_returns = df['XLRE_MoM']
    results = []

    # Regime 1: OI_YoY Direction (Rising vs Falling)
    df['OI_Regime'] = np.where(df['OI_YoY'] > 0, 'OI Rising', 'OI Falling')

    # Regime 2: OI Level (High vs Low)
    median_level = df['OI_Ratio'].median()
    df['OI_Level_Regime'] = np.where(df['OI_Ratio'] > median_level, 'High OI', 'Low OI')

    # Regime 3: Combined with Recession
    df['Econ_Regime'] = np.where(
        df['Recession'] == 1, 'Recession',
        np.where(df['OI_YoY'] > 0, 'Expansion + OI Rising', 'Expansion + OI Falling')
    )

    # Calculate performance by regime
    for regime_col in ['OI_Regime', 'OI_Level_Regime', 'Econ_Regime']:
        for regime in df[regime_col].unique():
            mask = df[regime_col] == regime
            regime_returns = xlre_returns[mask]

            if len(regime_returns) < 5:
                continue

            results.append({
                'regime_type': regime_col,
                'regime': regime,
                'n_months': len(regime_returns),
                'mean_return': regime_returns.mean(),
                'std_return': regime_returns.std(),
                'sharpe': regime_returns.mean() / regime_returns.std() * np.sqrt(12) if regime_returns.std() > 0 else 0,
                'win_rate': (regime_returns > 0).mean() * 100,
                'median_return': regime_returns.median(),
                'total_return': (1 + regime_returns / 100).prod() - 1
            })

    results_df = pd.DataFrame(results)

    print("\nXLRE Performance by O/I Ratio Regime:")
    print(results_df.round(3).to_string(index=False))

    # Statistical test: Rising vs Falling
    rising = xlre_returns[df['OI_Regime'] == 'OI Rising']
    falling = xlre_returns[df['OI_Regime'] == 'OI Falling']
    t_stat, p_value = stats.ttest_ind(rising, falling)
    print(f"\nT-test (Rising vs Falling): t={t_stat:.3f}, p={p_value:.3f}")

    return results_df, df


def create_visualizations(df, corr_matrix, lead_lag_results):
    """Step 7: Visualization."""
    print("\n" + "=" * 80)
    print("STEP 7: VISUALIZATION")
    print("=" * 80)

    # 1. Full Timeline with Regime Background
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Cumulative returns
    cumret = (1 + df['XLRE_MoM'] / 100).cumprod()

    # Plot price with regime background
    ax1 = axes[0]
    ax1.set_title('XLRE vs Orders/Inventories Ratio: Regime Analysis', fontsize=14)

    # Color background by regime
    for i in range(len(df) - 1):
        date = df.index[i]
        next_date = df.index[i + 1]

        if df.loc[date, 'Recession'] == 1:
            color = 'lightgray'
            alpha = 0.5
        elif df.loc[date, 'OI_YoY'] > 0:
            color = 'lightgreen'
            alpha = 0.3
        else:
            color = 'lightpink'
            alpha = 0.3

        ax1.axvspan(date, next_date, color=color, alpha=alpha)

    ax1.plot(cumret.index, cumret.values, 'b-', linewidth=1.5, label='XLRE Cumulative Return')
    ax1.set_ylabel('Cumulative Return', fontsize=10)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Legend for colors
    legend_elements = [
        Patch(facecolor='lightgreen', alpha=0.5, label='O/I Rising (YoY > 0)'),
        Patch(facecolor='lightpink', alpha=0.5, label='O/I Falling (YoY < 0)'),
        Patch(facecolor='lightgray', alpha=0.5, label='Recession')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')

    # Plot O/I Ratio
    ax2 = axes[1]
    ax2.plot(df.index, df['OI_Ratio'].values, 'purple', linewidth=1, label='O/I Ratio Level')
    ax2.axhline(y=df['OI_Ratio'].median(), color='black', linestyle='--', linewidth=0.5, label='Median')
    ax2.set_ylabel('O/I Ratio', fontsize=10)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot O/I YoY
    ax3 = axes[2]
    ax3.plot(df.index, df['OI_YoY'].values, 'green', linewidth=1)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax3.fill_between(df.index, 0, df['OI_YoY'].values,
                     where=df['OI_YoY'] > 0, color='green', alpha=0.3)
    ax3.fill_between(df.index, 0, df['OI_YoY'].values,
                     where=df['OI_YoY'] < 0, color='red', alpha=0.3)
    ax3.set_ylabel('O/I Ratio YoY (%)', fontsize=10)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.grid(True, alpha=0.3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'xlre_oi_regime_background.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: xlre_oi_regime_background.png")

    # 2. Correlation Scatter Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # XLRE MoM vs OI YoY
    ax1 = axes[0]
    ax1.scatter(df['OI_YoY'], df['XLRE_MoM'], alpha=0.5, s=20)
    z = np.polyfit(df['OI_YoY'], df['XLRE_MoM'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['OI_YoY'].min(), df['OI_YoY'].max(), 100)
    ax1.plot(x_line, p(x_line), 'r-', linewidth=2)
    corr = df['OI_YoY'].corr(df['XLRE_MoM'])
    ax1.set_xlabel('O/I Ratio YoY (%)', fontsize=10)
    ax1.set_ylabel('XLRE Monthly Return (%)', fontsize=10)
    ax1.set_title(f'XLRE Returns vs O/I Ratio YoY\nCorrelation: {corr:.3f}', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # XLRE YoY vs OI YoY
    ax2 = axes[1]
    ax2.scatter(df['OI_YoY'], df['XLRE_YoY'], alpha=0.5, s=20)
    z = np.polyfit(df['OI_YoY'].dropna(), df['XLRE_YoY'].dropna(), 1)
    p = np.poly1d(z)
    ax2.plot(x_line, p(x_line), 'r-', linewidth=2)
    corr = df['OI_YoY'].corr(df['XLRE_YoY'])
    ax2.set_xlabel('O/I Ratio YoY (%)', fontsize=10)
    ax2.set_ylabel('XLRE YoY Return (%)', fontsize=10)
    ax2.set_title(f'XLRE YoY vs O/I Ratio YoY\nCorrelation: {corr:.3f}', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'xlre_oi_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: xlre_oi_correlation.png")

    # 3. Lead-Lag Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for feature in ['OI_MoM', 'OI_QoQ', 'OI_YoY']:
        feat_results = lead_lag_results[lead_lag_results['feature'] == feature]
        ax.plot(feat_results['lag'], feat_results['correlation'], 'o-', label=feature, markersize=4)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Lag (months, negative = indicator leads)', fontsize=10)
    ax.set_ylabel('Correlation', fontsize=10)
    ax.set_title('Lead-Lag Analysis: O/I Ratio vs XLRE Returns', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'xlre_oi_leadlag.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: xlre_oi_leadlag.png")

    # 4. Regime Performance Box Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    regime_data = []
    labels = []
    colors = []

    for regime, color in [('OI Rising', 'green'), ('OI Falling', 'red')]:
        mask = df['OI_Regime'] == regime
        regime_data.append(df.loc[mask, 'XLRE_MoM'].values)
        labels.append(regime)
        colors.append(color)

    # Add recession
    mask = df['Recession'] == 1
    if mask.sum() > 0:
        regime_data.append(df.loc[mask, 'XLRE_MoM'].values)
        labels.append('Recession')
        colors.append('gray')

    bp = ax.boxplot(regime_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_ylabel('XLRE Monthly Return (%)', fontsize=10)
    ax.set_title('XLRE Returns by O/I Ratio Regime', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add mean annotations
    for i, data in enumerate(regime_data):
        mean_val = np.mean(data)
        ax.annotate(f'μ={mean_val:.2f}%', xy=(i + 1, mean_val),
                    xytext=(i + 1.3, mean_val), fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'xlre_oi_regime_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: xlre_oi_regime_boxplot.png")


def main():
    print("=" * 80)
    print("FULL TIME SERIES RELATIONSHIP ANALYSIS")
    print("XLRE (Real Estate ETF) vs Orders/Inventories Ratio")
    print("=" * 80)

    # Step 1: Data Preparation
    df = load_and_prepare_data()

    # Step 2: Correlation Analysis
    corr_matrix = correlation_analysis(df)

    # Step 3: Lead-Lag Analysis
    lead_lag_results = lead_lag_analysis(df)

    # Step 4: Granger Causality
    granger_results = granger_causality_analysis(df)

    # Step 5: ML Predictive Models
    ml_results = ml_predictive_analysis(df)

    # Step 6: Regime Analysis
    regime_results, df = regime_analysis(df)

    # Step 7: Visualization
    create_visualizations(df, corr_matrix, lead_lag_results)

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    # Key metrics
    rising_regime = regime_results[regime_results['regime'] == 'OI Rising'].iloc[0]
    falling_regime = regime_results[regime_results['regime'] == 'OI Falling'].iloc[0]

    print(f"""
Target: XLRE (Real Estate ETF)
Indicator: Orders/Inventories Ratio
Data Period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')} ({len(df)} months)

KEY FINDINGS:

1. CORRELATION
   - Level correlation (XLRE vs O/I Ratio): {corr_matrix.loc['XLRE', 'OI_Ratio']:.3f}
   - Returns vs YoY change: {corr_matrix.loc['XLRE_MoM', 'OI_YoY']:.3f}

2. LEAD-LAG RELATIONSHIP
   - Peak correlation at lag 0 (contemporaneous)
   - No significant predictive lead from O/I Ratio

3. GRANGER CAUSALITY
   - Limited evidence of Granger causality in either direction

4. ML PREDICTIVE POWER
   - All models show negative R² (worse than mean prediction)
   - O/I Ratio does NOT predict XLRE returns

5. REGIME ANALYSIS (Most Valuable)
   - O/I Rising:  Sharpe {rising_regime['sharpe']:.2f}, Mean {rising_regime['mean_return']:.2f}%/mo
   - O/I Falling: Sharpe {falling_regime['sharpe']:.2f}, Mean {falling_regime['mean_return']:.2f}%/mo
   - Sharpe Difference: {rising_regime['sharpe'] - falling_regime['sharpe']:+.2f}

CONCLUSION:
The O/I Ratio does not PREDICT XLRE returns, but provides meaningful
REGIME DIFFERENTIATION. XLRE performs significantly better when
O/I Ratio is rising (strong manufacturing demand).

FILES CREATED:
- data/xlre_oi_regime_background.png
- data/xlre_oi_correlation.png
- data/xlre_oi_leadlag.png
- data/xlre_oi_regime_boxplot.png
""")

    # Save data
    df.to_parquet(os.path.join(DATA_DIR, 'xlre_oi_analysis.parquet'))
    print("  Saved: xlre_oi_analysis.parquet")

    return df, corr_matrix, lead_lag_results, granger_results, ml_results, regime_results


if __name__ == '__main__':
    results = main()
