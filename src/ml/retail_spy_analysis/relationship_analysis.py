#!/usr/bin/env python3
"""
ML Analysis: SPY vs Retail Inventories to Sales Ratio

Explores relationships between SPY price/returns and RETAILIRSA derivatives
using various ML and statistical techniques.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def load_data(data_path):
    """Load the SPY and RETAILIRSA dataset."""
    df = pd.read_parquet(data_path)
    return df


def create_spy_features(df):
    """Create SPY-derived features."""
    spy = df['SPY']

    features = pd.DataFrame(index=df.index)

    # SPY Returns
    features['SPY_MoM'] = spy.pct_change(1) * 100
    features['SPY_QoQ'] = spy.pct_change(3) * 100
    features['SPY_YoY'] = spy.pct_change(12) * 100

    # SPY Momentum
    features['SPY_3m_vs_12m'] = (spy.rolling(3).mean() / spy.rolling(12).mean() - 1) * 100

    # SPY Volatility (rolling std of returns)
    features['SPY_Vol_3m'] = features['SPY_MoM'].rolling(3).std()
    features['SPY_Vol_12m'] = features['SPY_MoM'].rolling(12).std()

    return features


def correlation_analysis(df):
    """
    Compute correlation matrix between all series.
    """
    # Select relevant columns
    cols = ['SPY', 'Retail_Inv_Sales_Ratio',
            'RETAILIRSA_MoM', 'RETAILIRSA_QoQ', 'RETAILIRSA_YoY']

    # Add SPY returns
    df_analysis = df[cols].copy()
    df_analysis['SPY_MoM'] = df['SPY'].pct_change(1) * 100
    df_analysis['SPY_QoQ'] = df['SPY'].pct_change(3) * 100
    df_analysis['SPY_YoY'] = df['SPY'].pct_change(12) * 100

    corr_matrix = df_analysis.corr()

    return corr_matrix


def lead_lag_analysis(df, max_lag=12):
    """
    Analyze lead-lag relationships between RETAILIRSA and SPY.

    Tests if RETAILIRSA changes lead/lag SPY returns.
    """
    results = []

    spy_returns = df['SPY'].pct_change(1) * 100

    for feature in ['RETAILIRSA_MoM', 'RETAILIRSA_QoQ', 'RETAILIRSA_YoY',
                    'RETAILIRSA_MoM_Dir', 'RETAILIRSA_QoQ_Dir', 'RETAILIRSA_YoY_Dir']:

        if feature not in df.columns:
            continue

        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # RETAILIRSA leads SPY (RETAILIRSA at t predicts SPY at t+|lag|)
                x = df[feature].shift(-lag)
                y = spy_returns
            else:
                # SPY leads RETAILIRSA (SPY at t predicts RETAILIRSA at t+lag)
                x = df[feature]
                y = spy_returns.shift(-lag)

            # Align and drop NaN
            valid = pd.DataFrame({'x': x, 'y': y}).dropna()

            if len(valid) < 50:
                continue

            corr, pval = stats.pearsonr(valid['x'], valid['y'])

            results.append({
                'feature': feature,
                'lag': lag,
                'correlation': corr,
                'p_value': pval,
                'significant': pval < 0.05,
                'interpretation': 'RETAILIRSA leads SPY' if lag < 0 else 'SPY leads RETAILIRSA' if lag > 0 else 'Contemporaneous'
            })

    return pd.DataFrame(results)


def granger_causality_test(df, max_lag=6):
    """
    Test Granger causality between RETAILIRSA and SPY returns.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    spy_returns = df['SPY'].pct_change(1).dropna()

    results = []

    for feature in ['RETAILIRSA_MoM', 'RETAILIRSA_QoQ', 'RETAILIRSA_YoY']:
        if feature not in df.columns:
            continue

        # Prepare data (must be stationary)
        data = pd.DataFrame({
            'SPY_Returns': spy_returns,
            feature: df[feature]
        }).dropna()

        if len(data) < 50:
            continue

        # Test: Does RETAILIRSA Granger-cause SPY?
        try:
            gc_results = grangercausalitytests(
                data[['SPY_Returns', feature]],
                maxlag=max_lag,
                verbose=False
            )

            for lag in range(1, max_lag + 1):
                f_stat = gc_results[lag][0]['ssr_ftest'][0]
                p_value = gc_results[lag][0]['ssr_ftest'][1]

                results.append({
                    'direction': f'{feature} -> SPY_Returns',
                    'lag': lag,
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
        except Exception as e:
            pass

        # Test: Does SPY Granger-cause RETAILIRSA?
        try:
            gc_results = grangercausalitytests(
                data[[feature, 'SPY_Returns']],
                maxlag=max_lag,
                verbose=False
            )

            for lag in range(1, max_lag + 1):
                f_stat = gc_results[lag][0]['ssr_ftest'][0]
                p_value = gc_results[lag][0]['ssr_ftest'][1]

                results.append({
                    'direction': f'SPY_Returns -> {feature}',
                    'lag': lag,
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
        except Exception as e:
            pass

    return pd.DataFrame(results)


def predictive_model_analysis(df, target='SPY_Forward', feature_lags=[1, 3, 6, 12]):
    """
    Build predictive models to forecast SPY returns using RETAILIRSA features.
    """
    # Create target: Forward SPY returns
    df_model = df.copy()
    df_model['SPY_Forward_1m'] = df['SPY'].pct_change(1).shift(-1) * 100
    df_model['SPY_Forward_3m'] = df['SPY'].pct_change(3).shift(-3) * 100

    results = []

    for horizon, target_col in [('1m', 'SPY_Forward_1m'), ('3m', 'SPY_Forward_3m')]:

        # Features: Lagged RETAILIRSA derivatives
        feature_cols = []
        for lag in feature_lags:
            for feat in ['RETAILIRSA_MoM', 'RETAILIRSA_QoQ', 'RETAILIRSA_YoY']:
                if feat in df_model.columns:
                    col_name = f'{feat}_lag{lag}'
                    df_model[col_name] = df_model[feat].shift(lag)
                    feature_cols.append(col_name)

        # Add recession indicator if available
        if 'Recession' in df_model.columns:
            feature_cols.append('Recession')

        # Prepare data
        X = df_model[feature_cols].dropna()
        y = df_model.loc[X.index, target_col].dropna()
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if len(X) < 100:
            continue

        # Time series cross-validation
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

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

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

            # Feature importance for tree-based models
            if model_name in ['RandomForest', 'GradientBoosting']:
                # Refit on full data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model.fit(X_scaled, y)

                importance = pd.Series(model.feature_importances_, index=feature_cols)
                results[-1]['top_features'] = importance.nlargest(5).to_dict()

    return pd.DataFrame(results)


def regime_analysis(df):
    """
    Analyze SPY behavior in different RETAILIRSA regimes.
    """
    df_analysis = df.copy()
    df_analysis['SPY_MoM'] = df['SPY'].pct_change(1) * 100

    results = []

    # Define regimes based on RETAILIRSA levels and changes
    # Regime 1: RETAILIRSA level (high vs low)
    median_level = df_analysis['Retail_Inv_Sales_Ratio'].median()
    df_analysis['Level_Regime'] = np.where(
        df_analysis['Retail_Inv_Sales_Ratio'] > median_level,
        'High Inv/Sales', 'Low Inv/Sales'
    )

    # Regime 2: RETAILIRSA direction (rising vs falling)
    df_analysis['Direction_Regime'] = np.where(
        df_analysis['RETAILIRSA_YoY'] > 0,
        'Rising Inv/Sales', 'Falling Inv/Sales'
    )

    # Regime 3: Combined with Recession
    if 'Recession' in df_analysis.columns:
        df_analysis['Econ_Regime'] = np.where(
            df_analysis['Recession'] == 1,
            'Recession',
            np.where(df_analysis['RETAILIRSA_YoY'] > 0, 'Expansion-Rising', 'Expansion-Falling')
        )

    # Calculate SPY returns by regime
    for regime_col in ['Level_Regime', 'Direction_Regime', 'Econ_Regime']:
        if regime_col not in df_analysis.columns:
            continue

        for regime in df_analysis[regime_col].dropna().unique():
            mask = df_analysis[regime_col] == regime
            spy_returns = df_analysis.loc[mask, 'SPY_MoM'].dropna()

            if len(spy_returns) < 10:
                continue

            results.append({
                'regime_type': regime_col,
                'regime': regime,
                'n_months': len(spy_returns),
                'mean_return': spy_returns.mean(),
                'std_return': spy_returns.std(),
                'sharpe': spy_returns.mean() / spy_returns.std() * np.sqrt(12) if spy_returns.std() > 0 else 0,
                'positive_pct': (spy_returns > 0).mean() * 100,
                'median_return': spy_returns.median()
            })

    return pd.DataFrame(results)


def run_full_analysis(data_path):
    """
    Run complete ML relationship analysis.
    """
    print("=" * 80)
    print("ML RELATIONSHIP ANALYSIS: SPY vs RETAILIRSA")
    print("=" * 80)

    # Load data
    df = load_data(data_path)
    print(f"\nData loaded: {len(df)} observations, {df.index[0]} to {df.index[-1]}")

    # 1. Correlation Analysis
    print("\n" + "-" * 80)
    print("1. CORRELATION ANALYSIS")
    print("-" * 80)

    corr = correlation_analysis(df)
    print("\nCorrelation Matrix:")
    print(corr.round(3).to_string())

    # 2. Lead-Lag Analysis
    print("\n" + "-" * 80)
    print("2. LEAD-LAG ANALYSIS")
    print("-" * 80)

    lead_lag = lead_lag_analysis(df, max_lag=12)

    # Find strongest relationships
    significant = lead_lag[lead_lag['significant'] == True].copy()
    if len(significant) > 0:
        significant['abs_corr'] = significant['correlation'].abs()
        top_relationships = significant.nlargest(10, 'abs_corr')
        print("\nTop 10 Significant Lead-Lag Relationships:")
        print(top_relationships[['feature', 'lag', 'correlation', 'p_value', 'interpretation']].to_string(index=False))
    else:
        print("\nNo significant lead-lag relationships found at p<0.05")

    # 3. Granger Causality
    print("\n" + "-" * 80)
    print("3. GRANGER CAUSALITY TESTS")
    print("-" * 80)

    try:
        granger = granger_causality_test(df, max_lag=6)
        significant_gc = granger[granger['significant'] == True]
        if len(significant_gc) > 0:
            print("\nSignificant Granger Causality (p < 0.05):")
            print(significant_gc.to_string(index=False))
        else:
            print("\nNo significant Granger causality found at p<0.05")
    except Exception as e:
        print(f"\nGranger causality test failed: {e}")

    # 4. Predictive Models
    print("\n" + "-" * 80)
    print("4. PREDICTIVE MODEL ANALYSIS")
    print("-" * 80)

    pred_results = predictive_model_analysis(df)
    if len(pred_results) > 0:
        print("\nModel Performance (Time Series CV):")
        print(pred_results[['horizon', 'model', 'cv_r2', 'cv_rmse', 'cv_mae']].round(4).to_string(index=False))

        # Best model feature importance
        best_rf = pred_results[(pred_results['model'] == 'RandomForest')].iloc[0] if len(pred_results[pred_results['model'] == 'RandomForest']) > 0 else None
        if best_rf is not None and 'top_features' in best_rf and best_rf['top_features']:
            print(f"\nTop Features (Random Forest, {best_rf['horizon']}):")
            for feat, imp in best_rf['top_features'].items():
                print(f"  {feat}: {imp:.4f}")

    # 5. Regime Analysis
    print("\n" + "-" * 80)
    print("5. REGIME ANALYSIS")
    print("-" * 80)

    regime_results = regime_analysis(df)
    if len(regime_results) > 0:
        print("\nSPY Returns by RETAILIRSA Regime:")
        print(regime_results.round(3).to_string(index=False))

    # 6. Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 80)

    return {
        'correlation': corr,
        'lead_lag': lead_lag,
        'granger': granger if 'granger' in dir() else None,
        'predictive': pred_results,
        'regime': regime_results
    }


if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    data_path = os.path.join(project_dir, 'data', 'spy_retail_recession.parquet')

    results = run_full_analysis(data_path)
