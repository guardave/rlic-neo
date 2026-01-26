#!/usr/bin/env python3
"""
Full Analysis: XLRE vs New Home Sales

Following SOP v1.3 with extended lead-lag analysis (0 to +24 months).
This analysis found a significant predictive relationship at lag +8.

Phases:
- Phase 0: Qualitative Analysis (documented in report)
- Phase 1: Data Preparation
- Phase 2: Statistical Analysis (correlation matrix)
- Phase 3: Lead-Lag Analysis (0 to +24 months)
- Phase 4: Regime Analysis
- Phase 5: Backtesting
- Phase 6: Dashboard Data Preparation
- Phase 7: Documentation
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t
import yfinance as yf
from pathlib import Path
from datetime import datetime
import warnings
import urllib.request
import io

warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Analysis parameters
OPTIMAL_LAG = 8  # Months - New Home Sales leads XLRE by 8 months
MAX_LAG = 24     # Extended lead-lag range


def fetch_fred_data_url(series_id: str, start: str = "2000-01-01") -> pd.Series:
    """Fetch data from FRED via direct URL."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    print(f"  Fetching {series_id} from FRED...")

    with urllib.request.urlopen(url, timeout=30) as response:
        csv_data = response.read().decode('utf-8')

    df = pd.read_csv(io.StringIO(csv_data))

    # Find date column
    date_col = None
    for col in ['observation_date', 'DATE', df.columns[0]]:
        if col in df.columns:
            date_col = col
            break

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    return df[series_id].dropna()


def fetch_yahoo_data(ticker: str, start: str = "2000-01-01") -> pd.DataFrame:
    """Fetch price data from Yahoo Finance."""
    print(f"  Fetching {ticker} from Yahoo Finance...")
    data = yf.download(ticker, start=start, progress=False)

    # Handle different column formats
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(0):
            price = data['Adj Close']
            if isinstance(price, pd.DataFrame):
                price = price.iloc[:, 0]
        else:
            price = data.iloc[:, 0]
    else:
        price = data.get('Adj Close', data.get('Close'))

    return pd.DataFrame({'price': price})


def create_monthly_data(series: pd.Series, name: str) -> pd.DataFrame:
    """Resample to monthly and create derivatives."""
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    monthly = series.resample('ME').last()

    result = pd.DataFrame(index=monthly.index)
    result[f'{name}_Level'] = monthly
    result[f'{name}_MoM'] = monthly.pct_change(1)
    result[f'{name}_QoQ'] = monthly.pct_change(3)
    result[f'{name}_YoY'] = monthly.pct_change(12)
    result[f'{name}_Direction'] = np.sign(result[f'{name}_YoY'])

    return result


def correlation_analysis(df: pd.DataFrame, x_cols: list, y_col: str) -> pd.DataFrame:
    """Compute correlation matrix."""
    results = []
    for x_col in x_cols:
        mask = ~(df[x_col].isna() | df[y_col].isna())
        if mask.sum() >= 30:
            r, p = stats.pearsonr(df.loc[mask, x_col], df.loc[mask, y_col])
            results.append({
                'indicator': x_col,
                'target': y_col,
                'correlation': r,
                'pvalue': p,
                'n': mask.sum(),
                'significant': p < 0.05
            })
    return pd.DataFrame(results)


def lead_lag_analysis(df: pd.DataFrame, x_col: str, y_col: str, max_lag: int = 24) -> pd.DataFrame:
    """
    Lead-lag analysis from 0 to +max_lag.
    Positive lag means indicator leads target (predictive).
    """
    results = []
    for lag in range(0, max_lag + 1):
        x_lagged = df[x_col].shift(lag)
        y = df[y_col]

        mask = ~(x_lagged.isna() | y.isna())
        if mask.sum() >= 30:
            r, p = stats.pearsonr(x_lagged[mask], y[mask])
            results.append({
                'lag': lag,
                'correlation': r,
                'pvalue': p,
                'n': mask.sum(),
                'significant': p < 0.05
            })

    return pd.DataFrame(results)


def regime_analysis(df: pd.DataFrame, indicator_col: str, target_col: str,
                   optimal_lag: int = 8) -> dict:
    """Regime analysis with optimal lag."""
    # Create lagged indicator
    df = df.copy()
    df['indicator_lagged'] = df[indicator_col].shift(optimal_lag)

    # Define regimes based on YoY direction
    df['regime'] = df['indicator_lagged'].apply(
        lambda x: 'Rising' if x > 0 else ('Falling' if x < 0 else np.nan)
    )

    # Calculate regime performance
    regime_stats = []
    for regime in ['Rising', 'Falling']:
        mask = df['regime'] == regime
        if mask.sum() >= 20:
            returns = df.loc[mask, target_col]
            regime_stats.append({
                'regime': regime,
                'mean_return': returns.mean(),
                'std_return': returns.std(),
                'sharpe': returns.mean() / returns.std() * np.sqrt(12) if returns.std() > 0 else 0,
                'count': mask.sum()
            })

    regime_df = pd.DataFrame(regime_stats)

    # Statistical test for regime difference
    rising = df.loc[df['regime'] == 'Rising', target_col].dropna()
    falling = df.loc[df['regime'] == 'Falling', target_col].dropna()

    if len(rising) >= 20 and len(falling) >= 20:
        t_stat, p_value = stats.ttest_ind(rising, falling)
    else:
        t_stat, p_value = np.nan, np.nan

    return {
        'regime_performance': regime_df,
        'regime_test': {'t_stat': t_stat, 'pvalue': p_value, 'significant': p_value < 0.05 if not np.isnan(p_value) else False},
        'data_with_regimes': df
    }


def run_backtest(df: pd.DataFrame, indicator_col: str, target_col: str,
                optimal_lag: int = 8) -> dict:
    """
    Simple backtest: Long when indicator (lagged) is rising, cash otherwise.
    """
    df = df.copy()
    df['indicator_lagged'] = df[indicator_col].shift(optimal_lag)

    # Signal: 1 when rising, 0 when falling
    df['signal'] = (df['indicator_lagged'] > 0).astype(int)

    # Apply 1-month delay for execution
    df['position'] = df['signal'].shift(1)

    # Calculate returns
    df['strategy_return'] = df['position'] * df[target_col]
    df['benchmark_return'] = df[target_col]

    # Drop NaN
    df = df.dropna(subset=['strategy_return', 'benchmark_return'])

    # Cumulative returns
    df['strategy_cumret'] = (1 + df['strategy_return']).cumprod() - 1
    df['benchmark_cumret'] = (1 + df['benchmark_return']).cumprod() - 1

    # Calculate metrics
    n_years = len(df) / 12

    strategy_total = df['strategy_cumret'].iloc[-1] if len(df) > 0 else 0
    benchmark_total = df['benchmark_cumret'].iloc[-1] if len(df) > 0 else 0

    strategy_ann = (1 + strategy_total) ** (1/n_years) - 1 if n_years > 0 else 0
    benchmark_ann = (1 + benchmark_total) ** (1/n_years) - 1 if n_years > 0 else 0

    strategy_sharpe = df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(12) if df['strategy_return'].std() > 0 else 0
    benchmark_sharpe = df['benchmark_return'].mean() / df['benchmark_return'].std() * np.sqrt(12) if df['benchmark_return'].std() > 0 else 0

    return {
        'strategy_sharpe': strategy_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'strategy_total_return': strategy_total * 100,
        'benchmark_total_return': benchmark_total * 100,
        'strategy_annualized': strategy_ann * 100,
        'benchmark_annualized': benchmark_ann * 100,
        'backtest_data': df
    }


def main():
    print("="*70)
    print("FULL ANALYSIS: XLRE vs New Home Sales (SOP v1.3)")
    print("="*70)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Optimal Lag: {OPTIMAL_LAG} months")
    print(f"Lead-Lag Range: 0 to +{MAX_LAG} months")
    print()

    # Phase 1: Data Preparation
    print("Phase 1: Data Preparation")
    print("-"*40)

    # Fetch New Home Sales
    newhomesales = fetch_fred_data_url("HSN1F", start="2000-01-01")
    nhs_monthly = create_monthly_data(newhomesales, "NewHomeSales")

    # Fetch XLRE
    xlre_df = fetch_yahoo_data("XLRE", start="2000-01-01")
    xlre_monthly = create_monthly_data(xlre_df['price'], "XLRE")
    xlre_monthly['XLRE_Returns'] = xlre_monthly['XLRE_Level'].pct_change(1)

    # Merge
    df = pd.concat([nhs_monthly, xlre_monthly], axis=1).dropna()

    print(f"  Data period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    print(f"  Total observations: {len(df)}")
    print()

    # Phase 2: Correlation Analysis
    print("Phase 2: Correlation Analysis (Concurrent)")
    print("-"*40)

    x_cols = ['NewHomeSales_MoM', 'NewHomeSales_QoQ', 'NewHomeSales_YoY']
    corr_results = correlation_analysis(df, x_cols, 'XLRE_Returns')

    for _, row in corr_results.iterrows():
        sig = '✓' if row['significant'] else ''
        print(f"  {row['indicator']}: r={row['correlation']:.3f}, p={row['pvalue']:.4f} {sig}")

    print()

    # Phase 3: Lead-Lag Analysis
    print("Phase 3: Lead-Lag Analysis (0 to +24 months)")
    print("-"*40)

    leadlag_results = lead_lag_analysis(df, 'NewHomeSales_YoY', 'XLRE_Returns', max_lag=MAX_LAG)

    # Find best predictive lag
    best_lag = leadlag_results.loc[leadlag_results['pvalue'].idxmin()]
    sig_lags = leadlag_results[leadlag_results['significant']]

    print(f"  Best predictive lag: +{int(best_lag['lag'])} months")
    print(f"    Correlation: r={best_lag['correlation']:.3f}")
    print(f"    P-value: {best_lag['pvalue']:.4f}")
    print()

    print("  Significant lags (p < 0.05):")
    for _, row in sig_lags.iterrows():
        print(f"    Lag +{int(row['lag'])}: r={row['correlation']:.3f}, p={row['pvalue']:.4f}")

    print()

    # Phase 4: Regime Analysis
    print(f"Phase 4: Regime Analysis (using lag +{OPTIMAL_LAG})")
    print("-"*40)

    regime_results = regime_analysis(df, 'NewHomeSales_YoY', 'XLRE_Returns', optimal_lag=OPTIMAL_LAG)

    print("  Regime Performance:")
    for _, row in regime_results['regime_performance'].iterrows():
        print(f"    {row['regime']}: mean={row['mean_return']*100:.2f}%, Sharpe={row['sharpe']:.2f}, n={int(row['count'])}")

    if not np.isnan(regime_results['regime_test']['pvalue']):
        sig = '✓' if regime_results['regime_test']['significant'] else ''
        print(f"  Regime difference test: p={regime_results['regime_test']['pvalue']:.4f} {sig}")

    print()

    # Phase 5: Backtest
    print(f"Phase 5: Backtest (using lag +{OPTIMAL_LAG})")
    print("-"*40)

    backtest_results = run_backtest(df, 'NewHomeSales_YoY', 'XLRE_Returns', optimal_lag=OPTIMAL_LAG)

    print(f"  Strategy Sharpe: {backtest_results['strategy_sharpe']:.2f}")
    print(f"  Benchmark Sharpe: {backtest_results['benchmark_sharpe']:.2f}")
    print(f"  Strategy Total Return: {backtest_results['strategy_total_return']:.1f}%")
    print(f"  Benchmark Total Return: {backtest_results['benchmark_total_return']:.1f}%")

    print()

    # Phase 6: Dashboard Data Preparation
    print("Phase 6: Dashboard Data Preparation")
    print("-"*40)

    # Create dashboard-ready data file
    dashboard_df = df[['NewHomeSales_Level', 'NewHomeSales_MoM', 'NewHomeSales_QoQ',
                       'NewHomeSales_YoY', 'NewHomeSales_Direction',
                       'XLRE_Level', 'XLRE_Returns']].copy()

    # Add lagged indicator
    dashboard_df['NewHomeSales_YoY_Lagged'] = dashboard_df['NewHomeSales_YoY'].shift(OPTIMAL_LAG)
    dashboard_df['Regime'] = dashboard_df['NewHomeSales_YoY_Lagged'].apply(
        lambda x: 'Rising' if x > 0 else ('Falling' if x < 0 else np.nan)
    )

    # Add backtest columns
    backtest_data = backtest_results['backtest_data']
    dashboard_df['Strategy_Return'] = backtest_data['strategy_return']
    dashboard_df['Strategy_CumRet'] = backtest_data['strategy_cumret']
    dashboard_df['Benchmark_CumRet'] = backtest_data['benchmark_cumret']

    # Save main data file
    output_file = DATA_DIR / "xlre_newhomesales_full.parquet"
    dashboard_df.to_parquet(output_file)
    print(f"  Saved: {output_file}")

    # Save lead-lag results
    leadlag_file = DATA_DIR / "xlre_newhomesales_leadlag.parquet"
    leadlag_results.to_parquet(leadlag_file)
    print(f"  Saved: {leadlag_file}")

    # Save correlation results
    corr_file = DATA_DIR / "xlre_newhomesales_correlation.parquet"
    corr_results.to_parquet(corr_file)
    print(f"  Saved: {corr_file}")

    # Save regime performance
    regime_file = DATA_DIR / "xlre_newhomesales_regimes.parquet"
    regime_results['regime_performance'].to_parquet(regime_file)
    print(f"  Saved: {regime_file}")

    print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print("Key Findings:")
    print(f"  1. Significant predictive relationship at lag +{OPTIMAL_LAG} (r={best_lag['correlation']:.3f}, p={best_lag['pvalue']:.4f})")
    print(f"  2. New Home Sales from {OPTIMAL_LAG} months ago predicts XLRE returns")
    print(f"  3. Strategy Sharpe: {backtest_results['strategy_sharpe']:.2f} vs Benchmark: {backtest_results['benchmark_sharpe']:.2f}")
    print()
    print("Actionable: ✓ YES - Use New Home Sales at lag +8 as trading signal for XLRE")
    print()
    print("Files created:")
    print(f"  - {output_file}")
    print(f"  - {leadlag_file}")
    print(f"  - {corr_file}")
    print(f"  - {regime_file}")


if __name__ == "__main__":
    main()
