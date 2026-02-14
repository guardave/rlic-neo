#!/usr/bin/env python3
"""
Full Analysis: SPY vs HY-IG Credit Spread
Following SOP v1.3 with bilateral lead-lag analysis (-18 to +18 months).

Data sources:
  - FRED BAMLH0A0HYM2: ICE BofA US High Yield OAS (daily, bps)
  - FRED BAMLC0A0CM: ICE BofA US Corporate (IG) OAS (daily, bps)
  - HY-IG Spread = BAMLH0A0HYM2 - BAMLC0A0CM
  - Yahoo Finance SPY: S&P 500 ETF

Key features:
- Direction-based regimes: Spread Tightening (risk-on) vs Spread Widening (risk-off)
- Bilateral lead-lag range (-18 to +18)
- Optimal lag discovered from analysis, not preset
- Fast-fail (SOP v1.3): skip phases 4-5 if best |r| < 0.10 AND p > 0.30
- Backtest signal INVERTED: Long when spread tightening (direction < 0)

Phases 0-7 per SOP v1.3
"""

import pandas as pd
import numpy as np
from scipy import stats
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
MAX_LAG = 18  # Lead-lag range: -18 to +18 months
DATA_START = "1996-12-01"  # FRED OAS series start ~1996-12-31


def fetch_fred_data_url(series_id: str, start: str = "1996-12-01") -> pd.Series:
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

    # Handle '.' as missing value in FRED CSV
    values = pd.to_numeric(df[series_id], errors='coerce')
    return values.dropna()


def fetch_yahoo_data(ticker: str, start: str = "1996-12-01") -> pd.DataFrame:
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


def lead_lag_analysis(df: pd.DataFrame, x_col: str, y_col: str, max_lag: int = 18) -> pd.DataFrame:
    """
    Bilateral lead-lag analysis from -max_lag to +max_lag.
    Positive lag = indicator leads target (predictive).
    Negative lag = target leads indicator (confirmatory).
    """
    results = []
    for lag in range(-max_lag, max_lag + 1):
        x_lagged = df[x_col].shift(lag)
        y = df[y_col]

        mask = ~(x_lagged.isna() | y.isna())
        if mask.sum() >= 30:
            r, p = stats.pearsonr(x_lagged[mask], y[mask])
            results.append({
                'lag': lag,
                'correlation': r,
                'abs_correlation': abs(r),
                'pvalue': p,
                'n': mask.sum(),
                'significant': p < 0.05
            })

    return pd.DataFrame(results)


def regime_analysis(df: pd.DataFrame, indicator_col: str, target_col: str,
                    optimal_lag: int = 0) -> dict:
    """
    Regime analysis using YoY direction of spread.
    Spread Tightening (YoY < 0) = risk-on, favorable for equities.
    Spread Widening (YoY >= 0) = risk-off, unfavorable.
    """
    df = df.copy()
    df['indicator_lagged'] = df[indicator_col].shift(optimal_lag)

    # Define regimes: tightening vs widening
    df['regime'] = df['indicator_lagged'].apply(
        lambda x: 'Spread Tightening' if x < 0 else ('Spread Widening' if x >= 0 else np.nan)
    )

    # Calculate regime performance
    regime_stats = []
    for regime in ['Spread Tightening', 'Spread Widening']:
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
    tightening = df.loc[df['regime'] == 'Spread Tightening', target_col].dropna()
    widening = df.loc[df['regime'] == 'Spread Widening', target_col].dropna()

    if len(tightening) >= 20 and len(widening) >= 20:
        t_stat, p_value = stats.ttest_ind(tightening, widening)
    else:
        t_stat, p_value = np.nan, np.nan

    return {
        'regime_performance': regime_df,
        'regime_test': {
            't_stat': t_stat,
            'pvalue': p_value,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False
        },
        'data_with_regimes': df
    }


def run_backtest(df: pd.DataFrame, indicator_col: str, target_col: str,
                 optimal_lag: int = 0) -> dict:
    """
    Backtest: Long SPY when spread is TIGHTENING (YoY < 0), cash when widening.
    Note: Signal is INVERTED relative to other analyses — long on negative direction.
    """
    df = df.copy()
    df['indicator_lagged'] = df[indicator_col].shift(optimal_lag)

    # Signal: 1 when spread tightening (YoY < 0 = risk-on), 0 when widening
    df['signal'] = (df['indicator_lagged'] < 0).astype(int)

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

    strategy_sharpe = (df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(12)
                       if df['strategy_return'].std() > 0 else 0)
    benchmark_sharpe = (df['benchmark_return'].mean() / df['benchmark_return'].std() * np.sqrt(12)
                        if df['benchmark_return'].std() > 0 else 0)

    # Exposure (% time in market)
    exposure = df['position'].mean() * 100

    return {
        'strategy_sharpe': strategy_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'strategy_total_return': strategy_total * 100,
        'benchmark_total_return': benchmark_total * 100,
        'strategy_annualized': strategy_ann * 100,
        'benchmark_annualized': benchmark_ann * 100,
        'exposure': exposure,
        'backtest_data': df
    }


def main():
    print("=" * 70)
    print("FULL ANALYSIS: SPY vs HY-IG Credit Spread (SOP v1.3)")
    print("=" * 70)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Lead-Lag Range: -{MAX_LAG} to +{MAX_LAG} months")
    print()

    # =========================================================================
    # Phase 1: Data Preparation
    # =========================================================================
    print("Phase 1: Data Preparation")
    print("-" * 40)

    # Fetch HY OAS
    hy_oas = fetch_fred_data_url("BAMLH0A0HYM2", start=DATA_START)
    print(f"    HY OAS: {len(hy_oas)} daily observations ({hy_oas.index[0].strftime('%Y-%m-%d')} to {hy_oas.index[-1].strftime('%Y-%m-%d')})")

    # Fetch IG OAS
    ig_oas = fetch_fred_data_url("BAMLC0A0CM", start=DATA_START)
    print(f"    IG OAS: {len(ig_oas)} daily observations ({ig_oas.index[0].strftime('%Y-%m-%d')} to {ig_oas.index[-1].strftime('%Y-%m-%d')})")

    # Compute daily HY-IG spread
    # Align dates and compute difference
    spread_df = pd.DataFrame({'hy': hy_oas, 'ig': ig_oas}).dropna()
    spread_daily = spread_df['hy'] - spread_df['ig']
    spread_daily.name = 'HY_IG_Spread'

    print(f"    Daily spread: {len(spread_daily)} observations")
    print(f"    Spread range: {spread_daily.min():.1f} to {spread_daily.max():.1f} bps")

    # Sanity check: spread should always be positive
    if spread_daily.min() < 0:
        print(f"    WARNING: Spread has {(spread_daily < 0).sum()} negative values! Check FRED series order.")
    else:
        print(f"    Sanity check: Spread always positive")

    # Create monthly indicator data
    spread_monthly = create_monthly_data(spread_daily, "HY_IG_Spread")

    # Fetch SPY
    spy_df = fetch_yahoo_data("SPY", start=DATA_START)
    spy_monthly = create_monthly_data(spy_df['price'], "SPY")
    spy_monthly['SPY_Returns'] = spy_monthly['SPY_Level'].pct_change(1)

    # Merge
    df = pd.concat([spread_monthly, spy_monthly], axis=1).dropna()

    print(f"  Merged data period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    print(f"  Total observations: {len(df)}")
    print(f"  HY-IG Spread (latest): {df['HY_IG_Spread_Level'].iloc[-1]:.1f} bps")
    print()

    # =========================================================================
    # Phase 2: Correlation Analysis (Concurrent)
    # =========================================================================
    print("Phase 2: Correlation Analysis (Concurrent)")
    print("-" * 40)

    x_cols = ['HY_IG_Spread_Level', 'HY_IG_Spread_MoM', 'HY_IG_Spread_QoQ',
              'HY_IG_Spread_YoY', 'HY_IG_Spread_Direction']
    corr_results = correlation_analysis(df, x_cols, 'SPY_Returns')

    for _, row in corr_results.iterrows():
        sig = '*' if row['significant'] else ''
        print(f"  {row['indicator']:30s}: r={row['correlation']:+.3f}, p={row['pvalue']:.4f} {sig}")

    print()

    # =========================================================================
    # Phase 3: Lead-Lag Analysis (-18 to +18 months)
    # =========================================================================
    print(f"Phase 3: Lead-Lag Analysis (-{MAX_LAG} to +{MAX_LAG} months)")
    print("-" * 40)

    leadlag_results = lead_lag_analysis(df, 'HY_IG_Spread_YoY', 'SPY_Returns', max_lag=MAX_LAG)

    # Find best lag by absolute correlation
    best_idx = leadlag_results['abs_correlation'].idxmax()
    best_lag = leadlag_results.loc[best_idx]
    sig_lags = leadlag_results[leadlag_results['significant']]

    print(f"  Best lag: {int(best_lag['lag']):+d} months")
    print(f"    Correlation: r={best_lag['correlation']:+.3f}")
    print(f"    P-value: {best_lag['pvalue']:.4f}")
    print()

    print(f"  Significant lags (p < 0.05): {len(sig_lags)}")
    for _, row in sig_lags.iterrows():
        direction = "indicator leads" if row['lag'] > 0 else "target leads" if row['lag'] < 0 else "concurrent"
        print(f"    Lag {int(row['lag']):+3d}: r={row['correlation']:+.3f}, p={row['pvalue']:.4f} ({direction})")

    print()

    # =========================================================================
    # Phase 3b: Fast-Fail Check
    # =========================================================================
    print("Phase 3b: Fast-Fail Check")
    print("-" * 40)

    best_abs_r = best_lag['abs_correlation']
    min_pvalue = leadlag_results['pvalue'].min()

    fast_fail = best_abs_r < 0.10 and min_pvalue > 0.30

    if fast_fail:
        print(f"  FAST-FAIL: best |r|={best_abs_r:.3f} < 0.10 AND min p={min_pvalue:.3f} > 0.30")
        print("  Skipping Phases 4-5 (no significant relationship found)")
        print()
        optimal_lag = 0
    else:
        print(f"  PASS: best |r|={best_abs_r:.3f}, min p={min_pvalue:.4f}")
        print("  Proceeding to Phases 4-5")
        print()

        # Use the lag with strongest absolute correlation among significant lags
        if len(sig_lags) > 0:
            best_sig_idx = sig_lags['abs_correlation'].idxmax()
            optimal_lag = int(sig_lags.loc[best_sig_idx, 'lag'])
        else:
            optimal_lag = int(best_lag['lag'])

    # =========================================================================
    # Phase 4: Regime Analysis
    # =========================================================================
    if not fast_fail:
        print(f"Phase 4: Regime Analysis (using lag {optimal_lag:+d})")
        print("-" * 40)

        regime_results = regime_analysis(df, 'HY_IG_Spread_YoY', 'SPY_Returns',
                                         optimal_lag=optimal_lag)

        print("  Regime Performance:")
        for _, row in regime_results['regime_performance'].iterrows():
            print(f"    {row['regime']:25s}: mean={row['mean_return']*100:+.2f}%/mo, "
                  f"Sharpe={row['sharpe']:.2f}, n={int(row['count'])}")

        if not np.isnan(regime_results['regime_test']['pvalue']):
            sig = '*' if regime_results['regime_test']['significant'] else ''
            print(f"  Regime difference test: t={regime_results['regime_test']['t_stat']:.2f}, "
                  f"p={regime_results['regime_test']['pvalue']:.4f} {sig}")
        print()

    # =========================================================================
    # Phase 5: Backtest
    # =========================================================================
    if not fast_fail:
        print(f"Phase 5: Backtest (using lag {optimal_lag:+d})")
        print("-" * 40)

        backtest_results = run_backtest(df, 'HY_IG_Spread_YoY', 'SPY_Returns',
                                        optimal_lag=optimal_lag)

        print(f"  Strategy Sharpe: {backtest_results['strategy_sharpe']:.2f}")
        print(f"  Benchmark Sharpe: {backtest_results['benchmark_sharpe']:.2f}")
        print(f"  Strategy Total Return: {backtest_results['strategy_total_return']:.1f}%")
        print(f"  Benchmark Total Return: {backtest_results['benchmark_total_return']:.1f}%")
        print(f"  Strategy Annualized: {backtest_results['strategy_annualized']:.1f}%")
        print(f"  Benchmark Annualized: {backtest_results['benchmark_annualized']:.1f}%")
        print(f"  Exposure: {backtest_results['exposure']:.1f}%")
        print()

    # =========================================================================
    # Phase 6: Dashboard Data Preparation
    # =========================================================================
    print("Phase 6: Dashboard Data Preparation")
    print("-" * 40)

    # Create dashboard-ready data file
    dashboard_df = df[['HY_IG_Spread_Level', 'HY_IG_Spread_MoM', 'HY_IG_Spread_QoQ',
                        'HY_IG_Spread_YoY', 'HY_IG_Spread_Direction',
                        'SPY_Level', 'SPY_Returns']].copy()

    # Add lagged indicator
    dashboard_df['HY_IG_Spread_YoY_Lagged'] = dashboard_df['HY_IG_Spread_YoY'].shift(optimal_lag)
    dashboard_df['Regime'] = dashboard_df['HY_IG_Spread_YoY_Lagged'].apply(
        lambda x: 'Spread Tightening' if x < 0 else ('Spread Widening' if x >= 0 else np.nan)
    )

    # Add backtest columns (only if not fast-failed)
    if not fast_fail:
        backtest_data = backtest_results['backtest_data']
        dashboard_df['Strategy_Return'] = backtest_data['strategy_return']
        dashboard_df['Strategy_CumRet'] = backtest_data['strategy_cumret']
        dashboard_df['Benchmark_CumRet'] = backtest_data['benchmark_cumret']

    # Save main data file
    output_file = DATA_DIR / "spy_hy_ig_spread_full.parquet"
    dashboard_df.to_parquet(output_file)
    print(f"  Saved: {output_file}")

    # Save lead-lag results
    leadlag_file = DATA_DIR / "spy_hy_ig_spread_leadlag.parquet"
    leadlag_results.to_parquet(leadlag_file)
    print(f"  Saved: {leadlag_file}")

    # Save correlation results
    corr_file = DATA_DIR / "spy_hy_ig_spread_correlation.parquet"
    corr_results.to_parquet(corr_file)
    print(f"  Saved: {corr_file}")

    # Save regime performance (only if not fast-failed)
    if not fast_fail:
        regime_file = DATA_DIR / "spy_hy_ig_spread_regimes.parquet"
        regime_results['regime_performance'].to_parquet(regime_file)
        print(f"  Saved: {regime_file}")

    print()

    # =========================================================================
    # Phase 7: Summary
    # =========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Key Findings:")
    print(f"  1. Best lag: {optimal_lag:+d} months (r={best_lag['correlation']:+.3f}, p={best_lag['pvalue']:.4f})")
    print(f"  2. Significant lags: {len(sig_lags)} (of {len(leadlag_results)} tested)")

    if not fast_fail:
        print(f"  3. Regime difference: p={regime_results['regime_test']['pvalue']:.4f}")
        print(f"  4. Strategy Sharpe: {backtest_results['strategy_sharpe']:.2f} vs "
              f"Benchmark: {backtest_results['benchmark_sharpe']:.2f}")
        actionable = backtest_results['strategy_sharpe'] > backtest_results['benchmark_sharpe']
        print()
        if actionable:
            print(f"  Actionable: YES - Use HY-IG spread direction at lag {optimal_lag:+d} as signal for SPY")
        else:
            print(f"  Actionable: MARGINAL - Strategy underperforms benchmark despite significant regime difference")
    else:
        print(f"  3. FAST-FAILED: No significant relationship found")
        print()
        print("  Actionable: NO - HY-IG spread does not predict SPY returns at any tested lag")

    print()
    print("Dashboard data files created in data/")
    print(f"  Optimal lag for seed config: default_lag = {optimal_lag}")


if __name__ == "__main__":
    main()
