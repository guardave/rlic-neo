#!/usr/bin/env python3
"""
Full Analysis: SPY vs Cass Freight Index
Following SOP v1.3 with bilateral lead-lag analysis (-18 to +18 months).

Data sources:
  - FRED FRGSHPUSM649NCIS: Cass Freight Index - Shipments (primary indicator)
  - FRED FRGEXPUSM649NCIS: Cass Freight Index - Expenditures (secondary column)
  - Yahoo Finance SPY: S&P 500 ETF

Key features:
- Two freight components: Shipments as primary indicator, Expenditures as secondary
- Direction-based regimes: Freight Rising vs Freight Falling (based on YoY)
- Bilateral lead-lag range (-18 to +18)
- Optimal lag discovered from analysis, not preset
- Fast-fail (SOP v1.3): skip phases 4-5 if best |r| < 0.10 AND p > 0.30
- Backtest: Long SPY when freight shipments YoY is rising (positive direction)

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
DATA_START = "2000-01-01"  # Cass Freight Index available from ~1990


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

    # Handle '.' as missing value in FRED CSV
    values = pd.to_numeric(df[series_id], errors='coerce')
    return values.dropna()


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
    Regime analysis using YoY direction of Cass Freight Shipments.
    Rising (YoY > 0) = economic expansion, favorable for equities.
    Falling (YoY <= 0) = economic contraction, unfavorable.
    """
    df = df.copy()
    df['indicator_lagged'] = df[indicator_col].shift(optimal_lag)

    # Define regimes: rising vs falling
    df['regime'] = df['indicator_lagged'].apply(
        lambda x: 'Freight Rising' if x > 0 else ('Freight Falling' if x <= 0 else np.nan)
    )

    # Calculate regime performance
    regime_stats = []
    for regime in ['Freight Rising', 'Freight Falling']:
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
    rising = df.loc[df['regime'] == 'Freight Rising', target_col].dropna()
    falling = df.loc[df['regime'] == 'Freight Falling', target_col].dropna()

    if len(rising) >= 20 and len(falling) >= 20:
        t_stat, p_value = stats.ttest_ind(rising, falling)
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
    Backtest: Long SPY when freight is RISING (YoY > 0), cash when falling.
    """
    df = df.copy()
    df['indicator_lagged'] = df[indicator_col].shift(optimal_lag)

    # Signal: 1 when freight rising (YoY > 0), 0 when falling
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
    print("FULL ANALYSIS: SPY vs Cass Freight Index (SOP v1.3)")
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

    # Fetch Cass Freight Shipments (primary)
    shipments = fetch_fred_data_url("FRGSHPUSM649NCIS", start=DATA_START)
    print(f"    Shipments: {len(shipments)} observations "
          f"({shipments.index[0].strftime('%Y-%m-%d')} to {shipments.index[-1].strftime('%Y-%m-%d')})")

    # Fetch Cass Freight Expenditures (secondary)
    expenditures = fetch_fred_data_url("FRGEXPUSM649NCIS", start=DATA_START)
    print(f"    Expenditures: {len(expenditures)} observations "
          f"({expenditures.index[0].strftime('%Y-%m-%d')} to {expenditures.index[-1].strftime('%Y-%m-%d')})")

    # Create monthly indicator data for Shipments (primary)
    ship_monthly = create_monthly_data(shipments, "CassShip")

    # Create monthly indicator data for Expenditures (secondary)
    exp_monthly = create_monthly_data(expenditures, "CassExp")

    # Fetch SPY
    spy_df = fetch_yahoo_data("SPY", start=DATA_START)
    spy_monthly = create_monthly_data(spy_df['price'], "SPY")
    spy_monthly['SPY_Returns'] = spy_monthly['SPY_Level'].pct_change(1)

    # Merge all three
    df = pd.concat([ship_monthly, exp_monthly, spy_monthly], axis=1).dropna()

    print(f"  Merged data period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    print(f"  Total observations: {len(df)}")
    print(f"  Shipments (latest): {df['CassShip_Level'].iloc[-1]:.2f}")
    print(f"  Expenditures (latest): {df['CassExp_Level'].iloc[-1]:.2f}")
    print()

    # =========================================================================
    # Phase 2: Correlation Analysis (Concurrent)
    # =========================================================================
    print("Phase 2: Correlation Analysis (Concurrent)")
    print("-" * 40)

    # Shipments correlations
    print("  Shipments vs SPY_Returns:")
    ship_cols = ['CassShip_Level', 'CassShip_MoM', 'CassShip_QoQ',
                 'CassShip_YoY', 'CassShip_Direction']
    corr_ship = correlation_analysis(df, ship_cols, 'SPY_Returns')

    for _, row in corr_ship.iterrows():
        sig = '*' if row['significant'] else ''
        print(f"    {row['indicator']:30s}: r={row['correlation']:+.3f}, p={row['pvalue']:.4f} {sig}")

    print()

    # Expenditures correlations
    print("  Expenditures vs SPY_Returns:")
    exp_cols = ['CassExp_Level', 'CassExp_MoM', 'CassExp_QoQ',
                'CassExp_YoY', 'CassExp_Direction']
    corr_exp = correlation_analysis(df, exp_cols, 'SPY_Returns')

    for _, row in corr_exp.iterrows():
        sig = '*' if row['significant'] else ''
        print(f"    {row['indicator']:30s}: r={row['correlation']:+.3f}, p={row['pvalue']:.4f} {sig}")

    print()

    # Combine all correlation results
    corr_results = pd.concat([corr_ship, corr_exp], ignore_index=True)

    # =========================================================================
    # Phase 3: Lead-Lag Analysis (-18 to +18 months) — Shipments (primary)
    # =========================================================================
    print(f"Phase 3: Lead-Lag Analysis (-{MAX_LAG} to +{MAX_LAG} months)")
    print("-" * 40)

    print("  Primary: CassShip_YoY vs SPY_Returns")
    leadlag_results = lead_lag_analysis(df, 'CassShip_YoY', 'SPY_Returns', max_lag=MAX_LAG)

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

    # Also run lead-lag for Expenditures (secondary, for reference)
    print("  Secondary: CassExp_YoY vs SPY_Returns")
    leadlag_exp = lead_lag_analysis(df, 'CassExp_YoY', 'SPY_Returns', max_lag=MAX_LAG)
    best_exp_idx = leadlag_exp['abs_correlation'].idxmax()
    best_exp_lag = leadlag_exp.loc[best_exp_idx]
    sig_exp_lags = leadlag_exp[leadlag_exp['significant']]

    print(f"  Best lag: {int(best_exp_lag['lag']):+d} months")
    print(f"    Correlation: r={best_exp_lag['correlation']:+.3f}")
    print(f"    P-value: {best_exp_lag['pvalue']:.4f}")
    print(f"  Significant lags: {len(sig_exp_lags)}")
    print()

    # =========================================================================
    # Phase 3b: Fast-Fail Check (based on primary indicator)
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

        regime_results = regime_analysis(df, 'CassShip_YoY', 'SPY_Returns',
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

        backtest_results = run_backtest(df, 'CassShip_YoY', 'SPY_Returns',
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

    # Create dashboard-ready data file with both indicators
    dashboard_df = df[['CassShip_Level', 'CassShip_MoM', 'CassShip_QoQ',
                        'CassShip_YoY', 'CassShip_Direction',
                        'CassExp_Level', 'CassExp_MoM', 'CassExp_QoQ',
                        'CassExp_YoY', 'CassExp_Direction',
                        'SPY_Level', 'SPY_Returns']].copy()

    # Add lagged primary indicator
    dashboard_df['CassShip_YoY_Lagged'] = dashboard_df['CassShip_YoY'].shift(optimal_lag)
    dashboard_df['Regime'] = dashboard_df['CassShip_YoY_Lagged'].apply(
        lambda x: 'Freight Rising' if x > 0 else ('Freight Falling' if x <= 0 else np.nan)
    )

    # Add backtest columns (only if not fast-failed)
    if not fast_fail:
        backtest_data = backtest_results['backtest_data']
        dashboard_df['Strategy_Return'] = backtest_data['strategy_return']
        dashboard_df['Strategy_CumRet'] = backtest_data['strategy_cumret']
        dashboard_df['Benchmark_CumRet'] = backtest_data['benchmark_cumret']

    # Save main data file
    output_file = DATA_DIR / "spy_cass_freight_full.parquet"
    dashboard_df.to_parquet(output_file)
    print(f"  Saved: {output_file}")

    # Save lead-lag results (primary)
    leadlag_file = DATA_DIR / "spy_cass_freight_leadlag.parquet"
    leadlag_results.to_parquet(leadlag_file)
    print(f"  Saved: {leadlag_file}")

    # Save lead-lag results (secondary expenditures)
    leadlag_exp_file = DATA_DIR / "spy_cass_freight_exp_leadlag.parquet"
    leadlag_exp.to_parquet(leadlag_exp_file)
    print(f"  Saved: {leadlag_exp_file}")

    # Save correlation results
    corr_file = DATA_DIR / "spy_cass_freight_correlation.parquet"
    corr_results.to_parquet(corr_file)
    print(f"  Saved: {corr_file}")

    # Save regime performance (only if not fast-failed)
    if not fast_fail:
        regime_file = DATA_DIR / "spy_cass_freight_regimes.parquet"
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
    print("Key Findings (Shipments - Primary):")
    print(f"  1. Best lag: {optimal_lag:+d} months (r={best_lag['correlation']:+.3f}, p={best_lag['pvalue']:.4f})")
    print(f"  2. Significant lags: {len(sig_lags)} (of {len(leadlag_results)} tested)")

    print()
    print("Key Findings (Expenditures - Secondary):")
    print(f"  1. Best lag: {int(best_exp_lag['lag']):+d} months (r={best_exp_lag['correlation']:+.3f}, p={best_exp_lag['pvalue']:.4f})")
    print(f"  2. Significant lags: {len(sig_exp_lags)} (of {len(leadlag_exp)} tested)")

    if not fast_fail:
        print()
        print("Regime & Backtest (Shipments):")
        print(f"  3. Regime difference: p={regime_results['regime_test']['pvalue']:.4f}")
        print(f"  4. Strategy Sharpe: {backtest_results['strategy_sharpe']:.2f} vs "
              f"Benchmark: {backtest_results['benchmark_sharpe']:.2f}")
        actionable = backtest_results['strategy_sharpe'] > backtest_results['benchmark_sharpe']
        print()
        if actionable:
            print(f"  Actionable: YES - Use Cass Freight Shipments direction at lag {optimal_lag:+d} as signal for SPY")
        else:
            print(f"  Actionable: MARGINAL - Strategy underperforms benchmark despite significant regime difference")
    else:
        print(f"  3. FAST-FAILED: No significant relationship found")
        print()
        print("  Actionable: NO - Cass Freight does not predict SPY returns at any tested lag")

    print()
    print("Dashboard data files created in data/")
    print(f"  Optimal lag for seed config: default_lag = {optimal_lag}")


if __name__ == "__main__":
    main()
