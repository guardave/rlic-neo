"""
XLRE vs Housing Indicators Analysis

Following SOP v1.2: Full 7-phase analysis for:
1. XLRE vs New Home Sales (HSN1F)
2. XLRE vs Building Permits (PERMIT)

Phase 0: Qualitative Analysis
Phase 2: Statistical Analysis (correlation, p-values, effect size)
Phase 3: Lead-Lag Analysis
Phase 4: Regime Analysis
Phase 5: Backtest (regime-conditional)
Phase 6: Dashboard Data Preparation
Phase 7: Documentation
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t
from numpy.linalg import LinAlgError
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import yfinance as yf
from fredapi import Fred
from pathlib import Path
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs" / "analysis_reports"

# FRED API key (from environment)
FRED_API_KEY = os.environ.get('FRED_API_KEY', None)

# Minimum thresholds from SOP v1.2
MIN_SAMPLE_SIZE_CORRELATION = 30
MIN_SAMPLE_SIZE_GRANGER = 60
MIN_SAMPLE_SIZE_REGIME = 20
MIN_EFFECT_SIZE = 0.15  # |r| >= 0.15 for actionable correlation
SIGNIFICANCE_LEVEL = 0.05


def fetch_fred_data(series_id: str, start: str = "1990-01-01") -> pd.Series:
    """Fetch data from FRED using multiple methods."""
    # Method 1: Try existing cached data
    existing_data = {
        'PERMIT': 'building_permits',  # In monthly_leading_indicators.parquet
        'HOUST': 'housing_starts',
    }

    if series_id in existing_data:
        cache_file = DATA_DIR / "monthly_leading_indicators.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            col = existing_data[series_id]
            if col in df.columns:
                series = df[col].dropna()
                # Ensure datetime index
                if not isinstance(series.index, pd.DatetimeIndex):
                    series.index = pd.to_datetime(series.index)
                return series

    # Method 2: Try FRED API if key available
    if FRED_API_KEY is not None:
        try:
            fred = Fred(api_key=FRED_API_KEY)
            data = fred.get_series(series_id, observation_start=start)
            return data
        except Exception as e:
            print(f"  FRED API error: {e}")

    # Method 3: Try yfinance for economic data (some FRED data is mirrored)
    try:
        import yfinance as yf
        # Some indicators available via Yahoo Finance
        yf_mapping = {
            'HSN1F': None,  # Not available on YF
        }
        if series_id in yf_mapping and yf_mapping[series_id]:
            data = yf.download(yf_mapping[series_id], start=start, progress=False)
            return data['Close']
    except Exception:
        pass

    # Method 4: Try direct URL fetch from FRED
    try:
        import urllib.request
        import io
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
        print(f"  Fetching from FRED URL: {series_id}")

        with urllib.request.urlopen(url, timeout=30) as response:
            csv_data = response.read().decode('utf-8')

        # Parse CSV - FRED format has observation_date column
        df = pd.read_csv(io.StringIO(csv_data))
        # Find the date column (could be 'observation_date', 'DATE', or first column)
        date_col = None
        for col in ['observation_date', 'DATE', df.columns[0]]:
            if col in df.columns:
                date_col = col
                break
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        return df[series_id].dropna()
    except Exception as e:
        print(f"  URL fetch error for {series_id}: {e}")

    raise ValueError(f"Could not fetch {series_id} from any source")


def fetch_yahoo_data(ticker: str, start: str = "1990-01-01") -> pd.DataFrame:
    """Fetch price data from Yahoo Finance."""
    data = yf.download(ticker, start=start, progress=False)

    # Handle different column formats
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(0):
            price = data['Adj Close']
        else:
            price = data['Close']
        if isinstance(price, pd.DataFrame):
            price = price.iloc[:, 0]
    else:
        price = data.get('Adj Close', data.get('Close'))

    return pd.DataFrame({'price': price})


def create_monthly_data(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Resample to monthly frequency and create derivatives."""
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    monthly = df.resample('ME').last()

    result = pd.DataFrame(index=monthly.index)
    result[f'{col}_Level'] = monthly[col] if col in monthly.columns else monthly.iloc[:, 0]
    result[f'{col}_MoM'] = result[f'{col}_Level'].pct_change(1)
    result[f'{col}_QoQ'] = result[f'{col}_Level'].pct_change(3)
    result[f'{col}_YoY'] = result[f'{col}_Level'].pct_change(12)

    return result


def correlation_with_pvalue(x: pd.Series, y: pd.Series) -> dict:
    """Compute Pearson correlation with p-value."""
    valid = pd.concat([x, y], axis=1).dropna()
    n = len(valid)

    if n < MIN_SAMPLE_SIZE_CORRELATION:
        return {'correlation': np.nan, 'pvalue': np.nan, 'n': n, 'significant': False, 'effect_size_adequate': False}

    corr, pval = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])

    return {
        'correlation': corr,
        'pvalue': pval,
        'n': n,
        'significant': pval < SIGNIFICANCE_LEVEL,
        'effect_size_adequate': abs(corr) >= MIN_EFFECT_SIZE
    }


def leadlag_analysis(df: pd.DataFrame, x_col: str, y_col: str, max_lag: int = 12) -> pd.DataFrame:
    """Compute cross-correlation at different lags."""
    results = []

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            x_series = df[x_col].iloc[:-lag]
            y_series = df[y_col].shift(-lag).iloc[:-lag]
        elif lag < 0:
            x_series = df[x_col].iloc[-lag:]
            y_series = df[y_col].shift(-lag).iloc[-lag:]
        else:
            x_series = df[x_col]
            y_series = df[y_col]

        valid = pd.concat([x_series, y_series], axis=1).dropna()
        if len(valid) >= MIN_SAMPLE_SIZE_CORRELATION:
            corr, pval = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
        else:
            corr, pval = np.nan, np.nan

        results.append({
            'lag': lag,
            'correlation': corr,
            'pvalue': pval,
            'n_obs': len(valid),
            'significant': pval < SIGNIFICANCE_LEVEL if not np.isnan(pval) else False
        })

    return pd.DataFrame(results)


def granger_test(df: pd.DataFrame, x_col: str, y_col: str, max_lag: int = 6) -> pd.DataFrame:
    """Test Granger causality."""
    data = df[[y_col, x_col]].dropna()

    if len(data) < MIN_SAMPLE_SIZE_GRANGER:
        return pd.DataFrame()

    try:
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        output = []
        for lag in range(1, max_lag + 1):
            test_result = results[lag][0]
            f_stat = test_result['ssr_ftest'][0]
            pval = test_result['ssr_ftest'][1]
            output.append({
                'lag': lag,
                'f_statistic': f_stat,
                'pvalue': pval,
                'significant': pval < SIGNIFICANCE_LEVEL
            })

        return pd.DataFrame(output)
    except (ValueError, LinAlgError) as e:
        print(f"Granger test error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Granger test unexpected error: {e}")
        return pd.DataFrame()


def define_regimes(series: pd.Series, method: str = 'direction') -> pd.Series:
    """Define regimes based on direction or level."""
    if method == 'direction':
        # Use YoY change for direction
        yoy = series.pct_change(12)
        return (yoy > 0).map({True: 'Rising', False: 'Falling'})
    else:
        # Use median level
        median = series.median()
        return (series > median).map({True: 'High', False: 'Low'})


def regime_performance(df: pd.DataFrame, regime_col: str, return_col: str) -> pd.DataFrame:
    """Calculate performance by regime with statistical testing."""
    results = []
    regimes = df[regime_col].dropna().unique()

    for regime in regimes:
        mask = df[regime_col] == regime
        returns = df.loc[mask, return_col].dropna()

        if len(returns) < MIN_SAMPLE_SIZE_REGIME:
            continue

        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = mean_ret / std_ret * np.sqrt(12) if std_ret > 0 else 0

        results.append({
            'regime': regime,
            'mean_return': mean_ret,
            'std_return': std_ret,
            'sharpe_ratio': sharpe,
            'n_periods': len(returns),
            'pct_positive': (returns > 0).mean(),
            'annualized_return': mean_ret * 12
        })

    return pd.DataFrame(results)


def regime_difference_test(df: pd.DataFrame, regime_col: str, return_col: str) -> dict:
    """Test if regime difference is statistically significant."""
    regimes = df[regime_col].dropna().unique()

    if len(regimes) != 2:
        return {'pvalue': np.nan, 'significant': False}

    regime1, regime2 = regimes
    returns1 = df.loc[df[regime_col] == regime1, return_col].dropna()
    returns2 = df.loc[df[regime_col] == regime2, return_col].dropna()

    if len(returns1) < MIN_SAMPLE_SIZE_REGIME or len(returns2) < MIN_SAMPLE_SIZE_REGIME:
        return {'pvalue': np.nan, 'significant': False}

    # Two-sample t-test
    t_stat, pval = stats.ttest_ind(returns1, returns2)

    return {
        'regime1': regime1,
        'regime2': regime2,
        'mean_diff': returns1.mean() - returns2.mean(),
        't_statistic': t_stat,
        'pvalue': pval,
        'significant': pval < SIGNIFICANCE_LEVEL
    }


def check_fast_fail(corr_result: dict) -> bool:
    """
    SOP v1.2: Fast-fail check.
    If |r| < 0.10 AND p > 0.30, skip to Phase 7 documentation.
    """
    if np.isnan(corr_result['correlation']):
        return False
    return abs(corr_result['correlation']) < 0.10 and corr_result['pvalue'] > 0.30


def run_full_analysis(indicator_name: str, indicator_series: pd.Series,
                      target_name: str, target_df: pd.DataFrame,
                      random_seed: int = 42) -> dict:
    """
    Run full 7-phase analysis following SOP v1.2.

    Args:
        indicator_name: Name of the economic indicator
        indicator_series: FRED indicator data
        target_name: Name of target asset (e.g., 'XLRE')
        target_df: DataFrame with price data
        random_seed: For reproducibility (Monte Carlo if applicable)

    Returns:
        dict with all analysis results
    """
    np.random.seed(random_seed)

    print(f"\n{'='*60}")
    print(f"Analysis: {target_name} vs {indicator_name}")
    print(f"{'='*60}")

    # Prepare monthly data
    print("\nPhase 1: Data Preparation...")

    # Indicator data - ensure proper DataFrame construction
    if isinstance(indicator_series, pd.Series):
        indicator_df = pd.DataFrame({indicator_name: indicator_series})
    else:
        indicator_df = pd.DataFrame(indicator_series)
        indicator_df.columns = [indicator_name]
    indicator_monthly = create_monthly_data(indicator_df, indicator_name)

    # Target data - ensure proper DataFrame with target name
    if target_name not in target_df.columns:
        target_df = target_df.copy()
        target_df.columns = [target_name]
    target_monthly = create_monthly_data(target_df, target_name)
    target_monthly[f'{target_name}_Returns'] = target_monthly[f'{target_name}_Level'].pct_change(1)

    # Merge
    df = pd.concat([indicator_monthly, target_monthly], axis=1).dropna()

    print(f"  Data period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    print(f"  Total observations: {len(df)}")

    results = {
        'indicator_name': indicator_name,
        'target_name': target_name,
        'data_start': df.index[0],
        'data_end': df.index[-1],
        'n_observations': len(df),
        'random_seed': random_seed
    }

    # Phase 2: Statistical Analysis
    print("\nPhase 2: Statistical Analysis...")

    # Correlation matrix
    x_cols = [c for c in df.columns if indicator_name in c and 'Level' not in c]
    y_cols = [f'{target_name}_Returns']

    correlations = {}
    for x_col in x_cols:
        for y_col in y_cols:
            key = f"{x_col}_vs_{y_col}"
            correlations[key] = correlation_with_pvalue(df[x_col], df[y_col])

    results['correlations'] = correlations

    # Find best correlation
    best_corr_key = max(correlations.keys(),
                        key=lambda k: abs(correlations[k]['correlation']) if not np.isnan(correlations[k]['correlation']) else 0)
    best_corr = correlations[best_corr_key]

    print(f"  Best correlation: {best_corr_key}")
    print(f"    r = {best_corr['correlation']:.3f}, p = {best_corr['pvalue']:.3f}")
    print(f"    Effect size adequate (|r|>={MIN_EFFECT_SIZE}): {best_corr['effect_size_adequate']}")
    print(f"    Statistically significant (p<{SIGNIFICANCE_LEVEL}): {best_corr['significant']}")

    # Fast-fail check
    if check_fast_fail(best_corr):
        print(f"\n  ⚠️ FAST-FAIL: |r| < 0.10 AND p > 0.30 - Skipping to Phase 7")
        results['fast_fail'] = True
        results['conclusion'] = f"No meaningful relationship found between {indicator_name} and {target_name}"
        # Set placeholder values for report generator
        results['optimal_lag'] = {'lag': 0, 'correlation': best_corr['correlation'], 'pvalue': best_corr['pvalue'], 'significant': False}
        results['leadlag'] = pd.DataFrame()
        results['granger'] = pd.DataFrame()
        results['regime_performance'] = pd.DataFrame()
        results['regime_test'] = {'pvalue': np.nan, 'significant': False}
        results['backtest'] = {
            'strategy_sharpe': 0, 'benchmark_sharpe': 0,
            'strategy_total_return': 0, 'benchmark_total_return': 0,
            'strategy_annualized': 0, 'benchmark_annualized': 0
        }
        results['data'] = df
        return results

    results['fast_fail'] = False

    # Phase 3: Lead-Lag Analysis
    print("\nPhase 3: Lead-Lag Analysis...")

    x_col = f'{indicator_name}_MoM'
    y_col = f'{target_name}_Returns'

    leadlag_results = leadlag_analysis(df, x_col, y_col, max_lag=12)
    results['leadlag'] = leadlag_results

    # Find optimal lag
    significant_lags = leadlag_results[leadlag_results['significant']]
    if len(significant_lags) > 0:
        optimal = significant_lags.loc[significant_lags['correlation'].abs().idxmax()]
        print(f"  Optimal lag: {int(optimal['lag'])} months")
        print(f"    r = {optimal['correlation']:.3f}, p = {optimal['pvalue']:.3f}")
    else:
        optimal = leadlag_results.loc[leadlag_results['correlation'].abs().idxmax()]
        print(f"  No significant lags found. Best (non-significant): {int(optimal['lag'])} months")
        print(f"    r = {optimal['correlation']:.3f}, p = {optimal['pvalue']:.3f}")

    results['optimal_lag'] = {
        'lag': int(optimal['lag']),
        'correlation': optimal['correlation'],
        'pvalue': optimal['pvalue'],
        'significant': optimal.get('significant', False)
    }

    # Granger causality
    print("\n  Granger Causality Test...")
    granger_results = granger_test(df, x_col, y_col, max_lag=6)
    results['granger'] = granger_results

    if not granger_results.empty:
        significant_granger = granger_results[granger_results['significant']]
        if len(significant_granger) > 0:
            print(f"  Granger-causes at lags: {list(significant_granger['lag'])}")
        else:
            print("  No significant Granger causality found")

    # Phase 4: Regime Analysis
    print("\nPhase 4: Regime Analysis...")

    # Define regimes using YoY direction
    df['regime'] = define_regimes(df[f'{indicator_name}_Level'], 'direction')

    regime_perf = regime_performance(df, 'regime', y_col)
    results['regime_performance'] = regime_perf

    print("  Performance by Regime:")
    for _, row in regime_perf.iterrows():
        print(f"    {row['regime']}: Mean={row['mean_return']*100:.2f}%, Sharpe={row['sharpe_ratio']:.2f}, N={row['n_periods']}")

    # Test regime difference significance
    regime_test = regime_difference_test(df, 'regime', y_col)
    results['regime_test'] = regime_test

    print(f"  Regime difference p-value: {regime_test['pvalue']:.3f}")
    print(f"  Significant (p<{SIGNIFICANCE_LEVEL}): {regime_test['significant']}")

    # Phase 5: Backtest
    print("\nPhase 5: Regime-Conditional Backtest...")

    # Go long when indicator is falling (historically better for equities)
    df['signal'] = (df['regime'] == 'Falling').astype(int)
    df['signal_lagged'] = df['signal'].shift(1)
    df['strategy_return'] = df['signal_lagged'] * df[y_col]
    df['benchmark_return'] = df[y_col]

    # Calculate metrics
    strategy_returns = df['strategy_return'].dropna()
    benchmark_returns = df['benchmark_return'].dropna()

    strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(12) if strategy_returns.std() > 0 else 0
    benchmark_sharpe = benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(12) if benchmark_returns.std() > 0 else 0

    strategy_total = (1 + strategy_returns).prod() - 1
    benchmark_total = (1 + benchmark_returns).prod() - 1

    backtest_results = {
        'strategy_sharpe': strategy_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'strategy_total_return': strategy_total,
        'benchmark_total_return': benchmark_total,
        'strategy_annualized': strategy_returns.mean() * 12,
        'benchmark_annualized': benchmark_returns.mean() * 12
    }
    results['backtest'] = backtest_results

    print(f"  Strategy Sharpe: {strategy_sharpe:.2f}")
    print(f"  Benchmark Sharpe: {benchmark_sharpe:.2f}")
    print(f"  Strategy Total Return: {strategy_total*100:.1f}%")
    print(f"  Benchmark Total Return: {benchmark_total*100:.1f}%")

    # Store data for dashboard
    results['data'] = df

    # Generate conclusion
    if best_corr['significant'] and best_corr['effect_size_adequate']:
        conclusion = f"Significant relationship found (r={best_corr['correlation']:.3f}, p={best_corr['pvalue']:.3f})"
        if regime_test['significant']:
            conclusion += f" with significant regime differentiation"
    elif best_corr['significant']:
        conclusion = f"Statistically significant but weak relationship (r={best_corr['correlation']:.3f})"
    else:
        conclusion = f"No statistically significant relationship (r={best_corr['correlation']:.3f}, p={best_corr['pvalue']:.3f})"

    results['conclusion'] = conclusion

    print(f"\n📋 Conclusion: {conclusion}")

    return results


def generate_report(results: dict, output_path: Path):
    """Generate markdown report for analysis results."""

    r = results

    report = f"""# {r['target_name']} vs {r['indicator_name']} Analysis

## Overview

This analysis explores the relationship between {r['target_name']} (Real Estate Select Sector ETF) and {r['indicator_name']}.

**Data Period**: {r['data_start'].strftime('%B %Y')} to {r['data_end'].strftime('%B %Y')} (~{r['n_observations']} months)

**Analysis conducted following SOP v1.2** with random_seed={r['random_seed']} for reproducibility.

---

## Qualitative Analysis

### {r['indicator_name']} as Leading Indicator

"""

    if 'HSN1F' in r['indicator_name'] or 'New Home' in r['indicator_name']:
        report += """**New Home Sales** measures the number of newly constructed homes sold each month. It is a leading indicator because:

1. **Forward-Looking**: New home sales require mortgage applications, credit checks, and planning - all occurring before actual purchase
2. **Construction Pipeline**: Sales drive future construction activity and employment
3. **Consumer Confidence**: New home purchases are major financial decisions reflecting consumer outlook
4. **Real Estate Connection**: Directly impacts XLRE holdings (residential REITs, home-related companies)

**Economic Rationale for XLRE Relationship:**
- Rising new home sales → increased real estate activity → positive for XLRE
- Home sales lead construction spending by 2-6 months
- Strong home sales indicate healthy housing demand benefiting real estate sector
"""
    elif 'PERMIT' in r['indicator_name'] or 'Building' in r['indicator_name']:
        report += """**Building Permits** measures the number of new privately-owned housing units authorized by building permits. It is a leading indicator because:

1. **Most Forward-Looking Housing Indicator**: Permits precede construction starts by weeks/months
2. **Developer Confidence**: Reflects builder outlook on future demand
3. **Pipeline Indicator**: Strong permits signal robust future housing supply
4. **Planning Horizon**: Permits obtained 3-6 months before construction completes

**Economic Rationale for XLRE Relationship:**
- Rising permits → future construction → real estate sector growth → positive for XLRE
- Permits lead new home sales by 2-4 months
- Strong permit activity indicates expanding real estate market
"""

    report += f"""
---

## Key Findings Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Returns Correlation | {r['correlations'].get(f"{r['indicator_name']}_MoM_vs_{r['target_name']}_Returns", {}).get('correlation', 0):.3f} | {'Weak' if abs(r['correlations'].get(f"{r['indicator_name']}_MoM_vs_{r['target_name']}_Returns", {}).get('correlation', 0)) < 0.15 else 'Moderate' if abs(r['correlations'].get(f"{r['indicator_name']}_MoM_vs_{r['target_name']}_Returns", {}).get('correlation', 0)) < 0.30 else 'Strong'} |
| P-value | {r['correlations'].get(f"{r['indicator_name']}_MoM_vs_{r['target_name']}_Returns", {}).get('pvalue', 1):.3f} | {'Significant' if r['correlations'].get(f"{r['indicator_name']}_MoM_vs_{r['target_name']}_Returns", {}).get('pvalue', 1) < 0.05 else 'Not significant'} |
| Optimal Lag | {r['optimal_lag']['lag']} months | {'Indicator leads target' if r['optimal_lag']['lag'] > 0 else 'Concurrent' if r['optimal_lag']['lag'] == 0 else 'Target leads indicator'} |
| Regime Difference | {'Significant' if r['regime_test'].get('significant', False) else 'Not significant'} | p={r['regime_test'].get('pvalue', np.nan):.3f} |

**Fast-Fail**: {'Yes - relationship too weak for practical use' if r.get('fast_fail', False) else 'No - continued to full analysis'}

---

## Correlation Analysis

"""

    for key, corr in r['correlations'].items():
        if not np.isnan(corr['correlation']):
            report += f"- **{key}**: r={corr['correlation']:.3f}, p={corr['pvalue']:.3f}, n={corr['n']}"
            if corr['significant']:
                report += " ✓ Significant"
            if corr['effect_size_adequate']:
                report += " ✓ Effect size adequate"
            report += "\n"

    report += f"""
---

## Lead-Lag Analysis

| Lag (months) | Correlation | P-value | Significant |
|--------------|-------------|---------|-------------|
"""

    if 'leadlag' in r and not r['leadlag'].empty:
        for _, row in r['leadlag'].iterrows():
            if not np.isnan(row['correlation']):
                sig = "✓" if row['significant'] else ""
                report += f"| {int(row['lag'])} | {row['correlation']:.3f} | {row['pvalue']:.3f} | {sig} |\n"

    report += f"""
**Optimal Lag**: {r['optimal_lag']['lag']} months (r={r['optimal_lag']['correlation']:.3f}, p={r['optimal_lag']['pvalue']:.3f})

---

## Regime Analysis

"""

    if 'regime_performance' in r and not r['regime_performance'].empty:
        report += "| Regime | Mean Monthly Return | Sharpe Ratio | N Months |\n"
        report += "|--------|---------------------|--------------|----------|\n"
        for _, row in r['regime_performance'].iterrows():
            report += f"| {row['regime']} | {row['mean_return']*100:.2f}% | {row['sharpe_ratio']:.2f} | {row['n_periods']} |\n"

    report += f"""
**Regime Difference Test**: p={r['regime_test'].get('pvalue', np.nan):.3f} ({'Significant' if r['regime_test'].get('significant', False) else 'Not significant'})

---

## Backtest Results

| Metric | Strategy | Benchmark |
|--------|----------|-----------|
| Sharpe Ratio | {r['backtest']['strategy_sharpe']:.2f} | {r['backtest']['benchmark_sharpe']:.2f} |
| Total Return | {r['backtest']['strategy_total_return']*100:.1f}% | {r['backtest']['benchmark_total_return']*100:.1f}% |
| Annualized Return | {r['backtest']['strategy_annualized']*100:.1f}% | {r['backtest']['benchmark_annualized']*100:.1f}% |

**Strategy**: Long {r['target_name']} when {r['indicator_name']} is falling (YoY), cash otherwise.

---

## Conclusion

**{r['conclusion']}**

### Practical Implications

"""

    if r.get('fast_fail', False):
        report += f"""1. **Do NOT use {r['indicator_name']} as a trading signal for {r['target_name']}**
2. The relationship is too weak to provide actionable information
3. Consider other housing indicators or sector-specific data
"""
    elif r['regime_test'].get('significant', False):
        report += f"""1. **{r['indicator_name']} shows potential as a regime indicator for {r['target_name']}**
2. Consider incorporating into a broader investment framework
3. Use with caution - single indicator signals are generally not sufficient alone
"""
    else:
        report += f"""1. **{r['indicator_name']} has limited value for timing {r['target_name']}**
2. Economically intuitive pattern exists but lacks statistical significance
3. May be useful as one input among many, but not standalone
"""

    report += f"""
---

## Files Created

| File | Description |
|------|-------------|
| `data/{r['target_name'].lower()}_{r['indicator_name'].lower().replace(' ', '_')}.parquet` | Analysis data |
| `docs/analysis_reports/{r['target_name'].lower()}_{r['indicator_name'].lower().replace(' ', '_')}_analysis.md` | This document |

---

## Appendix: Audit Trail

- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Random Seed**: {r['random_seed']}
- **SOP Version**: 1.2
- **Data Period**: {r['data_start'].strftime('%Y-%m-%d')} to {r['data_end'].strftime('%Y-%m-%d')}
- **Observations**: {r['n_observations']}
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\n📝 Report saved to: {output_path}")


def main():
    """Run full analysis for XLRE vs housing indicators."""

    print("=" * 70)
    print("XLRE vs Housing Indicators - Full SOP v1.2 Analysis")
    print("=" * 70)

    # Fetch XLRE data
    print("\nFetching XLRE data...")
    xlre_df = fetch_yahoo_data("XLRE", start="2000-01-01")
    xlre_df.columns = ['XLRE']

    print(f"  XLRE data range: {xlre_df.index[0].strftime('%Y-%m-%d')} to {xlre_df.index[-1].strftime('%Y-%m-%d')}")

    # Analysis 1: XLRE vs New Home Sales (HSN1F)
    print("\n" + "="*70)
    print("ANALYSIS 1: XLRE vs New Home Sales")
    print("="*70)

    try:
        # Fetch New Home Sales data using flexible method
        print("  Fetching New Home Sales (HSN1F)...")
        newhomesales = fetch_fred_data("HSN1F", start="2000-01-01")

        newhomesales_df = pd.DataFrame(newhomesales, columns=['NewHomeSales'])

        results_nhs = run_full_analysis(
            indicator_name="NewHomeSales",
            indicator_series=newhomesales,
            target_name="XLRE",
            target_df=xlre_df,
            random_seed=42
        )

        # Save data
        if 'data' in results_nhs:
            output_file = DATA_DIR / "xlre_newhomesales.parquet"
            results_nhs['data'].to_parquet(output_file)
            print(f"\n💾 Data saved to: {output_file}")

        # Generate report
        report_path = DOCS_DIR / "xlre_newhomesales_analysis.md"
        generate_report(results_nhs, report_path)

    except Exception as e:
        print(f"  ❌ Error in New Home Sales analysis: {e}")
        results_nhs = None

    # Analysis 2: XLRE vs Building Permits (PERMIT)
    print("\n" + "="*70)
    print("ANALYSIS 2: XLRE vs Building Permits")
    print("="*70)

    try:
        # Fetch Building Permits data (available in cached data)
        print("  Fetching Building Permits (PERMIT)...")
        permits = fetch_fred_data("PERMIT", start="2000-01-01")

        results_bp = run_full_analysis(
            indicator_name="BuildingPermits",
            indicator_series=permits,
            target_name="XLRE",
            target_df=xlre_df,
            random_seed=42
        )

        # Save data
        if 'data' in results_bp:
            output_file = DATA_DIR / "xlre_buildingpermits.parquet"
            results_bp['data'].to_parquet(output_file)
            print(f"\n💾 Data saved to: {output_file}")

        # Generate report
        report_path = DOCS_DIR / "xlre_buildingpermits_analysis.md"
        generate_report(results_bp, report_path)

    except Exception as e:
        print(f"  ❌ Error in Building Permits analysis: {e}")
        results_bp = None

    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    if results_nhs:
        print(f"\n1. XLRE vs New Home Sales: {results_nhs['conclusion']}")
    if results_bp:
        print(f"\n2. XLRE vs Building Permits: {results_bp['conclusion']}")

    return results_nhs, results_bp


if __name__ == "__main__":
    main()
