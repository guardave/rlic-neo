#!/usr/bin/env python3
"""
Full Analysis: XLI vs ISM Manufacturing PMI
Following SOP v1.3 with lead-lag analysis (-18 to +18 months).

Data sources for ISM Manufacturing PMI:
  1. FRED (NAPM) - discontinued ~2016 due to ISM licensing
  2. forecasts.org - historical data (1948 to ~2014)
  3. DBnomics (ISM/pmi/pm) - recent data (2020+)
  4. ycharts - most recent 50 months
  5. Gap period (2014-09 to 2020-04) from ISM press releases

Key features:
- ISM PMI uses level threshold (50) for regimes, NOT YoY direction
- Bilateral lead-lag range (-18 to +18)
- Optimal lag discovered from analysis, not preset
- Fast-fail (SOP v1.3): skip phases 4-5 if best |r| < 0.10 AND p > 0.30

Phases:
- Phase 0: Qualitative Analysis (documented in report)
- Phase 1: Data Preparation
- Phase 2: Statistical Analysis (correlation matrix)
- Phase 3: Lead-Lag Analysis (-18 to +18 months)
- Phase 3b: Fast-fail check
- Phase 4: Regime Analysis (if not fast-failed)
- Phase 5: Backtesting (if not fast-failed)
- Phase 6: Dashboard Data Preparation
- Phase 7: Documentation
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
import re
import json

warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Analysis parameters
MAX_LAG = 18  # Lead-lag range: -18 to +18 months

# Regime threshold: PMI > 50 = expansion, <= 50 = contraction
PMI_THRESHOLD = 50.0

# ISM Manufacturing PMI: gap period values (2014-09 to 2020-04)
# Sources: ISM press releases, verified against Trading Economics, MacroMicro, Reuters
# These are the official ISM Manufacturing PMI Composite Index values
ISM_PMI_GAP_VALUES = {
    '2014-09-01': 56.6, '2014-10-01': 59.0, '2014-11-01': 58.7, '2014-12-01': 55.5,
    '2015-01-01': 53.5, '2015-02-01': 52.9, '2015-03-01': 51.5, '2015-04-01': 51.5,
    '2015-05-01': 52.8, '2015-06-01': 53.5, '2015-07-01': 52.7, '2015-08-01': 51.1,
    '2015-09-01': 50.2, '2015-10-01': 50.1, '2015-11-01': 48.6, '2015-12-01': 48.0,
    '2016-01-01': 48.2, '2016-02-01': 49.5, '2016-03-01': 51.8, '2016-04-01': 50.8,
    '2016-05-01': 51.3, '2016-06-01': 53.2, '2016-07-01': 52.6, '2016-08-01': 49.4,
    '2016-09-01': 51.5, '2016-10-01': 51.9, '2016-11-01': 53.2, '2016-12-01': 54.7,
    '2017-01-01': 56.0, '2017-02-01': 57.7, '2017-03-01': 57.2, '2017-04-01': 54.8,
    '2017-05-01': 54.9, '2017-06-01': 57.8, '2017-07-01': 56.3, '2017-08-01': 58.8,
    '2017-09-01': 60.8, '2017-10-01': 58.7, '2017-11-01': 58.2, '2017-12-01': 59.3,
    '2018-01-01': 59.1, '2018-02-01': 60.8, '2018-03-01': 59.3, '2018-04-01': 57.3,
    '2018-05-01': 58.7, '2018-06-01': 60.2, '2018-07-01': 58.1, '2018-08-01': 61.3,
    '2018-09-01': 59.8, '2018-10-01': 57.7, '2018-11-01': 59.3, '2018-12-01': 54.3,
    '2019-01-01': 56.6, '2019-02-01': 54.2, '2019-03-01': 55.3, '2019-04-01': 52.8,
    '2019-05-01': 52.1, '2019-06-01': 51.7, '2019-07-01': 51.2, '2019-08-01': 49.1,
    '2019-09-01': 47.8, '2019-10-01': 48.3, '2019-11-01': 48.1, '2019-12-01': 47.2,
    '2020-01-01': 50.9, '2020-02-01': 50.1, '2020-03-01': 49.1, '2020-04-01': 41.5,
}


def fetch_fred_napm(start: str = "1998-01-01") -> pd.Series:
    """Try to fetch NAPM (ISM PMI) from FRED. May fail if series is discontinued."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=NAPM&cosd={start}"
    print(f"    Trying FRED NAPM...")
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            csv_data = response.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        series = pd.to_numeric(df['NAPM'], errors='coerce').dropna()
        if len(series) > 0 and series.min() > 20 and series.max() < 80:
            print(f"    FRED NAPM: {len(series)} obs ({series.index[0].strftime('%Y-%m')} to {series.index[-1].strftime('%Y-%m')})")
            return series
    except Exception as e:
        print(f"    FRED NAPM failed: {e}")
    return pd.Series(dtype=float)


def fetch_forecasts_org() -> pd.Series:
    """Fetch ISM PMI historical data from forecasts.org (JavaScript data)."""
    url = 'https://www.forecasts.org/data/data/NAPM.htm'
    print(f"    Trying forecasts.org...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8')
        matches = re.findall(r"\['(\d{4}-\d{2}-\d{2})',\s*([\d.]+)\]", html)
        if matches:
            data = [(m[0], float(m[1])) for m in matches]
            df = pd.DataFrame(data, columns=['date', 'ISM_PMI'])
            df['date'] = pd.to_datetime(df['date'])
            series = df.set_index('date')['ISM_PMI'].sort_index()
            print(f"    forecasts.org: {len(series)} obs ({series.index[0].strftime('%Y-%m')} to {series.index[-1].strftime('%Y-%m')})")
            return series
    except Exception as e:
        print(f"    forecasts.org failed: {e}")
    return pd.Series(dtype=float)


def fetch_dbnomics() -> pd.Series:
    """Fetch ISM Manufacturing PMI from DBnomics API."""
    url = 'https://api.db.nomics.world/v22/series/ISM/pmi/pm?observations=1'
    print(f"    Trying DBnomics (ISM/pmi/pm)...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        doc = data['series']['docs'][0]
        df = pd.DataFrame({'date': doc['period'], 'ISM_PMI': doc['value']})
        df['date'] = pd.to_datetime(df['date'])
        df['ISM_PMI'] = pd.to_numeric(df['ISM_PMI'], errors='coerce')
        df = df.dropna()
        # Filter out corrupted values (ISM PMI is always 25-75 range)
        df = df[(df['ISM_PMI'] > 20) & (df['ISM_PMI'] < 80)]
        series = df.set_index('date')['ISM_PMI'].sort_index()
        print(f"    DBnomics: {len(series)} obs ({series.index[0].strftime('%Y-%m')} to {series.index[-1].strftime('%Y-%m')})")
        return series
    except Exception as e:
        print(f"    DBnomics failed: {e}")
    return pd.Series(dtype=float)


def fetch_ycharts() -> pd.Series:
    """Fetch recent ISM PMI data from ycharts.com (most recent ~50 months)."""
    url = 'https://ycharts.com/indicators/us_pmi'
    print(f"    Trying ycharts...")
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8')
        matches = re.findall(r'(\w+ \d{1,2}, \d{4})\s*</td>\s*<td[^>]*>\s*([\d.]+)', html)
        if matches:
            df = pd.DataFrame([(m[0], float(m[1])) for m in matches], columns=['date', 'ISM_PMI'])
            df['date'] = pd.to_datetime(df['date'])
            series = df.set_index('date')['ISM_PMI'].sort_index()
            print(f"    ycharts: {len(series)} obs ({series.index[0].strftime('%Y-%m')} to {series.index[-1].strftime('%Y-%m')})")
            return series
    except Exception as e:
        print(f"    ycharts failed: {e}")
    return pd.Series(dtype=float)


def get_gap_data() -> pd.Series:
    """Return hardcoded ISM PMI values for the gap period (2014-09 to 2020-04)."""
    series = pd.Series(ISM_PMI_GAP_VALUES, dtype=float)
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    print(f"    Gap data (hardcoded): {len(series)} obs ({series.index[0].strftime('%Y-%m')} to {series.index[-1].strftime('%Y-%m')})")
    return series


def fetch_ism_manufacturing_pmi(start: str = "1998-01-01") -> tuple:
    """
    Fetch ISM Manufacturing PMI from multiple sources and combine.
    Returns (combined_series, source_description).

    Assembly strategy:
    1. forecasts.org for historical (1948 to ~2014)
    2. Hardcoded gap values (2014-09 to 2020-04) from ISM press releases
    3. DBnomics for 2020+ data
    4. ycharts for most recent months (takes priority in overlaps)
    """
    print("  Fetching ISM Manufacturing PMI (multi-source):")

    sources_used = []
    all_series = []

    # Source 1: FRED NAPM (likely discontinued)
    fred = fetch_fred_napm(start)
    if len(fred) > 0:
        all_series.append(('fred', fred))
        sources_used.append(f"FRED NAPM ({len(fred)} obs)")

    # Source 2: forecasts.org (historical)
    forg = fetch_forecasts_org()
    if len(forg) > 0:
        all_series.append(('forecasts_org', forg))
        sources_used.append(f"forecasts.org ({len(forg)} obs)")

    # Source 3: Hardcoded gap
    gap = get_gap_data()
    all_series.append(('gap', gap))
    sources_used.append(f"ISM press releases ({len(gap)} obs)")

    # Source 4: DBnomics
    dbn = fetch_dbnomics()
    if len(dbn) > 0:
        all_series.append(('dbnomics', dbn))
        sources_used.append(f"DBnomics ({len(dbn)} obs)")

    # Source 5: ycharts (highest priority for recent data)
    yc = fetch_ycharts()
    if len(yc) > 0:
        all_series.append(('ycharts', yc))
        sources_used.append(f"ycharts ({len(yc)} obs)")

    # Combine: later sources take priority for overlapping dates
    # Resample each to month-start dates for consistency
    combined = pd.Series(dtype=float)
    for name, s in all_series:
        # Normalize to first of month
        s_normalized = s.copy()
        s_normalized.index = s_normalized.index.to_period('M').to_timestamp()
        # Update (later sources overwrite earlier for same date)
        combined = combined.combine_first(s_normalized) if name in ('forecasts_org', 'fred') else \
                   s_normalized.combine_first(combined) if name == 'gap' else \
                   s_normalized.combine_first(combined)

    # Actually, use a clearer priority: ycharts > dbnomics > gap > forecasts_org > fred
    # Build from lowest to highest priority
    combined = pd.Series(dtype=float)
    priority_order = [('fred', fred)] if len(fred) > 0 else []
    priority_order += [('forecasts_org', forg)] if len(forg) > 0 else []
    priority_order += [('gap', gap)]
    priority_order += [('dbnomics', dbn)] if len(dbn) > 0 else []
    priority_order += [('ycharts', yc)] if len(yc) > 0 else []

    for name, s in priority_order:
        s_norm = s.copy()
        s_norm.index = s_norm.index.to_period('M').to_timestamp()
        # Higher priority overwrites lower
        for date, val in s_norm.items():
            combined[date] = val

    combined = combined.sort_index()

    # Filter to start date
    start_dt = pd.Timestamp(start)
    combined = combined[combined.index >= start_dt]

    source_desc = " + ".join(sources_used)
    print(f"  Combined: {len(combined)} observations ({combined.index[0].strftime('%Y-%m')} to {combined.index[-1].strftime('%Y-%m')})")
    print(f"  Sources: {source_desc}")
    print(f"  PMI range: {combined.min():.1f} to {combined.max():.1f}")
    print(f"  Below 50 (contraction): {(combined <= 50).sum()} months ({(combined <= 50).sum()/len(combined)*100:.0f}%)")

    return combined, source_desc


def fetch_yahoo_data(ticker: str, start: str = "1998-01-01") -> pd.DataFrame:
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


def lead_lag_analysis(df: pd.DataFrame, x_col: str, y_col: str,
                      max_lag: int = 18) -> pd.DataFrame:
    """
    Lead-lag analysis from -max_lag to +max_lag.
    Positive lag means indicator leads target (predictive).
    Negative lag means target leads indicator.
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


def check_fast_fail(leadlag_results: pd.DataFrame) -> bool:
    """
    SOP v1.3 fast-fail: After Phase 3, check if best |r| < 0.10 AND p > 0.30
    across ALL lags. If so, skip Phases 4-5.
    Returns True if fast-fail triggered.
    """
    if len(leadlag_results) == 0:
        return True

    best_abs_r = leadlag_results['abs_correlation'].max()

    # Check: is the minimum p-value across all lags > 0.30?
    min_p = leadlag_results['pvalue'].min()

    triggered = (best_abs_r < 0.10) and (min_p > 0.30)
    return triggered


def regime_analysis(df: pd.DataFrame, indicator_level_col: str, target_col: str,
                    optimal_lag: int, threshold: float = 50.0) -> dict:
    """
    Regime analysis with optimal lag.
    Regimes defined by PMI level vs threshold (50).
    """
    df = df.copy()
    df['indicator_lagged'] = df[indicator_level_col].shift(optimal_lag)

    # Define regimes based on PMI level threshold
    df['regime'] = df['indicator_lagged'].apply(
        lambda x: 'Mfg Expansion' if x > threshold else (
            'Mfg Contraction' if x <= threshold and pd.notna(x) else np.nan
        )
    )

    # Calculate regime performance
    regime_stats = []
    for regime in ['Mfg Expansion', 'Mfg Contraction']:
        mask = df['regime'] == regime
        if mask.sum() >= 5:
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
    expansion = df.loc[df['regime'] == 'Mfg Expansion', target_col].dropna()
    contraction = df.loc[df['regime'] == 'Mfg Contraction', target_col].dropna()

    if len(expansion) >= 10 and len(contraction) >= 10:
        t_stat, p_value = stats.ttest_ind(expansion, contraction)
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


def run_backtest(df: pd.DataFrame, indicator_level_col: str, target_col: str,
                 optimal_lag: int, threshold: float = 50.0) -> dict:
    """
    Simple backtest: Long when PMI (lagged) > threshold, cash otherwise.
    1-month execution delay.
    """
    df = df.copy()
    df['indicator_lagged'] = df[indicator_level_col].shift(optimal_lag)

    # Signal: 1 when PMI > threshold (expansion), 0 when contraction
    df['signal'] = (df['indicator_lagged'] > threshold).astype(int)

    # Apply 1-month delay for execution
    df['position'] = df['signal'].shift(1)

    # Calculate returns
    df['strategy_return'] = df['position'] * df[target_col]
    df['benchmark_return'] = df[target_col]

    # Drop NaN
    df = df.dropna(subset=['strategy_return', 'benchmark_return'])

    if len(df) == 0:
        return {
            'strategy_sharpe': 0,
            'benchmark_sharpe': 0,
            'strategy_total_return': 0,
            'benchmark_total_return': 0,
            'strategy_annualized': 0,
            'benchmark_annualized': 0,
            'exposure': 0,
            'backtest_data': df
        }

    # Cumulative returns
    df['strategy_cumret'] = (1 + df['strategy_return']).cumprod() - 1
    df['benchmark_cumret'] = (1 + df['benchmark_return']).cumprod() - 1

    # Calculate metrics
    n_years = len(df) / 12

    strategy_total = df['strategy_cumret'].iloc[-1]
    benchmark_total = df['benchmark_cumret'].iloc[-1]

    strategy_ann = (1 + strategy_total) ** (1/n_years) - 1 if n_years > 0 else 0
    benchmark_ann = (1 + benchmark_total) ** (1/n_years) - 1 if n_years > 0 else 0

    strategy_sharpe = (df['strategy_return'].mean() / df['strategy_return'].std()
                       * np.sqrt(12)) if df['strategy_return'].std() > 0 else 0
    benchmark_sharpe = (df['benchmark_return'].mean() / df['benchmark_return'].std()
                        * np.sqrt(12)) if df['benchmark_return'].std() > 0 else 0

    # Exposure (fraction of time invested)
    exposure = df['position'].mean()

    return {
        'strategy_sharpe': strategy_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'strategy_total_return': strategy_total * 100,
        'benchmark_total_return': benchmark_total * 100,
        'strategy_annualized': strategy_ann * 100,
        'benchmark_annualized': benchmark_ann * 100,
        'exposure': exposure * 100,
        'backtest_data': df
    }


def main():
    print("=" * 70)
    print("FULL ANALYSIS: XLI vs ISM Manufacturing PMI (SOP v1.3)")
    print("=" * 70)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Lead-Lag Range: -{MAX_LAG} to +{MAX_LAG} months")
    print(f"Regime Threshold: PMI > {PMI_THRESHOLD} = Expansion")
    print()

    # ================================================================
    # Phase 1: Data Preparation
    # ================================================================
    print("Phase 1: Data Preparation")
    print("-" * 40)

    # Fetch ISM Manufacturing PMI (multi-source)
    ism_data, source_desc = fetch_ism_manufacturing_pmi(start="1998-01-01")
    ism_monthly = create_monthly_data(ism_data, "ISM_Mfg_PMI")

    # Fetch XLI
    xli_df = fetch_yahoo_data("XLI", start="1998-01-01")
    xli_monthly = create_monthly_data(xli_df['price'], "XLI")
    xli_monthly['XLI_Returns'] = xli_monthly['XLI_Level'].pct_change(1)

    # Merge
    df = pd.concat([ism_monthly, xli_monthly], axis=1).dropna()

    print(f"  Data period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    print(f"  Total observations: {len(df)}")
    print(f"  ISM PMI range in merged data: {df['ISM_Mfg_PMI_Level'].min():.1f} to {df['ISM_Mfg_PMI_Level'].max():.1f}")
    print()

    # Sanity check: PMI should have values both above and below 50
    above_50 = (df['ISM_Mfg_PMI_Level'] > PMI_THRESHOLD).sum()
    below_50 = (df['ISM_Mfg_PMI_Level'] <= PMI_THRESHOLD).sum()
    print(f"  PMI > 50 (expansion):   {above_50} months ({above_50/len(df)*100:.0f}%)")
    print(f"  PMI <= 50 (contraction): {below_50} months ({below_50/len(df)*100:.0f}%)")
    print()

    # ================================================================
    # Phase 2: Correlation Analysis
    # ================================================================
    print("Phase 2: Correlation Analysis (Concurrent)")
    print("-" * 40)

    x_cols = ['ISM_Mfg_PMI_Level', 'ISM_Mfg_PMI_MoM', 'ISM_Mfg_PMI_QoQ',
              'ISM_Mfg_PMI_YoY', 'ISM_Mfg_PMI_Direction']
    corr_results = correlation_analysis(df, x_cols, 'XLI_Returns')

    for _, row in corr_results.iterrows():
        sig = '*' if row['significant'] else ''
        print(f"  {row['indicator']}: r={row['correlation']:.3f}, p={row['pvalue']:.4f} {sig}")

    print()

    # ================================================================
    # Phase 3: Lead-Lag Analysis
    # ================================================================
    print(f"Phase 3: Lead-Lag Analysis (-{MAX_LAG} to +{MAX_LAG} months)")
    print("-" * 40)

    # Use Level for lead-lag (since regimes are level-based)
    leadlag_results = lead_lag_analysis(df, 'ISM_Mfg_PMI_Level', 'XLI_Returns',
                                         max_lag=MAX_LAG)

    # Find best lag by absolute correlation
    best_idx = leadlag_results['abs_correlation'].idxmax()
    best_lag_row = leadlag_results.loc[best_idx]
    sig_lags = leadlag_results[leadlag_results['significant']]

    print(f"  Best lag: {int(best_lag_row['lag']):+d} months")
    print(f"    |r| = {best_lag_row['abs_correlation']:.3f}, r = {best_lag_row['correlation']:.3f}")
    print(f"    P-value: {best_lag_row['pvalue']:.4f}")
    print(f"    N: {int(best_lag_row['n'])}")
    print()

    if len(sig_lags) > 0:
        print(f"  Significant lags (p < 0.05): {len(sig_lags)} found")
        for _, row in sig_lags.sort_values('abs_correlation', ascending=False).head(10).iterrows():
            print(f"    Lag {int(row['lag']):+3d}: r={row['correlation']:.3f}, "
                  f"p={row['pvalue']:.4f}")
    else:
        print("  No significant lags found (p < 0.05)")

    print()

    # ================================================================
    # Phase 3b: Fast-fail check (SOP v1.3)
    # ================================================================
    print("Phase 3b: Fast-Fail Check (SOP v1.3)")
    print("-" * 40)

    fast_fail = check_fast_fail(leadlag_results)
    best_abs_r = leadlag_results['abs_correlation'].max()
    min_p = leadlag_results['pvalue'].min()

    print(f"  Best |r| across all lags: {best_abs_r:.4f}")
    print(f"  Min p-value across all lags: {min_p:.4f}")
    print(f"  Fast-fail criteria: |r| < 0.10 AND p > 0.30")
    print(f"  Fast-fail triggered: {'YES - skipping Phases 4-5' if fast_fail else 'NO - proceeding'}")
    print()

    # Determine optimal lag for subsequent phases
    # Use the lag with highest |r| among significant lags; if none significant, use best overall
    if len(sig_lags) > 0:
        optimal_lag_row = sig_lags.loc[sig_lags['abs_correlation'].idxmax()]
        optimal_lag = int(optimal_lag_row['lag'])
    else:
        optimal_lag = int(best_lag_row['lag'])

    print(f"  Optimal lag for phases 4-5: {optimal_lag:+d} months")
    print()

    # ================================================================
    # Phase 4: Regime Analysis (skip if fast-fail)
    # ================================================================
    regime_results = None
    if not fast_fail:
        print(f"Phase 4: Regime Analysis (using lag {optimal_lag:+d}, threshold={PMI_THRESHOLD})")
        print("-" * 40)

        regime_results = regime_analysis(df, 'ISM_Mfg_PMI_Level', 'XLI_Returns',
                                          optimal_lag=optimal_lag, threshold=PMI_THRESHOLD)

        print("  Regime Performance:")
        for _, row in regime_results['regime_performance'].iterrows():
            print(f"    {row['regime']}: mean={row['mean_return']*100:.2f}%/mo, "
                  f"Sharpe={row['sharpe']:.2f}, n={int(row['count'])}")

        if not np.isnan(regime_results['regime_test']['pvalue']):
            sig = '*' if regime_results['regime_test']['significant'] else ''
            print(f"  Regime difference test: t={regime_results['regime_test']['t_stat']:.2f}, "
                  f"p={regime_results['regime_test']['pvalue']:.4f} {sig}")

        print()
    else:
        print("Phase 4: SKIPPED (fast-fail)")
        print()

    # ================================================================
    # Phase 5: Backtest (skip if fast-fail)
    # ================================================================
    backtest_results = None
    if not fast_fail:
        print(f"Phase 5: Backtest (using lag {optimal_lag:+d}, PMI > {PMI_THRESHOLD} = Long)")
        print("-" * 40)

        backtest_results = run_backtest(df, 'ISM_Mfg_PMI_Level', 'XLI_Returns',
                                         optimal_lag=optimal_lag, threshold=PMI_THRESHOLD)

        print(f"  Strategy Sharpe:  {backtest_results['strategy_sharpe']:.2f}")
        print(f"  Benchmark Sharpe: {backtest_results['benchmark_sharpe']:.2f}")
        print(f"  Strategy Total Return:  {backtest_results['strategy_total_return']:.1f}%")
        print(f"  Benchmark Total Return: {backtest_results['benchmark_total_return']:.1f}%")
        print(f"  Strategy Ann. Return:   {backtest_results['strategy_annualized']:.1f}%")
        print(f"  Benchmark Ann. Return:  {backtest_results['benchmark_annualized']:.1f}%")
        print(f"  Exposure (% time in market): {backtest_results['exposure']:.1f}%")

        print()
    else:
        print("Phase 5: SKIPPED (fast-fail)")
        print()

    # ================================================================
    # Phase 6: Dashboard Data Preparation
    # ================================================================
    print("Phase 6: Dashboard Data Preparation")
    print("-" * 40)

    # Create dashboard-ready data file
    dashboard_df = df[['ISM_Mfg_PMI_Level', 'ISM_Mfg_PMI_MoM', 'ISM_Mfg_PMI_QoQ',
                        'ISM_Mfg_PMI_YoY', 'ISM_Mfg_PMI_Direction',
                        'XLI_Level', 'XLI_Returns']].copy()

    # Add lagged indicator (use optimal lag if found, else 0)
    effective_lag = optimal_lag if not fast_fail else 0
    dashboard_df['ISM_Mfg_PMI_Level_Lagged'] = dashboard_df['ISM_Mfg_PMI_Level'].shift(effective_lag)

    # Define regimes based on lagged PMI level
    dashboard_df['Regime'] = dashboard_df['ISM_Mfg_PMI_Level_Lagged'].apply(
        lambda x: 'Mfg Expansion' if x > PMI_THRESHOLD else (
            'Mfg Contraction' if x <= PMI_THRESHOLD and pd.notna(x) else np.nan
        )
    )

    # Add backtest columns if available
    if backtest_results is not None:
        bt_data = backtest_results['backtest_data']
        dashboard_df['Strategy_Return'] = bt_data['strategy_return']
        dashboard_df['Strategy_CumRet'] = bt_data['strategy_cumret']
        dashboard_df['Benchmark_CumRet'] = bt_data['benchmark_cumret']
    else:
        dashboard_df['Strategy_Return'] = np.nan
        dashboard_df['Strategy_CumRet'] = np.nan
        dashboard_df['Benchmark_CumRet'] = np.nan

    # Save main data file
    output_file = DATA_DIR / "xli_ism_mfg_full.parquet"
    dashboard_df.to_parquet(output_file)
    print(f"  Saved: {output_file}")

    # Save lead-lag results
    leadlag_file = DATA_DIR / "xli_ism_mfg_leadlag.parquet"
    leadlag_results.to_parquet(leadlag_file)
    print(f"  Saved: {leadlag_file}")

    # Save correlation results
    corr_file = DATA_DIR / "xli_ism_mfg_correlation.parquet"
    corr_results.to_parquet(corr_file)
    print(f"  Saved: {corr_file}")

    # Save regime performance (even if empty)
    regime_file = DATA_DIR / "xli_ism_mfg_regimes.parquet"
    if regime_results is not None:
        regime_results['regime_performance'].to_parquet(regime_file)
    else:
        pd.DataFrame(columns=['regime', 'mean_return', 'std_return', 'sharpe', 'count']).to_parquet(regime_file)
    print(f"  Saved: {regime_file}")

    print()

    # ================================================================
    # Summary
    # ================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  Data sources: {source_desc}")
    print(f"  Observations: {len(df)}")
    print(f"  Period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    print()
    print(f"  Best lag: {int(best_lag_row['lag']):+d} months")
    print(f"  Best |r|: {best_lag_row['abs_correlation']:.3f}")
    print(f"  Best r: {best_lag_row['correlation']:.3f}")
    print(f"  Best p-value: {best_lag_row['pvalue']:.4f}")
    print(f"  Fast-fail: {'YES' if fast_fail else 'NO'}")
    print()

    if not fast_fail:
        print("  Regime Performance:")
        if regime_results is not None:
            for _, row in regime_results['regime_performance'].iterrows():
                print(f"    {row['regime']}: mean={row['mean_return']*100:.2f}%/mo, "
                      f"Sharpe={row['sharpe']:.2f}, n={int(row['count'])}")
            if not np.isnan(regime_results['regime_test']['pvalue']):
                print(f"    Regime difference p={regime_results['regime_test']['pvalue']:.4f}")
        print()

        if backtest_results is not None:
            print("  Backtest:")
            print(f"    Strategy Sharpe:  {backtest_results['strategy_sharpe']:.2f}")
            print(f"    Benchmark Sharpe: {backtest_results['benchmark_sharpe']:.2f}")
            print(f"    Strategy Total:   {backtest_results['strategy_total_return']:.1f}%")
            print(f"    Benchmark Total:  {backtest_results['benchmark_total_return']:.1f}%")
            print(f"    Exposure:         {backtest_results['exposure']:.1f}%")
        print()

        # Actionable determination
        actionable = False
        if regime_results is not None and regime_results['regime_test']['significant']:
            actionable = True
        if backtest_results is not None and backtest_results['strategy_sharpe'] > backtest_results['benchmark_sharpe']:
            actionable = True

        print(f"  Actionable: {'YES' if actionable else 'NO'}")
        if actionable:
            print(f"    Use ISM Mfg PMI at lag {optimal_lag:+d} as trading signal for XLI")
            print(f"    Long when PMI > {PMI_THRESHOLD}, cash otherwise")
    else:
        print("  Actionable: NO (fast-fail triggered - no significant relationship found)")

    print()
    print("Files created:")
    print(f"  - {output_file}")
    print(f"  - {leadlag_file}")
    print(f"  - {corr_file}")
    print(f"  - {regime_file}")
    print()


if __name__ == "__main__":
    main()
