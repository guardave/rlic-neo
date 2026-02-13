# XLI vs ISM PMI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two independent analyses (XLI vs ISM Manufacturing PMI, XLI vs ISM Services PMI) to the RLIC dashboard following SOP v1.3.

**Architecture:** Each analysis is fully independent — own script, own data files, own dashboard handlers, own report. Manufacturing runs first; lessons inform Services. Both follow the existing template pattern from `analyze_xlre_newhomesales_full.py`.

**Tech Stack:** Python 3, pandas, scipy, yfinance, urllib (FRED), Streamlit dashboard, Docker

**Design doc:** `docs/plans/2026-02-13-xli-ism-pmi-design.md`

---

## PART 1: XLI vs ISM Manufacturing PMI (`xli_ism_mfg`)

### Task 1: Create analysis script

**Files:**
- Create: `script/analyze_xli_ism_mfg.py`

**Step 1: Write the analysis script**

Create `script/analyze_xli_ism_mfg.py` based on the template `script/analyze_xlre_newhomesales_full.py` with these changes:

- FRED series: Try `NAPM` first, fallback to `MANEMP`, then `MMNRNJ`
- Yahoo ticker: `XLI` (not `XLRE`)
- Column prefix: `ISM_Mfg_PMI_` (not `NewHomeSales_`)
- Returns column: `XLI_Returns` (not `XLRE_Returns`)
- Lead-lag range: -18 to +18 months (standard, not extended 24)
- Regime definition: PMI > 50 = "Mfg Expansion", PMI <= 50 = "Mfg Contraction" (level-based, not YoY direction-based)
- `OPTIMAL_LAG`: Set to `None` initially — determine from lead-lag results
- Output files: `data/xli_ism_mfg_full.parquet`, `data/xli_ism_mfg_leadlag.parquet`, `data/xli_ism_mfg_correlation.parquet`, `data/xli_ism_mfg_regimes.parquet`

Key differences from template:
1. **FRED fallback chain**: Try multiple series IDs
2. **PMI-specific regime**: Use level threshold (50) not YoY direction
3. **No hardcoded optimal lag**: Discover from data, then set

```python
#!/usr/bin/env python3
"""
Full Analysis: XLI vs ISM Manufacturing PMI
SOP v1.3 — Seven-phase pipeline.

FRED series: NAPM (ISM Manufacturing: PMI Composite Index)
Target: XLI (Industrial Select Sector SPDR Fund)
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

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MAX_LAG = 18  # Standard range for ISM PMI

# FRED series fallback chain for ISM Manufacturing PMI
ISM_MFG_SERIES = ['NAPM', 'MANEMP', 'MMNRNJ']


def fetch_fred_data_url(series_id: str, start: str = "1998-01-01") -> pd.Series:
    """Fetch data from FRED via direct URL. Returns empty Series on failure."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    print(f"  Trying FRED series: {series_id}...")
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            csv_data = response.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))
        date_col = None
        for col in ['observation_date', 'DATE', df.columns[0]]:
            if col in df.columns:
                date_col = col
                break
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        result = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
        if len(result) < 30:
            print(f"    {series_id}: Only {len(result)} observations (need 30+), skipping")
            return pd.Series()
        print(f"    {series_id}: OK, {len(result)} observations")
        return result
    except Exception as e:
        print(f"    {series_id}: FAILED ({e})")
        return pd.Series()


def fetch_ism_mfg_pmi(start: str = "1998-01-01") -> pd.Series:
    """Try each FRED series in fallback chain until one works."""
    for series_id in ISM_MFG_SERIES:
        data = fetch_fred_data_url(series_id, start)
        if not data.empty:
            print(f"  Using FRED series: {series_id}")
            return data
    raise ValueError(f"All ISM Manufacturing PMI series failed: {ISM_MFG_SERIES}")


def fetch_yahoo_data(ticker: str, start: str = "1998-01-01") -> pd.DataFrame:
    """Fetch price data from Yahoo Finance."""
    print(f"  Fetching {ticker} from Yahoo Finance...")
    data = yf.download(ticker, start=start, progress=False)
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
    result[f'{name}_Direction'] = np.sign(result[f'{name}_MoM'])
    return result


def correlation_analysis(df, x_cols, y_col):
    """Compute correlation matrix."""
    results = []
    for x_col in x_cols:
        mask = ~(df[x_col].isna() | df[y_col].isna())
        if mask.sum() >= 30:
            r, p = stats.pearsonr(df.loc[mask, x_col], df.loc[mask, y_col])
            results.append({
                'indicator': x_col, 'target': y_col,
                'correlation': r, 'pvalue': p, 'n': mask.sum(),
                'significant': p < 0.05
            })
    return pd.DataFrame(results)


def lead_lag_analysis(df, x_col, y_col, max_lag=18):
    """Lead-lag analysis from -max_lag to +max_lag."""
    results = []
    for lag in range(-max_lag, max_lag + 1):
        x_lagged = df[x_col].shift(lag)
        y = df[y_col]
        mask = ~(x_lagged.isna() | y.isna())
        if mask.sum() >= 30:
            r, p = stats.pearsonr(x_lagged[mask], y[mask])
            results.append({
                'lag': lag, 'correlation': r, 'pvalue': p,
                'n': mask.sum(), 'significant': p < 0.05
            })
    return pd.DataFrame(results)


def regime_analysis(df, indicator_level_col, target_col, optimal_lag=None):
    """Regime analysis using PMI > 50 threshold."""
    df = df.copy()
    if optimal_lag and optimal_lag > 0:
        df['indicator_for_regime'] = df[indicator_level_col].shift(optimal_lag)
    else:
        df['indicator_for_regime'] = df[indicator_level_col]

    df['regime'] = df['indicator_for_regime'].apply(
        lambda x: 'Mfg Expansion' if x > 50 else ('Mfg Contraction' if x <= 50 else np.nan)
    )

    regime_stats = []
    for regime in ['Mfg Expansion', 'Mfg Contraction']:
        mask = df['regime'] == regime
        if mask.sum() >= 10:
            returns = df.loc[mask, target_col]
            regime_stats.append({
                'regime': regime,
                'mean_return': returns.mean(),
                'std_return': returns.std(),
                'sharpe': returns.mean() / returns.std() * np.sqrt(12) if returns.std() > 0 else 0,
                'count': mask.sum()
            })

    regime_df = pd.DataFrame(regime_stats)

    # t-test for regime difference
    exp = df.loc[df['regime'] == 'Mfg Expansion', target_col].dropna()
    con = df.loc[df['regime'] == 'Mfg Contraction', target_col].dropna()
    if len(exp) >= 10 and len(con) >= 10:
        t_stat, p_value = stats.ttest_ind(exp, con)
    else:
        t_stat, p_value = np.nan, np.nan

    return {
        'regime_performance': regime_df,
        'regime_test': {'t_stat': t_stat, 'pvalue': p_value,
                        'significant': p_value < 0.05 if not np.isnan(p_value) else False},
        'data_with_regimes': df
    }


def run_backtest(df, indicator_level_col, target_col, optimal_lag=None):
    """Backtest: Long when PMI > 50 (with lag), cash otherwise."""
    df = df.copy()
    if optimal_lag and optimal_lag > 0:
        df['indicator_for_signal'] = df[indicator_level_col].shift(optimal_lag)
    else:
        df['indicator_for_signal'] = df[indicator_level_col]

    df['signal'] = (df['indicator_for_signal'] > 50).astype(int)
    df['position'] = df['signal'].shift(1)  # 1-month execution delay
    df['strategy_return'] = df['position'] * df[target_col]
    df['benchmark_return'] = df[target_col]
    df = df.dropna(subset=['strategy_return', 'benchmark_return'])
    df['strategy_cumret'] = (1 + df['strategy_return']).cumprod() - 1
    df['benchmark_cumret'] = (1 + df['benchmark_return']).cumprod() - 1

    n_years = len(df) / 12
    strategy_total = df['strategy_cumret'].iloc[-1] if len(df) > 0 else 0
    benchmark_total = df['benchmark_cumret'].iloc[-1] if len(df) > 0 else 0
    strategy_sharpe = df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(12) if df['strategy_return'].std() > 0 else 0
    benchmark_sharpe = df['benchmark_return'].mean() / df['benchmark_return'].std() * np.sqrt(12) if df['benchmark_return'].std() > 0 else 0

    return {
        'strategy_sharpe': strategy_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'strategy_total_return': strategy_total * 100,
        'benchmark_total_return': benchmark_total * 100,
        'backtest_data': df
    }


def main():
    print("=" * 70)
    print("FULL ANALYSIS: XLI vs ISM Manufacturing PMI (SOP v1.3)")
    print("=" * 70)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Lead-Lag Range: -{MAX_LAG} to +{MAX_LAG} months")
    print()

    # Phase 1: Data Preparation
    print("Phase 1: Data Preparation")
    print("-" * 40)

    ism_mfg = fetch_ism_mfg_pmi(start="1998-01-01")
    ism_monthly = create_monthly_data(ism_mfg, "ISM_Mfg_PMI")

    xli_df = fetch_yahoo_data("XLI", start="1998-01-01")
    xli_monthly = create_monthly_data(xli_df['price'], "XLI")
    xli_monthly['XLI_Returns'] = xli_monthly['XLI_Level'].pct_change(1)

    df = pd.concat([ism_monthly, xli_monthly], axis=1).dropna()
    print(f"  Data period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    print(f"  Total observations: {len(df)}")
    print()

    # Phase 2: Correlation Analysis
    print("Phase 2: Correlation Analysis (Concurrent)")
    print("-" * 40)

    x_cols = ['ISM_Mfg_PMI_MoM', 'ISM_Mfg_PMI_QoQ', 'ISM_Mfg_PMI_YoY', 'ISM_Mfg_PMI_Level']
    corr_results = correlation_analysis(df, x_cols, 'XLI_Returns')

    for _, row in corr_results.iterrows():
        sig = 'SIGNIFICANT' if row['significant'] else ''
        print(f"  {row['indicator']}: r={row['correlation']:.3f}, p={row['pvalue']:.4f} {sig}")
    print()

    # Phase 3: Lead-Lag Analysis
    print(f"Phase 3: Lead-Lag Analysis (-{MAX_LAG} to +{MAX_LAG} months)")
    print("-" * 40)

    leadlag_results = lead_lag_analysis(df, 'ISM_Mfg_PMI_YoY', 'XLI_Returns', max_lag=MAX_LAG)

    # Find best lag (by p-value)
    best_idx = leadlag_results['pvalue'].idxmin()
    best_lag = leadlag_results.loc[best_idx]
    sig_lags = leadlag_results[leadlag_results['significant']]

    print(f"  Best lag: {int(best_lag['lag'])}")
    print(f"    Correlation: r={best_lag['correlation']:.3f}")
    print(f"    P-value: {best_lag['pvalue']:.4f}")
    print()

    # Significant positive lags (predictive)
    sig_positive = sig_lags[sig_lags['lag'] > 0]
    sig_negative = sig_lags[sig_lags['lag'] < 0]

    if len(sig_positive) > 0:
        print("  Significant PREDICTIVE lags (positive, indicator leads):")
        for _, row in sig_positive.iterrows():
            print(f"    Lag +{int(row['lag'])}: r={row['correlation']:.3f}, p={row['pvalue']:.4f}")
    else:
        print("  NO significant predictive lags")

    if len(sig_negative) > 0:
        print("  Significant REVERSE lags (negative, target leads):")
        for _, row in sig_negative.iterrows():
            print(f"    Lag {int(row['lag'])}: r={row['correlation']:.3f}, p={row['pvalue']:.4f}")

    print()

    # Fast-fail decision (SOP v1.3)
    best_abs_r = leadlag_results['correlation'].abs().max()
    best_p = leadlag_results.loc[leadlag_results['correlation'].abs().idxmax(), 'pvalue']

    if best_abs_r < 0.10 and best_p > 0.30:
        print("FAST-FAIL: Best |r| < 0.10 AND p > 0.30 across ALL lags")
        print("Skipping Phases 4-5, proceeding to documentation")
        optimal_lag = None
        fast_fail = True
    elif len(sig_positive) == 0 and len(sig_lags) > 0:
        print("NOTE: Significant lags exist only at NEGATIVE lags (reverse causality)")
        print("Proceeding to Phase 4 but flagging as potentially not actionable")
        optimal_lag = int(best_lag['lag']) if best_lag['significant'] else None
        fast_fail = False
    else:
        optimal_lag = int(best_lag['lag']) if best_lag['significant'] else None
        fast_fail = False
        if optimal_lag is not None:
            print(f"  OPTIMAL LAG: {optimal_lag}")

    print()

    # Phase 4: Regime Analysis
    if not fast_fail:
        print(f"Phase 4: Regime Analysis (PMI > 50 threshold, lag={optimal_lag})")
        print("-" * 40)

        regime_results = regime_analysis(df, 'ISM_Mfg_PMI_Level', 'XLI_Returns',
                                         optimal_lag=optimal_lag)

        print("  Regime Performance:")
        for _, row in regime_results['regime_performance'].iterrows():
            print(f"    {row['regime']}: mean={row['mean_return']*100:.2f}%, "
                  f"Sharpe={row['sharpe']:.2f}, n={int(row['count'])}")

        if not np.isnan(regime_results['regime_test']['pvalue']):
            sig = 'SIGNIFICANT' if regime_results['regime_test']['significant'] else 'not significant'
            print(f"  Regime difference test: p={regime_results['regime_test']['pvalue']:.4f} ({sig})")
        print()

        # Phase 5: Backtest
        print(f"Phase 5: Backtest (lag={optimal_lag})")
        print("-" * 40)

        backtest_results = run_backtest(df, 'ISM_Mfg_PMI_Level', 'XLI_Returns',
                                        optimal_lag=optimal_lag)

        print(f"  Strategy Sharpe: {backtest_results['strategy_sharpe']:.2f}")
        print(f"  Benchmark Sharpe: {backtest_results['benchmark_sharpe']:.2f}")
        print(f"  Strategy Total Return: {backtest_results['strategy_total_return']:.1f}%")
        print(f"  Benchmark Total Return: {backtest_results['benchmark_total_return']:.1f}%")
        print()
    else:
        regime_results = None
        backtest_results = None

    # Phase 6: Dashboard Data Preparation
    print("Phase 6: Dashboard Data Preparation")
    print("-" * 40)

    dashboard_df = df[['ISM_Mfg_PMI_Level', 'ISM_Mfg_PMI_MoM', 'ISM_Mfg_PMI_QoQ',
                        'ISM_Mfg_PMI_YoY', 'ISM_Mfg_PMI_Direction',
                        'XLI_Level', 'XLI_Returns']].copy()

    if optimal_lag is not None and optimal_lag > 0:
        dashboard_df['ISM_Mfg_PMI_Level_Lagged'] = dashboard_df['ISM_Mfg_PMI_Level'].shift(optimal_lag)
    else:
        dashboard_df['ISM_Mfg_PMI_Level_Lagged'] = dashboard_df['ISM_Mfg_PMI_Level']

    dashboard_df['Regime'] = dashboard_df['ISM_Mfg_PMI_Level_Lagged'].apply(
        lambda x: 'Mfg Expansion' if x > 50 else ('Mfg Contraction' if x <= 50 else np.nan)
    )

    if backtest_results is not None:
        bt_data = backtest_results['backtest_data']
        dashboard_df['Strategy_Return'] = bt_data.get('strategy_return')
        dashboard_df['Strategy_CumRet'] = bt_data.get('strategy_cumret')
        dashboard_df['Benchmark_CumRet'] = bt_data.get('benchmark_cumret')

    output_file = DATA_DIR / "xli_ism_mfg_full.parquet"
    dashboard_df.to_parquet(output_file)
    print(f"  Saved: {output_file}")

    leadlag_file = DATA_DIR / "xli_ism_mfg_leadlag.parquet"
    leadlag_results.to_parquet(leadlag_file)
    print(f"  Saved: {leadlag_file}")

    corr_file = DATA_DIR / "xli_ism_mfg_correlation.parquet"
    corr_results.to_parquet(corr_file)
    print(f"  Saved: {corr_file}")

    if regime_results is not None:
        regime_file = DATA_DIR / "xli_ism_mfg_regimes.parquet"
        regime_results['regime_performance'].to_parquet(regime_file)
        print(f"  Saved: {regime_file}")

    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    if fast_fail:
        print("RESULT: FAST-FAIL - No significant relationship at any lag")
        print("ACTIONABLE: NO")
    elif optimal_lag is not None and best_lag['significant']:
        print(f"Key Findings:")
        print(f"  1. Best lag: {optimal_lag} (r={best_lag['correlation']:.3f}, p={best_lag['pvalue']:.4f})")
        if backtest_results:
            print(f"  2. Strategy Sharpe: {backtest_results['strategy_sharpe']:.2f} "
                  f"vs Benchmark: {backtest_results['benchmark_sharpe']:.2f}")
        if regime_results:
            sig_text = "SIGNIFICANT" if regime_results['regime_test']['significant'] else "not significant"
            print(f"  3. Regime difference: {sig_text} (p={regime_results['regime_test']['pvalue']:.4f})")
        print()
        actionable = "YES" if best_lag['lag'] > 0 else "NO (reverse causality)"
        print(f"ACTIONABLE: {actionable}")
    else:
        print("RESULT: No statistically significant relationship found")
        print("ACTIONABLE: NO")

    print()
    print("Files created:")
    print(f"  - {output_file}")
    print(f"  - {leadlag_file}")
    print(f"  - {corr_file}")
    if regime_results:
        print(f"  - {DATA_DIR / 'xli_ism_mfg_regimes.parquet'}")


if __name__ == "__main__":
    main()
```

**Step 2: Run the analysis script**

Run: `cd /home/david/knows/rlic && python3 script/analyze_xli_ism_mfg.py`

Expected: Script outputs Phase 1-6 results. Check:
- Did FRED series resolve? (Which series ID worked?)
- How many observations?
- What's the best lag and is it significant?
- Does the fast-fail trigger?

**Step 3: Evaluate results and record optimal lag**

Based on the script output:
- If **fast-fail triggered**: Skip Tasks 3-5 for regime/backtest, but still add to dashboard as a documented negative result
- If **significant positive lag found**: Note the optimal lag value — it will be needed for dashboard handlers
- If **only negative lags significant**: Document as reverse causality

**Step 4: Commit analysis script and data**

```bash
git add script/analyze_xli_ism_mfg.py data/xli_ism_mfg_*.parquet
git commit --author="RA Cheryl <ra-cheryl@idficient.com>" -m "Add XLI vs ISM Manufacturing PMI analysis script and data

Phase 1-6 of SOP v1.3. Results: [fill in based on output]

🤖 Agent: RA Cheryl

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Add to dashboard navigation and data loader

**Files:**
- Modify: `src/dashboard/navigation.py:50-56` (after xlre_newhomesales entry)
- Modify: `src/dashboard/data_loader.py:587` (in existing_files dict)

**Step 1: Add navigation entry**

In `src/dashboard/navigation.py`, add after the `xlre_newhomesales` entry (before the closing `}`):

```python
    'xli_ism_mfg': {
        'name': 'XLI vs ISM Manufacturing PMI',
        'icon': '🏭',
        'short': 'XLI-MFG',
        'description': 'Industrials sector vs ISM Manufacturing PMI'
    }
```

**Step 2: Add data loader mapping**

In `src/dashboard/data_loader.py`, add to the `existing_files` dict (line ~587):

```python
        'xli_ism_mfg': 'xli_ism_mfg_full.parquet',
```

**Step 3: Verify syntax**

Run: `python3 -c "from src.dashboard.navigation import ANALYSES; print(len(ANALYSES), 'analyses')"`
Expected: `8 analyses`

Run: `python3 -c "from src.dashboard.data_loader import load_analysis_data; print('OK')"`
Expected: `OK`

---

### Task 3: Update Home page

**Files:**
- Modify: `src/dashboard/Home.py:35-43` (cards list)
- Modify: `src/dashboard/Home.py:70` (analysis count metric)

**Step 1: Add card to Home page**

In `src/dashboard/Home.py`, add to the `cards` list (line ~43, before the `]`):

```python
    ('xli_ism_mfg', col2, "ISM Mfg PMI • XLI • Industrials"),
```

**Step 2: Update count metric**

Change line ~70 from:
```python
stat_cols[0].metric("Analyses", "7")
```
to:
```python
stat_cols[0].metric("Analyses", "8")
```

---

### Task 4: Add column detection to Overview page

**Files:**
- Modify: `src/dashboard/pages/2_📊_Overview.py:87-93` (add elif block)

**Step 1: Add column detection handler**

In `src/dashboard/pages/2_📊_Overview.py`, add before the `else:` fallback (after the xlre_newhomesales block, around line 93):

```python
    elif analysis_id == 'xli_ism_mfg':
        indicator_cols = [c for c in data.columns if 'ISM_Mfg_PMI' in c and ('Level' in c or 'YoY' in c)]
        return_cols = ['XLI_Returns'] if 'XLI_Returns' in data.columns else []
        if not indicator_cols:
            indicator_cols = ['ISM_Mfg_PMI_Level'] if 'ISM_Mfg_PMI_Level' in data.columns else []
        if 'Regime' in data.columns and 'regime' not in data.columns:
            data['regime'] = data['Regime']
```

---

### Task 5: Add qualitative content

**Files:**
- Modify: `src/dashboard/pages/3_📖_Qualitative.py:665` (add elif block before else)

**Step 1: Add qualitative content for ISM Manufacturing PMI**

In `src/dashboard/pages/3_📖_Qualitative.py`, add before the `else:` fallback (after the xlre_newhomesales block, around line 665):

```python
elif analysis_id == 'xli_ism_mfg':
    st.header("XLI vs ISM Manufacturing PMI (NAPM)")

    st.subheader("What is ISM Manufacturing PMI?")
    st.markdown("""
    The **ISM Manufacturing PMI** (Purchasing Managers' Index) is a monthly survey of 300+ manufacturing
    purchasing managers conducted by the Institute for Supply Management. Published on the **1st business
    day** of each month, making it one of the earliest economic releases.

    **Key Characteristics:**
    - **Diffusion Index**: PMI > 50 = expansion, PMI = 50 = no change, PMI < 50 = contraction
    - **Sub-components**: New Orders, Production, Employment, Supplier Deliveries, Inventories
    - **Leading Property**: Survey data captures real-time sentiment before hard production data
    - **History**: Available since 1948, one of the longest-running economic surveys
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Economic Signal Interpretation")
        signal_table = pd.DataFrame({
            'PMI Level': ['Above 50', 'Below 50', 'Above 55', 'Below 45'],
            'Interpretation': [
                'Manufacturing sector expanding',
                'Manufacturing sector contracting',
                'Strong expansion, capacity pressure building',
                'Severe contraction, recession risk elevated'
            ]
        })
        st.dataframe(signal_table, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Why XLI (Industrials)?")
        st.markdown("""
        **XLI Composition:** Boeing, Caterpillar, Honeywell, GE Aerospace, Union Pacific,
        Lockheed Martin, RTX, Deere & Co.

        **Direct Link**: XLI holdings ARE the manufacturers being surveyed by ISM.
        When purchasing managers report expansion, XLI companies are seeing stronger orders.

        **Timing Advantage**: PMI is released before industrial production data (INDPRO),
        giving an earlier signal about manufacturing health.
        """)

    st.subheader("Academic and Professional Research")
    research = pd.DataFrame({
        'Finding': [
            'ISM PMI leads industrial production by 1-2 months',
            'PMI above 50 correlates with GDP growth',
            'New Orders sub-index is most predictive component',
            'PMI is published before most hard economic data',
            'ISM PMI is watched as a recession early warning'
        ],
        'Source': ['Federal Reserve', 'ISM Research', 'Stock & Watson', 'BLS Calendar', 'Advisor Perspectives'],
        'Implication': [
            'Potential leading indicator for industrial stocks',
            'Strong macro signal for economic cycle positioning',
            'Sub-components may add value beyond headline PMI',
            'Information advantage for trading strategies',
            'Useful for risk management'
        ]
    })
    st.dataframe(research, hide_index=True, use_container_width=True)

    st.subheader("Limitations")
    st.markdown("""
    1. **Survey Bias**: Self-reported data from purchasing managers, not objective measurement
    2. **Diffusion vs Magnitude**: PMI measures breadth (% reporting growth) not magnitude of growth
    3. **Manufacturing Focus**: Only ~12% of GDP is manufacturing; services dominate the modern economy
    4. **Market Pricing**: PMI is closely watched; significant moves may already be priced in
    5. **Seasonal Adjustments**: ISM applies its own seasonal adjustments which can introduce noise
    """)
```

---

### Task 6: Add column detection to Correlation page

**Files:**
- Modify: `src/dashboard/pages/4_📈_Correlation.py:92-97` (add elif block)

**Step 1: Add handler**

In `src/dashboard/pages/4_📈_Correlation.py`, add before the `else:` (after xlre_newhomesales block):

```python
    elif analysis_id == 'xli_ism_mfg':
        indicator_cols = [c for c in data.columns if 'ISM_Mfg_PMI' in c and ('Level' in c or 'YoY' in c)]
        return_cols = ['XLI_Returns'] if 'XLI_Returns' in data.columns else []
        if not indicator_cols:
            indicator_cols = ['ISM_Mfg_PMI_Level'] if 'ISM_Mfg_PMI_Level' in data.columns else []
```

---

### Task 7: Add column detection to Lead-Lag page

**Files:**
- Modify: `src/dashboard/pages/5_🔄_Lead_Lag.py:91-95` (add elif block)

**Step 1: Add handler**

In `src/dashboard/pages/5_🔄_Lead_Lag.py`, add before the `else:` (after xlre_newhomesales block):

```python
    elif analysis_id == 'xli_ism_mfg':
        indicator_cols = [c for c in data.columns if 'ISM_Mfg_PMI' in c and ('Level' in c or 'YoY' in c)]
        return_cols = ['XLI_Returns'] if 'XLI_Returns' in data.columns else []
        if not indicator_cols:
            indicator_cols = ['ISM_Mfg_PMI_Level'] if 'ISM_Mfg_PMI_Level' in data.columns else []
```

Note: Lead-Lag slider default stays at 12 for this analysis (ISM is faster-moving than housing).

---

### Task 8: Add column detection to Regimes page

**Files:**
- Modify: `src/dashboard/pages/6_🎯_Regimes.py:102-108` (add elif block)

**Step 1: Add handler**

In `src/dashboard/pages/6_🎯_Regimes.py`, add before the `else:` (after xlre_newhomesales block):

```python
    elif analysis_id == 'xli_ism_mfg':
        indicator_cols = ['ISM_Mfg_PMI_Level_Lagged'] if 'ISM_Mfg_PMI_Level_Lagged' in data.columns else ['ISM_Mfg_PMI_Level']
        return_cols = ['XLI_Returns'] if 'XLI_Returns' in data.columns else []
        if 'Regime' in data.columns and 'regime' not in data.columns:
            data['regime'] = data['Regime']
```

---

### Task 9: Add column detection to Backtests page

**Files:**
- Modify: `src/dashboard/pages/7_💰_Backtests.py:103-109` (add elif block)

**Step 1: Add handler**

In `src/dashboard/pages/7_💰_Backtests.py`, add before the `else:` (after xlre_newhomesales block):

```python
    elif analysis_id == 'xli_ism_mfg':
        indicator_cols = ['ISM_Mfg_PMI_Level_Lagged'] if 'ISM_Mfg_PMI_Level_Lagged' in data.columns else ['ISM_Mfg_PMI_Level']
        return_cols = ['XLI_Returns'] if 'XLI_Returns' in data.columns else []
        if 'Regime' in data.columns and 'regime' not in data.columns:
            data['regime'] = data['Regime']
```

---

### Task 10: Write analysis report

**Files:**
- Create: `docs/analysis_reports/xli_ism_mfg_analysis.md`

**Step 1: Write report based on script output**

Use the template structure from `docs/analysis_reports/xlre_newhomesales_analysis.md`:

1. Overview
2. Qualitative Analysis
3. Key Findings Summary (table)
4. Detailed Analysis (correlation, lead-lag, regime, backtest)
5. Actionable Recommendations (YES/NO with rationale)
6. Files Created

Fill in actual numbers from the script output (Task 1 Step 2).

---

### Task 11: Test via Docker and commit dashboard changes

**Step 1: Rebuild and test Docker**

```bash
cd /home/david/knows/rlic
docker compose -f docker-compose.dev.yml up -d --build
```

Wait for healthy status, then verify:
```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8501
```
Expected: `200`

**Step 2: Commit all dashboard changes**

```bash
git add src/dashboard/ docs/analysis_reports/xli_ism_mfg_analysis.md
git commit --author="RA Cheryl <ra-cheryl@idficient.com>" -m "Add XLI vs ISM Mfg PMI to dashboard (all 7 pages)

- Navigation: xli_ism_mfg entry with 🏭 icon
- Home: Card added, count updated to 8
- All 6 pages: Column detection handlers for ISM_Mfg_PMI columns
- Qualitative: Full economic rationale for ISM Mfg PMI + XLI
- Analysis report: Complete Phase 0-7 documentation
Docker tested: health check passed

🤖 Agent: RA Cheryl

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## PART 2: XLI vs ISM Services PMI (`xli_ism_svc`)

### Task 12: Create Services PMI analysis script

**Files:**
- Create: `script/analyze_xli_ism_svc.py`

**Step 1: Copy and adapt from Manufacturing script**

Key differences from `analyze_xli_ism_mfg.py`:
- FRED series: Try `NMFBAI` first, fallback to `ISMNSA`
- Column prefix: `ISM_Svc_PMI_` (not `ISM_Mfg_PMI_`)
- Regime labels: `Svc Expansion` / `Svc Contraction` (not `Mfg`)
- Output files: `data/xli_ism_svc_*.parquet`
- Start date: `2008-01-01` (ISM Services data starts ~1997 but NMFBAI may start later)
- Leverage findings from Manufacturing analysis (if Manufacturing found lag +N, check if Services shows similar)

**Step 2: Run the analysis**

```bash
python3 script/analyze_xli_ism_svc.py
```

**Step 3: Evaluate and commit**

Same evaluation criteria as Task 1 Step 3.

```bash
git add script/analyze_xli_ism_svc.py data/xli_ism_svc_*.parquet
git commit --author="RA Cheryl <ra-cheryl@idficient.com>" -m "Add XLI vs ISM Services PMI analysis script and data

🤖 Agent: RA Cheryl

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 13: Add Services PMI to dashboard (all files)

**Files to modify (same pattern as Tasks 2-9 but for xli_ism_svc):**
- `src/dashboard/navigation.py` — add `xli_ism_svc` entry
- `src/dashboard/data_loader.py` — add file mapping
- `src/dashboard/Home.py` — add card, update count to 9
- `src/dashboard/pages/2_📊_Overview.py` — add column detection
- `src/dashboard/pages/3_📖_Qualitative.py` — add full qualitative content for ISM Services PMI
- `src/dashboard/pages/4_📈_Correlation.py` — add column detection
- `src/dashboard/pages/5_🔄_Lead_Lag.py` — add column detection
- `src/dashboard/pages/6_🎯_Regimes.py` — add column detection (use `ISM_Svc_PMI_Level_Lagged`, regime labels `Svc Expansion`/`Svc Contraction`)
- `src/dashboard/pages/7_💰_Backtests.py` — add column detection

**Navigation entry:**
```python
    'xli_ism_svc': {
        'name': 'XLI vs ISM Services PMI',
        'icon': '🏢',
        'short': 'XLI-SVC',
        'description': 'Industrials sector vs ISM Services PMI'
    }
```

**Qualitative content key points for Services PMI:**
- ISM Non-Manufacturing (Services) PMI surveys services sector purchasing managers
- Published 3rd business day of month (2 days after Manufacturing)
- Services = ~80% of GDP, so captures broader economy
- Less direct link to XLI (Industrials are manufacturers, not services)
- Hypothesis: May show weaker relationship than Manufacturing PMI

**Column detection pattern (all pages):**
```python
    elif analysis_id == 'xli_ism_svc':
        indicator_cols = [c for c in data.columns if 'ISM_Svc_PMI' in c and ('Level' in c or 'YoY' in c)]
        return_cols = ['XLI_Returns'] if 'XLI_Returns' in data.columns else []
        if not indicator_cols:
            indicator_cols = ['ISM_Svc_PMI_Level'] if 'ISM_Svc_PMI_Level' in data.columns else []
```

For Regimes and Backtests pages, use `ISM_Svc_PMI_Level_Lagged` and set regime from `Regime` column.

---

### Task 14: Write Services PMI analysis report, test, and commit

**Files:**
- Create: `docs/analysis_reports/xli_ism_svc_analysis.md`

**Step 1: Write report based on script output**

Same template as Manufacturing report.

**Step 2: Docker test**

```bash
docker compose -f docker-compose.dev.yml up -d --build
curl -s -o /dev/null -w "%{http_code}" http://localhost:8501
```

**Step 3: Final commit**

```bash
git add src/dashboard/ docs/analysis_reports/xli_ism_svc_analysis.md
git commit --author="RA Cheryl <ra-cheryl@idficient.com>" -m "Add XLI vs ISM Services PMI to dashboard (9 analyses total)

- Navigation: xli_ism_svc entry with 🏢 icon
- All 7 pages updated with column detection handlers
- Qualitative: ISM Services PMI economic rationale
- Analysis report: Complete Phase 0-7 documentation
- Dashboard now has 9 analyses total

🤖 Agent: RA Cheryl

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 15: Push and update status board

**Step 1: Push all commits**

```bash
git push origin main
```

**Step 2: Update status board**

Add entry to `_pws/_team/status-board.md` with results of both analyses.

---

## Summary of All Files

| Action | File |
|--------|------|
| Create | `script/analyze_xli_ism_mfg.py` |
| Create | `script/analyze_xli_ism_svc.py` |
| Create | `data/xli_ism_mfg_*.parquet` (4 files) |
| Create | `data/xli_ism_svc_*.parquet` (4 files) |
| Create | `docs/analysis_reports/xli_ism_mfg_analysis.md` |
| Create | `docs/analysis_reports/xli_ism_svc_analysis.md` |
| Modify | `src/dashboard/navigation.py` |
| Modify | `src/dashboard/data_loader.py` |
| Modify | `src/dashboard/Home.py` |
| Modify | `src/dashboard/pages/2_📊_Overview.py` |
| Modify | `src/dashboard/pages/3_📖_Qualitative.py` |
| Modify | `src/dashboard/pages/4_📈_Correlation.py` |
| Modify | `src/dashboard/pages/5_🔄_Lead_Lag.py` |
| Modify | `src/dashboard/pages/6_🎯_Regimes.py` |
| Modify | `src/dashboard/pages/7_💰_Backtests.py` |
| Modify | `_pws/_team/status-board.md` |
