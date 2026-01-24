# Unified Time Series Analysis and Backtesting SOP

**Version:** 1.0
**Date:** 2026-01-24
**Author:** RA Cheryl

---

## Executive Summary

This Standard Operating Procedure (SOP) unifies multiple analysis frameworks into a comprehensive methodology for:
1. Analyzing relationships between economic indicators and asset returns
2. Validating investment strategies across economic regimes
3. Conducting rigorous backtesting with state-of-the-art methods
4. Presenting results through interactive dashboards

The framework synthesizes:
- Time Series Relationship Analysis Framework
- Investment Clock Sector Analysis Framework
- Cass Freight Index Analysis Methodology
- State-of-the-art backtesting techniques (Walk-Forward, Monte Carlo, etc.)

---

## Table of Contents

1. [Framework Overview](#1-framework-overview)
2. [Phase 0: Qualitative Analysis](#phase-0-qualitative-analysis)
3. [Phase 1: Data Preparation](#phase-1-data-preparation)
4. [Phase 2: Statistical Analysis](#phase-2-statistical-analysis)
5. [Phase 3: Lead-Lag and Causality](#phase-3-lead-lag-and-causality)
6. [Phase 4: Regime Analysis](#phase-4-regime-analysis)
7. [Phase 5: Backtesting Methodologies](#phase-5-backtesting-methodologies)
8. [Phase 6: Visualization and Dashboard](#phase-6-visualization-and-dashboard)
9. [Phase 7: Documentation](#phase-7-documentation)
10. [Appendices](#appendices)

---

## 1. Framework Overview

### 1.1 Analysis Types Supported

| Analysis Type | Indicators | Targets | Regimes | Use Case |
|--------------|------------|---------|---------|----------|
| **Single Indicator** | 1 | 1 | 2 (Rising/Falling) | Test predictive relationship |
| **Investment Clock** | 2 (Growth + Inflation) | Multiple | 4 Phases | Sector allocation validation |
| **Multi-Factor** | N | Multiple | Custom | Complex regime strategies |

### 1.2 Unified Process Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    UNIFIED ANALYSIS PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   PHASE 0    │ -> │   PHASE 1    │ -> │   PHASE 2    │          │
│  │  Qualitative │    │    Data      │    │  Statistical │          │
│  │   Analysis   │    │ Preparation  │    │   Analysis   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                                       │                   │
│         v                                       v                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   PHASE 3    │ <- │   PHASE 4    │ <- │   PHASE 5    │          │
│  │  Lead-Lag &  │    │    Regime    │    │  Backtesting │          │
│  │   Causality  │    │   Analysis   │    │              │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                                       │                   │
│         v                                       v                   │
│  ┌──────────────┐    ┌──────────────┐                              │
│  │   PHASE 6    │ -> │   PHASE 7    │                              │
│  │  Dashboard & │    │Documentation │                              │
│  │Visualization │    │   & Report   │                              │
│  └──────────────┘    └──────────────┘                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Qualitative Analysis

**CRITICAL: Always perform qualitative analysis BEFORE quantitative analysis.**

### 0.1 Indicator Definition

Document fundamental characteristics:

| Aspect | Description | Example |
|--------|-------------|---------|
| **Definition** | What does the indicator measure? | Ratio of new orders to inventories |
| **Source** | Who publishes it? | Federal Reserve, Census Bureau |
| **Frequency** | Monthly, weekly, daily? | Monthly |
| **Release Timing** | How long after reference period? | ~2-4 weeks |
| **Historical Range** | Record high, low, typical values | 0.8 - 1.5 |
| **Revisions** | Is data revised? How significantly? | Minor monthly revisions |

### 0.2 Literature Review

Conduct comprehensive review from multiple source types:

**Academic Sources:**
- Google Scholar, SSRN, JSTOR
- Seminal papers (Fama 1981, Chen-Roll-Ross 1986)
- Peer-reviewed methodology papers

**Professional Sources:**
- Federal Reserve research notes
- BofA/Merrill Lynch research
- Fidelity, Invesco sector studies
- IMF working papers

**Public Sources:**
- Financial media (Bloomberg, CNBC)
- Advisor Perspectives, industry blogs
- Real-time market commentary

### 0.3 Market Interpretation

Document how market participants use the indicator:

```markdown
#### How Investors Use [Indicator Name]

1. **Primary Signal**: What does rising/falling indicate?
2. **Secondary Uses**: Risk assessment, regime identification
3. **Combined With**: What other indicators complement this one?
4. **Thresholds**: Key levels considered significant
5. **Caution**: Known limitations or misinterpretations
```

### 0.4 Seasonality Analysis

**From Cass Freight Index methodology:**

1. **Seasonal Decomposition**
   - Model: Additive or Multiplicative (auto-select based on data)
   - Components: Trend + Seasonal + Residual
   - Measure: Standard deviation of seasonal component

2. **Autocorrelation Analysis**
   - Test lag 12 months for yearly patterns
   - Compare against 95% confidence interval
   - ACF > 0.5 at lag 12 indicates strong seasonality

3. **Monthly Means Analysis**
   - ANOVA test for significant monthly differences
   - Identify peak and trough months
   - Calculate month-over-month typical patterns

```python
def analyze_seasonality(series, freq=12):
    """
    Comprehensive seasonality analysis.

    Returns:
        dict with decomposition, autocorr, and monthly stats
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import acf
    from scipy import stats

    # Decomposition
    decomp = seasonal_decompose(series, model='additive', period=freq)
    seasonal_std = decomp.seasonal.std()

    # Autocorrelation at seasonal lag
    acf_values = acf(series.dropna(), nlags=freq)
    seasonal_acf = acf_values[freq]

    # Monthly means ANOVA
    monthly_groups = [series[series.index.month == m] for m in range(1, 13)]
    f_stat, p_value = stats.f_oneway(*[g.dropna() for g in monthly_groups])

    return {
        'decomposition': decomp,
        'seasonal_std': seasonal_std,
        'seasonal_acf': seasonal_acf,
        'anova_p_value': p_value,
        'is_seasonal': p_value < 0.05 and seasonal_acf > 0.3
    }
```

### 0.5 Key Insights Summary

| Finding | Source | Implication for Analysis |
|---------|--------|--------------------------|
| Indicator is leading | Literature | Can predict returns directly |
| Strong seasonality | Data analysis | Adjust for seasonal patterns |
| Relationship unstable over time | Academic paper | Use rolling windows |

---

## Phase 1: Data Preparation

### 1.1 Data Loading and Alignment

```python
def prepare_analysis_data(indicator_series, target_series, freq='ME'):
    """
    Align indicator and target series to common frequency.

    Args:
        indicator_series: Economic indicator (e.g., Orders/Inv Ratio)
        target_series: Target asset (e.g., SPY returns)
        freq: Target frequency ('ME' for month-end)

    Returns:
        DataFrame with aligned series
    """
    # Resample to target frequency
    indicator = indicator_series.resample(freq).last()
    target = target_series.resample(freq).last()

    # Combine and align
    df = pd.DataFrame({
        'indicator': indicator,
        'target': target
    })

    # Handle missing values
    df = df.dropna()

    return df
```

### 1.2 Derivative Series Creation

Create standard transformations for any time series:

| Derivative | Formula | Purpose |
|------------|---------|---------|
| **MoM** | `pct_change(1)` | Short-term momentum |
| **QoQ** | `pct_change(3)` | Medium-term trend |
| **YoY** | `pct_change(12)` | Long-term, seasonality-adjusted |
| **Direction** | `sign(change)` | Binary regime indicator |
| **Z-Score** | `(x - rolling_mean) / rolling_std` | Normalized level |
| **MA Crossover** | `3MA vs 6MA` | Trend direction signal |

```python
def create_derivatives(df, col, prefix):
    """Create standard derivative series."""
    series = df[col]

    # Percentage changes
    df[f'{prefix}_MoM'] = series.pct_change(1) * 100
    df[f'{prefix}_QoQ'] = series.pct_change(3) * 100
    df[f'{prefix}_YoY'] = series.pct_change(12) * 100

    # Direction indicators
    df[f'{prefix}_MoM_Dir'] = np.sign(df[f'{prefix}_MoM'])
    df[f'{prefix}_YoY_Dir'] = np.sign(df[f'{prefix}_YoY'])

    # Z-score (60-month rolling)
    rolling_mean = series.rolling(60).mean()
    rolling_std = series.rolling(60).std()
    df[f'{prefix}_ZScore'] = (series - rolling_mean) / rolling_std

    # MA Crossover
    df[f'{prefix}_3MA'] = series.rolling(3).mean()
    df[f'{prefix}_6MA'] = series.rolling(6).mean()
    df[f'{prefix}_MA_Signal'] = np.where(
        df[f'{prefix}_3MA'] > df[f'{prefix}_6MA'], 1, -1
    )

    return df
```

### 1.3 Sector Data Options

| Source | History | Sectors | Pros | Cons |
|--------|---------|---------|------|------|
| S&P Sector ETFs | 1998+ | 11 | Direct, tradeable | Short history |
| Fama-French 12 | 1926+ | 12 | Long history | Not exact S&P match |
| Fama-French 49 | 1926+ | 49 | Granular | Too many for clean analysis |

**Recommended:** Fama-French 12 Industries for historical analysis, S&P Sector ETFs for implementation.

---

## Phase 2: Statistical Analysis

### 2.1 Correlation Analysis

```python
from scipy import stats

def correlation_analysis(df, x_cols, y_cols):
    """
    Compute correlation matrix with significance testing.
    """
    results = []

    for x_col in x_cols:
        for y_col in y_cols:
            valid = df[[x_col, y_col]].dropna()
            if len(valid) < 30:
                continue

            corr, pval = stats.pearsonr(valid[x_col], valid[y_col])

            results.append({
                'X': x_col,
                'Y': y_col,
                'correlation': corr,
                'p_value': pval,
                'significant': pval < 0.05,
                'n_obs': len(valid)
            })

    return pd.DataFrame(results)
```

### 2.2 Correlation Interpretation Guide

| Correlation Type | Typical Finding | Interpretation |
|-----------------|-----------------|----------------|
| Level vs Level | Often high | Usually **spurious** (common trends) |
| Change vs Change | Lower but meaningful | **Contemporaneous** relationship |
| Direction vs Direction | Moderate | **Regime** relationship |

**Key Insight:** Always check if level correlation is spurious before drawing conclusions.

---

## Phase 3: Lead-Lag and Causality

### 3.1 Cross-Correlation Analysis

```python
def lead_lag_analysis(df, x_col, y_col, max_lag=12):
    """
    Test correlations at various leads and lags.

    Convention:
    - Negative lag = X leads Y (X at t predicts Y at t+|lag|)
    - Positive lag = Y leads X
    - Zero lag = Contemporaneous
    """
    results = []

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x = df[x_col].shift(-lag)  # X leads
            y = df[y_col]
        else:
            x = df[x_col]
            y = df[y_col].shift(-lag)  # Y leads

        valid = pd.DataFrame({'x': x, 'y': y}).dropna()
        if len(valid) < 30:
            continue

        corr, pval = stats.pearsonr(valid['x'], valid['y'])

        results.append({
            'lag': lag,
            'correlation': corr,
            'p_value': pval,
            'abs_corr': abs(corr),
            'interpretation': (
                'X leads Y' if lag < 0 else
                'Y leads X' if lag > 0 else
                'Contemporaneous'
            )
        })

    return pd.DataFrame(results)
```

### 3.2 Lead-Lag Interpretation

| Finding | Meaning | Implication |
|---------|---------|-------------|
| Peak at lag=0 | Contemporaneous | No predictive value |
| Peak at lag<0 | X leads Y | X may predict Y |
| Peak at lag>0 | Y leads X | Reverse causality |
| Flat across lags | No relationship | Series are independent |

### 3.3 Granger Causality Testing

```python
from statsmodels.tsa.stattools import grangercausalitytests

def granger_test(df, x_col, y_col, max_lag=6):
    """
    Test if X Granger-causes Y (and vice versa).
    """
    data = df[[y_col, x_col]].dropna()
    results = []

    # Test X -> Y
    try:
        gc = grangercausalitytests(
            data[[y_col, x_col]], maxlag=max_lag, verbose=False
        )
        for lag in range(1, max_lag + 1):
            f_stat = gc[lag][0]['ssr_ftest'][0]
            p_val = gc[lag][0]['ssr_ftest'][1]
            results.append({
                'direction': f'{x_col} -> {y_col}',
                'lag': lag,
                'f_statistic': f_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            })
    except:
        pass

    # Test Y -> X (bidirectional)
    try:
        gc = grangercausalitytests(
            data[[x_col, y_col]], maxlag=max_lag, verbose=False
        )
        for lag in range(1, max_lag + 1):
            f_stat = gc[lag][0]['ssr_ftest'][0]
            p_val = gc[lag][0]['ssr_ftest'][1]
            results.append({
                'direction': f'{y_col} -> {x_col}',
                'lag': lag,
                'f_statistic': f_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            })
    except:
        pass

    return pd.DataFrame(results)
```

---

## Phase 4: Regime Analysis

### 4.1 Single-Indicator Regimes

```python
def define_regimes(df, indicator_col, method='direction'):
    """
    Create regime indicators based on indicator.

    Methods:
    - 'median': Above/below median
    - 'direction': Positive/negative MoM change
    - 'ma_crossover': 3MA vs 6MA
    - 'zscore': High/normal/low based on z-score
    """
    if method == 'median':
        median = df[indicator_col].median()
        df['Regime'] = np.where(df[indicator_col] > median, 'High', 'Low')

    elif method == 'direction':
        change = df[indicator_col].pct_change()
        df['Regime'] = np.where(change > 0, 'Rising', 'Falling')

    elif method == 'ma_crossover':
        ma3 = df[indicator_col].rolling(3).mean()
        ma6 = df[indicator_col].rolling(6).mean()
        df['Regime'] = np.where(ma3 > ma6, 'Rising', 'Falling')

    elif method == 'zscore':
        zscore = (df[indicator_col] - df[indicator_col].rolling(60).mean()) / \
                  df[indicator_col].rolling(60).std()
        df['Regime'] = pd.cut(
            zscore, bins=[-np.inf, -1, 1, np.inf],
            labels=['Low', 'Normal', 'High']
        )

    return df
```

### 4.2 Investment Clock Phases (Two-Indicator)

```python
def classify_investment_clock_phase(growth_signal, inflation_signal):
    """
    Classify into Investment Clock phase.

    | Growth | Inflation | Phase       |
    |--------|-----------|-------------|
    | +1     | -1        | Recovery    |
    | +1     | +1        | Overheat    |
    | -1     | +1        | Stagflation |
    | -1     | -1        | Reflation   |
    """
    if growth_signal == 1 and inflation_signal == -1:
        return 'Recovery'
    elif growth_signal == 1 and inflation_signal == 1:
        return 'Overheat'
    elif growth_signal == -1 and inflation_signal == 1:
        return 'Stagflation'
    elif growth_signal == -1 and inflation_signal == -1:
        return 'Reflation'
    return np.nan
```

### 4.3 Regime Performance Analysis

```python
def regime_performance(df, regime_col, target_col):
    """
    Calculate target series statistics by regime.
    """
    results = []

    for regime in df[regime_col].dropna().unique():
        subset = df[df[regime_col] == regime][target_col].dropna()

        if len(subset) < 10:
            continue

        # Calculate metrics
        mean_ret = subset.mean()
        std_ret = subset.std()

        results.append({
            'regime': regime,
            'n_periods': len(subset),
            'mean_return': mean_ret * 100,
            'volatility': std_ret * 100,
            'sharpe': (mean_ret * 12) / (std_ret * np.sqrt(12)) if std_ret > 0 else 0,
            'ann_sharpe': (mean_ret * 12 - 0.02) / (std_ret * np.sqrt(12)) if std_ret > 0 else 0,
            'positive_pct': (subset > 0).mean() * 100,
            'median_return': subset.median() * 100
        })

    return pd.DataFrame(results)
```

---

## Phase 5: Backtesting Methodologies

### 5.1 Backtest Methodology Catalog

This section catalogs all available backtesting methods, from basic to state-of-the-art.

#### 5.1.1 Simple Historical Backtest

**Description:** Basic train-test split with fixed cutoff date.

```python
def simple_backtest(df, signal_col, return_col, train_end_date):
    """
    Simple historical backtest with train-test split.
    """
    train = df[df.index <= train_end_date]
    test = df[df.index > train_end_date]

    # Calculate strategy returns
    test['strategy_return'] = test[signal_col].shift(1) * test[return_col]

    return calculate_metrics(test['strategy_return'])
```

**Pros:** Simple, fast
**Cons:** Single test period, prone to overfitting

---

#### 5.1.2 Walk-Forward Validation (WFV)

**Description:** Rolling window optimization with out-of-sample testing.

**Reference:** [Interactive Brokers - Walk Forward Analysis](https://www.interactivebrokers.com/campus/ibkr-quant-news/the-future-of-backtesting-a-deep-dive-into-walk-forward-analysis/)

```python
def walk_forward_validation(df, signal_generator,
                            train_window=60, test_window=12,
                            step_size=12, purge_gap=3):
    """
    Walk-Forward Validation with purge gap.

    Args:
        df: DataFrame with features and target
        signal_generator: Function that generates signals from training data
        train_window: Training period in months
        test_window: Test period in months
        step_size: Step size between windows
        purge_gap: Gap between train and test to prevent lookahead

    Returns:
        DataFrame with OOS predictions and metrics
    """
    results = []

    for start in range(0, len(df) - train_window - purge_gap - test_window, step_size):
        # Define windows
        train_start = start
        train_end = start + train_window
        test_start = train_end + purge_gap
        test_end = test_start + test_window

        if test_end > len(df):
            break

        # Train on in-sample
        train_data = df.iloc[train_start:train_end]

        # Generate signal
        signal = signal_generator(train_data)

        # Test on out-of-sample
        test_data = df.iloc[test_start:test_end].copy()
        test_data['signal'] = signal
        test_data['strategy_return'] = test_data['signal'].shift(1) * test_data['return']

        results.append({
            'period': df.index[test_start],
            'train_sharpe': calculate_sharpe(train_data['strategy_return']),
            'test_sharpe': calculate_sharpe(test_data['strategy_return']),
            'test_return': test_data['strategy_return'].sum()
        })

    return pd.DataFrame(results)
```

**Walk-Forward Efficiency Ratio (WFER):**
```python
WFER = test_sharpe / train_sharpe
# WFER close to 1.0 indicates strategy generalizes well
# WFER < 0.5 indicates overfitting
```

---

#### 5.1.3 Combinatorial Purged Cross-Validation (CPCV)

**Description:** Lopez de Prado methodology for financial ML.

**Reference:** [Advances in Financial Machine Learning](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110)

```python
def combinatorial_purged_cv(df, n_splits=5, purge_gap=3, embargo_pct=0.01):
    """
    Combinatorial Purged Cross-Validation.

    Key features:
    - Purge gap prevents information leakage
    - Embargo period after test set
    - Combinatorial provides more test paths
    """
    from sklearn.model_selection import KFold

    n_samples = len(df)
    embargo = int(n_samples * embargo_pct)

    kf = KFold(n_splits=n_splits, shuffle=False)

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        # Apply purge gap
        test_start = test_idx[0]
        test_end = test_idx[-1]

        # Remove samples within purge gap before test
        train_idx = train_idx[train_idx < test_start - purge_gap]

        # Apply embargo after test
        if fold < n_splits - 1:
            train_idx = train_idx[
                (train_idx < test_start - purge_gap) |
                (train_idx > test_end + embargo)
            ]

        yield train_idx, test_idx
```

---

#### 5.1.4 Monte Carlo Simulation

**Description:** Assess strategy robustness via randomization.

**Reference:** [MDPI - Monte Carlo-Based VaR Estimation](https://www.mdpi.com/2227-9091/13/8/146)

```python
def monte_carlo_backtest(returns, n_simulations=10000, confidence=0.95):
    """
    Monte Carlo robustness testing.

    Methods:
    1. Bootstrap: Resample historical returns
    2. Permutation: Shuffle return order
    3. Parametric: Generate from fitted distribution
    """
    actual_sharpe = calculate_sharpe(returns)
    simulated_sharpes = []

    for _ in range(n_simulations):
        # Bootstrap resample
        bootstrap_returns = np.random.choice(
            returns.dropna(),
            size=len(returns.dropna()),
            replace=True
        )
        simulated_sharpes.append(calculate_sharpe(bootstrap_returns))

    simulated_sharpes = np.array(simulated_sharpes)

    return {
        'actual_sharpe': actual_sharpe,
        'mean_simulated': simulated_sharpes.mean(),
        'std_simulated': simulated_sharpes.std(),
        'ci_lower': np.percentile(simulated_sharpes, (1-confidence)/2 * 100),
        'ci_upper': np.percentile(simulated_sharpes, (1+confidence)/2 * 100),
        'p_value': (simulated_sharpes >= actual_sharpe).mean()
    }
```

---

#### 5.1.5 Signal Impact Analysis

**From Cass Freight Index methodology:**

```python
def signal_impact_analysis(df, signal_col, return_col,
                          holding_periods=[1, 3, 6, 12]):
    """
    Analyze impact of signal on forward returns.

    Tests: When signal triggers, what are the forward returns?
    """
    results = []

    for period in holding_periods:
        # Calculate forward returns
        forward_returns = df[return_col].rolling(period).sum().shift(-period)

        # Split by signal
        signal_true = df[signal_col] == 1
        signal_false = df[signal_col] == -1

        returns_when_true = forward_returns[signal_true].dropna()
        returns_when_false = forward_returns[signal_false].dropna()

        # Statistical test
        t_stat, p_val = stats.ttest_ind(returns_when_true, returns_when_false)

        results.append({
            'holding_period': period,
            'return_signal_true': returns_when_true.mean() * 100,
            'return_signal_false': returns_when_false.mean() * 100,
            'alpha': (returns_when_true.mean() - returns_when_false.mean()) * 100,
            'win_rate_true': (returns_when_true > 0).mean() * 100,
            'win_rate_false': (returns_when_false > 0).mean() * 100,
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        })

    return pd.DataFrame(results)
```

---

#### 5.1.6 Forecasting Methods

**From Cass Freight Index methodology:**

| Method | Description | Best For |
|--------|-------------|----------|
| **Historical Peak** | Analyze historical patterns | Seasonal indicators |
| **Seasonal Decomposition** | Trend + Seasonal + Residual | Strong seasonality |
| **SARIMA** | Seasonal ARIMA model | Complex patterns |
| **Historical + Trend** | Average adjusted for recent trend | Trending data |

```python
def multi_method_forecast(series, forecast_horizon=12):
    """
    Generate forecasts using multiple methods.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    results = {}

    # Method 1: Historical Average
    monthly_means = series.groupby(series.index.month).mean()
    results['historical_avg'] = [
        monthly_means[(series.index[-1].month + i) % 12 + 1]
        for i in range(1, forecast_horizon + 1)
    ]

    # Method 2: Seasonal Decomposition
    decomp = seasonal_decompose(series, model='additive', period=12)
    trend_slope = np.polyfit(range(len(decomp.trend.dropna())),
                            decomp.trend.dropna(), 1)[0]
    last_trend = decomp.trend.dropna().iloc[-1]

    forecast_trend = [last_trend + trend_slope * i for i in range(1, forecast_horizon + 1)]
    seasonal_pattern = decomp.seasonal.iloc[-12:].values

    results['seasonal_decomp'] = [
        forecast_trend[i] + seasonal_pattern[i % 12]
        for i in range(forecast_horizon)
    ]

    # Method 3: SARIMA
    try:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(0,1,1,12))
        fitted = model.fit(disp=False)
        results['sarima'] = fitted.forecast(forecast_horizon).values
    except:
        results['sarima'] = [np.nan] * forecast_horizon

    # Method 4: Historical + Trend Adjustment
    trend_adj = series.diff(12).mean()  # Annual trend
    results['historical_trend'] = [
        monthly_means[(series.index[-1].month + i) % 12 + 1] + trend_adj * (i/12)
        for i in range(1, forecast_horizon + 1)
    ]

    return results
```

---

### 5.2 Backtest Quality Metrics

| Metric | Formula | Good Value | Interpretation |
|--------|---------|------------|----------------|
| **Sharpe Ratio** | (Return - Rf) / Vol | > 0.5 | Risk-adjusted return |
| **Sortino Ratio** | (Return - Rf) / Downside Vol | > 1.0 | Downside risk-adjusted |
| **Calmar Ratio** | CAGR / Max Drawdown | > 1.0 | Return vs worst case |
| **WFER** | OOS Sharpe / IS Sharpe | > 0.7 | Generalization ability |
| **Win Rate** | % positive periods | > 55% | Consistency |
| **Profit Factor** | Gross Profit / Gross Loss | > 1.5 | Profit vs loss ratio |

---

### 5.3 Signal Lag Requirements

**CRITICAL: Always apply signal lag to avoid look-ahead bias.**

| Indicator Type | Typical Publication Lag | Recommended Signal Lag |
|---------------|------------------------|----------------------|
| Monthly FRED | 2-4 weeks | 1 month |
| Weekly data | 1 week | 1 week |
| Daily market | Same day | 1 day |
| Quarterly GDP | 4-8 weeks | 1 quarter |

```python
def apply_signal_lag(signals, lag=1):
    """Apply lag to prevent look-ahead bias."""
    return signals.shift(lag)
```

---

## Phase 6: Visualization and Dashboard

### 6.1 Dashboard Architecture

**Technology Stack:**
- **Backend:** Python with Plotly Dash
- **Charts:** Plotly.js (50+ chart types)
- **Interactive Features:** Crosshairs, zoom, hover cards
- **Deployment:** Flask-based web app

**Reference:** [Plotly Dash Documentation](https://dash.plotly.com/interactive-graphing)

### 6.2 Dashboard Components

#### 6.2.1 Main Navigation

```
┌─────────────────────────────────────────────────────────────────┐
│  RLIC Analysis Dashboard                                        │
├─────────────────────────────────────────────────────────────────┤
│  [Overview] [Qualitative] [Correlation] [Lead-Lag] [Regimes]    │
│  [Backtests] [Forecasts] [Reports]                              │
└─────────────────────────────────────────────────────────────────┘
```

#### 6.2.2 Interactive Chart Specifications

All charts must support:

| Feature | Implementation | User Interaction |
|---------|---------------|------------------|
| **Zoom** | `config={'scrollZoom': True}` | Mouse wheel / pinch |
| **Pan** | Drag to pan | Click and drag |
| **Crosshair** | `hovermode='x unified'` | Hover shows vertical line |
| **Data Card** | Custom hover template | Popup with detailed data |
| **Range Selector** | `rangeselector` buttons | 1M, 3M, YTD, 1Y, All |
| **Export** | Download as PNG/SVG | Button in toolbar |

```python
def create_interactive_timeseries(df, y_cols, title):
    """
    Create interactive time series with all required features.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    for col in y_cols:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            name=col,
            mode='lines',
            hovertemplate=(
                '<b>%{x|%Y-%m-%d}</b><br>'
                f'{col}: %{{y:.2f}}<br>'
                '<extra></extra>'
            )
        ))

    fig.update_layout(
        title=title,
        hovermode='x unified',
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month"),
                    dict(count=3, label="3M", step="month"),
                    dict(count=6, label="6M", step="month"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year"),
                    dict(step="all", label="All")
                ]
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(fixedrange=False)
    )

    fig.update_xaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        showline=True
    )

    fig.update_yaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor'
    )

    return fig
```

#### 6.2.3 Hover Data Card Template

```python
hover_template = '''
<div style="background: white; padding: 10px; border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
    <b style="font-size: 14px;">%{x|%Y-%m-%d}</b>
    <hr style="margin: 5px 0;">
    <table style="font-size: 12px;">
        <tr><td>Value:</td><td><b>%{y:.4f}</b></td></tr>
        <tr><td>Change:</td><td style="color: %{customdata[0]}">%{customdata[1]:.2%}</td></tr>
        <tr><td>Regime:</td><td>%{customdata[2]}</td></tr>
        <tr><td>Signal:</td><td>%{customdata[3]}</td></tr>
    </table>
</div>
'''
```

### 6.3 Dashboard Pages

#### Page 1: Overview Dashboard

```
┌─────────────────────────────────────────────────────────────────┐
│  OVERVIEW                                         [Date Range ▼] │
├────────────────────────┬────────────────────────────────────────┤
│  KPI Cards             │  Main Time Series Chart                │
│  ┌──────┐ ┌──────┐    │  [Interactive chart with regime        │
│  │Sharpe│ │ CAGR │    │   background coloring]                  │
│  │ 0.62 │ │ 8.2% │    │                                         │
│  └──────┘ └──────┘    │                                         │
│  ┌──────┐ ┌──────┐    │                                         │
│  │MaxDD │ │WinRt │    │                                         │
│  │-12.2%│ │ 58%  │    │                                         │
│  └──────┘ └──────┘    │                                         │
├────────────────────────┼────────────────────────────────────────┤
│  Current Regime        │  Recent Signals Table                  │
│  ┌──────────────────┐ │  Date    | Signal | Action | Return    │
│  │    OVERHEAT      │ │  2026-01 | +1     | Long   | +2.3%     │
│  │  Growth: Rising  │ │  2025-12 | +1     | Long   | +1.8%     │
│  │  Infl: Rising    │ │  2025-11 | -1     | Short  | -0.5%     │
│  └──────────────────┘ │                                         │
└────────────────────────┴────────────────────────────────────────┘
```

#### Page 2: Qualitative Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│  QUALITATIVE ANALYSIS                                           │
├─────────────────────────────────────────────────────────────────┤
│  [Indicator Selector ▼]                                         │
├────────────────────────┬────────────────────────────────────────┤
│  Indicator Profile     │  Literature Summary                    │
│  ┌──────────────────┐ │  ┌────────────────────────────────────┐│
│  │ Definition       │ │  │ Academic Sources (3)               ││
│  │ Source           │ │  │ Professional Sources (5)           ││
│  │ Frequency        │ │  │ Public Sources (2)                 ││
│  │ Release Lag      │ │  └────────────────────────────────────┘│
│  │ Seasonality      │ │                                        │
│  └──────────────────┘ │  Key Insights Table                    │
├────────────────────────┼────────────────────────────────────────┤
│  Seasonal Pattern      │  Historical Examples                   │
│  [Monthly boxplot]     │  [Annotated timeline with events]      │
└────────────────────────┴────────────────────────────────────────┘
```

#### Page 3: Correlation Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│  CORRELATION ANALYSIS                                           │
├────────────────────────┬────────────────────────────────────────┤
│  Correlation Matrix    │  Scatter Plot with Regression          │
│  [Interactive heatmap] │  [Zoom, crosshair, hover cards]        │
│                        │                                         │
│  Click cell to see     │  R² = 0.XX, p < 0.001                  │
│  detailed scatter      │  Equation: y = ax + b                  │
├────────────────────────┼────────────────────────────────────────┤
│  Correlation Over Time │  Rolling Correlation                   │
│  [Line chart showing   │  [12M, 36M, 60M windows]               │
│   stability]           │                                         │
└────────────────────────┴────────────────────────────────────────┘
```

#### Page 4: Lead-Lag Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│  LEAD-LAG ANALYSIS                                              │
├────────────────────────┬────────────────────────────────────────┤
│  Cross-Correlation     │  Optimal Lag Summary                   │
│  [Bar chart: corr vs   │  ┌────────────────────────────────────┐│
│   lag from -12 to +12] │  │ Indicator    | Optimal Lag | Corr  ││
│                        │  │ Orders/Inv   | -2 months   | 0.18  ││
│  Peak: lag = -2        │  │ PPI          | -1 month    | 0.15  ││
│  Interpretation:       │  └────────────────────────────────────┘│
│  Indicator leads by    │                                        │
│  2 months              │  Granger Causality Results             │
├────────────────────────┼────────────────────────────────────────┤
│  Sector Lead-Lag       │  Dimension Heatmaps                    │
│  Heatmap               │  [Growth vs Sectors]                   │
│  [Sectors x Lags]      │  [Inflation vs Sectors]                │
└────────────────────────┴────────────────────────────────────────┘
```

#### Page 5: Regime Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│  REGIME ANALYSIS                                                │
├─────────────────────────────────────────────────────────────────┤
│  Phase Timeline [Interactive - zoom, pan, click for details]    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Recovery | Overheat | Stagflation | Overheat | Reflation   ││
│  │ ████████ | █████████████████████ | ████████ | ████████████ ││
│  └─────────────────────────────────────────────────────────────┘│
├────────────────────────┬────────────────────────────────────────┤
│  Sector-Phase Heatmap  │  Performance by Phase                  │
│  [Click cell for       │  [Box plots showing return             │
│   detailed stats]      │   distributions]                       │
│                        │                                         │
│  Annualized Returns    │  Click phase for details               │
├────────────────────────┼────────────────────────────────────────┤
│  Phase Statistics      │  Best Sectors by Phase                 │
│  Duration, Frequency   │  [Horizontal bar chart]                │
└────────────────────────┴────────────────────────────────────────┘
```

#### Page 6: Backtesting

```
┌─────────────────────────────────────────────────────────────────┐
│  BACKTESTING                              [Strategy Selector ▼]  │
├─────────────────────────────────────────────────────────────────┤
│  Equity Curve [Interactive with drawdown overlay]               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Strategy vs Benchmark cumulative returns                   ││
│  │  Hover for: Date, Value, Drawdown, Regime                   ││
│  └─────────────────────────────────────────────────────────────┘│
├────────────────────────┬────────────────────────────────────────┤
│  Walk-Forward Results  │  Monte Carlo Distribution              │
│  [In-sample vs         │  [Histogram with confidence            │
│   Out-of-sample]       │   intervals]                           │
│                        │                                         │
│  WFER: 0.87            │  95% CI: [0.45, 0.79]                  │
├────────────────────────┼────────────────────────────────────────┤
│  Performance Metrics   │  Trade Log                             │
│  Sharpe: 0.62          │  [Sortable, filterable table]          │
│  CAGR: 8.2%            │                                         │
│  Max DD: -12.2%        │  Export: [CSV] [Excel]                 │
└────────────────────────┴────────────────────────────────────────┘
```

#### Page 7: Forecasts

```
┌─────────────────────────────────────────────────────────────────┐
│  FORECASTS                                                      │
├─────────────────────────────────────────────────────────────────┤
│  Multi-Method Forecast Comparison                               │
│  [Line chart with historical + forecast lines per method]       │
│                                                                 │
│  Methods: Historical Avg | Seasonal Decomp | SARIMA | Trend Adj │
├────────────────────────┬────────────────────────────────────────┤
│  Forecast Table        │  Method Comparison                     │
│  ┌──────────────────┐ │  [Error metrics by method]             │
│  │ Month | M1 | M2  │ │                                         │
│  │ Jan26 |1.03|0.99 │ │  RMSE | MAE | MAPE                     │
│  │ Feb26 |1.07|1.04 │ │                                         │
│  └──────────────────┘ │                                         │
├────────────────────────┼────────────────────────────────────────┤
│  Seasonality Analysis  │  Forecast Confidence                   │
│  [Monthly pattern]     │  [Fan chart with intervals]            │
└────────────────────────┴────────────────────────────────────────┘
```

### 6.4 Dash App Structure

```python
# app.py - Main Dashboard Application

import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

# Initialize app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Layout
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="RLIC Analysis Dashboard",
        children=[
            dbc.NavItem(dbc.NavLink("Overview", href="/")),
            dbc.NavItem(dbc.NavLink("Qualitative", href="/qualitative")),
            dbc.NavItem(dbc.NavLink("Correlation", href="/correlation")),
            dbc.NavItem(dbc.NavLink("Lead-Lag", href="/leadlag")),
            dbc.NavItem(dbc.NavLink("Regimes", href="/regimes")),
            dbc.NavItem(dbc.NavLink("Backtests", href="/backtests")),
            dbc.NavItem(dbc.NavLink("Forecasts", href="/forecasts")),
        ],
        color="primary",
        dark=True
    ),

    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
], fluid=True)

# Callback for page routing
@callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/qualitative':
        return qualitative_layout
    elif pathname == '/correlation':
        return correlation_layout
    # ... etc
    return overview_layout

if __name__ == '__main__':
    app.run_server(debug=True)
```

### 6.5 Interactive Features Implementation

```python
# Interactive crosshair and hover card
@callback(
    Output('hover-data-card', 'children'),
    Input('main-chart', 'hoverData')
)
def display_hover_data(hoverData):
    if hoverData is None:
        return "Hover over chart for details"

    point = hoverData['points'][0]
    date = point['x']
    value = point['y']

    return dbc.Card([
        dbc.CardHeader(f"Date: {date}"),
        dbc.CardBody([
            html.P(f"Value: {value:.4f}"),
            html.P(f"Change: {point['customdata'][0]:.2%}"),
            html.P(f"Regime: {point['customdata'][1]}"),
        ])
    ])
```

---

## Phase 7: Documentation

### 7.1 Report Structure

Every analysis report MUST follow this structure:

```markdown
# {Target} vs {Indicator} Analysis

## Overview
- Brief description
- Data period

---

## Qualitative Analysis
### What is {Indicator}?
### Market Interpretation
### Literature Review
### Limitations

---

## Key Findings Summary
### 1. Level Relationship
### 2. Change Relationship
### 3. Predictive Power
### 4. Regime-Based Insights

## Detailed Analysis
### Correlation Matrix
### Lead-Lag Analysis
### Granger Causality
### Predictive Modeling
### Regime Analysis

## Backtesting Results
### Walk-Forward Validation
### Monte Carlo Analysis
### Signal Impact Analysis

## Visualizations
[Embedded interactive dashboard screenshots]

## Actionable Recommendations

## Files Created
```

### 7.2 File Naming Conventions

**Reports:**
```
docs/analysis_reports/{target}_{indicator}_analysis.md
```

**Visualizations:**
```
data/{target}_{indicator}_{plot_type}.png
```

**Data Files:**
```
data/{analysis_name}_{data_type}.parquet
data/{analysis_name}_{data_type}.csv
```

---

## Appendices

### Appendix A: Decision Framework

```
                    ┌─────────────────────────┐
                    │ Correlation Analysis    │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ Is correlation > 0.3?   │
                    └───────────┬─────────────┘
                          No    │    Yes
                    ┌───────────┴───────────┐
                    ▼                       ▼
               Weak/No                 Check if spurious
               relationship            (level vs change)
                    │                       │
                    │           ┌───────────▼───────────┐
                    │           │     Lead-Lag          │
                    │           │     Analysis          │
                    │           └───────────┬───────────┘
                    │                       │
                    │           ┌───────────▼───────────┐
                    │           │    Peak at lag ≠ 0?   │
                    │           └───────────┬───────────┘
                    │                 No    │    Yes
                    │           ┌───────────┴───────────┐
                    │           ▼                       ▼
                    │      Contemporaneous         Leading indicator
                    │      (no prediction)         (test causality)
                    │           │                       │
                    │           │           ┌───────────▼───────────┐
                    │           │           │  Granger significant? │
                    │           │           └───────────┬───────────┘
                    │           │                 No    │    Yes
                    │           │           ┌───────────┴───────────┐
                    │           │           ▼                       ▼
                    │           │      No causality            Predictive!
                    │           │           │                  Build strategy
                    └───────────┴───────────┘
                                │
                    ┌───────────▼───────────┐
                    │    Regime Analysis    │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Regime differences?  │
                    └───────────┬───────────┘
                          No    │    Yes
                    ┌───────────┴───────────┐
                    ▼                       ▼
                No value              Use as FILTER
                                     (not predictor)
```

### Appendix B: Quality Checklist

Before finalizing any analysis:

- [ ] Qualitative analysis completed with literature citations
- [ ] Seasonality checked and documented
- [ ] Spurious correlations ruled out
- [ ] Lead-lag analysis shows meaningful pattern
- [ ] Signal lag applied (minimum 1 month for monthly data)
- [ ] Walk-forward validation performed
- [ ] Monte Carlo robustness tested
- [ ] WFER > 0.5 (strategy generalizes)
- [ ] Visual examples validated against data
- [ ] All charts have interactive features
- [ ] Report follows standard structure

### Appendix C: References

**Academic:**
- Fama (1981) - Stock Returns, Real Activity, Inflation, and Money
- Chen, Roll & Ross (1986) - Economic Forces and the Stock Market
- Lopez de Prado - Advances in Financial Machine Learning

**Methodological:**
- [Walk-Forward Analysis - Interactive Brokers](https://www.interactivebrokers.com/campus/ibkr-quant-news/the-future-of-backtesting-a-deep-dive-into-walk-forward-analysis/)
- [Monte Carlo Backtesting - MDPI](https://www.mdpi.com/2227-9091/13/8/146)
- [Plotly Dash Interactive Graphing](https://dash.plotly.com/interactive-graphing)

**Data Sources:**
- [FRED Economic Data](https://fred.stlouisfed.org/)
- [Cass Freight Index](https://www.cassinfo.com/freight-audit-payment/cass-transportation-indexes/cass-freight-index)
- [Fama-French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-24 | RA Cheryl | Initial unified SOP |

