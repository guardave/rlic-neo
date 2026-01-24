# Backtest Methodology Catalog

**Version:** 1.0
**Date:** 2026-01-24
**Author:** RA Cheryl
**Related:** [Unified Analysis SOP](./unified_analysis_sop.md)

---

## Executive Summary

This catalog documents all backtesting methodologies available for the RLIC analysis framework. Each method includes:
- Theoretical foundation
- Implementation code
- Pros and cons
- When to use
- Quality metrics

**Key Principle:** No single backtest method is sufficient. Robust validation requires multiple complementary approaches.

---

## Table of Contents

1. [Methodology Overview](#1-methodology-overview)
2. [Basic Methods](#2-basic-methods)
3. [Cross-Validation Methods](#3-cross-validation-methods)
4. [Simulation Methods](#4-simulation-methods)
5. [Signal Analysis Methods](#5-signal-analysis-methods)
6. [Regime-Based Methods](#6-regime-based-methods)
7. [Robustness Testing](#7-robustness-testing)
8. [Quality Metrics](#8-quality-metrics)
9. [Method Selection Guide](#9-method-selection-guide)

---

## 1. Methodology Overview

### 1.1 The Backtesting Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    BACKTESTING HIERARCHY                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Level 1: BASIC VALIDATION                                      │
│  ├── Simple Train/Test Split                                    │
│  └── Historical Replay                                          │
│                                                                 │
│  Level 2: CROSS-VALIDATION                                      │
│  ├── Walk-Forward Validation (WFV)                              │
│  ├── Combinatorial Purged CV (CPCV)                             │
│  └── Rolling Window CV                                          │
│                                                                 │
│  Level 3: SIMULATION                                            │
│  ├── Monte Carlo Bootstrapping                                  │
│  ├── Block Bootstrapping                                        │
│  └── Parametric Simulation                                      │
│                                                                 │
│  Level 4: ROBUSTNESS                                            │
│  ├── Sensitivity Analysis                                       │
│  ├── Parameter Stability                                        │
│  └── Regime Stress Testing                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Common Biases to Avoid

| Bias | Description | Mitigation |
|------|-------------|------------|
| **Look-ahead** | Using future data in signals | Apply signal lag |
| **Survivorship** | Missing delisted/failed securities | Use point-in-time data |
| **Overfitting** | Tuning to historical noise | Walk-forward validation |
| **Data snooping** | Multiple testing without correction | CPCV, Bonferroni adjustment |
| **Selection** | Cherry-picking favorable periods | Full sample + robustness |

---

## 2. Basic Methods

### 2.1 Simple Train/Test Split

**Description:** Divide data into fixed training and testing periods.

**Use When:**
- Initial exploration
- Quick hypothesis testing
- Sufficient data for meaningful split

**Code:**

```python
def simple_train_test_backtest(
    data,
    signal_column,
    return_column,
    train_pct=0.7,
    signal_lag=1
):
    """
    Simple train/test split backtest.

    Args:
        data: DataFrame with signal and returns
        signal_column: Column name for trading signal
        return_column: Column name for asset returns
        train_pct: Percentage of data for training
        signal_lag: Months to lag signal (avoid look-ahead)

    Returns:
        dict with train and test metrics
    """
    n = len(data)
    split_idx = int(n * train_pct)

    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()

    # Apply signal lag
    train['lagged_signal'] = train[signal_column].shift(signal_lag)
    test['lagged_signal'] = test[signal_column].shift(signal_lag)

    # Calculate strategy returns
    train['strategy_return'] = train['lagged_signal'] * train[return_column]
    test['strategy_return'] = test['lagged_signal'] * test[return_column]

    return {
        'train': calculate_metrics(train['strategy_return'].dropna()),
        'test': calculate_metrics(test['strategy_return'].dropna()),
        'split_date': data.index[split_idx],
        'train_periods': split_idx,
        'test_periods': n - split_idx
    }
```

**Pros:**
- Simple to implement
- Easy to understand
- Fast execution

**Cons:**
- Single test period (high variance)
- Sensitive to split point
- Doesn't test stability over time

**Quality Metrics:**
- Compare train vs test Sharpe
- Test Sharpe > 0.5 indicates potential
- Large train-test gap indicates overfitting

---

### 2.2 Historical Replay

**Description:** Simulate trading decisions as if made in real-time.

**Use When:**
- Final validation before live trading
- Understanding practical implementation
- Verifying signal timing

**Code:**

```python
def historical_replay(
    data,
    signal_generator,
    return_column,
    signal_lag=1,
    transaction_costs=0.001
):
    """
    Historical replay with realistic constraints.

    Args:
        data: Full dataset
        signal_generator: Function that generates signal from available data
        return_column: Column for returns
        signal_lag: Publication lag
        transaction_costs: Round-trip costs as decimal

    Returns:
        DataFrame with strategy performance
    """
    results = []
    position = 0

    for i in range(signal_lag, len(data)):
        # Only use data available at decision time
        available_data = data.iloc[:i-signal_lag+1]

        # Generate signal
        signal = signal_generator(available_data)

        # Record trade
        trade = {
            'date': data.index[i],
            'signal': signal,
            'return': data.iloc[i][return_column],
            'position_change': signal - position
        }

        # Apply transaction costs on position change
        if trade['position_change'] != 0:
            trade['costs'] = abs(trade['position_change']) * transaction_costs
        else:
            trade['costs'] = 0

        # Calculate net return
        trade['strategy_return'] = signal * trade['return'] - trade['costs']

        position = signal
        results.append(trade)

    return pd.DataFrame(results).set_index('date')
```

---

## 3. Cross-Validation Methods

### 3.1 Walk-Forward Validation (WFV)

**Description:** Rolling window optimization with out-of-sample testing.

**Reference:** [Interactive Brokers - Walk Forward Analysis](https://www.interactivebrokers.com/campus/ibkr-quant-news/the-future-of-backtesting-a-deep-dive-into-walk-forward-analysis/)

**Use When:**
- Testing strategy stability over time
- Parameter optimization
- Realistic performance estimation

**Key Parameters:**
| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `train_window` | 60 months | In-sample period |
| `test_window` | 12 months | Out-of-sample period |
| `step_size` | 12 months | How often to retrain |
| `purge_gap` | 3 months | Gap between train/test |

**Code:**

```python
def walk_forward_validation(
    data,
    signal_generator,
    return_column,
    train_window=60,
    test_window=12,
    step_size=12,
    purge_gap=3
):
    """
    Walk-Forward Validation with purge gap.

    The purge gap prevents information leakage between train and test sets.

    Returns:
        DataFrame with period-by-period results
        dict with aggregate metrics
    """
    results = []
    all_oos_returns = []

    for start in range(0, len(data) - train_window - purge_gap - test_window, step_size):
        # Define window boundaries
        train_start = start
        train_end = start + train_window
        test_start = train_end + purge_gap
        test_end = min(test_start + test_window, len(data))

        # Extract data
        train_data = data.iloc[train_start:train_end]
        test_data = data.iloc[test_start:test_end].copy()

        # Train: optimize/fit on in-sample
        optimized_signal = signal_generator(train_data)

        # Test: apply to out-of-sample
        test_data['signal'] = optimized_signal
        test_data['strategy_return'] = (
            test_data['signal'].shift(1) * test_data[return_column]
        )

        oos_returns = test_data['strategy_return'].dropna()
        all_oos_returns.extend(oos_returns.tolist())

        # Calculate period metrics
        is_metrics = calculate_metrics(train_data['strategy_return'].dropna())
        oos_metrics = calculate_metrics(oos_returns)

        results.append({
            'period_start': data.index[test_start],
            'period_end': data.index[test_end - 1],
            'is_sharpe': is_metrics['sharpe'],
            'oos_sharpe': oos_metrics['sharpe'],
            'oos_return': oos_returns.sum(),
            'is_volatility': is_metrics['volatility'],
            'oos_volatility': oos_metrics['volatility'],
            'wfer': oos_metrics['sharpe'] / is_metrics['sharpe'] if is_metrics['sharpe'] != 0 else 0
        })

    results_df = pd.DataFrame(results)

    # Aggregate metrics
    aggregate = {
        'mean_is_sharpe': results_df['is_sharpe'].mean(),
        'mean_oos_sharpe': results_df['oos_sharpe'].mean(),
        'std_oos_sharpe': results_df['oos_sharpe'].std(),
        'mean_wfer': results_df['wfer'].mean(),
        'pct_profitable_periods': (results_df['oos_return'] > 0).mean() * 100,
        'full_oos_metrics': calculate_metrics(pd.Series(all_oos_returns))
    }

    return results_df, aggregate


def calculate_wfer(is_sharpe, oos_sharpe):
    """
    Calculate Walk-Forward Efficiency Ratio.

    Interpretation:
    - WFER > 0.7: Strategy generalizes well
    - WFER 0.5-0.7: Moderate generalization
    - WFER < 0.5: Significant overfitting
    """
    if is_sharpe == 0:
        return 0
    return oos_sharpe / is_sharpe
```

---

### 3.2 Combinatorial Purged Cross-Validation (CPCV)

**Description:** Lopez de Prado methodology generating multiple backtest paths.

**Reference:** Advances in Financial Machine Learning (2018)

**Use When:**
- Limited data history
- Need statistical significance of results
- Testing for data snooping

**Key Concept:** Instead of single train/test splits, generate all possible combinations of k-fold splits to estimate distribution of performance metrics.

**Code:**

```python
from itertools import combinations
import numpy as np

def combinatorial_purged_cv(
    data,
    signal_generator,
    return_column,
    n_splits=5,
    purge_gap=3,
    embargo_pct=0.01
):
    """
    Combinatorial Purged Cross-Validation.

    Generates multiple backtest paths by combining different test folds.

    Args:
        data: Full dataset
        signal_generator: Signal generation function
        n_splits: Number of folds
        purge_gap: Gap between train and test
        embargo_pct: Embargo period as percentage of data

    Returns:
        List of all path results
        Distribution statistics
    """
    n_samples = len(data)
    fold_size = n_samples // n_splits
    embargo = max(1, int(n_samples * embargo_pct))

    # Create fold indices
    folds = []
    for i in range(n_splits):
        start = i * fold_size
        end = start + fold_size if i < n_splits - 1 else n_samples
        folds.append(list(range(start, end)))

    # Generate all path combinations (test on 2 folds each time)
    all_paths = []
    for test_folds in combinations(range(n_splits), 2):
        test_indices = []
        for f in test_folds:
            test_indices.extend(folds[f])

        # Training indices with purge and embargo
        train_indices = []
        for idx in range(n_samples):
            # Check if in test set
            if idx in test_indices:
                continue
            # Check if within purge gap of test
            in_purge = any(
                abs(idx - t) < purge_gap for t in test_indices
            )
            if in_purge:
                continue
            # Check embargo (after test)
            in_embargo = any(
                0 < idx - t < embargo for t in test_indices
            )
            if in_embargo:
                continue

            train_indices.append(idx)

        if len(train_indices) < fold_size:
            continue

        # Train and test
        train_data = data.iloc[train_indices]
        test_data = data.iloc[sorted(test_indices)].copy()

        signal = signal_generator(train_data)
        test_data['signal'] = signal
        test_data['strategy_return'] = (
            test_data['signal'].shift(1) * test_data[return_column]
        )

        path_metrics = calculate_metrics(test_data['strategy_return'].dropna())
        path_metrics['path_id'] = f"{test_folds}"
        all_paths.append(path_metrics)

    # Distribution statistics
    sharpes = [p['sharpe'] for p in all_paths]
    distribution = {
        'mean_sharpe': np.mean(sharpes),
        'std_sharpe': np.std(sharpes),
        'median_sharpe': np.median(sharpes),
        'min_sharpe': np.min(sharpes),
        'max_sharpe': np.max(sharpes),
        'n_paths': len(all_paths),
        'pct_positive': np.mean([s > 0 for s in sharpes]) * 100
    }

    return all_paths, distribution
```

---

### 3.3 Rolling Window Cross-Validation

**Description:** Fixed-size rolling window for continuous evaluation.

**Use When:**
- Testing for regime changes
- Continuous strategy monitoring
- Time-varying parameter estimation

**Code:**

```python
def rolling_window_cv(
    data,
    signal_generator,
    return_column,
    window_size=60,
    step_size=1,
    signal_lag=1
):
    """
    Rolling window cross-validation.

    Evaluates strategy performance at each time step using
    a fixed lookback window.

    Returns:
        DataFrame with rolling metrics
    """
    results = []

    for end in range(window_size, len(data)):
        start = end - window_size

        # Training window
        train_data = data.iloc[start:end]

        # Generate signal and calculate return
        signal = signal_generator(train_data)

        # Out-of-sample return (next period)
        if end < len(data):
            oos_return = data.iloc[end][return_column]
            strategy_return = signal * oos_return

            # Rolling Sharpe (last 12 periods)
            if len(results) >= 12:
                recent_returns = [r['strategy_return'] for r in results[-11:]] + [strategy_return]
                rolling_sharpe = calculate_sharpe(pd.Series(recent_returns))
            else:
                rolling_sharpe = np.nan

            results.append({
                'date': data.index[end],
                'signal': signal,
                'oos_return': oos_return,
                'strategy_return': strategy_return,
                'rolling_sharpe_12m': rolling_sharpe
            })

    return pd.DataFrame(results).set_index('date')
```

---

## 4. Simulation Methods

### 4.1 Monte Carlo Bootstrapping

**Description:** Resample historical returns to estimate distribution of metrics.

**Reference:** [MDPI - Monte Carlo-Based VaR Estimation](https://www.mdpi.com/2227-9091/13/8/146)

**Use When:**
- Assessing statistical significance
- Estimating confidence intervals
- Testing for luck vs skill

**Code:**

```python
def monte_carlo_bootstrap(
    strategy_returns,
    n_simulations=10000,
    confidence_levels=[0.90, 0.95, 0.99]
):
    """
    Monte Carlo bootstrap analysis.

    Resamples historical returns to estimate distribution
    of performance metrics.

    Returns:
        dict with metric distributions and confidence intervals
    """
    returns = strategy_returns.dropna().values
    n_periods = len(returns)

    # Store simulated metrics
    simulated = {
        'sharpe': [],
        'total_return': [],
        'max_drawdown': [],
        'volatility': []
    }

    for _ in range(n_simulations):
        # Bootstrap resample
        resampled = np.random.choice(returns, size=n_periods, replace=True)
        resampled_series = pd.Series(resampled)

        # Calculate metrics
        simulated['sharpe'].append(calculate_sharpe(resampled_series))
        simulated['total_return'].append(resampled_series.sum())
        simulated['max_drawdown'].append(calculate_max_drawdown(resampled_series))
        simulated['volatility'].append(resampled_series.std() * np.sqrt(12))

    # Convert to numpy arrays
    for key in simulated:
        simulated[key] = np.array(simulated[key])

    # Calculate actual metrics
    actual = {
        'sharpe': calculate_sharpe(strategy_returns),
        'total_return': strategy_returns.sum(),
        'max_drawdown': calculate_max_drawdown(strategy_returns),
        'volatility': strategy_returns.std() * np.sqrt(12)
    }

    # Calculate confidence intervals and p-values
    results = {}
    for metric in simulated:
        ci = {}
        for level in confidence_levels:
            lower_pct = (1 - level) / 2 * 100
            upper_pct = (1 + level) / 2 * 100
            ci[f'{int(level*100)}%'] = {
                'lower': np.percentile(simulated[metric], lower_pct),
                'upper': np.percentile(simulated[metric], upper_pct)
            }

        results[metric] = {
            'actual': actual[metric],
            'mean_simulated': simulated[metric].mean(),
            'std_simulated': simulated[metric].std(),
            'median_simulated': np.median(simulated[metric]),
            'p_value': (simulated[metric] >= actual[metric]).mean(),
            'confidence_intervals': ci,
            'simulated_values': simulated[metric]
        }

    return results
```

---

### 4.2 Block Bootstrap

**Description:** Resample blocks of consecutive returns to preserve autocorrelation.

**Use When:**
- Returns show serial correlation
- Momentum/trend-following strategies
- Regime persistence analysis

**Code:**

```python
def block_bootstrap(
    strategy_returns,
    block_size=12,
    n_simulations=10000
):
    """
    Block bootstrap preserving serial correlation.

    Resamples blocks of consecutive returns rather than
    individual observations.

    Args:
        strategy_returns: Series of returns
        block_size: Number of consecutive periods per block
        n_simulations: Number of bootstrap samples
    """
    returns = strategy_returns.dropna().values
    n_periods = len(returns)
    n_blocks = n_periods // block_size

    # Create blocks
    blocks = []
    for i in range(n_periods - block_size + 1):
        blocks.append(returns[i:i + block_size])
    blocks = np.array(blocks)

    simulated_sharpes = []

    for _ in range(n_simulations):
        # Sample blocks with replacement
        sampled_blocks = blocks[
            np.random.choice(len(blocks), size=n_blocks, replace=True)
        ]

        # Flatten to single series
        resampled = sampled_blocks.flatten()[:n_periods]
        resampled_series = pd.Series(resampled)

        simulated_sharpes.append(calculate_sharpe(resampled_series))

    return {
        'actual_sharpe': calculate_sharpe(strategy_returns),
        'mean_simulated': np.mean(simulated_sharpes),
        'std_simulated': np.std(simulated_sharpes),
        'p_value': (np.array(simulated_sharpes) >= calculate_sharpe(strategy_returns)).mean(),
        'ci_95_lower': np.percentile(simulated_sharpes, 2.5),
        'ci_95_upper': np.percentile(simulated_sharpes, 97.5)
    }
```

---

### 4.3 Parametric Simulation

**Description:** Generate synthetic returns from fitted distribution.

**Use When:**
- Tail risk analysis
- Stress testing
- Limited historical data

**Code:**

```python
from scipy import stats

def parametric_simulation(
    strategy_returns,
    n_simulations=10000,
    distribution='t'
):
    """
    Parametric simulation from fitted distribution.

    Args:
        strategy_returns: Historical returns
        n_simulations: Number of simulations
        distribution: 't' for Student-t, 'normal' for Gaussian

    Returns:
        Simulated metrics distribution
    """
    returns = strategy_returns.dropna().values

    # Fit distribution
    if distribution == 't':
        # Fit Student-t distribution
        params = stats.t.fit(returns)
        df, loc, scale = params
        dist = stats.t(df=df, loc=loc, scale=scale)
    else:
        # Fit normal distribution
        loc, scale = stats.norm.fit(returns)
        dist = stats.norm(loc=loc, scale=scale)

    # Generate simulated paths
    n_periods = len(returns)
    simulated_sharpes = []
    simulated_returns = []

    for _ in range(n_simulations):
        # Generate synthetic returns
        synthetic = dist.rvs(size=n_periods)
        synthetic_series = pd.Series(synthetic)

        simulated_sharpes.append(calculate_sharpe(synthetic_series))
        simulated_returns.append(synthetic_series.sum())

    return {
        'distribution': distribution,
        'fitted_params': params if distribution == 't' else (loc, scale),
        'actual_sharpe': calculate_sharpe(strategy_returns),
        'simulated_sharpe_mean': np.mean(simulated_sharpes),
        'simulated_sharpe_std': np.std(simulated_sharpes),
        'simulated_return_mean': np.mean(simulated_returns),
        'simulated_return_std': np.std(simulated_returns),
        'var_95': np.percentile(simulated_returns, 5),
        'cvar_95': np.mean([r for r in simulated_returns if r <= np.percentile(simulated_returns, 5)])
    }
```

---

## 5. Signal Analysis Methods

### 5.1 Signal Impact Analysis

**Description:** Measure forward returns conditional on signal state.

**Use When:**
- Evaluating signal predictive power
- Comparing signal variations
- Determining optimal holding period

**Code:**

```python
def signal_impact_analysis(
    data,
    signal_column,
    return_column,
    holding_periods=[1, 3, 6, 12]
):
    """
    Analyze signal impact on forward returns.

    Args:
        data: DataFrame with signal and returns
        signal_column: Column with trading signal (-1, 0, +1)
        return_column: Column with asset returns
        holding_periods: List of forward periods to test

    Returns:
        DataFrame with impact statistics per holding period
    """
    results = []

    for period in holding_periods:
        # Calculate forward returns
        forward_return = data[return_column].rolling(period).sum().shift(-period)

        # Split by signal
        long_mask = data[signal_column] == 1
        short_mask = data[signal_column] == -1

        long_returns = forward_return[long_mask].dropna()
        short_returns = forward_return[short_mask].dropna()

        # Statistical tests
        if len(long_returns) > 10 and len(short_returns) > 10:
            t_stat, t_pval = stats.ttest_ind(long_returns, short_returns)
            mw_stat, mw_pval = stats.mannwhitneyu(long_returns, short_returns)
        else:
            t_stat, t_pval, mw_stat, mw_pval = np.nan, np.nan, np.nan, np.nan

        results.append({
            'holding_period': period,
            'n_long': len(long_returns),
            'n_short': len(short_returns),
            'avg_return_long': long_returns.mean() * 100,
            'avg_return_short': short_returns.mean() * 100,
            'alpha': (long_returns.mean() - short_returns.mean()) * 100,
            'long_win_rate': (long_returns > 0).mean() * 100,
            'short_win_rate': (short_returns > 0).mean() * 100,
            't_statistic': t_stat,
            't_p_value': t_pval,
            'mannwhitney_p_value': mw_pval,
            'significant_at_05': t_pval < 0.05 if not np.isnan(t_pval) else False
        })

    return pd.DataFrame(results)
```

---

### 5.2 Signal Stability Analysis

**Description:** Test if signal performance is stable over time.

**Code:**

```python
def signal_stability_analysis(
    data,
    signal_column,
    return_column,
    window_size=60,
    min_observations=30
):
    """
    Analyze signal stability over time.

    Tests whether signal performance is consistent across
    different time periods.

    Returns:
        DataFrame with rolling performance
        Stability metrics
    """
    results = []

    for end in range(window_size, len(data), 12):  # Annual steps
        window_data = data.iloc[end-window_size:end]

        if len(window_data) < min_observations:
            continue

        # Calculate signal performance in window
        strategy_return = (
            window_data[signal_column].shift(1) *
            window_data[return_column]
        ).dropna()

        long_returns = window_data[window_data[signal_column] == 1][return_column]
        short_returns = window_data[window_data[signal_column] == -1][return_column]

        results.append({
            'period_end': data.index[end-1],
            'sharpe': calculate_sharpe(strategy_return),
            'total_return': strategy_return.sum(),
            'volatility': strategy_return.std() * np.sqrt(12),
            'win_rate': (strategy_return > 0).mean(),
            'avg_long_return': long_returns.mean() if len(long_returns) > 0 else np.nan,
            'avg_short_return': short_returns.mean() if len(short_returns) > 0 else np.nan
        })

    results_df = pd.DataFrame(results).set_index('period_end')

    # Stability metrics
    stability = {
        'sharpe_mean': results_df['sharpe'].mean(),
        'sharpe_std': results_df['sharpe'].std(),
        'sharpe_stability': results_df['sharpe'].mean() / results_df['sharpe'].std() if results_df['sharpe'].std() > 0 else np.nan,
        'pct_positive_sharpe': (results_df['sharpe'] > 0).mean() * 100,
        'worst_period_sharpe': results_df['sharpe'].min(),
        'best_period_sharpe': results_df['sharpe'].max()
    }

    return results_df, stability
```

---

## 6. Regime-Based Methods

### 6.1 Regime-Conditional Backtest

**Description:** Evaluate strategy separately by market regime.

**Code:**

```python
def regime_conditional_backtest(
    data,
    signal_column,
    return_column,
    regime_column
):
    """
    Backtest strategy conditional on regime.

    Evaluates performance separately for each regime to
    identify regime-dependent alpha.
    """
    results = {}

    for regime in data[regime_column].dropna().unique():
        regime_data = data[data[regime_column] == regime].copy()

        if len(regime_data) < 20:
            continue

        strategy_return = (
            regime_data[signal_column].shift(1) *
            regime_data[return_column]
        ).dropna()

        benchmark_return = regime_data[return_column]

        results[regime] = {
            'n_periods': len(regime_data),
            'pct_of_sample': len(regime_data) / len(data) * 100,
            'strategy_sharpe': calculate_sharpe(strategy_return),
            'benchmark_sharpe': calculate_sharpe(benchmark_return),
            'strategy_total_return': strategy_return.sum() * 100,
            'benchmark_total_return': benchmark_return.sum() * 100,
            'alpha': (strategy_return.sum() - benchmark_return.sum()) * 100,
            'win_rate': (strategy_return > 0).mean() * 100,
            'avg_monthly_return': strategy_return.mean() * 100,
            'volatility': strategy_return.std() * 100
        }

    return pd.DataFrame(results).T
```

---

### 6.2 Regime Transition Analysis

**Description:** Analyze performance around regime changes.

**Code:**

```python
def regime_transition_analysis(
    data,
    signal_column,
    return_column,
    regime_column,
    window=3
):
    """
    Analyze strategy performance around regime transitions.

    Tests whether strategy handles regime changes well.
    """
    # Find transition points
    regime_changes = data[regime_column] != data[regime_column].shift(1)
    transition_dates = data.index[regime_changes]

    transition_results = []

    for date in transition_dates:
        date_idx = data.index.get_loc(date)

        # Skip if not enough data around transition
        if date_idx < window or date_idx >= len(data) - window:
            continue

        # Before transition
        before_data = data.iloc[date_idx-window:date_idx]
        before_returns = (
            before_data[signal_column].shift(1) *
            before_data[return_column]
        ).dropna()

        # After transition
        after_data = data.iloc[date_idx:date_idx+window]
        after_returns = (
            after_data[signal_column].shift(1) *
            after_data[return_column]
        ).dropna()

        transition_results.append({
            'date': date,
            'from_regime': data.iloc[date_idx-1][regime_column],
            'to_regime': data.iloc[date_idx][regime_column],
            'return_before': before_returns.sum() * 100,
            'return_after': after_returns.sum() * 100,
            'performance_diff': (after_returns.sum() - before_returns.sum()) * 100,
            'correct_signal_before': (before_returns > 0).mean() * 100,
            'correct_signal_after': (after_returns > 0).mean() * 100
        })

    results_df = pd.DataFrame(transition_results)

    # Summary statistics
    summary = {
        'n_transitions': len(results_df),
        'avg_return_before': results_df['return_before'].mean(),
        'avg_return_after': results_df['return_after'].mean(),
        'avg_accuracy_before': results_df['correct_signal_before'].mean(),
        'avg_accuracy_after': results_df['correct_signal_after'].mean(),
        'pct_improved_after': (results_df['performance_diff'] > 0).mean() * 100
    }

    return results_df, summary
```

---

## 7. Robustness Testing

### 7.1 Parameter Sensitivity Analysis

**Description:** Test how performance changes with parameter variations.

**Code:**

```python
def parameter_sensitivity(
    data,
    signal_generator_factory,
    return_column,
    param_ranges
):
    """
    Parameter sensitivity analysis.

    Args:
        data: Full dataset
        signal_generator_factory: Function that creates signal generator from params
        return_column: Return column name
        param_ranges: Dict of {param_name: [values]}

    Example:
        param_ranges = {
            'lookback': [30, 60, 90, 120],
            'threshold': [0.0, 0.5, 1.0]
        }
    """
    from itertools import product

    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())

    results = []

    for combo in product(*param_values):
        params = dict(zip(param_names, combo))

        # Create signal generator with these params
        signal_gen = signal_generator_factory(**params)

        # Run backtest
        strategy_return = (
            signal_gen(data).shift(1) * data[return_column]
        ).dropna()

        metrics = calculate_metrics(strategy_return)
        metrics.update(params)

        results.append(metrics)

    results_df = pd.DataFrame(results)

    # Sensitivity summary
    summary = {}
    for param in param_names:
        grouped = results_df.groupby(param)['sharpe'].agg(['mean', 'std'])
        summary[param] = {
            'best_value': results_df.loc[results_df['sharpe'].idxmax(), param],
            'sensitivity': grouped['std'].mean() / grouped['mean'].mean() if grouped['mean'].mean() != 0 else np.nan,
            'range': [grouped['mean'].min(), grouped['mean'].max()]
        }

    return results_df, summary
```

---

### 7.2 Structural Break Test

**Description:** Test for performance deterioration over time.

**Code:**

```python
def structural_break_test(
    data,
    signal_column,
    return_column,
    break_candidates=None,
    min_segment_size=36
):
    """
    Test for structural breaks in strategy performance.

    Uses Chow test to detect if performance changed significantly
    at potential break points.
    """
    from scipy import stats

    strategy_return = (
        data[signal_column].shift(1) * data[return_column]
    ).dropna()

    if break_candidates is None:
        # Test at each year boundary
        break_candidates = strategy_return.resample('YE').last().index[1:-1]

    results = []

    for break_date in break_candidates:
        before = strategy_return[strategy_return.index < break_date]
        after = strategy_return[strategy_return.index >= break_date]

        if len(before) < min_segment_size or len(after) < min_segment_size:
            continue

        # Chow test (simplified: compare means)
        t_stat, p_value = stats.ttest_ind(before, after)

        results.append({
            'break_date': break_date,
            'n_before': len(before),
            'n_after': len(after),
            'mean_before': before.mean() * 100,
            'mean_after': after.mean() * 100,
            'sharpe_before': calculate_sharpe(before),
            'sharpe_after': calculate_sharpe(after),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        })

    return pd.DataFrame(results)
```

---

## 8. Quality Metrics

### 8.1 Core Performance Metrics

```python
def calculate_metrics(returns):
    """
    Calculate comprehensive performance metrics.
    """
    if len(returns) < 12:
        return {'error': 'Insufficient data'}

    returns = returns.dropna()

    # Basic stats
    mean_ret = returns.mean()
    std_ret = returns.std()
    total_ret = (1 + returns).prod() - 1

    # Annualized metrics (assuming monthly data)
    ann_ret = (1 + mean_ret) ** 12 - 1
    ann_vol = std_ret * np.sqrt(12)

    # Sharpe Ratio (assuming 2% risk-free rate)
    rf_monthly = 0.02 / 12
    sharpe = (mean_ret - rf_monthly) / std_ret * np.sqrt(12) if std_ret > 0 else 0

    # Sortino Ratio
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(12) if len(downside) > 0 else 0
    sortino = (ann_ret - 0.02) / downside_vol if downside_vol > 0 else 0

    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    # Calmar Ratio
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    win_rate = (returns > 0).mean()

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'ann_return': ann_ret,
        'ann_volatility': ann_vol,
        'total_return': total_ret,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'n_periods': len(returns),
        'mean_return': mean_ret,
        'volatility': std_ret
    }
```

---

### 8.2 Quality Thresholds

| Metric | Minimum | Good | Excellent | Interpretation |
|--------|---------|------|-----------|----------------|
| **Sharpe Ratio** | > 0.3 | > 0.5 | > 1.0 | Risk-adjusted return |
| **Sortino Ratio** | > 0.5 | > 1.0 | > 2.0 | Downside-adjusted return |
| **Calmar Ratio** | > 0.3 | > 0.5 | > 1.0 | Return vs worst drawdown |
| **WFER** | > 0.4 | > 0.6 | > 0.8 | Out-of-sample generalization |
| **Win Rate** | > 50% | > 55% | > 60% | Consistency |
| **Profit Factor** | > 1.0 | > 1.5 | > 2.0 | Gross profit vs loss |
| **Max Drawdown** | < 30% | < 20% | < 10% | Worst peak-to-trough |

---

## 9. Method Selection Guide

### 9.1 Decision Tree

```
                        START
                          │
                          ▼
              ┌───────────────────────┐
              │   Data History Length │
              └───────────┬───────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
      < 5 years     5-15 years       > 15 years
          │               │               │
          ▼               ▼               ▼
     Block Boot       All Methods      Full Suite
     + CPCV                            + Structural
                                       Break Tests

                          │
                          ▼
              ┌───────────────────────┐
              │  Strategy Complexity  │
              └───────────┬───────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
       Simple      Parameter-based    ML-based
          │               │               │
          ▼               ▼               ▼
     Basic +         + Sensitivity     + CPCV
     WFV             Analysis          + Monte Carlo
                                       + Stability
```

### 9.2 Method Combination Matrix

| Scenario | Primary Method | Secondary Methods |
|----------|---------------|-------------------|
| **Initial exploration** | Simple split | Monte Carlo |
| **Parameter optimization** | WFV | Sensitivity analysis |
| **Statistical significance** | CPCV | Monte Carlo |
| **Regime strategy** | Regime conditional | Transition analysis |
| **Production validation** | Full WFV + CPCV | All robustness tests |

### 9.3 Minimum Validation Requirements

For any strategy to be considered validated:

**Level 1 (Minimum):**
- [ ] Walk-Forward Validation with WFER > 0.5
- [ ] Monte Carlo p-value < 0.05

**Level 2 (Recommended):**
- [ ] CPCV with > 80% positive paths
- [ ] Stable performance across regimes
- [ ] Parameter sensitivity within reasonable bounds

**Level 3 (Full):**
- [ ] No significant structural breaks
- [ ] Performance robust to transaction costs
- [ ] Positive regime transition performance
- [ ] Consistent across sub-periods

---

## Helper Functions

```python
def calculate_sharpe(returns, rf=0.02):
    """Calculate annualized Sharpe ratio."""
    excess = returns.mean() - rf/12
    return excess / returns.std() * np.sqrt(12) if returns.std() > 0 else 0


def calculate_max_drawdown(returns):
    """Calculate maximum drawdown from return series."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_calmar(returns, rf=0.02):
    """Calculate Calmar ratio."""
    ann_return = (1 + returns.mean()) ** 12 - 1 - rf
    max_dd = abs(calculate_max_drawdown(returns))
    return ann_return / max_dd if max_dd > 0 else 0
```

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-24 | RA Cheryl | Initial backtest methodology catalog |
