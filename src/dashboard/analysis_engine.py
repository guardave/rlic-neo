"""
Analysis Engine for RLIC Dashboard.

Implements unified analysis pipeline from SOP:
- Correlation analysis
- Lead-lag analysis
- Granger causality testing
- Regime definition and performance
- Backtesting functions
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import correlate
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import warnings

warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# =============================================================================
# Data Loading
# =============================================================================

def load_cached_data(filename: str) -> pd.DataFrame:
    """Load cached parquet data."""
    filepath = DATA_DIR / filename
    if filepath.exists():
        return pd.read_parquet(filepath)
    raise FileNotFoundError(f"Data file not found: {filepath}")


def load_indicator_data() -> pd.DataFrame:
    """Load monthly indicator data with phases."""
    return load_cached_data("monthly_with_best_phases.parquet")


def load_sector_prices() -> pd.DataFrame:
    """Load sector ETF prices."""
    return load_cached_data("sector_prices.parquet")


def load_ff12_industries() -> pd.DataFrame:
    """Load Fama-French 12 industry returns."""
    return load_cached_data("ff_12_industries.parquet")


# =============================================================================
# Derivative Series Creation
# =============================================================================

def create_derivatives(series: pd.Series, name: str = None) -> pd.DataFrame:
    """
    Create derivative series: Level, MoM, QoQ, YoY.

    Args:
        series: Input time series
        name: Base name for columns (defaults to series name)

    Returns:
        DataFrame with Level, MoM, QoQ, YoY columns
    """
    if name is None:
        name = series.name or "value"

    derivatives = pd.DataFrame(index=series.index)
    derivatives[f"{name}_Level"] = series
    derivatives[f"{name}_MoM"] = series.pct_change(1)
    derivatives[f"{name}_QoQ"] = series.pct_change(3)
    derivatives[f"{name}_YoY"] = series.pct_change(12)

    return derivatives


def create_direction_signal(series: pd.Series, short_window: int = 3,
                            long_window: int = 6) -> pd.Series:
    """
    Create direction signal (+1/-1) based on moving average crossover.

    Args:
        series: Input series
        short_window: Short MA window
        long_window: Long MA window

    Returns:
        Series with +1 (rising) or -1 (falling)
    """
    short_ma = series.rolling(short_window).mean()
    long_ma = series.rolling(long_window).mean()
    return np.sign(short_ma - long_ma)


# =============================================================================
# Correlation Analysis
# =============================================================================

def correlation_analysis(df: pd.DataFrame, x_cols: List[str],
                        y_cols: List[str]) -> pd.DataFrame:
    """
    Compute correlation matrix between X and Y columns.

    Args:
        df: DataFrame with all columns
        x_cols: List of X column names (indicators)
        y_cols: List of Y column names (targets)

    Returns:
        Correlation matrix DataFrame
    """
    # Filter to valid data
    all_cols = x_cols + y_cols
    valid_df = df[all_cols].dropna()

    # Compute correlation
    corr_matrix = valid_df.corr()

    # Return X vs Y portion
    return corr_matrix.loc[x_cols, y_cols]


def correlation_with_pvalues(x: pd.Series, y: pd.Series) -> Dict:
    """
    Compute Pearson correlation with p-value.

    Returns:
        Dict with 'correlation', 'pvalue', 'n_obs'
    """
    # Align and drop NaN
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 10:
        return {'correlation': np.nan, 'pvalue': np.nan, 'n_obs': len(valid)}

    corr, pval = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
    return {'correlation': corr, 'pvalue': pval, 'n_obs': len(valid)}


def rolling_correlation(x: pd.Series, y: pd.Series,
                       window: int = 36) -> pd.Series:
    """
    Compute rolling correlation between two series.

    Args:
        x, y: Input series
        window: Rolling window size

    Returns:
        Series of rolling correlations
    """
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(index=aligned.index, dtype=float)

    return aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])


# =============================================================================
# Lead-Lag Analysis
# =============================================================================

def leadlag_analysis(df: pd.DataFrame, x_col: str, y_col: str,
                    max_lag: int = 12) -> pd.DataFrame:
    """
    Compute cross-correlation at different lags.

    Positive lag: X leads Y (X at t predicts Y at t+lag)
    Negative lag: Y leads X

    Args:
        df: DataFrame with x_col and y_col
        x_col: Indicator column name
        y_col: Target column name
        max_lag: Maximum lag to test (both directions)

    Returns:
        DataFrame with lag, correlation, pvalue columns
    """
    results = []

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            # X leads Y: correlate X(t) with Y(t+lag)
            x_series = df[x_col].iloc[:-lag] if lag > 0 else df[x_col]
            y_series = df[y_col].shift(-lag).iloc[:-lag] if lag > 0 else df[y_col]
        elif lag < 0:
            # Y leads X: correlate X(t) with Y(t+lag) where lag is negative
            x_series = df[x_col].iloc[-lag:]
            y_series = df[y_col].shift(-lag).iloc[-lag:]
        else:
            x_series = df[x_col]
            y_series = df[y_col]

        # Align and compute correlation
        valid = pd.concat([x_series, y_series], axis=1).dropna()
        if len(valid) >= 10:
            corr, pval = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
        else:
            corr, pval = np.nan, np.nan

        results.append({
            'lag': lag,
            'correlation': corr,
            'pvalue': pval,
            'n_obs': len(valid)
        })

    return pd.DataFrame(results)


def find_optimal_lag(leadlag_results: pd.DataFrame) -> Dict:
    """
    Find the lag with highest absolute correlation.

    Returns:
        Dict with optimal lag info
    """
    if leadlag_results.empty:
        return {'optimal_lag': 0, 'correlation': np.nan}

    # Filter to significant correlations (p < 0.05)
    sig = leadlag_results[leadlag_results['pvalue'] < 0.05]
    if sig.empty:
        sig = leadlag_results

    # Find max absolute correlation
    idx = sig['correlation'].abs().idxmax()
    row = sig.loc[idx]

    return {
        'optimal_lag': int(row['lag']),
        'correlation': row['correlation'],
        'pvalue': row['pvalue'],
        'n_obs': int(row['n_obs'])
    }


# =============================================================================
# Granger Causality
# =============================================================================

def granger_causality_test(df: pd.DataFrame, x_col: str, y_col: str,
                           max_lag: int = 6) -> pd.DataFrame:
    """
    Test if X Granger-causes Y.

    Args:
        df: DataFrame with x_col and y_col
        x_col: Potential cause variable
        y_col: Potential effect variable
        max_lag: Maximum lag to test

    Returns:
        DataFrame with lag, f_stat, pvalue for each lag
    """
    # Prepare data
    data = df[[y_col, x_col]].dropna()

    if len(data) < max_lag * 3:
        return pd.DataFrame()

    try:
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        output = []
        for lag in range(1, max_lag + 1):
            test_result = results[lag][0]
            # Use F-test result
            f_stat = test_result['ssr_ftest'][0]
            pval = test_result['ssr_ftest'][1]
            output.append({
                'lag': lag,
                'f_statistic': f_stat,
                'pvalue': pval
            })

        return pd.DataFrame(output)

    except Exception as e:
        return pd.DataFrame()


def is_stationary(series: pd.Series, significance: float = 0.05) -> bool:
    """Test if series is stationary using ADF test."""
    clean = series.dropna()
    if len(clean) < 20:
        return False

    try:
        result = adfuller(clean, autolag='AIC')
        return result[1] < significance
    except:
        return False


def granger_bidirectional(df: pd.DataFrame, x_col: str, y_col: str,
                          max_lag: int = 6, alpha: float = 0.05) -> Dict:
    """
    Test Granger causality in both directions and classify relationship.

    Args:
        df: DataFrame with x_col and y_col
        x_col: Indicator column
        y_col: Target column
        max_lag: Maximum lag to test
        alpha: Significance level

    Returns:
        Dict with:
            'forward': DataFrame (x→y results)
            'reverse': DataFrame (y→x results)
            'fwd_best': {'lag', 'f_stat', 'pvalue'}
            'rev_best': {'lag', 'f_stat', 'pvalue'}
            'direction': 'predictive'|'confirmatory'|'bi-directional'|'independent'
    """
    fwd = granger_causality_test(df, x_col, y_col, max_lag=max_lag)
    rev = granger_causality_test(df, y_col, x_col, max_lag=max_lag)

    def _best(results):
        if results.empty:
            return {'lag': None, 'f_stat': None, 'pvalue': 1.0}
        idx = results['pvalue'].idxmin()
        row = results.loc[idx]
        return {'lag': int(row['lag']), 'f_stat': row['f_statistic'], 'pvalue': row['pvalue']}

    fwd_best = _best(fwd)
    rev_best = _best(rev)

    fwd_sig = fwd_best['pvalue'] < alpha
    rev_sig = rev_best['pvalue'] < alpha

    if fwd_sig and not rev_sig:
        direction = 'predictive'
    elif not fwd_sig and rev_sig:
        direction = 'confirmatory'
    elif fwd_sig and rev_sig:
        direction = 'bi-directional'
    else:
        direction = 'independent'

    return {
        'forward': fwd,
        'reverse': rev,
        'fwd_best': fwd_best,
        'rev_best': rev_best,
        'direction': direction,
    }


def identify_deepdive_lags(leadlag_results: pd.DataFrame,
                            granger_fwd: pd.DataFrame = None,
                            granger_rev: pd.DataFrame = None,
                            top_n: int = 3) -> List[Dict]:
    """
    Identify lags worth deep-diving with scatter plots.

    Selection criteria (priority order):
    1. Optimal lag from cross-correlation (always included)
    2. Granger-significant lags (if any, from either direction)
    3. Top-N significant cross-correlation lags by |r|

    Returns:
        List of {lag, source, r, p} dicts, deduplicated by lag value.
    """
    candidates = {}

    # 1. Optimal lag (always first)
    if not leadlag_results.empty:
        sig = leadlag_results[leadlag_results['pvalue'] < 0.05]
        search = sig if not sig.empty else leadlag_results
        idx = search['correlation'].abs().idxmax()
        row = search.loc[idx]
        candidates[int(row['lag'])] = {
            'lag': int(row['lag']),
            'source': 'optimal',
            'r': row['correlation'],
            'p': row['pvalue']
        }

    # 2. Granger-significant lags
    for label, gresults in [('granger_fwd', granger_fwd), ('granger_rev', granger_rev)]:
        if gresults is not None and not gresults.empty:
            sig_g = gresults[gresults['pvalue'] < 0.05]
            for _, grow in sig_g.iterrows():
                lag_val = int(grow['lag'])
                if label == 'granger_rev':
                    lag_val = -lag_val  # Reverse direction
                if lag_val not in candidates:
                    # Find corresponding cross-correlation
                    ll_match = leadlag_results[leadlag_results['lag'] == lag_val]
                    r_val = ll_match['correlation'].iloc[0] if not ll_match.empty else None
                    p_val = ll_match['pvalue'].iloc[0] if not ll_match.empty else None
                    candidates[lag_val] = {
                        'lag': lag_val,
                        'source': label,
                        'r': r_val,
                        'p': p_val
                    }

    # 3. Top-N significant lags by |r|
    if not leadlag_results.empty:
        sig_ll = leadlag_results[leadlag_results['pvalue'] < 0.05].copy()
        if not sig_ll.empty:
            sig_ll['abs_r'] = sig_ll['correlation'].abs()
            sig_ll = sig_ll.sort_values('abs_r', ascending=False)
            for _, row in sig_ll.head(top_n).iterrows():
                lag_val = int(row['lag'])
                if lag_val not in candidates:
                    candidates[lag_val] = {
                        'lag': lag_val,
                        'source': 'crosscorr',
                        'r': row['correlation'],
                        'p': row['pvalue']
                    }

    # Sort by |r| descending, cap at top_n
    result = sorted(candidates.values(), key=lambda x: abs(x.get('r') or 0), reverse=True)
    return result[:top_n]


def lag_scatter_data(df: pd.DataFrame, x_col: str, y_col: str,
                     lag: int) -> pd.DataFrame:
    """
    Create scatter-ready DataFrame at a specific lag.

    Args:
        df: DataFrame with x_col and y_col
        x_col: Indicator column
        y_col: Target column
        lag: Lag value (positive = x leads y)

    Returns:
        DataFrame with 'x_lagged' and 'y' columns, NaN-dropped.
    """
    result = pd.DataFrame(index=df.index)
    if lag != 0:
        result['x_lagged'] = df[x_col].shift(lag)
    else:
        result['x_lagged'] = df[x_col]
    result['y'] = df[y_col]
    return result.dropna()


# =============================================================================
# Regime Definition
# =============================================================================

def define_regimes_direction(df: pd.DataFrame, indicator_col: str,
                             short_window: int = 3, long_window: int = 6) -> pd.Series:
    """
    Define regimes based on direction (Rising/Falling).

    Args:
        df: DataFrame with indicator
        indicator_col: Column to use for regime
        short_window: Short MA window
        long_window: Long MA window

    Returns:
        Series with regime labels ('Rising' or 'Falling')
    """
    series = df[indicator_col]
    direction = create_direction_signal(series, short_window, long_window)
    return direction.map({1: 'Rising', -1: 'Falling'})


def define_regimes_level(df: pd.DataFrame, indicator_col: str,
                        threshold: str = 'median') -> pd.Series:
    """
    Define regimes based on level (High/Low).

    Args:
        df: DataFrame with indicator
        indicator_col: Column to use for regime
        threshold: 'median', 'mean', or numeric value

    Returns:
        Series with regime labels ('High' or 'Low')
    """
    series = df[indicator_col]

    if threshold == 'median':
        thresh = series.median()
    elif threshold == 'mean':
        thresh = series.mean()
    else:
        thresh = float(threshold)

    return (series >= thresh).map({True: 'High', False: 'Low'})


def define_investment_clock_phases(df: pd.DataFrame,
                                   growth_col: str = 'orders_inv_ratio',
                                   inflation_col: str = 'ppi_all') -> pd.Series:
    """
    Define Investment Clock phases.

    Args:
        df: DataFrame with growth and inflation indicators
        growth_col: Column for growth signal
        inflation_col: Column for inflation signal

    Returns:
        Series with phase labels
    """
    # Get direction signals
    growth_dir = create_direction_signal(df[growth_col])
    inflation_dir = create_direction_signal(df[inflation_col])

    # Classify phases
    phases = pd.Series(index=df.index, dtype=str)

    phases[(growth_dir == 1) & (inflation_dir == -1)] = 'Recovery'
    phases[(growth_dir == 1) & (inflation_dir == 1)] = 'Overheat'
    phases[(growth_dir == -1) & (inflation_dir == 1)] = 'Stagflation'
    phases[(growth_dir == -1) & (inflation_dir == -1)] = 'Reflation'

    return phases


# =============================================================================
# Regime Performance Analysis
# =============================================================================

def regime_performance(df: pd.DataFrame, regime_col: str,
                      return_col: str) -> pd.DataFrame:
    """
    Calculate performance statistics by regime.

    Args:
        df: DataFrame with regime and return columns
        regime_col: Column with regime labels
        return_col: Column with returns

    Returns:
        DataFrame with performance stats per regime
    """
    results = []

    for regime in df[regime_col].dropna().unique():
        mask = df[regime_col] == regime
        returns = df.loc[mask, return_col].dropna()

        if len(returns) < 2:
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
            'min_return': returns.min(),
            'max_return': returns.max()
        })

    return pd.DataFrame(results)


def regime_transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """
    Compute regime transition probability matrix.

    Args:
        regimes: Series of regime labels

    Returns:
        DataFrame transition matrix
    """
    clean = regimes.dropna()

    if len(clean) < 2:
        return pd.DataFrame()

    try:
        # Get previous and current regimes
        prev_regime = clean.shift(1).dropna()
        curr_regime = clean.iloc[1:]  # Align with shifted series

        if len(prev_regime) == 0 or len(curr_regime) == 0:
            return pd.DataFrame()

        # Compute transition counts
        transitions = pd.crosstab(prev_regime, curr_regime, normalize='index')

        # Ensure numeric values
        transitions = transitions.astype(float)

        return transitions
    except Exception:
        return pd.DataFrame()


# =============================================================================
# Backtesting
# =============================================================================

def simple_backtest(df: pd.DataFrame, signal_col: str, return_col: str,
                   lag: int = 1) -> pd.DataFrame:
    """
    Run simple backtest: go long when signal > 0, flat otherwise.

    Args:
        df: DataFrame with signal and return columns
        signal_col: Column with trading signal
        return_col: Column with returns
        lag: Signal lag (1 = use yesterday's signal for today's trade)

    Returns:
        DataFrame with strategy returns and cumulative performance
    """
    results = df[[signal_col, return_col]].copy()

    # Apply lag to signal
    results['signal_lagged'] = results[signal_col].shift(lag)

    # Position: 1 if signal > 0, 0 otherwise
    results['position'] = (results['signal_lagged'] > 0).astype(int)

    # Strategy returns
    results['strategy_return'] = results['position'] * results[return_col]
    results['benchmark_return'] = results[return_col]

    # Cumulative returns
    results['strategy_cumulative'] = (1 + results['strategy_return']).cumprod()
    results['benchmark_cumulative'] = (1 + results['benchmark_return']).cumprod()

    return results


def backtest_metrics(returns: pd.Series) -> Dict:
    """
    Calculate backtest performance metrics.

    Args:
        returns: Series of returns

    Returns:
        Dict with performance metrics
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return {}

    total_return = (1 + clean).prod() - 1
    mean_return = clean.mean()
    std_return = clean.std()
    sharpe = mean_return / std_return * np.sqrt(12) if std_return > 0 else 0

    # Max drawdown
    cumulative = (1 + clean).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    max_drawdown = drawdowns.min()

    # Win rate
    win_rate = (clean > 0).mean()

    return {
        'total_return': total_return,
        'annualized_return': mean_return * 12,
        'annualized_volatility': std_return * np.sqrt(12),
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_periods': len(clean)
    }


def regime_conditional_backtest(df: pd.DataFrame, regime_col: str,
                                return_col: str,
                                long_regimes: List[str],
                                short_regimes: List[str] = None,
                                lag: int = 1) -> pd.DataFrame:
    """
    Backtest based on regime signals.

    Args:
        df: DataFrame with regime and return columns
        regime_col: Column with regime labels
        return_col: Column with returns
        long_regimes: List of regimes to go long
        short_regimes: List of regimes to go short (optional)
        lag: Signal lag

    Returns:
        DataFrame with strategy performance
    """
    results = df[[regime_col, return_col]].copy()

    # Create position based on regime
    lagged_regime = results[regime_col].shift(lag)

    position = pd.Series(0, index=df.index)
    position[lagged_regime.isin(long_regimes)] = 1
    if short_regimes:
        position[lagged_regime.isin(short_regimes)] = -1

    results['position'] = position
    results['strategy_return'] = position * results[return_col]
    results['benchmark_return'] = results[return_col]

    # Cumulative
    results['strategy_cumulative'] = (1 + results['strategy_return']).cumprod()
    results['benchmark_cumulative'] = (1 + results['benchmark_return']).cumprod()

    return results


# =============================================================================
# Analysis Summary
# =============================================================================

def run_full_analysis(indicator_series: pd.Series,
                      target_series: pd.Series,
                      indicator_name: str = "Indicator",
                      target_name: str = "Target") -> Dict:
    """
    Run complete analysis pipeline on indicator-target pair.

    Args:
        indicator_series: Economic indicator series
        target_series: Target return series
        indicator_name: Name for indicator
        target_name: Name for target

    Returns:
        Dict with all analysis results
    """
    # Create derivatives
    indicator_derivs = create_derivatives(indicator_series, indicator_name)
    target_derivs = create_derivatives(target_series, target_name)

    # Merge
    df = pd.concat([indicator_derivs, target_derivs], axis=1).dropna()

    # Correlation matrix
    x_cols = [c for c in df.columns if indicator_name in c]
    y_cols = [c for c in df.columns if target_name in c]
    corr_matrix = correlation_analysis(df, x_cols, y_cols)

    # Lead-lag on MoM
    x_mom = f"{indicator_name}_MoM"
    y_mom = f"{target_name}_MoM"
    leadlag_results = leadlag_analysis(df, x_mom, y_mom, max_lag=12)
    optimal_lag = find_optimal_lag(leadlag_results)

    # Granger causality
    granger_results = granger_causality_test(df, x_mom, y_mom, max_lag=6)

    # Regime analysis (Rising/Falling)
    df['regime'] = define_regimes_direction(df, f"{indicator_name}_Level")
    regime_perf = regime_performance(df, 'regime', y_mom)

    return {
        'correlation_matrix': corr_matrix,
        'leadlag_results': leadlag_results,
        'optimal_lag': optimal_lag,
        'granger_results': granger_results,
        'regime_performance': regime_perf,
        'data': df
    }


# =============================================================================
# Utility Functions
# =============================================================================

def get_current_regime(regimes: pd.Series) -> str:
    """Get the most recent regime label."""
    clean = regimes.dropna()
    if len(clean) == 0:
        return "Unknown"
    return clean.iloc[-1]


def get_regime_color(regime: str) -> str:
    """Get color for regime visualization."""
    colors = {
        'Recovery': '#2ecc71',      # Green
        'Overheat': '#e74c3c',      # Red
        'Stagflation': '#9b59b6',   # Purple
        'Reflation': '#3498db',     # Blue
        'Rising': '#27ae60',        # Dark green
        'Falling': '#c0392b',       # Dark red
        'High': '#2980b9',          # Blue
        'Low': '#7f8c8d',           # Gray
        'Unknown': '#bdc3c7'        # Light gray
    }
    return colors.get(regime, '#bdc3c7')


def format_pct(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with specified decimals."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"
