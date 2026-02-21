#!/usr/bin/env python3
"""
Populate analysis_results DB table for all analyses.

Loads each analysis parquet, computes standardized metrics, and stores
structured results in the SQLite config database.

Usage:
    python script/populate_results.py              # All analyses
    python script/populate_results.py spy_hy_ig_spread  # Single analysis
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from scipy import stats

from src.dashboard.config_db import (
    get_all_analyses, store_results_batch, init_db
)
from src.dashboard.analysis_engine import (
    leadlag_analysis, find_optimal_lag, granger_bidirectional,
    identify_deepdive_lags, rolling_correlation, correlation_with_pvalues,
    create_derivatives, regime_performance
)
from src.dashboard.data_loader import load_analysis_data
from src.dashboard.config_db import resolve_columns


def _to_float(val):
    """Convert numpy types to Python float for JSON serialization."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return float(val)


def compute_results(analysis_id: str) -> list:
    """Compute all standard metrics for one analysis."""
    results = []

    # Load data
    data = load_analysis_data(analysis_id)
    if data.empty:
        print(f"  SKIP {analysis_id}: no data")
        return []

    resolved = resolve_columns(analysis_id, data)
    data = resolved['data']
    indicator_col = resolved['indicator_col']
    return_col = resolved['return_col']
    price_col = resolved.get('price_col')

    if not indicator_col or not return_col:
        print(f"  SKIP {analysis_id}: missing columns")
        return []

    # --- Correlation section ---
    # Level correlation (indicator vs price or return)
    level_target = price_col or return_col
    corr_level = correlation_with_pvalues(data[indicator_col], data[level_target])
    results.extend([
        {'section': 'correlation', 'metric': 'pearson_r_level', 'value': _to_float(corr_level['correlation'])},
        {'section': 'correlation', 'metric': 'pearson_p_level', 'value': _to_float(corr_level['pvalue'])},
        {'section': 'correlation', 'metric': 'pearson_n_level', 'value': _to_float(corr_level['n_obs'])},
    ])

    # Change correlation (MoM vs returns)
    def _stem(col_name):
        return col_name[:-6] if col_name.endswith('_Level') else col_name

    ind_stem = _stem(indicator_col)
    mom_col = f"{ind_stem}_MoM"
    if mom_col not in data.columns:
        derivs = create_derivatives(data[indicator_col], ind_stem)
        for c in derivs.columns:
            if c not in data.columns:
                data[c] = derivs[c]

    if mom_col in data.columns:
        corr_change = correlation_with_pvalues(data[mom_col], data[return_col])
        results.extend([
            {'section': 'correlation', 'metric': 'pearson_r_change', 'value': _to_float(corr_change['correlation'])},
            {'section': 'correlation', 'metric': 'pearson_p_change', 'value': _to_float(corr_change['pvalue'])},
            {'section': 'correlation', 'metric': 'pearson_n_change', 'value': _to_float(corr_change['n_obs'])},
        ])

        # Rolling correlation (12M)
        rc = rolling_correlation(data[mom_col], data[return_col], window=12)
        if not rc.empty and rc.notna().any():
            results.extend([
                {'section': 'correlation', 'metric': 'rolling_corr_mean', 'value': _to_float(rc.mean())},
                {'section': 'correlation', 'metric': 'rolling_corr_std', 'value': _to_float(rc.std())},
                {'section': 'correlation', 'metric': 'rolling_corr_min', 'value': _to_float(rc.min())},
                {'section': 'correlation', 'metric': 'rolling_corr_max', 'value': _to_float(rc.max())},
            ])

        # --- Lead-Lag section ---
        ll = leadlag_analysis(data, mom_col, return_col, max_lag=12)
        if not ll.empty:
            opt = find_optimal_lag(ll)
            results.extend([
                {'section': 'leadlag', 'metric': 'optimal_lag', 'value': _to_float(opt['optimal_lag'])},
                {'section': 'leadlag', 'metric': 'optimal_lag_r', 'value': _to_float(opt['correlation'])},
                {'section': 'leadlag', 'metric': 'optimal_lag_p', 'value': _to_float(opt.get('pvalue'))},
                {'section': 'leadlag', 'metric': 'optimal_lag_n', 'value': _to_float(opt.get('n_obs'))},
            ])

            # Significant lags
            sig = ll[ll['pvalue'] < 0.05]
            sig_list = [{'lag': int(r['lag']), 'r': round(float(r['correlation']), 4),
                         'p': round(float(r['pvalue']), 4)}
                        for _, r in sig.iterrows()]
            results.append({'section': 'leadlag', 'metric': 'significant_lags',
                           'metadata': sig_list})

        # --- Granger section (bi-directional) ---
        bg = granger_bidirectional(data, mom_col, return_col, max_lag=6)
        results.extend([
            {'section': 'granger', 'metric': 'fwd_best_pvalue', 'value': _to_float(bg['fwd_best']['pvalue'])},
            {'section': 'granger', 'metric': 'fwd_best_lag', 'value': _to_float(bg['fwd_best']['lag'])},
            {'section': 'granger', 'metric': 'fwd_best_fstat', 'value': _to_float(bg['fwd_best']['f_stat'])},
            {'section': 'granger', 'metric': 'rev_best_pvalue', 'value': _to_float(bg['rev_best']['pvalue'])},
            {'section': 'granger', 'metric': 'rev_best_lag', 'value': _to_float(bg['rev_best']['lag'])},
            {'section': 'granger', 'metric': 'rev_best_fstat', 'value': _to_float(bg['rev_best']['f_stat'])},
            {'section': 'granger', 'metric': 'direction', 'value_text': bg['direction']},
        ])

        # Full Granger results for display
        if not bg['forward'].empty:
            fwd_list = [{'lag': int(r['lag']), 'f': round(float(r['f_statistic']), 2),
                         'p': round(float(r['pvalue']), 4)}
                        for _, r in bg['forward'].iterrows()]
            results.append({'section': 'granger', 'metric': 'fwd_all', 'metadata': fwd_list})
        if not bg['reverse'].empty:
            rev_list = [{'lag': int(r['lag']), 'f': round(float(r['f_statistic']), 2),
                         'p': round(float(r['pvalue']), 4)}
                        for _, r in bg['reverse'].iterrows()]
            results.append({'section': 'granger', 'metric': 'rev_all', 'metadata': rev_list})

        # Deep-dive lags
        dd = identify_deepdive_lags(ll, bg['forward'], bg['reverse'], top_n=3)
        dd_serializable = [{'lag': d['lag'], 'source': d['source'],
                            'r': round(float(d['r']), 4) if d['r'] is not None else None,
                            'p': round(float(d['p']), 4) if d['p'] is not None else None}
                           for d in dd]
        results.append({'section': 'leadlag', 'metric': 'deepdive_lags',
                       'metadata': dd_serializable})

    # --- Regime section ---
    regime_col = 'Regime' if 'Regime' in data.columns else 'regime' if 'regime' in data.columns else None
    if regime_col:
        rp = regime_performance(data, regime_col, return_col)
        if not rp.empty:
            perf_dict = {}
            for _, row in rp.iterrows():
                perf_dict[row['regime']] = {
                    'mean': round(float(row['mean_return']), 4),
                    'std': round(float(row['std_return']), 4),
                    'sharpe': round(float(row['sharpe_ratio']), 2),
                    'n': int(row['n_periods']),
                    'pct_positive': round(float(row['pct_positive']), 3),
                }
            results.append({'section': 'regime', 'metric': 'perf_summary',
                           'metadata': perf_dict})

            # t-test between regimes (if exactly 2)
            regimes = data[regime_col].dropna().unique()
            if len(regimes) == 2:
                r1 = data.loc[data[regime_col] == regimes[0], return_col].dropna()
                r2 = data.loc[data[regime_col] == regimes[1], return_col].dropna()
                if len(r1) >= 5 and len(r2) >= 5:
                    t_stat, t_pval = stats.ttest_ind(r1, r2, equal_var=False)
                    results.extend([
                        {'section': 'regime', 'metric': 't_test_pvalue', 'value': _to_float(t_pval)},
                        {'section': 'regime', 'metric': 't_test_statistic', 'value': _to_float(t_stat)},
                    ])

    return results


def main():
    target_id = sys.argv[1] if len(sys.argv) > 1 else None

    init_db()
    analyses = get_all_analyses()

    for analysis in analyses:
        aid = analysis['id']
        if target_id and aid != target_id:
            continue

        print(f"Processing {aid}...")
        try:
            results = compute_results(aid)
            if results:
                store_results_batch(aid, results)
                print(f"  Stored {len(results)} results")
            else:
                print(f"  No results computed")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == '__main__':
    main()
