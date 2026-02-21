# Annotations, Bi-directional Lead-Lag & Deep-Dive Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add DB-stored analysis results, auto-generated textual interpretations with human overrides, bi-directional Granger causality, and lag deep-dive scatter verification to the RLIC dashboard.

**Architecture:** New `analysis_results` and `analysis_annotations` tables in the existing SQLite config DB. Analysis scripts produce structured results; an interpretation engine generates template-based prose; dashboard pages render annotations alongside charts. Bi-directional Granger and lag deep-dive are new analysis engine functions.

**Tech Stack:** Python, SQLite, Streamlit, Plotly, scipy, statsmodels (all existing deps)

**Design doc:** `docs/plans/2026-02-21-annotations-bidirectional-deepdive-design.md`

---

## Task 1: DB Schema — Add `analysis_results` and `analysis_annotations` Tables

**Files:**
- Modify: `src/dashboard/config_db.py` (add tables in `init_db()`, lines 36-88)
- Modify: `script/seed_config_db.py` (add seed functions for new tables)

**Step 1: Add table creation to `init_db()`**

In `src/dashboard/config_db.py`, append to the `cursor.executescript(...)` block inside `init_db()` (after the `CREATE INDEX` for `analysis_indicators`, before the closing `"""`):

```sql
CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id TEXT NOT NULL,
    section TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL,
    value_text TEXT,
    metadata TEXT,
    computed_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE,
    UNIQUE(analysis_id, section, metric)
);

CREATE INDEX IF NOT EXISTS idx_results_analysis_section
    ON analysis_results(analysis_id, section);

CREATE TABLE IF NOT EXISTS analysis_annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id TEXT NOT NULL,
    section_key TEXT NOT NULL,
    intro TEXT,
    finding TEXT,
    interpretation TEXT,
    verdict TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE,
    UNIQUE(analysis_id, section_key)
);
```

**Step 2: Add CRUD helper functions to `config_db.py`**

Add after the `_resolve_fallback` function (after line 403):

```python
# =============================================================================
# Analysis Results — Structured Quantitative Findings
# =============================================================================

def store_result(analysis_id: str, section: str, metric: str,
                 value: float = None, value_text: str = None,
                 metadata: Any = None):
    """Store or update a single analysis result."""
    conn = get_connection()
    meta_json = json.dumps(metadata) if metadata is not None else None
    conn.execute("""
        INSERT OR REPLACE INTO analysis_results
        (analysis_id, section, metric, value, value_text, metadata, computed_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    """, (analysis_id, section, metric, value, value_text, meta_json))
    conn.commit()
    conn.close()


def store_results_batch(analysis_id: str, results: List[Dict]):
    """Store multiple results in one transaction.

    Each dict: {section, metric, value?, value_text?, metadata?}
    """
    conn = get_connection()
    cursor = conn.cursor()
    for r in results:
        meta_json = json.dumps(r.get('metadata')) if r.get('metadata') is not None else None
        cursor.execute("""
            INSERT OR REPLACE INTO analysis_results
            (analysis_id, section, metric, value, value_text, metadata, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """, (analysis_id, r['section'], r['metric'],
              r.get('value'), r.get('value_text'), meta_json))
    conn.commit()
    conn.close()


def get_results(analysis_id: str, section: str = None) -> List[Dict[str, Any]]:
    """Get results for an analysis, optionally filtered by section."""
    conn = get_connection()
    if section:
        cursor = conn.execute(
            "SELECT * FROM analysis_results WHERE analysis_id = ? AND section = ?",
            (analysis_id, section))
    else:
        cursor = conn.execute(
            "SELECT * FROM analysis_results WHERE analysis_id = ?",
            (analysis_id,))
    results = []
    for row in cursor.fetchall():
        d = _row_to_dict(row)
        if d.get('metadata'):
            try:
                d['metadata'] = json.loads(d['metadata'])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(d)
    conn.close()
    return results


def get_result(analysis_id: str, section: str, metric: str) -> Optional[Dict]:
    """Get a single result value."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT * FROM analysis_results WHERE analysis_id = ? AND section = ? AND metric = ?",
        (analysis_id, section, metric))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    d = _row_to_dict(row)
    if d.get('metadata'):
        try:
            d['metadata'] = json.loads(d['metadata'])
        except (json.JSONDecodeError, TypeError):
            pass
    return d


# =============================================================================
# Analysis Annotations — Human Override Text
# =============================================================================

def get_annotation(analysis_id: str, section_key: str) -> Optional[Dict]:
    """Get annotation override for a section."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT * FROM analysis_annotations WHERE analysis_id = ? AND section_key = ?",
        (analysis_id, section_key))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return _row_to_dict(row)


def store_annotation(analysis_id: str, section_key: str,
                     intro: str = None, finding: str = None,
                     interpretation: str = None, verdict: str = None):
    """Store or update an annotation override."""
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO analysis_annotations
        (analysis_id, section_key, intro, finding, interpretation, verdict, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    """, (analysis_id, section_key, intro, finding, interpretation, verdict))
    conn.commit()
    conn.close()
```

**Step 3: Verify schema creation**

Run:
```bash
cd /home/david/knows/rlic && python3 -c "
from src.dashboard.config_db import init_db, get_connection
init_db()
conn = get_connection()
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
print([t[0] for t in tables])
conn.close()
"
```
Expected: list includes `analysis_results` and `analysis_annotations`

**Step 4: Commit**

```bash
git add src/dashboard/config_db.py
git commit -m "feat: add analysis_results and analysis_annotations DB tables and CRUD helpers"
```

---

## Task 2: Analysis Engine — Bi-directional Granger & Deep-Dive Functions

**Files:**
- Modify: `src/dashboard/analysis_engine.py` (add 3 new functions after `is_stationary`, line 298)

**Step 1: Add `granger_bidirectional()` function**

Add after `is_stationary()` (after line 298):

```python
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
```

**Step 2: Add `identify_deepdive_lags()` function**

```python
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
```

**Step 3: Add `lag_scatter_data()` function**

```python
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
    if lag > 0:
        result['x_lagged'] = df[x_col].shift(lag)
    elif lag < 0:
        result['x_lagged'] = df[x_col].shift(lag)
    else:
        result['x_lagged'] = df[x_col]
    result['y'] = df[y_col]
    return result.dropna()
```

**Step 4: Verify new functions work**

Run:
```bash
cd /home/david/knows/rlic && python3 -c "
import pandas as pd
from src.dashboard.analysis_engine import granger_bidirectional, identify_deepdive_lags, lag_scatter_data, leadlag_analysis

df = pd.read_parquet('data/spy_hy_ig_spread_full.parquet')
mom_col = 'HY_IG_Spread_MoM' if 'HY_IG_Spread_MoM' in df.columns else None
if not mom_col:
    df['HY_IG_Spread_MoM'] = df['HY_IG_Spread_Level'].pct_change(1)
    mom_col = 'HY_IG_Spread_MoM'

# Bi-directional Granger
bg = granger_bidirectional(df, mom_col, 'SPY_Returns', max_lag=6)
print(f'Direction: {bg[\"direction\"]}')
print(f'Fwd best: {bg[\"fwd_best\"]}')
print(f'Rev best: {bg[\"rev_best\"]}')

# Lead-lag + deep-dive lags
ll = leadlag_analysis(df, mom_col, 'SPY_Returns', max_lag=12)
dd = identify_deepdive_lags(ll, bg['forward'], bg['reverse'], top_n=3)
print(f'Deep-dive lags: {dd}')

# Lag scatter
for d in dd[:2]:
    scatter = lag_scatter_data(df, mom_col, 'SPY_Returns', d['lag'])
    print(f'Lag {d[\"lag\"]}: {len(scatter)} points')
"
```
Expected: direction = 'independent', deep-dive lags identified, scatter data created

**Step 5: Commit**

```bash
git add src/dashboard/analysis_engine.py
git commit -m "feat: add bi-directional Granger, deep-dive lag identification, lag scatter data"
```

---

## Task 3: Interpretation Engine — Template-Based Prose Generation

**Files:**
- Create: `src/dashboard/interpretation.py`

**Step 1: Create the interpretation module**

```python
"""
Interpretation Engine for RLIC Dashboard.

Generates template-based textual annotations from analysis_results DB values.
Human overrides from analysis_annotations replace auto-generated text when present.
"""

import json
from typing import Dict, Optional, Any
from src.dashboard.config_db import (
    get_results, get_result, get_annotation, get_analysis_config
)


# =============================================================================
# Helpers
# =============================================================================

def _correlation_strength(r: float) -> str:
    abs_r = abs(r)
    if abs_r < 0.10: return "negligible"
    if abs_r < 0.30: return "weak"
    if abs_r < 0.50: return "moderate"
    if abs_r < 0.70: return "strong"
    return "very strong"


def _correlation_direction(r: float) -> str:
    return "positive" if r > 0 else "negative"


def _p_qualifier(p: float) -> str:
    if p < 0.001: return "< 0.001"
    if p < 0.01:  return f"= {p:.3f}"
    if p < 0.05:  return f"= {p:.3f}"
    return f"= {p:.2f}"


def _format_r(r: float) -> str:
    return f"{r:.4f}" if r is not None else "N/A"


def _granger_direction_label(direction: str) -> str:
    labels = {
        'predictive': 'Predictive',
        'confirmatory': 'Confirmatory',
        'bi-directional': 'Bi-directional',
        'independent': 'Independent',
    }
    return labels.get(direction, direction)


# =============================================================================
# Section Interpreters — each returns {intro, finding, interpretation, verdict}
# =============================================================================

def _interp_overview_summary(analysis_id: str, indicator_name: str,
                              target_name: str) -> Dict[str, str]:
    r_level = get_result(analysis_id, 'correlation', 'pearson_r_level')
    r_change = get_result(analysis_id, 'correlation', 'pearson_r_change')
    granger_dir = get_result(analysis_id, 'granger', 'direction')
    regime_pval = get_result(analysis_id, 'regime', 't_test_pvalue')

    parts = []
    if r_level:
        rv = r_level['value']
        parts.append(f"{indicator_name} shows a **{_correlation_strength(rv)} "
                     f"{_correlation_direction(rv)}** level correlation (r = {_format_r(rv)}) with {target_name}.")
    if r_change:
        rv = r_change['value']
        parts.append(f"The MoM change correlation is **{_correlation_strength(rv)} "
                     f"{_correlation_direction(rv)}** (r = {_format_r(rv)}).")
    if granger_dir:
        label = _granger_direction_label(granger_dir.get('value_text', ''))
        parts.append(f"Granger causality classification: **{label}**.")
    if regime_pval:
        pv = regime_pval['value']
        sig = "highly significant" if pv < 0.01 else "significant" if pv < 0.05 else "not significant"
        parts.append(f"Regime performance difference is **{sig}** (p {_p_qualifier(pv)}).")

    return {
        'finding': ' '.join(parts) if parts else None,
        'intro': None,
        'interpretation': None,
        'verdict': None,
    }


def _interp_correlation_level(analysis_id: str, indicator_name: str,
                               target_name: str) -> Dict[str, str]:
    r = get_result(analysis_id, 'correlation', 'pearson_r_level')
    p = get_result(analysis_id, 'correlation', 'pearson_p_level')
    n = get_result(analysis_id, 'correlation', 'pearson_n_level')

    if not r:
        return {'intro': None, 'finding': None, 'interpretation': None, 'verdict': None}

    rv, pv, nv = r['value'], p['value'] if p else None, int(n['value']) if n else None

    finding = (f"The level correlation between {indicator_name} and {target_name} is "
               f"**r = {_format_r(rv)}** (p {_p_qualifier(pv)}, n = {nv}). "
               f"This is a **{_correlation_strength(rv)} {_correlation_direction(rv)}** relationship.")

    interpretation = ("Level correlations between trending series can be inflated by common trends. "
                      "The change-based analysis below controls for this.")

    return {'intro': None, 'finding': finding, 'interpretation': interpretation, 'verdict': None}


def _interp_correlation_change(analysis_id: str, indicator_name: str,
                                target_name: str) -> Dict[str, str]:
    r = get_result(analysis_id, 'correlation', 'pearson_r_change')
    p = get_result(analysis_id, 'correlation', 'pearson_p_change')
    n = get_result(analysis_id, 'correlation', 'pearson_n_change')
    r_level = get_result(analysis_id, 'correlation', 'pearson_r_level')

    if not r:
        return {'intro': None, 'finding': None, 'interpretation': None, 'verdict': None}

    rv, pv, nv = r['value'], p['value'] if p else None, int(n['value']) if n else None

    finding = (f"The MoM change correlation is **r = {_format_r(rv)}** "
               f"(p {_p_qualifier(pv)}, n = {nv}), "
               f"which is **{_correlation_strength(rv)} {_correlation_direction(rv)}**.")

    interpretation = None
    if r_level:
        rl = r_level['value']
        if abs(rv) > abs(rl):
            interpretation = ("The change correlation is stronger than the level correlation, "
                              "confirming the relationship is not driven by spurious trend overlap.")
        elif abs(rv) < abs(rl) * 0.7:
            interpretation = ("The change correlation is weaker than the level correlation, "
                              "suggesting some of the level relationship may be driven by common trends.")

    return {'intro': None, 'finding': finding, 'interpretation': interpretation, 'verdict': None}


def _interp_correlation_rolling(analysis_id: str, indicator_name: str,
                                 target_name: str) -> Dict[str, str]:
    mean = get_result(analysis_id, 'correlation', 'rolling_corr_mean')
    std = get_result(analysis_id, 'correlation', 'rolling_corr_std')
    mn = get_result(analysis_id, 'correlation', 'rolling_corr_min')
    mx = get_result(analysis_id, 'correlation', 'rolling_corr_max')

    if not mean:
        return {'intro': None, 'finding': None, 'interpretation': None, 'verdict': None}

    finding = (f"The rolling correlation averages **{_format_r(mean['value'])}** "
               f"with range [{_format_r(mn['value'])} to {_format_r(mx['value'])}].")

    # Check stability
    if std and std['value'] > 0.3:
        interpretation = ("The high variability indicates the relationship is **not stable** over time. "
                          "Periods where the correlation flips sign suggest structural regime changes.")
    elif std and std['value'] > 0.15:
        interpretation = "The relationship shows **moderate variability** over time."
    else:
        interpretation = "The relationship is **relatively stable** over the sample period."

    return {'intro': None, 'finding': finding, 'interpretation': interpretation, 'verdict': None}


def _interp_leadlag_crosscorr(analysis_id: str, indicator_name: str,
                                target_name: str) -> Dict[str, str]:
    opt_lag = get_result(analysis_id, 'leadlag', 'optimal_lag')
    opt_r = get_result(analysis_id, 'leadlag', 'optimal_lag_r')
    sig_lags = get_result(analysis_id, 'leadlag', 'significant_lags')

    if not opt_lag:
        return {'intro': None, 'finding': None, 'interpretation': None, 'verdict': None}

    lag_val = int(opt_lag['value'])
    r_val = opt_r['value'] if opt_r else None
    n_sig = len(sig_lags['metadata']) if sig_lags and sig_lags.get('metadata') else 0

    finding = (f"The strongest cross-correlation is at **lag {lag_val}** "
               f"(r = {_format_r(r_val)}). "
               f"{n_sig} lag(s) are statistically significant (p < 0.05).")

    if lag_val > 0:
        interpretation = (f"{indicator_name} **leads** {target_name} by {lag_val} month(s). "
                          "This suggests predictive potential.")
    elif lag_val < 0:
        interpretation = (f"{target_name} **leads** {indicator_name} by {abs(lag_val)} month(s). "
                          "The target moves before the indicator (confirmatory signal).")
    else:
        interpretation = ("The relationship is **contemporaneous** — movements occur simultaneously "
                          "with no leading signal detected.")

    return {'intro': None, 'finding': finding, 'interpretation': interpretation, 'verdict': None}


def _interp_granger_fwd(analysis_id: str, indicator_name: str,
                         target_name: str) -> Dict[str, str]:
    best_p = get_result(analysis_id, 'granger', 'fwd_best_pvalue')
    best_lag = get_result(analysis_id, 'granger', 'fwd_best_lag')
    best_f = get_result(analysis_id, 'granger', 'fwd_best_fstat')

    if not best_p:
        return {'intro': None, 'finding': None, 'interpretation': None, 'verdict': None}

    intro = f"**Does {indicator_name} Granger-cause {target_name}?**"
    pv = best_p['value']
    if pv < 0.05:
        finding = (f"Yes. Best p-value = {pv:.4f} at lag {int(best_lag['value'])} "
                   f"(F = {best_f['value']:.2f}).")
        interpretation = (f"Past values of {indicator_name} contain information that helps "
                          f"predict future {target_name} beyond what past {target_name} alone can explain.")
    else:
        finding = f"No. Best p-value = {pv:.4f} at lag {int(best_lag['value'])} (not significant at 0.05)."
        interpretation = (f"Past {indicator_name} values do not add predictive power for {target_name} "
                          f"beyond what past {target_name} already provides.")

    return {'intro': intro, 'finding': finding, 'interpretation': interpretation, 'verdict': None}


def _interp_granger_rev(analysis_id: str, indicator_name: str,
                         target_name: str) -> Dict[str, str]:
    best_p = get_result(analysis_id, 'granger', 'rev_best_pvalue')
    best_lag = get_result(analysis_id, 'granger', 'rev_best_lag')
    best_f = get_result(analysis_id, 'granger', 'rev_best_fstat')

    if not best_p:
        return {'intro': None, 'finding': None, 'interpretation': None, 'verdict': None}

    intro = f"**Does {target_name} Granger-cause {indicator_name}?**"
    pv = best_p['value']
    if pv < 0.05:
        finding = (f"Yes. Best p-value = {pv:.4f} at lag {int(best_lag['value'])} "
                   f"(F = {best_f['value']:.2f}).")
        interpretation = (f"There is a feedback effect where {target_name} movements help predict "
                          f"subsequent {indicator_name} changes.")
    elif pv < 0.10:
        finding = (f"Weak evidence. Best p-value = {pv:.4f} at lag {int(best_lag['value'])} — "
                   "suggestive but not significant at 0.05.")
        interpretation = (f"There may be a mild feedback where {target_name} influences subsequent "
                          f"{indicator_name} changes, but the evidence is not conclusive.")
    else:
        finding = f"No. Best p-value = {pv:.4f} (not significant)."
        interpretation = f"Past {target_name} values do not help predict {indicator_name}."

    return {'intro': intro, 'finding': finding, 'interpretation': interpretation, 'verdict': None}


def _interp_granger_verdict(analysis_id: str, indicator_name: str,
                             target_name: str) -> Dict[str, str]:
    direction = get_result(analysis_id, 'granger', 'direction')
    if not direction:
        return {'intro': None, 'finding': None, 'interpretation': None, 'verdict': None}

    d = direction.get('value_text', 'unknown')
    label = _granger_direction_label(d)

    verdicts = {
        'predictive': (f"{indicator_name} is a **leading signal** for {target_name}. "
                       "Past indicator values help predict future target returns."),
        'confirmatory': (f"{target_name} moves first — {indicator_name} **confirms** afterward. "
                         "The indicator is useful for regime classification but not for timing entries."),
        'bi-directional': (f"The relationship is **bi-directional** — both {indicator_name} and "
                           f"{target_name} influence each other. The indicator has predictive power "
                           "but is also influenced by the target (feedback loop)."),
        'independent': (f"**No Granger-causal relationship** detected at monthly frequency. "
                        "Any correlation is contemporaneous, not predictive."),
    }

    return {
        'intro': None,
        'finding': f"**Verdict: {label}**",
        'interpretation': verdicts.get(d, ''),
        'verdict': verdicts.get(d, ''),
    }


def _interp_regime_performance(analysis_id: str, indicator_name: str,
                                target_name: str) -> Dict[str, str]:
    perf = get_result(analysis_id, 'regime', 'perf_summary')
    t_pval = get_result(analysis_id, 'regime', 't_test_pvalue')

    if not perf or not perf.get('metadata'):
        return {'intro': None, 'finding': None, 'interpretation': None, 'verdict': None}

    regimes = perf['metadata']
    parts = []
    for regime_name, stats in regimes.items():
        mean_r = stats.get('mean', 0)
        sharpe = stats.get('sharpe', 0)
        parts.append(f"**{regime_name}**: mean return {mean_r*100:.2f}%/mo (Sharpe {sharpe:.2f})")

    finding = "Regime performance: " + " vs ".join(parts) + "."

    interpretation = None
    if t_pval:
        pv = t_pval['value']
        if pv < 0.01:
            interpretation = f"The regime performance difference is **highly significant** (p {_p_qualifier(pv)})."
        elif pv < 0.05:
            interpretation = f"The regime performance difference is **significant** (p {_p_qualifier(pv)})."
        else:
            interpretation = f"The regime performance difference is **not statistically significant** (p {_p_qualifier(pv)})."

    return {'intro': None, 'finding': finding, 'interpretation': interpretation, 'verdict': None}


# =============================================================================
# Registry: section_key → interpreter function
# =============================================================================

_INTERPRETERS = {
    'overview.summary': _interp_overview_summary,
    'correlation.level': _interp_correlation_level,
    'correlation.change': _interp_correlation_change,
    'correlation.rolling': _interp_correlation_rolling,
    'leadlag.crosscorr': _interp_leadlag_crosscorr,
    'leadlag.granger_fwd': _interp_granger_fwd,
    'leadlag.granger_rev': _interp_granger_rev,
    'leadlag.granger_verdict': _interp_granger_verdict,
    'regime.performance': _interp_regime_performance,
}


# =============================================================================
# Public API
# =============================================================================

def get_interpretation(analysis_id: str, section_key: str,
                       indicator_name: str = "Indicator",
                       target_name: str = "Target") -> Dict[str, Optional[str]]:
    """
    Get interpretation text for a dashboard section.

    Resolution order:
    1. Check analysis_annotations for human override
    2. For any NULL fields, auto-generate from analysis_results + templates
    3. Return dict with {intro, finding, interpretation, verdict}

    Args:
        analysis_id: Analysis identifier
        section_key: Section key (e.g., 'correlation.level')
        indicator_name: Display name for indicator
        target_name: Display name for target

    Returns:
        Dict with 'intro', 'finding', 'interpretation', 'verdict' (any may be None)
    """
    # Auto-generate
    interpreter = _INTERPRETERS.get(section_key)
    auto = interpreter(analysis_id, indicator_name, target_name) if interpreter else {}

    # Check for human override
    override = get_annotation(analysis_id, section_key)
    if override:
        for field in ('intro', 'finding', 'interpretation', 'verdict'):
            if override.get(field):
                auto[field] = override[field]

    return {
        'intro': auto.get('intro'),
        'finding': auto.get('finding'),
        'interpretation': auto.get('interpretation'),
        'verdict': auto.get('verdict'),
    }


def render_annotation(analysis_id: str, section_key: str,
                      indicator_name: str = "Indicator",
                      target_name: str = "Target") -> None:
    """
    Render interpretation text in Streamlit.

    Convenience function that calls get_interpretation() and renders
    non-None fields as markdown.
    """
    import streamlit as st

    interp = get_interpretation(analysis_id, section_key, indicator_name, target_name)

    if interp.get('intro'):
        st.markdown(interp['intro'])
    if interp.get('finding'):
        st.markdown(interp['finding'])
    if interp.get('interpretation'):
        st.markdown(f"*{interp['interpretation']}*")
    if interp.get('verdict'):
        st.success(interp['verdict'])
```

**Step 2: Verify interpretation engine**

Run:
```bash
cd /home/david/knows/rlic && python3 -c "
from src.dashboard.interpretation import get_interpretation
# Will return empty dicts since no results are stored yet
result = get_interpretation('spy_hy_ig_spread', 'correlation.level', 'HY-IG Spread', 'SPY')
print(result)
"
```
Expected: dict with all None values (no results stored yet)

**Step 3: Commit**

```bash
git add src/dashboard/interpretation.py
git commit -m "feat: add interpretation engine with template-based auto-generation and DB override"
```

---

## Task 4: Results Population Script for All 10 Analyses

**Files:**
- Create: `script/populate_results.py`

**Step 1: Create the results population script**

This script loads each analysis parquet, computes all standard metrics, and stores them in the DB. It replaces the need to modify each individual analysis script.

```python
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
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from scipy import stats

from src.dashboard.config_db import (
    get_all_analyses, get_indicator_config, get_analysis_config,
    store_results_batch, init_db
)
from src.dashboard.analysis_engine import (
    leadlag_analysis, find_optimal_lag, granger_bidirectional,
    identify_deepdive_lags, rolling_correlation, correlation_with_pvalues,
    create_derivatives
)
from src.dashboard.data_loader import load_analysis_data
from src.dashboard.config_db import resolve_columns


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
        {'section': 'correlation', 'metric': 'pearson_r_level', 'value': corr_level['correlation']},
        {'section': 'correlation', 'metric': 'pearson_p_level', 'value': corr_level['pvalue']},
        {'section': 'correlation', 'metric': 'pearson_n_level', 'value': corr_level['n_obs']},
    ])

    # Change correlation (MoM vs returns)
    # Ensure MoM exists
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
            {'section': 'correlation', 'metric': 'pearson_r_change', 'value': corr_change['correlation']},
            {'section': 'correlation', 'metric': 'pearson_p_change', 'value': corr_change['pvalue']},
            {'section': 'correlation', 'metric': 'pearson_n_change', 'value': corr_change['n_obs']},
        ])

        # Rolling correlation (12M)
        rc = rolling_correlation(data[mom_col], data[return_col], window=12)
        if not rc.empty and rc.notna().any():
            results.extend([
                {'section': 'correlation', 'metric': 'rolling_corr_mean', 'value': float(rc.mean())},
                {'section': 'correlation', 'metric': 'rolling_corr_std', 'value': float(rc.std())},
                {'section': 'correlation', 'metric': 'rolling_corr_min', 'value': float(rc.min())},
                {'section': 'correlation', 'metric': 'rolling_corr_max', 'value': float(rc.max())},
            ])

        # --- Lead-Lag section ---
        ll = leadlag_analysis(data, mom_col, return_col, max_lag=12)
        if not ll.empty:
            opt = find_optimal_lag(ll)
            results.extend([
                {'section': 'leadlag', 'metric': 'optimal_lag', 'value': opt['optimal_lag']},
                {'section': 'leadlag', 'metric': 'optimal_lag_r', 'value': opt['correlation']},
                {'section': 'leadlag', 'metric': 'optimal_lag_p', 'value': opt.get('pvalue')},
                {'section': 'leadlag', 'metric': 'optimal_lag_n', 'value': opt.get('n_obs')},
            ])

            # Significant lags
            sig = ll[ll['pvalue'] < 0.05]
            sig_list = [{'lag': int(r['lag']), 'r': round(r['correlation'], 4),
                         'p': round(r['pvalue'], 4)}
                        for _, r in sig.iterrows()]
            results.append({'section': 'leadlag', 'metric': 'significant_lags',
                           'metadata': sig_list})

        # --- Granger section (bi-directional) ---
        bg = granger_bidirectional(data, mom_col, return_col, max_lag=6)
        results.extend([
            {'section': 'granger', 'metric': 'fwd_best_pvalue', 'value': bg['fwd_best']['pvalue']},
            {'section': 'granger', 'metric': 'fwd_best_lag', 'value': bg['fwd_best']['lag']},
            {'section': 'granger', 'metric': 'fwd_best_fstat', 'value': bg['fwd_best']['f_stat']},
            {'section': 'granger', 'metric': 'rev_best_pvalue', 'value': bg['rev_best']['pvalue']},
            {'section': 'granger', 'metric': 'rev_best_lag', 'value': bg['rev_best']['lag']},
            {'section': 'granger', 'metric': 'rev_best_fstat', 'value': bg['rev_best']['f_stat']},
            {'section': 'granger', 'metric': 'direction', 'value_text': bg['direction']},
        ])

        # Full Granger results for display
        if not bg['forward'].empty:
            fwd_list = [{'lag': int(r['lag']), 'f': round(r['f_statistic'], 2),
                         'p': round(r['pvalue'], 4)}
                        for _, r in bg['forward'].iterrows()]
            results.append({'section': 'granger', 'metric': 'fwd_all', 'metadata': fwd_list})
        if not bg['reverse'].empty:
            rev_list = [{'lag': int(r['lag']), 'f': round(r['f_statistic'], 2),
                         'p': round(r['pvalue'], 4)}
                        for _, r in bg['reverse'].iterrows()]
            results.append({'section': 'granger', 'metric': 'rev_all', 'metadata': rev_list})

        # Deep-dive lags
        dd = identify_deepdive_lags(ll, bg['forward'], bg['reverse'], top_n=3)
        dd_serializable = [{'lag': d['lag'], 'source': d['source'],
                            'r': round(d['r'], 4) if d['r'] is not None else None,
                            'p': round(d['p'], 4) if d['p'] is not None else None}
                           for d in dd]
        results.append({'section': 'leadlag', 'metric': 'deepdive_lags',
                       'metadata': dd_serializable})

    # --- Regime section ---
    regime_col = 'Regime' if 'Regime' in data.columns else 'regime' if 'regime' in data.columns else None
    if regime_col:
        from src.dashboard.analysis_engine import regime_performance
        rp = regime_performance(data, regime_col, return_col)
        if not rp.empty:
            perf_dict = {}
            for _, row in rp.iterrows():
                perf_dict[row['regime']] = {
                    'mean': round(row['mean_return'], 4),
                    'std': round(row['std_return'], 4),
                    'sharpe': round(row['sharpe_ratio'], 2),
                    'n': int(row['n_periods']),
                    'pct_positive': round(row['pct_positive'], 3),
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
                        {'section': 'regime', 'metric': 't_test_pvalue', 'value': t_pval},
                        {'section': 'regime', 'metric': 't_test_statistic', 'value': t_stat},
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
```

**Step 2: Run for spy_hy_ig_spread to validate**

Run:
```bash
cd /home/david/knows/rlic && python3 script/populate_results.py spy_hy_ig_spread
```
Expected: "Processing spy_hy_ig_spread... Stored N results"

**Step 3: Verify stored results**

Run:
```bash
cd /home/david/knows/rlic && python3 -c "
from src.dashboard.config_db import get_results
results = get_results('spy_hy_ig_spread')
for r in results:
    print(f'{r[\"section\"]:15s} {r[\"metric\"]:25s} val={r.get(\"value\")!s:>12s} txt={r.get(\"value_text\") or \"\":<20s}')
"
```
Expected: ~25+ rows covering correlation, leadlag, granger, regime sections

**Step 4: Verify interpretation engine with stored data**

Run:
```bash
cd /home/david/knows/rlic && python3 -c "
from src.dashboard.interpretation import get_interpretation
for key in ['overview.summary', 'correlation.level', 'correlation.change', 'leadlag.granger_verdict']:
    interp = get_interpretation('spy_hy_ig_spread', key, 'HY-IG Spread', 'SPY')
    print(f'=== {key} ===')
    for field, text in interp.items():
        if text: print(f'  {field}: {text[:100]}...' if len(text or '') > 100 else f'  {field}: {text}')
    print()
"
```
Expected: Auto-generated text for each section with correct values

**Step 5: Run for all 10 analyses**

Run:
```bash
cd /home/david/knows/rlic && python3 script/populate_results.py
```
Expected: All 10 analyses processed, results stored

**Step 6: Commit**

```bash
git add script/populate_results.py
git commit -m "feat: add results population script for all 10 analyses"
```

---

## Task 5: Update Lead-Lag Page — Bi-directional Granger & Deep-Dive

**Files:**
- Modify: `src/dashboard/pages/5_🔄_Lead_Lag.py`

**Step 1: Add imports**

At the top of the file, add to existing imports:

```python
from src.dashboard.analysis_engine import (
    create_derivatives, leadlag_analysis, find_optimal_lag,
    granger_causality_test, granger_bidirectional,
    identify_deepdive_lags, lag_scatter_data, is_stationary,
    correlation_with_pvalues
)
from src.dashboard.components import plot_leadlag_bars, plot_scatter, format_number
from src.dashboard.interpretation import render_annotation
```

**Step 2: Add annotation rendering after the cross-correlation chart**

After the cross-correlation chart section (after the optimal lag interpretation, ~line 94), add:

```python
    # Annotation: cross-correlation interpretation
    render_annotation(analysis_id, 'leadlag.crosscorr',
                     indicator_name=indicator_col, target_name=return_col)
```

**Step 3: Replace one-way Granger with bi-directional**

Replace the current Granger section (lines 97-150) with:

```python
    # Bi-directional Granger Causality
    st.subheader("Granger Causality Test (Bi-directional)")
    st.markdown("""
    Tests whether past values of one series help predict the other,
    tested in **both directions** to classify the relationship.
    """)

    # Stationarity checks (keep existing)
    col1, col2 = st.columns(2)
    with col1:
        ind_stationary = is_stationary(data[mom_col])
        if ind_stationary:
            st.success(f"✅ {mom_col} is stationary (ADF test p < 0.05)")
        else:
            st.warning(f"⚠️ {mom_col} may not be stationary")
    with col2:
        ret_stationary = is_stationary(data[return_col])
        if ret_stationary:
            st.success(f"✅ {return_col} is stationary")
        else:
            st.warning(f"⚠️ {return_col} may not be stationary")

    # Bi-directional Granger
    bg = granger_bidirectional(data, mom_col, return_col, max_lag=granger_max_lag)

    col_fwd, col_rev = st.columns(2)

    with col_fwd:
        st.markdown(f"**{mom_col} → {return_col}**")
        render_annotation(analysis_id, 'leadlag.granger_fwd',
                         indicator_name=mom_col, target_name=return_col)
        if not bg['forward'].empty:
            gf_display = bg['forward'].copy()
            gf_display['significant'] = gf_display['pvalue'] < 0.05
            st.dataframe(
                gf_display.style.format({'f_statistic': '{:.2f}', 'pvalue': '{:.4f}'})
                .map(lambda x: 'background-color: #d4edda' if x else '', subset=['significant']),
                width='stretch'
            )

    with col_rev:
        st.markdown(f"**{return_col} → {mom_col}**")
        render_annotation(analysis_id, 'leadlag.granger_rev',
                         indicator_name=mom_col, target_name=return_col)
        if not bg['reverse'].empty:
            gr_display = bg['reverse'].copy()
            gr_display['significant'] = gr_display['pvalue'] < 0.05
            st.dataframe(
                gr_display.style.format({'f_statistic': '{:.2f}', 'pvalue': '{:.4f}'})
                .map(lambda x: 'background-color: #d4edda' if x else '', subset=['significant']),
                width='stretch'
            )

    # Granger verdict
    render_annotation(analysis_id, 'leadlag.granger_verdict',
                     indicator_name=indicator_col, target_name=return_col)
```

**Step 4: Add deep-dive scatter section**

After the Granger verdict, add:

```python
    # Deep-Dive: Lag-Specific Scatter Verification
    st.subheader("Lag Deep-Dive Verification")
    st.markdown("Scatter plots at key lags to verify cross-correlation and Granger findings.")

    deepdive_lags = identify_deepdive_lags(
        leadlag_results, bg['forward'], bg['reverse'], top_n=3
    )

    if deepdive_lags:
        cols = st.columns(min(len(deepdive_lags), 3))
        for i, dd in enumerate(deepdive_lags[:3]):
            with cols[i]:
                lag_val = dd['lag']
                scatter_df = lag_scatter_data(data, mom_col, return_col, lag_val)
                if not scatter_df.empty:
                    # Compute Pearson r at this lag
                    corr = correlation_with_pvalues(scatter_df['x_lagged'], scatter_df['y'])
                    r_val = corr['correlation']
                    p_val = corr['pvalue']

                    lag_label = f"Lag {lag_val}" if lag_val != 0 else "Lag 0 (contemporaneous)"
                    fig = plot_scatter(
                        scatter_df, 'x_lagged', 'y',
                        title=f"{lag_label}\nr={r_val:.4f}, p={p_val:.4f}"
                    )
                    fig.update_layout(
                        xaxis_title=f"{mom_col} (t{lag_val:+d})" if lag_val != 0 else mom_col,
                        yaxis_title=return_col,
                        height=350
                    )
                    st.plotly_chart(fig, width='stretch')
                    st.caption(f"Source: {dd['source']} | n={len(scatter_df)}")
    else:
        st.info("No significant lags identified for deep-dive analysis.")
```

**Step 5: Docker build and test**

Run:
```bash
cd /home/david/knows/rlic && docker compose up --build -d && sleep 5 && docker compose logs --tail=30 dashboard
```
Expected: No Python errors, dashboard starts successfully

**Step 6: Commit**

```bash
git add src/dashboard/pages/5_🔄_Lead_Lag.py
git commit -m "feat: add bi-directional Granger, deep-dive scatter to Lead-Lag page"
```

---

## Task 6: Update Correlation Page — Add Annotations

**Files:**
- Modify: `src/dashboard/pages/4_📈_Correlation.py`

**Step 1: Add import**

Add to imports:
```python
from src.dashboard.interpretation import render_annotation
```

**Step 2: Add annotations to each section**

After the heatmap + interpretation block (~line 89), before the scatter plot section:

```python
    # Annotation: level correlation
    render_annotation(analysis_id, 'correlation.level',
                     indicator_name=indicator_col, target_name=return_col)
```

After the scatter plot stats (~line 115), before rolling correlation:

```python
    # Annotation: change correlation
    render_annotation(analysis_id, 'correlation.change',
                     indicator_name=indicator_col, target_name=return_col)
```

After the rolling correlation stats (~line 137):

```python
    # Annotation: rolling correlation
    render_annotation(analysis_id, 'correlation.rolling',
                     indicator_name=indicator_col, target_name=return_col)
```

**Step 3: Commit**

```bash
git add src/dashboard/pages/4_📈_Correlation.py
git commit -m "feat: add interpretation annotations to Correlation page"
```

---

## Task 7: Update Overview Page — Summary Annotation & Box Plot

**Files:**
- Modify: `src/dashboard/pages/2_📊_Overview.py`

**Step 1: Add imports**

Add to imports:
```python
from src.dashboard.interpretation import render_annotation
from src.dashboard.components import plot_regime_boxplot
```

**Step 2: Add summary annotation after title**

After `st.title(...)` (line 31), add:

```python
    # Summary annotation (auto-generated or override)
    render_annotation(analysis_id, 'overview.summary',
                     indicator_name=indicator_col if 'indicator_col' in dir() else '',
                     target_name=return_col if 'return_col' in dir() else '')
```

Note: This needs to be placed inside the try block, after `indicator_col` and `return_col` are resolved (after line 48). Move it to after the resolve_columns block.

**Step 3: Add box plot to regime section**

In the regime performance section (~line 138), add a box plot in `col2` alongside the existing bar chart:

Replace the current `col2` content (regime distribution bar chart) with:

```python
    with col2:
        st.subheader("Returns by Regime")
        if 'regime' in data.columns:
            fig_box = plot_regime_boxplot(
                data, 'regime', return_col,
                title="Return Distribution by Regime"
            )
            st.plotly_chart(fig_box, width='stretch')

    # Regime interpretation
    render_annotation(analysis_id, 'regime.performance',
                     indicator_name=indicator_col, target_name=return_col)
```

**Step 4: Commit**

```bash
git add src/dashboard/pages/2_📊_Overview.py
git commit -m "feat: add summary annotation and regime box plot to Overview page"
```

---

## Task 8: Update Seed Script — Schema + Results Seeding

**Files:**
- Modify: `script/seed_config_db.py`

**Step 1: Update main() to seed results after config**

Add to `main()` function, after `seed_indicators(conn)`:

```python
        # Populate analysis results (runs computations)
        from pathlib import Path
        populate_script = Path(__file__).parent / "populate_results.py"
        if populate_script.exists():
            import subprocess
            print("\nPopulating analysis results...")
            subprocess.run([sys.executable, str(populate_script)], check=False)
```

Note: This makes the seed idempotent — re-seeding also refreshes results. The `check=False` prevents seed failure if one analysis has data issues.

**Step 2: Commit**

```bash
git add script/seed_config_db.py
git commit -m "feat: seed script triggers results population after config seeding"
```

---

## Task 9: Update SOP Documents

**Files:**
- Modify: `docs/sop/unified_analysis_sop.md`

**Step 1: Add to Phase 2 (Statistical Analysis)**

Add a new subsection after existing Phase 2 content:

```markdown
### 2.4 Standard Correlation Outputs (NEW)

Every analysis MUST compute and store:

| Metric | Description | Storage |
|--------|-------------|---------|
| Level Pearson r | Indicator level vs target price/level | `analysis_results` |
| Change Pearson r | Indicator MoM vs target returns | `analysis_results` |
| Rolling correlation | 12-month rolling window stats | `analysis_results` |

These are stored via `store_results_batch()` and rendered with auto-generated
interpretations by the dashboard interpretation engine.
```

**Step 2: Add to Phase 3 (Lead-Lag)**

Add after existing Phase 3 content:

```markdown
### 3.4 Bi-directional Granger Causality (NEW — MANDATORY)

ALL analyses must test Granger causality in BOTH directions:

1. **Forward**: Does indicator Granger-cause target?
2. **Reverse**: Does target Granger-cause indicator?

**Classification:**

| Forward | Reverse | Classification | Meaning |
|---------|---------|----------------|---------|
| p < 0.05 | p >= 0.05 | Predictive | Indicator leads target |
| p >= 0.05 | p < 0.05 | Confirmatory | Target moves first |
| p < 0.05 | p < 0.05 | Bi-directional | Feedback loop |
| p >= 0.05 | p >= 0.05 | Independent | No causal relationship |

Use `granger_bidirectional()` from `analysis_engine.py`.

### 3.5 Deep-Dive Lag Verification (NEW)

For the top 3 significant lags identified by cross-correlation or Granger:

1. Create scatter plot at each lag
2. Compute direct Pearson r at that lag
3. If Granger significance contradicts simple Pearson, document the distinction:
   - Granger measures *incremental* predictive power after accounting for
     the target's own history
   - Simple correlation measures the *isolated* relationship
   - Both can be true: Granger-significant but weak Pearson means the signal
     is useful in combination with other data, not in isolation
```

**Step 3: Add to Phase 7 (Documentation)**

Add:

```markdown
### 7.3 Results Storage (NEW — MANDATORY)

Every analysis script MUST store structured results to the `analysis_results`
DB table. Use `script/populate_results.py` or call `store_results_batch()`
directly.

Required results per analysis:
- correlation: pearson_r_level, pearson_r_change, rolling_corr_mean/std/min/max
- leadlag: optimal_lag, optimal_lag_r, significant_lags, deepdive_lags
- granger: fwd_best_pvalue/lag/fstat, rev_best_pvalue/lag/fstat, direction
- regime: perf_summary, t_test_pvalue

Optional overrides:
- Store custom interpretations via `analysis_annotations` table
- Override any auto-generated text by populating the relevant section_key
```

**Step 4: Commit**

```bash
git add docs/sop/unified_analysis_sop.md
git commit -m "docs: update SOP with bi-directional Granger, deep-dive verification, results storage requirements"
```

---

## Task 10: Full Integration Test & Push

**Step 1: Rebuild Docker and test all pages**

```bash
cd /home/david/knows/rlic && docker compose up --build -d && sleep 8
```

**Step 2: Verify all 10 analyses have results**

```bash
cd /home/david/knows/rlic && python3 -c "
from src.dashboard.config_db import get_connection
conn = get_connection()
cursor = conn.execute('SELECT analysis_id, COUNT(*) as n FROM analysis_results GROUP BY analysis_id ORDER BY analysis_id')
for row in cursor.fetchall():
    print(f'{row[0]:25s} {row[1]:3d} results')
conn.close()
"
```
Expected: All 10 analyses with 15+ results each

**Step 3: Check Docker logs for errors**

```bash
docker compose logs --tail=50 dashboard 2>&1 | grep -i error || echo "No errors found"
```

**Step 4: Test interpretation engine across analyses**

```bash
cd /home/david/knows/rlic && python3 -c "
from src.dashboard.config_db import get_all_analyses
from src.dashboard.interpretation import get_interpretation
for a in get_all_analyses():
    aid = a['id']
    interp = get_interpretation(aid, 'leadlag.granger_verdict', 'Indicator', 'Target')
    verdict = interp.get('finding', 'N/A')
    print(f'{aid:25s} {verdict}')
"
```
Expected: Each analysis shows a Granger verdict (Predictive/Confirmatory/Bi-directional/Independent)

**Step 5: Push**

```bash
git push origin main
```

---

## Summary of Deliverables

| # | Task | Files | Commit Message |
|---|------|-------|---------------|
| 1 | DB Schema + CRUD | `config_db.py` | feat: add analysis_results and analysis_annotations tables |
| 2 | Analysis Engine | `analysis_engine.py` | feat: add bi-directional Granger, deep-dive functions |
| 3 | Interpretation Engine | `interpretation.py` (NEW) | feat: add interpretation engine with auto-generation |
| 4 | Results Population | `populate_results.py` (NEW) | feat: add results population script |
| 5 | Lead-Lag Page | `Lead_Lag.py` | feat: bi-directional Granger + deep-dive on Lead-Lag |
| 6 | Correlation Page | `Correlation.py` | feat: add annotations to Correlation page |
| 7 | Overview Page | `Overview.py` | feat: summary annotation + box plot on Overview |
| 8 | Seed Script | `seed_config_db.py` | feat: seed triggers results population |
| 9 | SOP Update | `unified_analysis_sop.md` | docs: update SOP with new requirements |
| 10 | Integration Test | — | Verify + push |
