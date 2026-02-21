# Design: Annotations, Bi-directional Lead-Lag, and Deep-Dive Verification

**Date:** 2026-02-21
**Status:** Approved
**Triggered by:** Comparison with KS HY-IG Credit Spread vs SPY Analysis PDF

---

## Problem Statement

The dashboard displays charts with minimal contextual text. Readers who are not
domain experts cannot interpret what the quantitative findings mean. Three gaps
were identified by comparing against a reference PDF analysis:

1. **No textual interpretations** — charts lack before/after narrative explaining
   context, findings, and actionable verdicts
2. **One-way Granger causality** — Lead-Lag page only tests indicator → target,
   missing the reverse direction and bi-directional classification
3. **No deep-dive verification** — when lead-lag identifies significant lags,
   there is no scatter-plot verification to confirm or explain discrepancies
   between Granger significance and simple Pearson correlation

## Solution Overview

### Architecture: DB-Stored Results + Auto-Generated Interpretations + Overrides

```
Analysis Script (producer)
    │
    ├─► Parquet file (time series data — unchanged)
    │
    └─► SQLite DB: analysis_results table (structured quantitative findings)

Dashboard (consumer)
    │
    ├─► Reads parquet for charts (unchanged)
    ├─► Reads analysis_results for numbers
    ├─► Interpretation engine generates prose from templates + results
    └─► Reads analysis_annotations for human overrides (replaces auto-text)
```

**Key principle:** Analysis scripts are the single producer of quantitative
results. Dashboard never recomputes what scripts already computed. The DB is
the single source of truth for both numbers and narrative.

---

## Schema Design

### Table: `analysis_results`

Stores pre-computed quantitative findings. One row per analysis per metric.

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
```

**Section values:** `correlation`, `leadlag`, `granger`, `regime`, `backtest`

**Standard metrics per section:**

| Section | Metric | Type | Description |
|---------|--------|------|-------------|
| correlation | pearson_r_level | REAL | Level-vs-level Pearson r |
| correlation | pearson_p_level | REAL | p-value for level correlation |
| correlation | pearson_n_level | REAL | Observation count |
| correlation | pearson_r_change | REAL | MoM-vs-returns Pearson r |
| correlation | pearson_p_change | REAL | p-value for change correlation |
| correlation | pearson_n_change | REAL | Observation count |
| correlation | rolling_corr_mean | REAL | Mean of rolling 12M correlation |
| correlation | rolling_corr_std | REAL | Std of rolling correlation |
| correlation | rolling_corr_min | REAL | Min rolling correlation |
| correlation | rolling_corr_max | REAL | Max rolling correlation |
| leadlag | optimal_lag | REAL | Best lag (months) |
| leadlag | optimal_lag_r | REAL | Correlation at optimal lag |
| leadlag | optimal_lag_p | REAL | P-value at optimal lag |
| leadlag | optimal_lag_n | REAL | Observations at optimal lag |
| leadlag | significant_lags | — | metadata: JSON array of {lag, r, p} |
| leadlag | deepdive_lags | — | metadata: JSON array of top lags for scatter |
| granger | fwd_best_pvalue | REAL | Best p-value: indicator → target |
| granger | fwd_best_lag | REAL | Lag of best forward p-value |
| granger | fwd_best_fstat | REAL | F-statistic at best forward lag |
| granger | rev_best_pvalue | REAL | Best p-value: target → indicator |
| granger | rev_best_lag | REAL | Lag of best reverse p-value |
| granger | rev_best_fstat | REAL | F-statistic at best reverse lag |
| granger | direction | — | value_text: predictive/confirmatory/bi-directional/independent |
| granger | fwd_all | — | metadata: JSON array of all lag results |
| granger | rev_all | — | metadata: JSON array of all lag results |
| regime | perf_summary | — | metadata: JSON {regime: {mean, std, sharpe, n, pct_positive}} |
| regime | t_test_pvalue | REAL | Regime difference significance |
| regime | t_test_statistic | REAL | t-statistic for regime difference |
| backtest | strategy_sharpe | REAL | Strategy Sharpe ratio |
| backtest | benchmark_sharpe | REAL | Benchmark Sharpe ratio |
| backtest | exposure_pct | REAL | % time in market |

### Table: `analysis_annotations`

Stores human-written text overrides. Optional — auto-generated text is used
when no override exists.

```sql
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

**Section keys:**

```
overview.summary
overview.timeseries
overview.regime
correlation.level
correlation.change
correlation.rolling
leadlag.crosscorr
leadlag.granger_fwd
leadlag.granger_rev
leadlag.granger_verdict
leadlag.deepdive
regime.performance
regime.acceleration
backtest.strategy
```

Each annotation has four optional fields:
- **intro**: Context shown BEFORE the chart (what to look for)
- **finding**: Key quantitative result (the fact)
- **interpretation**: What it means (the meaning)
- **verdict**: Actionable conclusion (the implication)

---

## Interpretation Engine

### Module: `src/dashboard/interpretation.py`

Generates template-based prose from `analysis_results` values. Each section
has a template function that produces a dict of `{intro, finding, interpretation, verdict}`.

**Resolution order:**
1. Check `analysis_annotations` for override on `(analysis_id, section_key)`
2. If override field is non-NULL, use it for that field
3. For NULL fields, auto-generate from `analysis_results` + templates

**Template logic example — correlation strength:**

```python
def _correlation_strength(r: float) -> str:
    abs_r = abs(r)
    if abs_r < 0.10: return "negligible"
    if abs_r < 0.30: return "weak"
    if abs_r < 0.50: return "moderate"
    if abs_r < 0.70: return "strong"
    return "very strong"

def _correlation_direction(r: float) -> str:
    return "positive" if r > 0 else "negative"
```

**Template output for `correlation.level`:**

> The level correlation between {indicator_name} and {target_name} is
> **r = {r:.4f}** (p {p_qualifier}, n = {n}). This is a **{strength}
> {direction}** relationship — as {indicator_name} {rises_or_falls},
> {target_name} tends to {opposite_or_same}.
>
> *Note: Level correlations between trending series can be inflated by
> common trends. The change-based analysis below controls for this.*

### Granger Bi-directional Classification

```python
def classify_granger_direction(fwd_pvalue, rev_pvalue, alpha=0.05):
    fwd_sig = fwd_pvalue < alpha
    rev_sig = rev_pvalue < alpha
    if fwd_sig and not rev_sig: return "predictive"
    if not fwd_sig and rev_sig: return "confirmatory"
    if fwd_sig and rev_sig:     return "bi-directional"
    return "independent"
```

| Classification | Indicator→Target | Target→Indicator | Dashboard Text |
|---|---|---|---|
| predictive | Significant | Not significant | "{indicator} is a **leading signal** for {target}" |
| confirmatory | Not significant | Significant | "{target} moves first — {indicator} **confirms** afterward" |
| bi-directional | Significant | Significant | "The relationship is **bi-directional** — both influence each other" |
| independent | Not significant | Not significant | "**No Granger-causal relationship** detected at monthly frequency" |

---

## Dashboard Page Enhancements

### Overview Page (`2_📊_Overview.py`)

**Added:**
- Summary annotation block at top (auto-generated overview.summary)
- Regime interpretation text below regime performance table
- Box plot alongside existing bar chart (using existing `plot_regime_boxplot()`)

### Correlation Page (`4_📈_Correlation.py`)

**Added:**
- Three sections with annotations: Level, Change, Rolling
- Scatter plots for level and change (using existing `plot_scatter()`)
- Rolling correlation chart (using existing `plot_rolling_correlation()`)
- Template-based interpretation text before each chart

### Lead-Lag Page (`5_🔄_Lead_Lag.py`)

**Added:**
- Confidence interval bands on cross-correlation chart (±2σ)
- Bi-directional Granger: test both directions, display side-by-side
- Granger verdict with classification badge
- Deep-dive section: scatter plots at top significant lags with regression lines and Pearson r annotation
- Interpretation text explaining Granger vs Pearson discrepancies

### Regimes Page (`6_🎯_Regimes.py`)

**Added:**
- Regime performance interpretation text

### Analysis Scripts (existing `script/analyze_*.py`)

**Added to each script (Phase 7):**
- Store results to `analysis_results` table via new helper function
- Compute bi-directional Granger causality
- Identify deep-dive lag candidates

---

## Analysis Engine Additions (`analysis_engine.py`)

### New Functions

```python
def granger_bidirectional(df, x_col, y_col, max_lag=6) -> Dict:
    """
    Test Granger causality in both directions.

    Returns:
        {
            'forward': DataFrame (x→y results),
            'reverse': DataFrame (y→x results),
            'fwd_best': {'lag', 'f_stat', 'pvalue'},
            'rev_best': {'lag', 'f_stat', 'pvalue'},
            'direction': 'predictive'|'confirmatory'|'bi-directional'|'independent'
        }
    """

def identify_deepdive_lags(leadlag_results, granger_results=None, top_n=2) -> List[Dict]:
    """
    Identify lags worth deep-diving.

    Selection criteria (in priority order):
    1. Granger-significant lags (if any)
    2. Optimal lag from cross-correlation
    3. Top-N significant cross-correlation lags

    Returns list of {lag, source, r, p} dicts.
    """

def lag_scatter_data(df, x_col, y_col, lag) -> pd.DataFrame:
    """
    Create scatter-ready DataFrame at a specific lag.

    Returns DataFrame with columns: x_lagged, y, suitable for plot_scatter().
    """
```

---

## Config DB Additions (`config_db.py`)

### New Functions

```python
def store_result(analysis_id, section, metric, value=None,
                 value_text=None, metadata=None):
    """Store or update a single analysis result."""

def store_results_batch(analysis_id, results: List[Dict]):
    """Store multiple results in one transaction."""

def get_results(analysis_id, section=None) -> List[Dict]:
    """Get results for an analysis, optionally filtered by section."""

def get_result(analysis_id, section, metric) -> Optional[Dict]:
    """Get a single result value."""

def get_annotation(analysis_id, section_key) -> Optional[Dict]:
    """Get annotation override for a section."""

def store_annotation(analysis_id, section_key, intro=None,
                     finding=None, interpretation=None, verdict=None):
    """Store or update an annotation override."""
```

---

## Seed Script Updates (`seed_config_db.py`)

Add `RESULTS_SEED` dict with pre-computed results for all 10 analyses.
This will be populated by running each analysis script once, extracting
the key metrics, and hardcoding them in the seed.

For the initial implementation, we will:
1. Add the DB schema (tables + indexes)
2. Add the config_db helper functions
3. Build the interpretation engine
4. Update dashboard pages to render annotations
5. Add bi-directional Granger + deep-dive to analysis engine
6. Update the Lead-Lag page to use new analysis functions
7. Populate results for spy_hy_ig_spread as the pilot
8. Run all 10 analysis scripts to populate results for remaining analyses
9. Update SOP docs

---

## Concrete Example: SPY vs HY-IG Credit Spread

### DB Rows in `analysis_results`

| section | metric | value | value_text | metadata |
|---------|--------|-------|------------|----------|
| correlation | pearson_r_level | -0.4535 | | |
| correlation | pearson_p_level | 0.0 | | |
| correlation | pearson_n_level | 339 | | |
| correlation | pearson_r_change | -0.6666 | | |
| correlation | pearson_p_change | 0.0 | | |
| correlation | pearson_n_change | 338 | | |
| correlation | rolling_corr_mean | -0.6329 | | |
| correlation | rolling_corr_std | 0.2541 | | |
| correlation | rolling_corr_min | -0.958 | | |
| correlation | rolling_corr_max | 0.376 | | |
| leadlag | optimal_lag | 0 | | |
| leadlag | optimal_lag_r | -0.6666 | | |
| leadlag | optimal_lag_p | 0.0 | | |
| leadlag | optimal_lag_n | 338 | | |
| leadlag | significant_lags | | | `[{"lag":0,"r":-0.667,"p":0.0},{"lag":-1,"r":-0.148,"p":0.006}]` |
| leadlag | deepdive_lags | | | `[{"lag":0,"r":-0.667,"p":0.0},{"lag":-1,"r":-0.148,"p":0.006}]` |
| granger | fwd_best_pvalue | 0.66 | | `{"lag":1,"f_stat":0.19}` |
| granger | fwd_best_lag | 1 | | |
| granger | rev_best_pvalue | 0.082 | | `{"lag":1,"f_stat":3.04}` |
| granger | rev_best_lag | 1 | | |
| granger | direction | | independent | |
| granger | fwd_all | | | `[{"lag":1,"f":0.19,"p":0.66},...,{"lag":6,"f":0.37,"p":0.90}]` |
| granger | rev_all | | | `[{"lag":1,"f":3.04,"p":0.082},...,{"lag":6,"f":1.29,"p":0.26}]` |
| regime | perf_summary | | | `{"Spread Tightening":{"mean":0.0166,"std":0.0304,"sharpe":1.889,"n":178,"pct_positive":0.769},"Spread Widening":{"mean":-0.0009,"std":0.0541,"sharpe":-0.058,"n":160,"pct_positive":0.494}}` |
| regime | t_test_pvalue | 0.0001 | | |

### Auto-Generated Text (no overrides)

**overview.summary:**
> The HY-IG Credit Spread shows a **moderate negative** level correlation
> (r = -0.45) with SPY. The MoM change correlation is **strong negative**
> (r = -0.67). Granger causality testing reveals an **independent**
> relationship — neither series causally leads the other at monthly frequency.
> Regime analysis shows **highly significant** performance differences:
> Spread Tightening delivers +1.66%/mo (Sharpe 1.89) vs -0.09%/mo
> (Sharpe -0.06) during Spread Widening.

**leadlag.granger_verdict:**
> **Verdict: Independent (at monthly frequency)**
> Neither direction shows significant Granger causality at p < 0.05.
> The strong contemporaneous correlation (r = -0.67) reflects simultaneous
> risk repricing across credit and equity markets, not a causal lead-lag
> dynamic.

---

## SOP Updates Required

### `docs/sop/unified_analysis_sop.md`

**Phase 2 additions:**
- Rolling correlation (12M window) as standard output
- Scatter plots for level and change correlations
- Interpretation text requirements

**Phase 3 additions:**
- Bi-directional Granger causality testing (MANDATORY)
- Granger direction classification: predictive / confirmatory / bi-directional / independent
- Deep-dive verification: scatter plot at top significant lags
- When Granger significance contradicts simple Pearson, document the distinction

**Phase 7 additions:**
- Store results to `analysis_results` table
- Store results via `store_results_batch()` helper
- Populate bi-directional Granger and deep-dive results

### `docs/sop/dashboard_component_specs.md`

- Add interpretation engine spec
- Add annotation rendering pattern
- Document new components usage (scatter, rolling correlation, box plot)

---

## Verification Plan

1. Build schema + helpers + interpretation engine
2. Pilot with spy_hy_ig_spread — populate results, verify auto-generated text
3. Update Lead-Lag page with bi-directional Granger + deep-dive
4. Update Correlation page with scatter + rolling + annotations
5. Update Overview page with summary annotation + box plot
6. Docker build + test all pages for spy_hy_ig_spread
7. Run remaining 9 analysis scripts to populate their results
8. Test all 10 analyses across all pages
9. Update SOP documents
10. Commit and push
