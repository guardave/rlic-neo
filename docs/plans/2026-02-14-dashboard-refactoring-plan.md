# Implementation Plan: Dashboard Refactoring to SQLite-Driven Architecture

**Date**: 2026-02-14
**Design Doc**: `docs/design_dashboard_refactoring.md`
**Estimated Tasks**: 20

---

## Phase 1: Foundation (Tasks 1-4)

### Task 1: Create `config_db.py` — Schema + DB Initialization

**File**: `src/dashboard/config_db.py` (NEW)

Create the SQLite database module with:
- `init_db()` — creates tables if not exist (analyses + analysis_indicators)
- `get_connection()` — returns sqlite3 connection with Row factory
- `get_all_analyses()` — returns list of analysis dicts
- `get_analysis_config(analysis_id)` — returns single analysis dict
- `get_indicator_config(analysis_id, axis='primary')` — returns indicator config
- Auto-init on import (create DB + tables if missing)

**DB Path**: `data/rlic_config.db`

**Acceptance**: `python -c "from src.dashboard.config_db import init_db; init_db()"` creates DB file.

---

### Task 2: Create `seed_config_db.py` — Populate from Current Hardcoded Values

**File**: `script/seed_config_db.py` (NEW)

Reads current values from:
- `navigation.py` ANALYSES dict → inserts into `analyses` table
- `data_loader.py` existing_files dict → populates `data_file` column
- `Home.py` cards list → populates `caption`, `home_column` columns
- Column detection patterns (manually transcribed from elif chains) → inserts into `analysis_indicators`

Must be idempotent (uses INSERT OR REPLACE).

**Acceptance**: `python script/seed_config_db.py` creates populated DB. Query returns 9 analyses and 10 indicator rows.

---

### Task 3: Create `resolve_columns()` — The Core Column Resolver

**File**: `src/dashboard/config_db.py` (append to Task 1's file)

```python
def resolve_columns(analysis_id, data, context='default'):
    """Replace all 64 elif branches with one function call."""
```

Algorithm:
1. Read indicator config from DB for `analysis_id`
2. If `context='trading'` and `trading_columns` is set → use those
3. Else apply `indicator_columns` (exact) or `indicator_pattern` (search) or `indicator_filter`
4. Apply `indicator_exclude` to filter out unwanted columns
5. Resolve return columns: `return_columns` (exact) or `return_pattern` (suffix) or compute from `price_column`
6. Handle regime: precomputed (copy column) or direction or threshold
7. Populate lag_config dict from DB fields
8. Return resolved dict

**Acceptance**: Write unit test `temp/test_resolve_columns.py` that:
- Loads each of the 9 parquet files
- Calls resolve_columns() for each
- Asserts indicator_col and return_col match what the old elif chain produces

---

### Task 4: Verify resolve_columns() parity with old elif chains

Run the test from Task 3 against all 9 analyses × 2 contexts (default + trading).
Fix any mismatches.

**Acceptance**: All 18 test cases pass (9 analyses × 2 contexts).

---

## Phase 2: Content Extraction (Tasks 5-6)

### Task 5: Extract Qualitative Content to Markdown Files

**Files**: `src/dashboard/content/*.md` (9 NEW files)

Extract each elif block from `Qualitative.py` into a separate markdown file.
Convert Streamlit-specific elements:
- `st.header("...")` → `# ...`
- `st.subheader("...")` → `## ...`
- `st.markdown("...")` → plain markdown
- `st.warning("...")` → `:::warning\n...\n:::`
- `st.info("...")` → `:::info\n...\n:::`
- `st.success("...")` → `:::success\n...\n:::`
- `st.dataframe(pd.DataFrame({...}))` → markdown table
- `st.columns([...])` → `:::columns\n...\n:::` (custom)

**Acceptance**: 9 markdown files exist. Content visually matches current Qualitative page.

---

### Task 6: Create Content Renderer

**File**: `src/dashboard/content_renderer.py` (NEW)

```python
def render_qualitative_content(analysis_id: str):
    """Load and render markdown file for analysis."""
```

Parses the custom extensions:
- `:::warning` blocks → `st.warning()`
- `:::info` blocks → `st.info()`
- `:::success` blocks → `st.success()`
- Markdown tables → `st.table()`
- Regular markdown → `st.markdown()`

**Acceptance**: `render_qualitative_content('spy_retailirsa')` produces same visual output as current elif block.

---

## Phase 3: Page Refactoring (Tasks 7-15)

### Task 7: Refactor `navigation.py`

Replace `ANALYSES` dict with DB query:
```python
def get_analyses_dict():
    """Replaces ANALYSES dict — reads from SQLite."""
    return {a['id']: a for a in get_all_analyses()}
```

Update `render_sidebar()`, `render_breadcrumb()`, `get_analysis_title()` to use DB.
Format analysis names with Material Icons: `:material/{icon}: {name}`.

Keep backward compat: `ANALYSES = get_analyses_dict()` at module level (lazy init).

**Acceptance**: Sidebar shows all 9 analyses with Material Icons. No regressions.

---

### Task 8: Refactor `data_loader.py`

Replace `existing_files` dict and elif fallback chain with:
```python
def load_analysis_data(analysis_id):
    config = get_analysis_config(analysis_id)
    if config['id'] == 'investment_clock':
        return load_investment_clock_data()
    data_path = DATA_DIR / config['data_file']
    return pd.read_parquet(data_path)
```

Keep Investment Clock's specialized loader (it merges sector returns).

**Acceptance**: `load_analysis_data(id)` works for all 9 analyses. No regressions.

---

### Task 9: Refactor `Home.py`

Replace hardcoded cards list with DB query:
```python
analyses = get_all_analyses()
for analysis in sorted(analyses, key=lambda a: a.get('display_order', 100)):
    col = col1 if analysis['home_column'] == 1 else col2
    render_card(analysis)
```

Use Material Icons in card headers. Update analysis count dynamically.

**Acceptance**: Home page shows all 9 cards with Material Icons. Cards match current layout.

---

### Task 10: Refactor `Overview.py`

Replace elif chain (lines 42-110) with:
```python
cols = resolve_columns(analysis_id, data, context='default')
indicator_col = cols['indicator_col']
return_col = cols['return_col']
```

Keep all page logic after column resolution unchanged.

**Acceptance**: Overview page renders identically for all 9 analyses.

---

### Task 11: Refactor `Correlation.py`

Same pattern as Task 10: replace elif chain with `resolve_columns()`.

**Acceptance**: Correlation page renders identically for all 9 analyses.

---

### Task 12: Refactor `Lead_Lag.py`

Replace elif chain with `resolve_columns()`.
Use `lag_config` from DB for sidebar slider defaults:
```python
lag_config = cols.get('lag_config', {})
default_max = lag_config.get('max', 12)
max_lag = st.slider("Max Lag (months)", 6, 24, default_max)
```

**Acceptance**: Lead-Lag page renders identically. NewHomeSales still defaults to max_lag=24.

---

### Task 13: Refactor `Regimes.py`

Replace elif chain with `resolve_columns(analysis_id, data, context='trading')`.
Add dynamic lag slider:
```python
if lag_config['base_col']:
    selected_lag = st.slider("Indicator Lag", lag_config['min'], lag_config['max'], lag_config['default'])
    if selected_lag != 0:
        data['indicator_shifted'] = data[lag_config['base_col']].shift(selected_lag)
```

**Acceptance**: Regimes page renders identically at default lag. Slider works for lag exploration.

---

### Task 14: Refactor `Backtests.py`

Same pattern as Task 13: `resolve_columns(context='trading')` + lag slider.

**Acceptance**: Backtests page renders identically at default lag. Slider functional.

---

### Task 15: Refactor `Forecasts.py`

Replace elif chain with `resolve_columns(analysis_id, data, context='default')`.

**Acceptance**: Forecasts page renders identically for all 9 analyses.

---

## Phase 4: Qualitative Page (Task 16)

### Task 16: Refactor `Qualitative.py`

Replace 851-line elif chain with:
```python
from src.dashboard.content_renderer import render_qualitative_content

render_qualitative_content(analysis_id)
```

Entire file shrinks from 851 lines to ~30 lines.

**Acceptance**: Qualitative page renders identically for all 9 analyses. File < 40 lines.

---

## Phase 5: Enhancements (Tasks 17-18)

### Task 17: Add Composite 4-Regime Support

Enhance `resolve_columns()` to handle `analysis_type='composite'`:
- Query both primary and secondary axis configs
- Add `combine_regimes()` function to config_db.py
- Update Regimes page to show 4-phase grid when composite

Verify Investment Clock still works correctly after migration.

**Acceptance**: Investment Clock shows 4 phases. Adding a new composite via SQL INSERT works.

---

### Task 18: Docker Build + Integration Test

1. Rebuild Docker image
2. Verify HTTP 200 on health endpoint
3. Navigate all 9 analyses × 7 pages manually
4. Test lag slider on Regimes/Backtests
5. Verify Material Icons render correctly

**Acceptance**: Docker runs. All pages load without errors.

---

## Phase 6: Cleanup + Push (Tasks 19-20)

### Task 19: Remove Dead Code

- Delete old `ANALYSES` dict from navigation.py (if fully replaced)
- Delete `existing_files` dict from data_loader.py
- Remove backward-compat shims if no longer needed
- Delete temp test files

**Acceptance**: No dead code. All imports resolve. Docker still works.

---

### Task 20: Commit, Push, Update Status Board

- Commit all changes with descriptive message
- Push to remote
- Update `_pws/_team/status-board.md` with refactoring summary

**Acceptance**: Clean git status. Remote up to date.

---

## Dependency Graph

```
Task 1 (config_db) ──┬──→ Task 3 (resolve_columns) ──→ Task 4 (verify parity)
                     │                                      │
Task 2 (seed DB)  ───┘                                      │
                                                             ↓
Task 5 (extract content) ──→ Task 6 (renderer)     Tasks 7-15 (refactor pages)
                                    │                        │
                                    ↓                        ↓
                             Task 16 (Qualitative)    Task 17 (composite)
                                    │                        │
                                    └────────┬───────────────┘
                                             ↓
                                      Task 18 (Docker test)
                                             ↓
                                      Task 19 (cleanup)
                                             ↓
                                      Task 20 (push)
```

**Parallelizable**: Tasks 5-6 can run in parallel with Tasks 7-15.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| resolve_columns() doesn't match old behavior | Task 4 verifies parity before any page changes |
| Material Icons not supported in selectbox | Fall back to text-only labels |
| Qualitative markdown renderer misses edge cases | Visual comparison against screenshots |
| SQLite file missing at runtime | Auto-init on first access (creates DB + schema) |
| Investment Clock special handling breaks | Keep specialized loader, only change config source |
