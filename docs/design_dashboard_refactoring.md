# Dashboard Refactoring: SQLite-Driven Dynamic Architecture

**Date**: 2026-02-14
**Author**: RA Cheryl
**Status**: Design (Pending Implementation)

## 1. Problem Statement

The RLIC dashboard currently contains **64 elif branches** across 7 pages, with **87.5% being duplicated column-detection logic**. Adding a new analysis requires editing 8+ files (7 pages + navigation + data_loader + Home), inserting near-identical elif blocks in each. The Qualitative page alone is 851 lines of hardcoded content.

### Current Pain Points

| Problem | Impact |
|---------|--------|
| 64 elif branches across 7 pages | Every new analysis = 7 copy-paste blocks |
| Column detection duplicated 87.5% | Bug fixes must be applied in 7 places |
| Qualitative content hardcoded (851 lines) | Not reusable, not editable outside code |
| Navigation, data_loader, Home all have parallel dicts | 3 more files to update per analysis |
| No dynamic lag exploration | Lag is hardcoded in analysis scripts |
| No composite analysis support beyond Investment Clock | Can't combine arbitrary indicators into 4-regime analyses |

### Current Architecture (Before)

```
User adds new analysis → Edit 8+ files:
  navigation.py      → Add ANALYSES dict entry
  data_loader.py     → Add existing_files dict entry + elif fallback
  Home.py            → Add card tuple
  Overview.py        → Add elif column detection block
  Qualitative.py     → Add elif content block (50-100 lines)
  Correlation.py     → Add elif column detection block
  Lead_Lag.py        → Add elif column detection block
  Regimes.py         → Add elif column detection block (may differ)
  Backtests.py       → Add elif column detection block (may differ)
  Forecasts.py       → Add elif column detection block
```

### Target Architecture (After)

```
User adds new analysis → 3 steps:
  1. Run analysis script          → outputs parquet file
  2. INSERT rows into SQLite DB   → config for all pages
  3. Add qualitative .md file     → content for Qualitative page
  Dashboard picks it up automatically — zero code changes.
```

## 2. Database Schema

### 2.1 Table: `analyses`

Replaces: `navigation.py` ANALYSES dict, `data_loader.py` existing_files dict, `Home.py` cards list.

```sql
CREATE TABLE analyses (
    id TEXT PRIMARY KEY,                    -- e.g., 'spy_retailirsa'
    name TEXT NOT NULL,                     -- e.g., 'SPY vs Retail Inv/Sales'
    icon TEXT DEFAULT 'analytics',          -- Material Icon name (e.g., 'factory', 'inventory_2')
    short_name TEXT,                        -- e.g., 'SPY-Retail' (breadcrumb)
    description TEXT,                       -- sidebar caption
    caption TEXT,                           -- Home page card subtitle
    home_column INTEGER DEFAULT 1,         -- 1=left, 2=right on Home grid
    display_order INTEGER DEFAULT 100,     -- sort order on Home page
    analysis_type TEXT DEFAULT 'single',   -- 'single' | 'composite'
    target_ticker TEXT,                    -- e.g., 'SPY', 'XLI', 'XLRE'
    target_return_col TEXT,                -- e.g., 'XLI_Returns', 'SPY_return'
    data_file TEXT NOT NULL,               -- parquet filename in data/
    -- Composite only: 4-regime phase labels (JSON)
    phase_labels TEXT,                     -- e.g., {"RR":"Recovery","RF":"Overheat",...}
    -- Metadata
    created_at TEXT DEFAULT (datetime('now')),
    notes TEXT                             -- free-form notes
);
```

### 2.2 Table: `analysis_indicators`

Replaces: 64 elif column-detection branches across 7 pages.

```sql
CREATE TABLE analysis_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id TEXT NOT NULL,
    axis TEXT NOT NULL DEFAULT 'primary',   -- 'primary' | 'secondary' (composite)
    -- Column detection for display pages (Overview, Correlation, Lead-Lag, Forecasts)
    indicator_pattern TEXT,                 -- case-insensitive search (e.g., 'retail')
    indicator_columns TEXT,                 -- JSON array of exact col names
    indicator_filter TEXT,                  -- JSON: {"contains":["ISM_Mfg_PMI"], "and_contains":["Level","YoY"]}
    indicator_exclude TEXT,                 -- JSON array of suffix patterns to exclude (e.g., ["_return"])
    -- Column detection for trading pages (Regimes, Backtests)
    trading_columns TEXT,                   -- JSON array of exact cols for regime/backtest pages
    -- Return column detection
    return_columns TEXT,                    -- JSON array of exact return col names
    return_pattern TEXT,                    -- suffix pattern for return detection (e.g., '_return')
    price_column TEXT,                      -- price col for computing returns if needed (e.g., 'SPY')
    -- Columns to exclude from generic fallback
    exclude_from_detection TEXT,            -- JSON array of col names to skip (e.g., ["SPY","regime"])
    -- Dynamic lag support
    base_column TEXT,                       -- base indicator col for shift() (e.g., 'ISM_Mfg_PMI_Level')
    default_lag INTEGER DEFAULT 0,         -- best lag from analysis (used as slider default)
    lag_min INTEGER DEFAULT -12,           -- slider minimum
    lag_max INTEGER DEFAULT 12,            -- slider maximum
    -- Per-axis regime definition
    regime_method TEXT DEFAULT 'direction', -- 'direction' | 'threshold' | 'precomputed'
    regime_threshold REAL,                 -- for threshold method (e.g., 50.0 for PMI)
    regime_labels TEXT,                    -- JSON: {"above":"Expansion","below":"Contraction"}
                                           -- or {"rising":"Rising","falling":"Falling"}
    regime_source_col TEXT,                -- for precomputed regimes (e.g., 'Regime')
    -- Foreign key
    FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
);

-- Index for fast lookup
CREATE INDEX idx_analysis_indicators_analysis ON analysis_indicators(analysis_id, axis);
```

### 2.3 Qualitative Content: Markdown Files

Replaces: 851-line elif chain in `Qualitative.py`.

```
src/dashboard/content/
├── investment_clock.md
├── spy_retailirsa.md
├── spy_indpro.md
├── xlre_orders_inv.md
├── xlp_retailirsa.md
├── xly_retailirsa.md
├── xlre_newhomesales.md
├── xli_ism_mfg.md
└── xli_ism_svc.md
```

Each file uses standard markdown with optional Streamlit-compatible extensions:

```markdown
# XLI vs ISM Manufacturing PMI (NAPM)

## What is ISM Manufacturing PMI?

The **ISM Manufacturing PMI** is a monthly survey of 300+ manufacturing
purchasing managers...

:::warning
**Our Analysis Found**: ISM Manufacturing PMI does NOT predict XLI returns...
:::

## Key Research

| Finding | Source | Implication |
|---------|--------|-------------|
| ISM PMI leads IP by 1-2 months | Federal Reserve | PMI leads real economy but not equities |
```

The Qualitative page renders these files using a lightweight parser that converts
`:::warning`, `:::info`, `:::success` blocks to `st.warning()`, `st.info()`, `st.success()` calls,
and markdown tables to `st.table()`.

## 3. Column Resolution Logic

### 3.1 The `resolve_columns()` Function

A single function replaces all 64 elif branches. Lives in a new module: `src/dashboard/config_db.py`.

```python
def resolve_columns(
    analysis_id: str,
    data: pd.DataFrame,
    context: str = 'default'  # 'default' for display pages, 'trading' for Regimes/Backtests
) -> dict:
    """
    Resolve indicator and return columns for a given analysis.

    Returns:
        {
            'indicator_cols': ['ISM_Mfg_PMI_Level', 'ISM_Mfg_PMI_YoY'],
            'return_cols': ['XLI_Returns'],
            'indicator_col': 'ISM_Mfg_PMI_Level',  # primary
            'return_col': 'XLI_Returns',            # primary
            'regime_method': 'threshold',
            'regime_config': {...},
            'lag_config': {'default': 0, 'min': -12, 'max': 12, 'base_col': '...'},
        }
    """
```

### 3.2 Resolution Algorithm

```
1. Query analysis_indicators for analysis_id, axis='primary'
2. For indicator columns:
   a. If context='trading' AND trading_columns is set → use trading_columns
   b. Else if indicator_columns is set → use exact column names
   c. Else if indicator_filter is set → apply contains/and_contains filter
   d. Else if indicator_pattern is set → case-insensitive pattern match
   e. Apply indicator_exclude to filter out unwanted suffixes
   f. Fallback: all columns not in returns/regime/exclude lists
3. For return columns:
   a. If return_columns is set → use exact names
   b. Else if return_pattern is set → find cols matching suffix
   c. Else if price_column is set → compute returns from price
   d. Fallback: cols ending with '_return' or '_Returns'
4. Handle regime:
   a. If regime_method='precomputed' → copy regime_source_col to 'regime'
   b. If regime_method='direction' → use analysis_engine.define_regimes_direction()
   c. If regime_method='threshold' → use regime_threshold value
5. Return resolved dict
```

### 3.3 Composite Resolution (4-Regime)

For composite analyses, resolve_columns also queries `axis='secondary'`:

```
1. Resolve primary indicator (axis='primary') → binary signal A
2. Resolve secondary indicator (axis='secondary') → binary signal B
3. Combine: A × B → 4 phases using analyses.phase_labels
4. Each axis can have independent lag sliders
```

## 4. Refactored Page Architecture

### 4.1 Generic Page Template (Before vs After)

**Before** (each page, ~130 lines):
```python
# 60+ lines of elif chains for column detection
if analysis_id == 'spy_retailirsa':
    indicator_cols = [c for c in data.columns if 'retail' in c.lower() ...]
    ...
elif analysis_id == 'spy_indpro':
    ...
# ... 9 more elif blocks

# 70+ lines of actual page logic
```

**After** (each page, ~70 lines):
```python
from src.dashboard.config_db import resolve_columns

# One line replaces all elif chains
cols = resolve_columns(analysis_id, data, context='default')
indicator_col = cols['indicator_col']
return_col = cols['return_col']

# Page logic unchanged
```

### 4.2 Page-Specific Changes

| Page | Context | Special Handling |
|------|---------|-----------------|
| Overview | `default` | None |
| Qualitative | N/A | Loads markdown file, no column detection needed |
| Correlation | `default` | None |
| Lead-Lag | `default` | Uses `lag_config` for sidebar slider defaults |
| Regimes | `trading` | Uses `trading_columns`; dynamic lag slider from `lag_config` |
| Backtests | `trading` | Uses `trading_columns`; dynamic lag slider |
| Forecasts | `default` | None |

### 4.3 Home Page

**Before**: Hardcoded list of 9 card tuples.

**After**:
```python
analyses = get_all_analyses()  # reads from SQLite
for analysis in sorted(analyses, key=lambda a: a['display_order']):
    col = col1 if analysis['home_column'] == 1 else col2
    with col:
        render_analysis_card(analysis)
```

### 4.4 Navigation/Sidebar

**Before**: `ANALYSES` dict in `navigation.py`.

**After**: `get_all_analyses()` reads from SQLite, populates dropdown dynamically.

### 4.5 Data Loader

**Before**: `existing_files` dict + elif fallbacks in `load_analysis_data()`.

**After**:
```python
def load_analysis_data(analysis_id):
    config = get_analysis_config(analysis_id)  # from SQLite
    data_path = DATA_DIR / config['data_file']
    return pd.read_parquet(data_path)
```

## 5. Dynamic Lag Exploration

### 5.1 Current State

Lag values are hardcoded in analysis scripts. The `_Lagged` columns are pre-computed
and baked into parquet files. Changing the lag requires re-running the script.

### 5.2 New Capability

The Regimes and Backtests pages gain a **lag slider**:

```python
lag_config = cols['lag_config']
selected_lag = st.slider(
    "Indicator Lag (months)",
    min_value=lag_config['min'],
    max_value=lag_config['max'],
    value=lag_config['default']
)

# Apply lag dynamically
if selected_lag != 0:
    data[f'{base_col}_shifted'] = data[lag_config['base_col']].shift(selected_lag)
    indicator_col = f'{base_col}_shifted'
```

This lets users explore different lag values without re-running analysis scripts.
The `default_lag` value (from the analysis result) serves as the slider's initial position.

### 5.3 Composite Lag Exploration

For composite 4-regime analyses, each axis gets its own lag slider:

```
Primary axis: Orders/Inventories ratio, lag slider [-12, +12], default 0
Secondary axis: PPI, lag slider [-12, +12], default 0
```

Users can independently shift each indicator to find the optimal 4-regime timing.

## 6. Composite 4-Regime Analysis Support

### 6.1 Current State

Only `investment_clock` supports 4-regime (Recovery/Overheat/Stagflation/Reflation).
It is entirely hardcoded — adding another composite analysis requires new code.

### 6.2 New Capability

Any two indicators can be combined into a 4-regime analysis by:

1. Creating a merged parquet (analysis script joins both indicators + target)
2. Inserting an `analyses` row with `analysis_type='composite'` and `phase_labels`
3. Inserting 2 `analysis_indicators` rows (primary + secondary axes)

### 6.3 Example: XLI vs (ISM Mfg PMI + New Home Sales)

```sql
-- Analysis definition
INSERT INTO analyses (id, name, icon, analysis_type, target_ticker,
                      target_return_col, data_file, phase_labels)
VALUES ('xli_composite_ism_nhs', 'XLI: ISM Mfg + New Home Sales', '🔄', 'composite',
        'XLI', 'XLI_Returns', 'xli_composite_ism_nhs_full.parquet',
        '{"expansion+rising":"Strong Growth","expansion+falling":"Divergent A",
          "contraction+rising":"Divergent B","contraction+falling":"Weakness"}');

-- Primary axis: ISM Mfg PMI (threshold-based)
INSERT INTO analysis_indicators
    (analysis_id, axis, base_column, regime_method, regime_threshold,
     regime_labels, indicator_columns, return_columns, default_lag)
VALUES ('xli_composite_ism_nhs', 'primary', 'ISM_Mfg_PMI_Level', 'threshold', 50.0,
        '{"above":"Expansion","below":"Contraction"}',
        '["ISM_Mfg_PMI_Level"]', '["XLI_Returns"]', 0);

-- Secondary axis: New Home Sales (direction-based)
INSERT INTO analysis_indicators
    (analysis_id, axis, base_column, regime_method,
     regime_labels, indicator_columns, default_lag)
VALUES ('xli_composite_ism_nhs', 'secondary', 'NewHomeSales_YoY', 'direction',
        '{"rising":"Rising","falling":"Falling"}',
        '["NewHomeSales_YoY"]', 8);
```

### 6.4 Regime Combination Logic

```python
def combine_regimes(primary_regime, secondary_regime, phase_labels):
    """
    Combine two binary regime signals into 4 phases.

    primary_regime: Series of 'Expansion'/'Contraction' or 'Rising'/'Falling'
    secondary_regime: Series of 'Rising'/'Falling' etc.
    phase_labels: dict mapping "primary_label+secondary_label" to phase name
    """
    combined = primary_regime.astype(str) + '+' + secondary_regime.astype(str)
    # Map to phase labels using lowercase matching
    label_map = {k.lower(): v for k, v in phase_labels.items()}
    return combined.str.lower().map(label_map)
```

## 7. Database Location and Docker Integration

### 7.1 File Location

```
data/rlic_config.db    ← SQLite database
```

Sits alongside parquet files in the `data/` directory.

### 7.2 Docker Integration

Two options depending on deployment needs:

**Option A: Baked into image** (current approach extended)
```dockerfile
COPY data/ /app/data/
# DB file is part of the image, read-only
```

**Option B: Volume mount** (recommended for production)
```yaml
volumes:
  - ./data:/app/data    # DB + parquet files externalized
```

Option B enables updating analyses without rebuilding the Docker image.

### 7.3 Connection Management

```python
# src/dashboard/config_db.py
import sqlite3
from pathlib import Path
from functools import lru_cache

DB_PATH = Path(__file__).parent.parent.parent / "data" / "rlic_config.db"

def get_connection():
    """Get SQLite connection (cached per thread in Streamlit)."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row  # dict-like access
    return conn
```

## 8. Migration Strategy

### 8.1 Seed Script

`script/seed_config_db.py` populates the database from current hardcoded values:

```python
# 1. Create tables
# 2. Insert 9 analyses rows (from navigation.py ANALYSES dict)
# 3. Insert 9 analysis_indicators rows (from column detection patterns)
# 4. Extract qualitative content to markdown files
```

### 8.2 Migration Order

| Step | What | Risk |
|------|------|------|
| 1 | Create `config_db.py` with schema + resolve_columns() | None — new file |
| 2 | Create `seed_config_db.py` and populate DB | None — new file |
| 3 | Extract qualitative content to markdown files | None — new files |
| 4 | Create markdown renderer for Qualitative page | None — new file |
| 5 | Refactor `navigation.py` to read from DB | Medium — affects all pages |
| 6 | Refactor `data_loader.py` to read from DB | Medium — affects all pages |
| 7 | Refactor `Home.py` to read from DB | Low — single page |
| 8 | Refactor each data page to use resolve_columns() | Low per page — 6 pages |
| 9 | Refactor `Qualitative.py` to load markdown files | Low — single page |
| 10 | Add dynamic lag slider to Regimes/Backtests | Low — enhancement |
| 11 | Add composite analysis support | Low — enhancement |
| 12 | Docker test + cleanup | Low |

### 8.3 Backward Compatibility

During migration, both old (elif) and new (DB) paths can coexist:

```python
# Transitional pattern
try:
    cols = resolve_columns(analysis_id, data)
except Exception:
    # Fall back to old elif chain during migration
    cols = resolve_columns_legacy(analysis_id, data)
```

This allows incremental migration — one page at a time.

## 9. Seed Data

### 9.1 Analyses Table Seed Data

| id | name | icon | short_name | analysis_type | target_ticker | data_file |
|----|------|------|------------|---------------|---------------|-----------|
| investment_clock | Investment Clock Sectors | cycle | IC Sectors | composite | SPY | monthly_with_best_phases.parquet |
| spy_retailirsa | SPY vs Retail Inv/Sales | inventory_2 | SPY-Retail | single | SPY | spy_retail_inv_sales.parquet |
| spy_indpro | SPY vs Industrial Production | precision_manufacturing | SPY-INDPRO | single | SPY | spy_ip_analysis.parquet |
| xlre_orders_inv | XLRE vs Orders/Inventories | real_estate_agent | XLRE-O/I | single | XLRE | xlre_oi_analysis.parquet |
| xlp_retailirsa | XLP vs Retail Inv/Sales | shopping_basket | XLP-Retail | single | XLP | xlp_retail_inv_sales.parquet |
| xly_retailirsa | XLY vs Retail Inv/Sales | storefront | XLY-Retail | single | XLY | xly_retail_inv_sales.parquet |
| xlre_newhomesales | XLRE vs New Home Sales | home_work | XLRE-NHS | single | XLRE | xlre_newhomesales_full.parquet |
| xli_ism_mfg | XLI vs ISM Manufacturing PMI | factory | XLI-MFG | single | XLI | xli_ism_mfg_full.parquet |
| xli_ism_svc | XLI vs ISM Services PMI | corporate_fare | XLI-SVC | single | XLI | xli_ism_svc_full.parquet |

### 9.2 Analysis Indicators Seed Data

| analysis_id | axis | indicator_pattern | indicator_columns | trading_columns | return_columns | return_pattern | price_column | base_column | default_lag | regime_method | regime_threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| spy_retailirsa | primary | retail | null | null | null | _return | SPY | retail_inv_sales | 0 | direction | null |
| spy_indpro | primary | indpro\|industrial | null | null | null | _return | SPY | industrial_prod | 0 | direction | null |
| xlre_orders_inv | primary | order\|oi | null | null | null | _return | XLRE | orders_inv_ratio | 0 | direction | null |
| xlp_retailirsa | primary | retail | null | null | null | _return | XLP | retail_inv_sales | 0 | direction | null |
| xly_retailirsa | primary | retail | null | null | null | _return | XLY | retail_inv_sales | 0 | direction | null |
| xlre_newhomesales | primary | null | ["NewHomeSales_Level","NewHomeSales_YoY"] | ["NewHomeSales_YoY_Lagged"] | ["XLRE_Returns"] | null | null | NewHomeSales_YoY | 8 | precomputed | null |
| xli_ism_mfg | primary | null | ["ISM_Mfg_PMI_Level","ISM_Mfg_PMI_YoY"] | ["ISM_Mfg_PMI_Level_Lagged"] | ["XLI_Returns"] | null | null | ISM_Mfg_PMI_Level | 0 | threshold | 50.0 |
| xli_ism_svc | primary | null | ["ISM_Svc_PMI_Level","ISM_Svc_PMI_YoY"] | ["ISM_Svc_PMI_Level_Lagged"] | ["XLI_Returns"] | null | null | ISM_Svc_PMI_Level | 0 | threshold | 50.0 |
| investment_clock | primary | null | ["orders_inv_ratio"] | null | null | _return | null | orders_inv_ratio | 0 | direction | null |
| investment_clock | secondary | null | ["ppi_all"] | null | null | null | null | ppi_all | 0 | direction | null |

## 10. Icon System: Material Icons

### 10.1 Current State

All icons are emoji characters (📈, 🏪, 🏭, etc.) hardcoded in the ANALYSES dict.
Emoji icons look informal and inconsistent across operating systems.

### 10.2 New Approach: Streamlit Native Material Icons

Streamlit supports [Google Material Symbols](https://fonts.google.com/icons) natively
via the `:material/icon_name:` syntax. No additional dependencies required.

Usage in Streamlit:
```python
st.markdown(":material/trending_up: Investment Clock")
st.button(":material/analytics: Select", icon=":material/analytics:")
```

### 10.3 Icon Mapping

The `analyses.icon` column stores Material Icon names instead of emoji:

| Analysis | Current (Emoji) | New (Material Icon) | Icon Name |
|----------|-----------------|---------------------|-----------|
| Investment Clock | 📈 | :material/cycle: | `cycle` |
| SPY vs RETAILIRSA | 🏪 | :material/inventory_2: | `inventory_2` |
| SPY vs INDPRO | 🏭 | :material/precision_manufacturing: | `precision_manufacturing` |
| XLRE vs Orders/Inv | 🏠 | :material/real_estate_agent: | `real_estate_agent` |
| XLP vs RETAILIRSA | 🛒 | :material/shopping_basket: | `shopping_basket` |
| XLY vs RETAILIRSA | 🛍️ | :material/storefront: | `storefront` |
| XLRE vs New Home Sales | 🏡 | :material/home_work: | `home_work` |
| XLI vs ISM Mfg PMI | 🏭 | :material/factory: | `factory` |
| XLI vs ISM Svc PMI | 🏢 | :material/corporate_fare: | `corporate_fare` |

### 10.4 Usage in Navigation Sidebar

```python
# Format function for analysis selector
def format_analysis(analysis_id):
    config = get_analysis_config(analysis_id)
    icon = config['icon']  # e.g., 'factory'
    return f":material/{icon}: {config['name']}"

selected = st.selectbox(
    "Focus Analysis",
    options=analysis_ids,
    format_func=format_analysis
)
```

### 10.5 Usage in Home Cards

```python
for analysis in analyses:
    with st.container(border=True):
        st.markdown(f"### :material/{analysis['icon']}: {analysis['name']}")
        st.markdown(analysis['description'])
        st.caption(analysis['caption'])
```

### 10.6 Page Navigation Icons

Streamlit page files use emoji in filenames (required by framework). These remain
unchanged since they are structural, not analysis-specific:

```
pages/2_📊_Overview.py      ← framework requirement, unchanged
pages/3_📖_Qualitative.py   ← framework requirement, unchanged
...
```

## 11. New Files Created

| File | Purpose |
|------|---------|
| `src/dashboard/config_db.py` | DB connection, schema init, resolve_columns(), get_all_analyses() |
| `script/seed_config_db.py` | Populate DB from current hardcoded values |
| `src/dashboard/content/*.md` | 9 qualitative markdown files |
| `src/dashboard/content_renderer.py` | Markdown → Streamlit widget renderer |
| `data/rlic_config.db` | SQLite database file |

## 11. Files Modified

| File | Change |
|------|--------|
| `src/dashboard/navigation.py` | Replace ANALYSES dict with DB query |
| `src/dashboard/data_loader.py` | Replace existing_files dict with DB query |
| `src/dashboard/Home.py` | Replace hardcoded cards with DB query |
| `src/dashboard/pages/2_📊_Overview.py` | Replace elif chains with resolve_columns() |
| `src/dashboard/pages/3_📖_Qualitative.py` | Replace elif chain with markdown file loader |
| `src/dashboard/pages/4_📈_Correlation.py` | Replace elif chains with resolve_columns() |
| `src/dashboard/pages/5_🔄_Lead_Lag.py` | Replace elif chains with resolve_columns() |
| `src/dashboard/pages/6_🎯_Regimes.py` | Replace elif chains with resolve_columns() + lag slider |
| `src/dashboard/pages/7_💰_Backtests.py` | Replace elif chains with resolve_columns() + lag slider |
| `src/dashboard/pages/8_🔮_Forecasts.py` | Replace elif chains with resolve_columns() |

## 12. Estimated Effort

| Phase | Tasks | Lines Changed |
|-------|-------|---------------|
| Phase 1: Foundation | config_db.py, seed script, DB creation | ~400 new |
| Phase 2: Content extraction | 9 markdown files + renderer | ~1000 new (mostly content) |
| Phase 3: Page refactoring | 6 data pages + navigation + data_loader + Home | ~800 removed, ~200 new |
| Phase 4: Enhancements | Dynamic lag slider, composite support | ~200 new |
| **Total** | | **~1800 new, ~800 removed** |

Net effect: **~1000 lines fewer** in the codebase, plus new capabilities.

## 13. Testing Strategy

1. **Before refactoring**: Screenshot every page for every analysis (9 × 7 = 63 screenshots)
2. **After each page**: Compare visually — output must be identical
3. **Docker test**: Rebuild image, verify HTTP 200 on all pages
4. **New analysis test**: Add a dummy analysis via SQL INSERT only — verify it appears on all pages
5. **Lag slider test**: Verify dynamic lag produces same results as pre-computed lag columns
6. **Composite test**: Verify Investment Clock still shows 4 phases correctly after migration
